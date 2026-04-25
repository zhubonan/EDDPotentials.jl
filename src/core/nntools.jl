#=
Routine for working with NN

Allow using forward autodiff for training small scale neutron networks with direct minimisation.
Direct minimisation require accessing individual predictions and the jacobian matrix.
=#
using Random
using Flux


"""
    setup_fg_backprop(model, data::AbstractVector, y)

Get a function for computing both the objective function and the 
jacobian matrix for the **mean atomic energy**.
"""
function setup_fg_backprop(model, data::AbstractVector, y)
    batch_sizes = unique(size.(data, 2))
    gbuffers = ChainGradients.(Ref(model), batch_sizes)
    function fg!(fvec, jmat, param)
        update_param!(model, param)
        if !(jmat === nothing)
            compute_jacobian_backprop!(jmat, gbuffers, model, data)
        end
        if !(fvec === nothing)
            compute_diff_with_forward!(fvec, gbuffers, model, data, y)
            return fvec
        end
    end
    return fg!
end


"""
    compute_jacobian_backprop!(jacmat, gbuffs, model, data::AbstractVector)

Compute jacobian of the NN weights/bias using back-propagation.
"""
function compute_jacobian_backprop!(jacmat, gbuffs, model, data::AbstractVector)
    g1 = zeros(eltype(jacmat), size(jacmat, 2))
    for (i, inp) in enumerate(data)
        sinp = size(inp, 2)
        # Find the buffer of the right shape
        gbuff = gbuffs[findfirst(x -> x.n == sinp, gbuffs)]
        forward!(gbuff, model, inp)
        backward!(gbuff, model)
        collect_gradients!(g1, gbuff)
        # Copy data to the jacobian matrix
        for j = 1:size(jacmat, 2)
            jacmat[i, j] = g1[j]
        end
    end
    jacmat
end

raw"""
    compute_diff_with_forward!(f, gbuffs, model, data::AbstractVector, y)

Compute the objective function ($y' - y_ref$).
"""
function compute_diff_with_forward!(f, gbuffs, model, data::AbstractVector, y)
    for (i, inp) in enumerate(data)
        sinp = size(inp, 2)
        # Find the buffer of the right shape
        gbuff = gbuffs[findfirst(x -> x.n == sinp, gbuffs)]
        out = forward!(gbuff, model, inp)
        f[i] = mean(out) - y[i]
    end
    f
end

"""
    setup_fj(model::AbstractNNInterface, data::AbstractVector, y)

Setup the function returning the residual and the jacobian matrix.
"""
function setup_fj(model::AbstractNNInterface, data::AbstractVector, y, weights=nothing)
    jtmp = similar(paramvector(model))
    function fj!(fvec, jmat, param)
        setparamvector!(model, param)
        # Compute the gradients
        compute_objectives_diff(fvec, jmat, model, data, y, weights; jtmp)
        fvec, jmat
    end
    function f!(fvec, param)
        setparamvector!(model, param)
        compute_objectives(fvec, model, data, y, weights)
        fvec
    end
    function j!(jmat, param)
        fj!(nothing, jmat, param)[2]
    end
    return f!, j!, fj!
end


"""
    setup_fj_batched(model::AbstractNNInterface, data, y, weights=nothing)

Setup closures for batched LM training.  Returns a tuple
`(fj_batch!, f_full!)` where:

- `fj_batch!(fvec, jmat, param, batch_indices)` computes the residual and
  Jacobian for only the data points in `batch_indices`.  `fvec` or `jmat`
  may be `nothing`.
- `f_full!(fvec, param)` computes the residual for the full dataset.
"""
function setup_fj_batched(
    model::AbstractNNInterface,
    data,
    y,
    weights=nothing,
)
    jtmp = similar(paramvector(model))

    function fj_batch!(fvec, jmat, param, batch_indices)
        setparamvector!(model, param)
        need_jacobian = !(jmat === nothing)
        for (local_i, global_i) in enumerate(batch_indices)
            inp = data[global_i]
            out = forward!(model, inp)

            if need_jacobian
                backward!(model)
                gradparam!(jtmp, model)
                if isnothing(weights)
                    jmat[local_i, :] .= jtmp
                else
                    jmat[local_i, :] .= jtmp .* weights[global_i]
                end
            end

            if !(fvec === nothing)
                if isnothing(weights)
                    fvec[local_i] = sum(out) - y[global_i]
                else
                    fvec[local_i] = (sum(out) - y[global_i]) * weights[global_i]
                end
            end
        end
        fvec, jmat
    end

    function f_full!(fvec, param)
        setparamvector!(model, param)
        compute_objectives(fvec, model, data, y, weights)
        fvec
    end

    return fj_batch!, f_full!
end


"""
    ThreadBufferPool

Pre-allocated pool of gradient buffers indexed by **chunk index**, not
`threadid()`.  This is safe with `:dynamic` scheduling because each iteration
of `Threads.@threads` processes a unique element of the iteration space.

One `ChainGradients` and one `jtmp` vector are stored per chunk slot.
The pool lazily creates `ChainGradients` of the right batch size on first use
per slot.
"""
struct ThreadBufferPool
    chain::Chain
    gchains::Vector{Vector}
    jtmps::Vector{Vector}
    yt_scale::Float64
    yt::Any  # ZScoreTransform or nothing — used for reconstruct! on output
    xt::Any  # ZScoreTransform or nothing — used for transform! on input
    ngps::Int  # granularity for _chunk_ranges — controls chunk count

    function ThreadBufferPool(model::AbstractNNInterface, nslots::Int; ngps::Int=10)
        chain = get_flux_model(model)
        T = eltype(paramvector(model))
        np = nparams(model)
        gchains = [Vector{ChainGradients}() for _ in 1:nslots]
        jtmps = [zeros(T, np) for _ in 1:nslots]
        yt = model isa ManualFluxBackPropInterface ? model.yt : nothing
        yt_scale = !isnothing(yt) ? yt.scale[1] : one(T)
        xt = model isa ManualFluxBackPropInterface ? model.xt : nothing
        new(chain, gchains, jtmps, yt_scale, yt, xt, ngps)
    end
end

function _get_pool_gchain!(pool::ThreadBufferPool, ci::Int, bsize::Int)
    bufs = pool.gchains[ci]
    idx = findfirst(x -> x.n == bsize, bufs)
    if isnothing(idx)
        cg = ChainGradients(pool.chain, bsize)
        push!(bufs, cg)
        idx = length(bufs)
    end
    bufs[idx]
end

"""
Apply input transform (xt) if present, then forward + backward + gradparam.
Uses pre-allocated ChainGradients from the pool.  Returns the output (with yt
inverse transform applied if present).
"""
function _threaded_forward_grad!(jtmp_, gchain, pool::ThreadBufferPool, inp)
    if !isnothing(pool.xt)
        # Apply ZScore transform to the last xt.len rows
        # forward! will copy our transformed input into gchain.layers[1].x internally
        transformed = copy(inp)
        transform!(pool.xt, @view(transformed[end-pool.xt.len+1:end, :]))
        out = forward!(gchain, pool.chain, transformed)
    else
        out = forward!(gchain, pool.chain, inp)
    end
    # Apply yt inverse transform to put output in physical units
    if !isnothing(pool.yt)
        reconstruct!(pool.yt, out)
    end
    backward!(gchain, pool.chain)
    gradparam!(jtmp_, gchain)
    if pool.yt_scale != 1
        jtmp_ .*= pool.yt_scale
    end
    out
end

"""
Apply input transform (xt) if present, then forward only.
Returns the output (with yt inverse transform applied if present).
"""
function _threaded_forward(gchain, pool::ThreadBufferPool, inp)
    out = if !isnothing(pool.xt)
        transformed = copy(inp)
        transform!(pool.xt, @view(transformed[end-pool.xt.len+1:end, :]))
        forward!(gchain, pool.chain, transformed)
    else
        forward!(gchain, pool.chain, inp)
    end
    if !isnothing(pool.yt)
        reconstruct!(pool.yt, out)
    end
    out
end

"""
    setup_fj_batched_threaded(model, data, y, weights=nothing; ngps=10)

Threaded variant of `setup_fj_batched`.  Uses a `ThreadBufferPool` with
pre-allocated per-chunk gradient buffers instead of `deepcopy` on every call.
Safe with `:dynamic` scheduling — buffers are indexed by chunk index, not
`threadid()`.

Threads write to disjoint rows of `jmat` and `fvec`, so no synchronisation
is needed.  Falls back to the sequential path when `nthreads() == 1` or the
batch is too small to benefit from threading.
"""
function setup_fj_batched_threaded(
    model::AbstractNNInterface,
    data,
    y,
    weights=nothing;
    ngps=10,
)
    jtmp = similar(paramvector(model))  # used by sequential fallback only
    nslots = nthreads() * ngps + 1
    pool = ThreadBufferPool(model, nslots)

    function fj_batch!(fvec, jmat, param, batch_indices)
        setparamvector!(model, param)
        need_jacobian = !(jmat === nothing)
        nt = nthreads()

        if nt > 1 && length(batch_indices) > nt * 5
            # Threaded path: chunk the batch indices across tasks
            batch_vec = collect(batch_indices)
            chunks = _chunk_ranges(length(batch_vec); ngps)
            Threads.@threads for ci in eachindex(chunks)
                jtmp_ = pool.jtmps[ci]
                for (local_i, gi) in enumerate(chunks[ci])
                    global_i = batch_vec[gi]
                    inp = data[global_i]
                    gchain = _get_pool_gchain!(pool, ci, size(inp, 2))

                    if need_jacobian
                        out = _threaded_forward_grad!(jtmp_, gchain, pool, inp)
                        if isnothing(weights)
                            jmat[gi, :] .= jtmp_
                        else
                            jmat[gi, :] .= jtmp_ .* weights[global_i]
                        end
                        if !(fvec === nothing)
                            if isnothing(weights)
                                fvec[gi] = sum(out) - y[global_i]
                            else
                                fvec[gi] = (sum(out) - y[global_i]) * weights[global_i]
                            end
                        end
                    else
                        out = _threaded_forward(gchain, pool, inp)
                        if !(fvec === nothing)
                            if isnothing(weights)
                                fvec[gi] = sum(out) - y[global_i]
                            else
                                fvec[gi] = (sum(out) - y[global_i]) * weights[global_i]
                            end
                        end
                    end
                end
            end
        else
            # Sequential fallback
            for (local_i, global_i) in enumerate(batch_indices)
                inp = data[global_i]
                out = forward!(model, inp)

                if need_jacobian
                    backward!(model)
                    gradparam!(jtmp, model)
                    if isnothing(weights)
                        jmat[local_i, :] .= jtmp
                    else
                        jmat[local_i, :] .= jtmp .* weights[global_i]
                    end
                end

                if !(fvec === nothing)
                    if isnothing(weights)
                        fvec[local_i] = sum(out) - y[global_i]
                    else
                        fvec[local_i] = (sum(out) - y[global_i]) * weights[global_i]
                    end
                end
            end
        end
        fvec, jmat
    end

    function f_full!(fvec, param)
        setparamvector!(model, param)
        compute_objectives_threaded(fvec, pool, data, y, weights)
        fvec
    end

    return fj_batch!, f_full!, pool
end


function compute_objectives(f, itf, data, y, weights=nothing)
    for (i, inp) in enumerate(data)
        out = forward!(itf, inp)
        if isnothing(weights)
            f[i] = (sum(out) - y[i])
        else
            f[i] = (sum(out) - y[i]) * weights[i]
        end
    end
    f
end

function compute_objectives_threaded(f, pool::ThreadBufferPool, data, y, weights=nothing)
    nt = nthreads()
    if nt > 1 && length(data) > nt * 5
        chunks = _chunk_ranges(length(data); ngps=pool.ngps)
        Threads.@threads for ci in eachindex(chunks)
            for i in chunks[ci]
                gchain = _get_pool_gchain!(pool, ci, size(data[i], 2))
                out = _threaded_forward(gchain, pool, data[i])
                if isnothing(weights)
                    f[i] = sum(out) - y[i]
                else
                    f[i] = (sum(out) - y[i]) * weights[i]
                end
            end
        end
    else
        for (i, inp) in enumerate(data)
            gchain = _get_pool_gchain!(pool, 1, size(inp, 2))
            out = _threaded_forward(gchain, pool, inp)
            if isnothing(weights)
                f[i] = sum(out) - y[i]
            else
                f[i] = (sum(out) - y[i]) * weights[i]
            end
        end
    end
    f
end

function compute_objectives_diff(
    f,
    jmat,
    itf,
    data,
    y,
    weights=nothing,
    ;
    jtmp=jmat[1, :],
    ngps=10,
)
    nt = nthreads()
    if nt > 1 && div(length(data), nt) > 10
        _compute_objectives_diff_threaded(f, jmat, itf, data, y, weights; jtmp, ngps)
    else
        for (i, inp) in enumerate(data)
            out = forward!(itf, inp)
            backward!(itf)
            gradparam!(jtmp, itf)
            if isnothing(weights)
                jmat[i, :] .= jtmp
            else
                jmat[i, :] .= jtmp .* weights[i]
            end
            isnothing(f) || (f[i] = sum(out) - y[i])
        end
    end
    jmat
end

"""
Threaded forward-only prediction of energies for RMSE computation.
Returns predictions vector (energies in physical space, i.e. with yt inverse applied).
"""
function _predict_energy_threaded(pool::ThreadBufferPool, data)
    n = length(data)
    preds = Vector{Float64}(undef, n)
    nt = nthreads()
    if nt > 1 && n > nt * 5
        chunks = _chunk_ranges(n; ngps=pool.ngps)
        Threads.@threads for ci in eachindex(chunks)
            for i in chunks[ci]
                gchain = _get_pool_gchain!(pool, ci, size(data[i], 2))
                out = _threaded_forward(gchain, pool, data[i])
                preds[i] = sum(out)
            end
        end
    else
        for (i, inp) in enumerate(data)
            gchain = _get_pool_gchain!(pool, 1, size(inp, 2))
            out = _threaded_forward(gchain, pool, inp)
            preds[i] = sum(out)
        end
    end
    preds
end

function rmse_per_atom_threaded(pool::ThreadBufferPool, data, y, nat)
    preds = _predict_energy_threaded(pool, data)
    return sqrt(mean(((preds .- y) ./ nat) .^ 2))
end

"""
    _chunk_ranges(ndata; ngps=1)

Divide a range into multiple chunks such that each thread receives up to `ngps` chunks in total.
"""
function _chunk_ranges(ndata; ngps=1)
    nt = nthreads()
    ndivide = nt * ngps
    ningroup = div(ndata, ndivide)
    if ningroup == 0
        out = [i:i for i = 1:ndata]
        @assert sum(length, out) == ndata
        return out
    end
    out = UnitRange{Int64}[]
    i = 1
    for _ = 1:ndivide
        push!(out, i:i+ningroup-1)
        i += ningroup
    end
    m = mod(ndata, ndivide)
    if m != 0
        push!(out, ndata-m+1:ndata)
    end
    @assert sum(length, out) == ndata
    out
end


function _compute_objectives_diff_threaded(
    f,
    jmat,
    itf,
    data::AbstractVector,
    y,
    weights=nothing;
    jtmp=jmat[1, :],
    ngps=10,
    pool::Union{Nothing,ThreadBufferPool}=nothing,
)
    if pool === nothing
        nslots = nthreads() * ngps + 1
        pool = ThreadBufferPool(itf, nslots)
    end
    chunks = _chunk_ranges(length(data); ngps)
    Threads.@threads for ci in eachindex(chunks)
        jtmp_ = pool.jtmps[ci]
        for i in chunks[ci]
            inp = data[i]
            gchain = _get_pool_gchain!(pool, ci, size(inp, 2))
            out = _threaded_forward_grad!(jtmp_, gchain, pool, inp)
            if isnothing(weights)
                jmat[i, :] .= jtmp_
            else
                jmat[i, :] .= jtmp_ .* weights[i]
            end
            isnothing(f) || (f[i] = sum(out) - y[i])
        end
    end
    jmat
end



"""
    collect_gradients!(gvec::AbstractVector, gbuff::ChainGradients)

Collect the gradients after back-propagration into a vector
"""
function collect_gradients!(gvec::AbstractVector, gbuff::ChainGradients)
    i = 1
    for layer in gbuff.layers
        _, i = _collect_gradient!(gvec, layer, i)
    end
    gvec
end

function _collect_gradient!(gvec, source::AbstractVecOrMat, i)
    for j in axes(source, 2), k in axes(source, 1)
        gvec[i] = source[k, j]
        i += 1
    end
    gvec, i
end

"""
    _collect_gradient!(gvec, g::DenseGradient, i)

Collect gradient from a DenseGradient layer
"""
function _collect_gradient!(gvec, g::DenseGradient, i)
    _, i = _collect_gradient!(gvec, g.gw, i)
    _, i = _collect_gradient!(gvec, g.gb, i)
    gvec, i
end

"""
    _collect_gradient!(gvec, g::CellEmbeddingGradient, i)

Collect gradient from a CellEmbeddingGradient layer
"""
function _collect_gradient!(gvec, g::CellEmbeddingGradient, i)
    _, i = _collect_gradient!(gvec, g.two_body.gw, i)
    _, i = _collect_gradient!(gvec, g.three_body.gw, i)
    gvec, i
end

"""
    atomic_rmse(f, x, y, yt)

Compute per atom based RMSE

Args:
* f: prediction function
* x: input data 
* y: reference data 
* yt: Normalisation transformation originally applied to obtain y.
"""
function atomic_rmse(pred, y, yt)
    y = StatsBase.reconstruct(yt, y)
    rec = StatsBase.reconstruct(yt, pred)
    sqrt(Flux.mse(rec, y))
end



# For reinitialising the weights
function reinit!(chain::Chain, init=ginit)
    for layer in chain.layers
        reinit!(layer, init)
    end
    chain
end

#=
Learnable embedding training closures.

These closures handle the combined parameter vector [nn_params; vec(embedding)]
and recompute features each LM step, producing a Jacobian with columns for both
NN and embedding parameters.
=#

"""
    setup_fj_with_embedding(model, structures, cf, y, weights=nothing;
        nmax=500, recompute_xt=false)

Create closures for full LM training with learnable species embedding.

The combined parameter vector is `[nn_params; vec(cf.embedding)]`.
"""
function setup_fj_with_embedding(
    model::AbstractNNInterface,
    structures::AbstractVector{<:Cell},
    cf::RandomEmbeddedCellFeature,
    y::AbstractVector,
    weights=nothing;
    nmax::Int=500,
    recompute_xt::Bool=false,
)
    n_nn = nparams(model)
    n_emb = length(cf.embedding)
    n_total = n_nn + n_emb

    jtmp_nn = zeros(eltype(paramvector(model)), n_nn)
    gE = zeros(eltype(cf.embedding), size(cf.embedding))

    # Precompute NLs and workspaces for each structure
    rcut = suggest_rcut(cf; shell=0.0)
    nls = [NeighbourList(s, rcut, nmax; savevec=true) for s in structures]
    workspaces = [get_workspace(cf, nl, false) for nl in nls]

    # Allocate a gv buffer large enough for the largest structure
    max_nat = maximum(natoms, structures)
    gv_mat = zeros(eltype(cf.embedding), nfeatures(cf), max_nat)

    function _update_params(param)
        nn_params = view(param, 1:n_nn)
        emb_params = view(param, n_nn+1:n_total)
        setparamvector!(model, nn_params)
        copyto!(cf.embedding, emb_params)
    end

    function _recompute_and_forward(i)
        @timeit to "embed: feature" recompute_features!(workspaces[i].fvec, cf, structures[i], workspaces[i], nls[i])
        @timeit to "embed: forward" forward!(model, workspaces[i].fvec)
    end

    function _update_xt()
        nf = nfeatures(cf)
        mean_vec = zeros(eltype(cf.embedding), nf)
        std_vec = zeros(eltype(cf.embedding), nf)
        compute_feature_stats!(mean_vec, std_vec, structures, cf, workspaces, nls)
        model.xt = ZScoreTransform(nf, 2, mean_vec, std_vec)
    end

    function f!(fvec, param)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        for i in eachindex(structures)
            out = _recompute_and_forward(i)
            if isnothing(weights)
                fvec[i] = sum(out) - y[i]
            else
                fvec[i] = (sum(out) - y[i]) * weights[i]
            end
        end
        fvec
    end

    function j!(jmat, param)
        fj!(nothing, jmat, param)[2]
    end

    function fj!(fvec, jmat, param)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        for i in eachindex(structures)
            out = _recompute_and_forward(i)
            @timeit to "embed: backward" backward!(model)

            if jmat !== nothing
                # NN parameter gradient
                @timeit to "embed: gradparam" gradparam!(jtmp_nn, model)
                if isnothing(weights)
                    jmat[i, 1:n_nn] .= jtmp_nn
                else
                    jmat[i, 1:n_nn] .= jtmp_nn .* weights[i]
                end

                # Embedding gradient
                nat = natoms(structures[i])
                gv_view = view(gv_mat, 1:nfeatures(cf), 1:nat)
                @timeit to "embed: gradinp" gradinp!(gv_view, model)
                fill!(gE, zero(eltype(gE)))
                @timeit to "embed: grad_emb" grad_embedding!(gE, cf, structures[i], nls[i], gv_view)
                if isnothing(weights)
                    jmat[i, n_nn+1:n_total] .= vec(gE)
                else
                    jmat[i, n_nn+1:n_total] .= vec(gE) .* weights[i]
                end
            end

            if fvec !== nothing
                if isnothing(weights)
                    fvec[i] = sum(out) - y[i]
                else
                    fvec[i] = (sum(out) - y[i]) * weights[i]
                end
            end
        end
        fvec, jmat
    end

    return f!, j!, fj!
end

"""
    setup_fj_with_embedding(model, structures, cf::MultiRandomEmbeddedCellFeature, y, weights=nothing;
        nmax=500, recompute_xt=false)

Create closures for full LM training with learnable multi-embedding.

The combined parameter vector is `[nn_params; vec(emb1.embedding); vec(emb2.embedding); ...]`.
"""
function setup_fj_with_embedding(
    model::AbstractNNInterface,
    structures::AbstractVector{<:Cell},
    cf::MultiRandomEmbeddedCellFeature,
    y::AbstractVector,
    weights=nothing;
    nmax::Int=500,
    recompute_xt::Bool=false,
)
    n_nn = nparams(model)
    n_emb_total = sum(length(emb.embedding) for emb in cf.embeddings)
    n_total = n_nn + n_emb_total

    jtmp_nn = zeros(eltype(paramvector(model)), n_nn)
    gE_vec = [zeros(eltype(emb.embedding), size(emb.embedding)) for emb in cf.embeddings]

    # Precompute NLs and workspaces for each structure
    rcut = suggest_rcut(cf; shell=0.0)
    nls = [NeighbourList(s, rcut, nmax; savevec=true) for s in structures]
    workspaces = [get_workspace(cf, nl, false) for nl in nls]

    max_nat = maximum(natoms, structures)
    gv_mat = zeros(eltype(gE_vec[1]), nfeatures(cf), max_nat)

    # Build index ranges for each embedding's parameters
    emb_ranges = Vector{UnitRange{Int}}(undef, length(cf.embeddings))
    pos = n_nn + 1
    for (i, emb) in enumerate(cf.embeddings)
        n_emb_i = length(emb.embedding)
        emb_ranges[i] = pos:(pos + n_emb_i - 1)
        pos += n_emb_i
    end

    function _update_params(param)
        nn_params = view(param, 1:n_nn)
        setparamvector!(model, nn_params)
        for (i, emb) in enumerate(cf.embeddings)
            copyto!(emb.embedding, view(param, emb_ranges[i]))
        end
    end

    function _recompute_and_forward(i)
        @timeit to "embed: feature" recompute_features!(workspaces[i].fvec, cf, structures[i], workspaces[i], nls[i])
        @timeit to "embed: forward" forward!(model, workspaces[i].fvec)
    end

    function _update_xt()
        nf = nfeatures(cf)
        T = eltype(gE_vec[1])
        mean_vec = zeros(T, nf)
        std_vec = zeros(T, nf)
        compute_feature_stats!(mean_vec, std_vec, structures, cf, workspaces, nls)
        model.xt = ZScoreTransform(nf, 2, mean_vec, std_vec)
    end

    function f!(fvec, param)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        for i in eachindex(structures)
            out = _recompute_and_forward(i)
            if isnothing(weights)
                fvec[i] = sum(out) - y[i]
            else
                fvec[i] = (sum(out) - y[i]) * weights[i]
            end
        end
        fvec
    end

    function j!(jmat, param)
        fj!(nothing, jmat, param)[2]
    end

    function fj!(fvec, jmat, param)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        for i in eachindex(structures)
            out = _recompute_and_forward(i)
            @timeit to "embed: backward" backward!(model)

            if jmat !== nothing
                # NN parameter gradient
                @timeit to "embed: gradparam" gradparam!(jtmp_nn, model)
                if isnothing(weights)
                    jmat[i, 1:n_nn] .= jtmp_nn
                else
                    jmat[i, 1:n_nn] .= jtmp_nn .* weights[i]
                end

                # Embedding gradient for each embedding
                nat = natoms(structures[i])
                gv_view = view(gv_mat, 1:nfeatures(cf), 1:nat)
                @timeit to "embed: gradinp" gradinp!(gv_view, model)
                for (k, emb) in enumerate(cf.embeddings)
                    fill!(gE_vec[k], zero(eltype(gE_vec[k])))
                end
                @timeit to "embed: grad_emb" grad_embedding!(gE_vec, cf, structures[i], nls[i], gv_view)
                for (k, emb) in enumerate(cf.embeddings)
                    if isnothing(weights)
                        jmat[i, emb_ranges[k]] .= vec(gE_vec[k])
                    else
                        jmat[i, emb_ranges[k]] .= vec(gE_vec[k]) .* weights[i]
                    end
                end
            end

            if fvec !== nothing
                if isnothing(weights)
                    fvec[i] = sum(out) - y[i]
                else
                    fvec[i] = (sum(out) - y[i]) * weights[i]
                end
            end
        end
        fvec, jmat
    end

    return f!, j!, fj!
end

"""
    setup_fj_batched_with_embedding(model, structures, cf, y, weights=nothing;
        nmax=500, recompute_xt=false)

Create closures for batched LM training with learnable species embedding.

Returns `(fj_batch!, f_full!)` where `fj_batch!` operates on batch indices
and `f_full!` evaluates the full dataset.
"""
function setup_fj_batched_with_embedding(
    model::AbstractNNInterface,
    structures::AbstractVector{<:Cell},
    cf::RandomEmbeddedCellFeature,
    y::AbstractVector,
    weights=nothing;
    nmax::Int=500,
    recompute_xt::Bool=false,
)
    n_nn = nparams(model)
    n_emb = length(cf.embedding)
    n_total = n_nn + n_emb

    jtmp_nn = zeros(eltype(paramvector(model)), n_nn)
    gE = zeros(eltype(cf.embedding), size(cf.embedding))

    # Precompute NLs and workspaces for each structure
    rcut = suggest_rcut(cf; shell=0.0)
    nls = [NeighbourList(s, rcut, nmax; savevec=true) for s in structures]
    workspaces = [get_workspace(cf, nl, false) for nl in nls]

    max_nat = maximum(natoms, structures)
    gv_mat = zeros(eltype(cf.embedding), nfeatures(cf), max_nat)

    function _update_params(param)
        nn_params = view(param, 1:n_nn)
        emb_params = view(param, n_nn+1:n_total)
        setparamvector!(model, nn_params)
        copyto!(cf.embedding, emb_params)
    end

    function _recompute_and_forward(i)
        @timeit to "embed: feature" recompute_features!(workspaces[i].fvec, cf, structures[i], workspaces[i], nls[i])
        @timeit to "embed: forward" forward!(model, workspaces[i].fvec)
    end

    function _update_xt()
        nf = nfeatures(cf)
        mean_vec = zeros(eltype(cf.embedding), nf)
        std_vec = zeros(eltype(cf.embedding), nf)
        compute_feature_stats!(mean_vec, std_vec, structures, cf, workspaces, nls)
        model.xt = ZScoreTransform(nf, 2, mean_vec, std_vec)
    end

    function fj_batch!(fvec, jmat, param, batch_indices)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        need_jacobian = !(jmat === nothing)

        for (local_i, global_i) in enumerate(batch_indices)
            out = _recompute_and_forward(global_i)

            if need_jacobian
                @timeit to "embed: backward" backward!(model)
                @timeit to "embed: gradparam" gradparam!(jtmp_nn, model)
                if isnothing(weights)
                    jmat[local_i, 1:n_nn] .= jtmp_nn
                else
                    jmat[local_i, 1:n_nn] .= jtmp_nn .* weights[global_i]
                end

                nat = natoms(structures[global_i])
                gv_view = view(gv_mat, 1:nfeatures(cf), 1:nat)
                @timeit to "embed: gradinp" gradinp!(gv_view, model)
                fill!(gE, zero(eltype(gE)))
                @timeit to "embed: grad_emb" grad_embedding!(gE, cf, structures[global_i], nls[global_i], gv_view)
                if isnothing(weights)
                    jmat[local_i, n_nn+1:end] .= vec(gE)
                else
                    jmat[local_i, n_nn+1:end] .= vec(gE) .* weights[global_i]
                end
            end

            if fvec !== nothing
                if isnothing(weights)
                    fvec[local_i] = sum(out) - y[global_i]
                else
                    fvec[local_i] = (sum(out) - y[global_i]) * weights[global_i]
                end
            end
        end
        fvec, jmat
    end

    function f_full!(fvec, param)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        for i in eachindex(structures)
            out = _recompute_and_forward(i)
            if isnothing(weights)
                fvec[i] = sum(out) - y[i]
            else
                fvec[i] = (sum(out) - y[i]) * weights[i]
            end
        end
        fvec
    end

    return fj_batch!, f_full!
end

function setup_fj_batched_with_embedding(
    model::AbstractNNInterface,
    structures::AbstractVector{<:Cell},
    cf::MultiRandomEmbeddedCellFeature,
    y::AbstractVector,
    weights=nothing;
    nmax::Int=500,
    recompute_xt::Bool=false,
)
    n_nn = nparams(model)
    n_emb = _emb_param_count(cf)
    n_total = n_nn + n_emb

    jtmp_nn = zeros(eltype(paramvector(model)), n_nn)
    gE_vec = [zeros(eltype(emb.embedding), size(emb.embedding)) for emb in cf.embeddings]

    # Precompute NLs and workspaces for each structure
    rcut = suggest_rcut(cf; shell=0.0)
    nls = [NeighbourList(s, rcut, nmax; savevec=true) for s in structures]
    workspaces = [get_workspace(cf, nl, false) for nl in nls]

    max_nat = maximum(natoms, structures)
    gv_mat = zeros(eltype(gE_vec[1]), nfeatures(cf), max_nat)

    # Build index ranges for each embedding's parameters
    emb_ranges = Vector{UnitRange{Int}}(undef, length(cf.embeddings))
    pos = n_nn + 1
    for (i, emb) in enumerate(cf.embeddings)
        n_emb_i = length(emb.embedding)
        emb_ranges[i] = pos:(pos + n_emb_i - 1)
        pos += n_emb_i
    end

    function _update_params(param)
        nn_params = view(param, 1:n_nn)
        setparamvector!(model, nn_params)
        for (i, emb) in enumerate(cf.embeddings)
            copyto!(emb.embedding, view(param, emb_ranges[i]))
        end
    end

    function _recompute_and_forward(i)
        @timeit to "embed: feature" recompute_features!(workspaces[i].fvec, cf, structures[i], workspaces[i], nls[i])
        @timeit to "embed: forward" forward!(model, workspaces[i].fvec)
    end

    function _update_xt()
        nf = nfeatures(cf)
        T = eltype(gE_vec[1])
        mean_vec = zeros(T, nf)
        std_vec = zeros(T, nf)
        compute_feature_stats!(mean_vec, std_vec, structures, cf, workspaces, nls)
        model.xt = ZScoreTransform(nf, 2, mean_vec, std_vec)
    end

    function fj_batch!(fvec, jmat, param, batch_indices)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        need_jacobian = !(jmat === nothing)

        for (local_i, global_i) in enumerate(batch_indices)
            out = _recompute_and_forward(global_i)

            if need_jacobian
                @timeit to "embed: backward" backward!(model)
                @timeit to "embed: gradparam" gradparam!(jtmp_nn, model)
                if isnothing(weights)
                    jmat[local_i, 1:n_nn] .= jtmp_nn
                else
                    jmat[local_i, 1:n_nn] .= jtmp_nn .* weights[global_i]
                end

                nat = natoms(structures[global_i])
                gv_view = view(gv_mat, 1:nfeatures(cf), 1:nat)
                @timeit to "embed: gradinp" gradinp!(gv_view, model)
                for (k, emb) in enumerate(cf.embeddings)
                    fill!(gE_vec[k], zero(eltype(gE_vec[k])))
                end
                @timeit to "embed: grad_emb" grad_embedding!(gE_vec, cf, structures[global_i], nls[global_i], gv_view)
                for (k, emb) in enumerate(cf.embeddings)
                    if isnothing(weights)
                        jmat[local_i, emb_ranges[k]] .= vec(gE_vec[k])
                    else
                        jmat[local_i, emb_ranges[k]] .= vec(gE_vec[k]) .* weights[global_i]
                    end
                end
            end

            if fvec !== nothing
                if isnothing(weights)
                    fvec[local_i] = sum(out) - y[global_i]
                else
                    fvec[local_i] = (sum(out) - y[global_i]) * weights[global_i]
                end
            end
        end
        fvec, jmat
    end

    function f_full!(fvec, param)
        _update_params(param)
        if recompute_xt
            _update_xt()
        end
        for i in eachindex(structures)
            out = _recompute_and_forward(i)
            if isnothing(weights)
                fvec[i] = sum(out) - y[i]
            else
                fvec[i] = (sum(out) - y[i]) * weights[i]
            end
        end
        fvec
    end

    return fj_batch!, f_full!
end

# --- Embedding parameter helpers (used by training.jl) ---

_emb_param_count(cf::RandomEmbeddedCellFeature) = length(cf.embedding)
_emb_param_count(cf::MultiRandomEmbeddedCellFeature) = sum(length(emb.embedding) for emb in cf.embeddings)

_emb_param_vec(cf::RandomEmbeddedCellFeature) = vec(cf.embedding)
_emb_param_vec(cf::MultiRandomEmbeddedCellFeature) = reduce(vcat, vec(emb.embedding) for emb in cf.embeddings)

function _emb_param_set!(cf::RandomEmbeddedCellFeature, params)
    copyto!(cf.embedding, params)
end
function _emb_param_set!(cf::MultiRandomEmbeddedCellFeature, params)
    pos = firstindex(params)
    for emb in cf.embeddings
        n = length(emb.embedding)
        copyto!(emb.embedding, view(params, pos:pos+n-1))
        pos += n
    end
end

function _emb_normalize!(cf::RandomEmbeddedCellFeature)
    for i in 1:size(cf.embedding, 1)
        rownorm = sqrt(sum(abs2, @view cf.embedding[i, :]))
        rownorm > 0 && (@views cf.embedding[i, :] ./= rownorm)
    end
end
function _emb_normalize!(cf::MultiRandomEmbeddedCellFeature)
    for emb in cf.embeddings
        _emb_normalize!(emb)
    end
end


function reinit!(layer::Dense, init=ginit)
    layer.weight .= init(size(layer.weight)...)
    layer.bias .= init(size(layer.bias)...)
    layer
end

function reinit!(itf::ManualFluxBackPropInterface, init=ginit)
    reinit!(itf.chain, init)
    itf
end

reinit(itf::AbstractNNInterface, init=ginit) = reinit!(deepcopy(itf), init)

function reinit!(itf::FluxInterface, init=ginit)
    reinit!(itf.model, init)
    itf
end

function reinit!(be::BodyEmbedding, init=ginit)
    be.weight .= init(size(be.weight)...)
    be
end

function reinit!(ce::CellEmbedding, init=ginit)
    reinit!(ce.two_body, init)
    reinit!(ce.three_body, init)
    ce
end
