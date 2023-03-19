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


function compute_objectives(f, itf, data::AbstractVector, y, weights=nothing)
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

function compute_objectives_diff(
    f,
    jmat,
    itf,
    data::AbstractVector,
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
)
    chunks = _chunk_ranges(length(data); ngps)
    Threads.@threads for idx in chunks
        # Thread local copy
        itf_ = deepcopy(itf)
        jtmp_ = copy(jtmp)
        for i in idx
            inp = data[i]
            out = forward!(itf_, inp)
            backward!(itf_)
            gradparam!(jtmp_, itf_)
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



if VERSION >= v"1.7"
    @doc """
        default_rng_value()
    Create an instance of the default RNG depending on Julia's version.
    - Julia version is < 1.7: `Random.GLOBAL_RNG`
    - Julia version is >= 1.7: `Random.default_rng()`
    """
    default_rng_value() = Random.default_rng()
else
    default_rng_value() = Random.GLOBAL_RNG
end


function glorot_uniform_f64(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = Float64(gain) * sqrt(24.0f0 / sum(Flux.nfan(dims...)))
    (rand(rng, Float32, dims...) .- 0.5f0) .* scale
end
glorot_uniform_f64(dims::Integer...; kw...) =
    glorot_uniform_f64(default_rng_value(), dims...; kw...)
glorot_uniform_f64(rng::AbstractRNG=default_rng_value(); init_kwargs...) =
    (dims...; kwargs...) -> glorot_uniform_f64(rng, dims...; init_kwargs..., kwargs...)

const ginit = glorot_uniform_f64


# For reinitialising the weights
function reinit!(chain::Chain, init=ginit)
    for layer in chain.layers
        reinit!(layer, init)
    end
    chain
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
