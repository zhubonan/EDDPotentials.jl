#=
Routine for working with NN

Allow using forward autodiff for training small scale neutron networks with direct minimisation.
Direct minimisation require accessing individual predcitions and the jacobian matrix.
In which case, N >> M, so backward autodiff is not efficent.
The standard ForwardDiff interface has drawback in that the NN has to be recreated everytime.
Here, the ¬`Dual` type is used directly for computing the gradients instead, with reduced overheads.


=#
using ForwardDiff
using Flux

"""
Update the parameters of a model from a given vector
"""
function update_param!(model, param)
    fparams = Flux.params(model)
    @assert sum(length, fparams) == length(param)
    i = 1
    for item in fparams
        l = length(item)
        item[:] .= param[i:i+l-1]
        i += l
    end
    model
end

function make_matrix(param, mat, offset=1)
    l = length(mat)
    reshape(param[offset:offset+l - 1], size(mat))
end

"""
Return a vector containing all parameters of a model
"""
function paramvector(model)
    i = 1
    fparam = Flux.params(model)
    np = sum(length, fparam)
    out = zeros(eltype(fparam[1]), np)
    for item in fparam
        l = length(item)
        out[i: i+l-1] .= item[:]
        i += l
    end
    out
end

"""
Create a copy of the model with parameters replaced as duals
"""
function dualize_model(model::Chain; duals::Vector{T}=dualize(paramvector(model))) where {T}
    i = 1
    nets = []
    for layer in model
        wt::Matrix{T} = make_matrix(duals, layer.weight, i)
        i += length(wt)
        bias::Vector{T} = make_matrix(duals, layer.bias, i)
        i += length(bias)
        layer_ = Dense(wt, bias, layer.σ)
        push!(nets, layer_)
    end
    Chain(nets...)
end

"""
Collect jacobian from output as duals
"""
function collect_jacobian!(jac, out)
    for (i, o) in enumerate(out)
        for (j, p) in enumerate(o.partials)
            jac[i, j] = p
        end
    end
    jac
end

function collect_jacobian(out::Vector{ForwardDiff.Dual{N, T, M}}) where {N, T, M}
    jac = zeros(T, length(out), length(out[1].partials))
    collect_jacobian!(jac, out)
end

function jac!(j, dm, p, cfg; x0)
    pduals = ForwardDiff.seed!(cfg.duals, p, cfg.seeds);
    update_param!(dm, pduals)
    collect_jacobian!(j, dm(x0))
end

get_jacobiancfg(p) = ForwardDiff.JacobianConfig(nothing, p, ForwardDiff.Chunk{length(p)}());

dualize(x::Vector;cfg=ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())) = ForwardDiff.seed!(cfg.duals, x, cfg.seeds)

"""
    setup_autodiff(model)

Return components for autodiff operation -  a tuple of (dm, cfg, p1)
"""
function setup_autodiff(model)
    p0 = paramvector(model)
    cfg = get_jacobiancfg(p0)
    pdual = dualize(p0;cfg)
    dm = dualize_model(model, pdual)
    dm, cfg, p0
end

"""
Setup the function for computing jacobian for each element of the data (N, M) matrix
"""
function setup_jacobian_func(model, data)
    dm, cfg, p0 = setup_autodiff(model)
    g!(j, p) = jac!(j, dm, p, cfg;x0=data)
    return g!, dm, p0
end

"""
    setup_atomic_energy_jacobian(;model, dm, data, cfg, total=reduce(hcat,data))

Setup the function for computing the jacobian of the mean atomic energy of each frame.
"""
function setup_atomic_energy_jacobian(;model, dm, x, cfg, total=reduce(hcat,x))
    function inner!(j, p)
        pduals = ForwardDiff.seed!(cfg.duals, p, cfg.seeds);
        update_param!(dm, pduals)
        update_param!(model, p)
        all_E = dm(total)
        out = zeros(eltype(all_E), length(x))
        ct = 1
        for i in 1:length(out)
            lv = size(x[i], 2)
            val = 0.
            for j = ct:ct+lv -1
                val += all_E[j]
            end
            out[i] = val / lv
            ct += lv
        end
        collect_jacobian!(j, out)
    end
    return inner!
end

"""
    setup_atomic_energy_diff(;model, x, y, total=reduce(hcat,x))

Setup the function for evaluting atomic energy
"""
function setup_atomic_energy_diff(;model, x, y, total=reduce(hcat,x))
    function inner!(f, p;)
        update_param!(model, p)
        ct = 1
        all_E = model(total)
        ct = 1
        for i in 1:length(f)
            lv = size(x[i], 2)
            f[i] = mean(all_E[ct:ct+lv-1]) - y[i]
            ct += lv
        end
        f
    end
    inner!
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
