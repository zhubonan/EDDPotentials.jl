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
        # Scale the gradient for mean atomic energy rather than the total energy
        g1 ./= sinp
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


function setup_fj(model::AbstractNNInterface, data::AbstractVector, y)
    jtmp = similar(paramvector(model))
    function fj!(fvec, jmat, param)
        setparamvector!(model, param)
        # Compute the gradients
        compute_objectives_diff(fvec, jmat, model, data, y;jtmp)
        fvec, jmat
    end
    function f!(fvec, param)
        setparamvector!(model, param)
        compute_objectives(fvec, model, data, y)
        fvec
    end
    function j!(jmat, param)
        fj!(nothing, jmat, param)[2]
    end
    return f!, j!, fj!
end


function compute_objectives(f, itf, data::AbstractVector, y)
    for (i, inp) in enumerate(data)
        out = forward!(itf, inp)
        f[i] = sum(out) - y[i]
    end
    f
end

function compute_objectives_diff(f, jmat, itf, data::AbstractVector, y;jtmp = jmat[1, :])
    for (i, inp) in enumerate(data)
        out = forward!(itf, inp)
        backward!(itf)
        gradparam!(jtmp, itf)
        jmat[i, :] .= jtmp
        isnothing(f) || (f[i] = sum(out) - y[i])
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
        for j = 1:size(layer.gw, 2), k = 1:size(layer.gw, 1)
            gvec[i] = layer.gw[k, j]
            i += 1
        end
        for j = 1:size(layer.gb, 2), k = 1:size(layer.gb, 1)
            gvec[i] = layer.gb[k, j]
            i += 1
        end
    end
    gvec
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
glorot_uniform_f64(dims::Integer...; kw...) = glorot_uniform_f64(default_rng_value(), dims...; kw...)
glorot_uniform_f64(rng::AbstractRNG=default_rng_value(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform_f64(rng, dims...; init_kwargs..., kwargs...)

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

reinit(itf::ManualFluxBackPropInterface, init=ginit) = reinit!(deepcopy(itf), init)
