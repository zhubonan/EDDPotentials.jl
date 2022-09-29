#=
Interface for Neutron network implementations
=#
using StatsBase: ZScoreTransform, fit, transform!, reconstruct!

abstract type AbstractNNInterface end

"Return the gradient against the parameters"
function gradparam!(gvec::AbstractVector, itf::AbstractNNInterface) end

"Return the gradient against the inputs"
function gradinp!(gvec::AbstractVecOrMat, itf::AbstractNNInterface) end

"Do the forward step"
function forward!(itf::AbstractNNInterface, inp::AbstractArray) end

"Do the backward step"
function backward!(itf::AbstractNNInterface) end

"Return a vector containing the parameters"
function paramvector(itf::AbstractNNInterface) end

"Return a vector containing the parameters"
function paramvector!(vec::AbstractVector, itf::AbstractNNInterface) end

"Return the number of parameters"
function nparams(itf::AbstractNNInterface) end

"Set the parameters with a vector"
function setparamvector!(itf::AbstractNNInterface, vec::AbstractVector) end

function gradinp!(gvec::AbstractVector, itf::AbstractNNInterface, inp::AbstractArray) 
    forward!(itf, inp)
    gradinp!(gvec, itf)
end

function gradparam!(gvec::AbstractVecOrMat, itf::AbstractNNInterface, inp::AbstractArray) 
    forward!(itf, inp)
    gradparam!(gvec, itf)
end

## Standardisation

include("backprop.jl")

#=
Interface for DenseGradient implementation 
=#

struct ManualFluxBackPropInterface{T, G, Z} <: AbstractNNInterface
    chain::T
    gchain::ChainGradients{G}
    xt::Z
    yt::Z
end

ManualFluxBackPropInterface(chain::Chain, n::Int) = ManualFluxBackPropInterface(chain, ChainGradients(chain, n), nothing, nothing)

function forward!(itf::ManualFluxBackPropInterface, inp)
    # Apply x transformation
    if !isnothing(itf.xt)
        transform!(itf.xt, @view(inp[end-nl:end, :]))
        out = forward!(itf.gchain, itf.chain, inp)
    end
    # Apply y transformation
    if !isnothing(itf.yt)
        reconstruct!(itf.yt, out)
    end
    out
end

function gradparam!(gvec::AbstractVector, itf::ManualFluxBackPropInterface)
    backward!(itf)
    grad = collect_gradients!(gvec, itf.gchain)
    if !isnothing(itf.yt)
        grad .*= itf.yt.scale
    end
    grad
end

"""
Return the gradient of the input matrix ``X`` against the sum of the output ``sum(G(X))``.
"""
function gradinp!(gvec::AbstractVecOrMat, itf::ManualFluxBackPropInterface)
    backward!(itf)
    # Collect 
    input_gradient(itf.gchain.layers[1])
    gvec .= input_gradient
    # If transform is applied then we have to scale the gradient
    if !isnothing(itf.xt)
        nl = length(itf.xt.scale)
        gvec[end-nl:end, :] ./= xt.scale
    end
    if !isnothing(itf.yt)
        gvec .*= itf.yt.scale
    end
    gvec
end

backward!(itf::ManualFluxBackPropInterface;kwargs...) = backward!(itf.gchain, itf.chain;kwargs...)


paramvector(itf::ManualFluxBackPropInterface) = paramvector(itf.chain)
paramvector!(vec, itf::ManualFluxBackPropInterface) = paramvector!(vec, itf.chain)


function setparamvector!(itf::ManualFluxBackPropInterface, vec::AbstractVector)
    update_param!(itf.chain, vec)
end

nparams(itf::ManualFluxBackPropInterface) = nparams(itf.chain)