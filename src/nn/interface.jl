#=
Interface for Neutron network implementations
=#

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

include("backprop.jl")

#=
Interface for DenseGradient implementation 
=#

struct ManualFluxBackPropInterface{T, G}
    chain::T
    gchain::ChainGradients{G}
end

ManualFluxBackPropInterface(chain::Chain, n::Int) = ManualFluxBackPropInterface(chain, ChainGradients(chain, n))

forward!(itf::ManualFluxBackPropInterface, inp) = forward!(itf.gchain, itf.chain, inp)

function gradparam!(gvec::AbstractVector, itf::ManualFluxBackPropInterface)
    backward!(itf)
    collect_gradients!(gvec, itf.gchain)
end

function gradinp!(gvec::AbstractVecOrMat, itf::ManualFluxBackPropInterface)
    backward!(itf)
    # Collect 
    input_gradient(itf.gchain.layers[1])
end

backward!(itf::ManualFluxBackPropInterface;kwargs...) = backward!(itf.gchain, itf.chain;kwargs...)


paramvector(itf::ManualFluxBackPropInterface) = paramvector(itf.chain)
paramvector!(vec, itf::ManualFluxBackPropInterface) = paramvector!(vec, itf.chain)


function setparamvector!(itf::ManualFluxBackPropInterface, vec::AbstractVector)
    update_param!(itf.chain, vec)
end

nparams(itf::ManualFluxBackPropInterface) = nparams(itf.chain)