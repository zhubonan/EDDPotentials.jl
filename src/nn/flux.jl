#=
Interface using standard Flux + Zygote
=#
using Flux



mutable struct FluxInterface{T, N} <: AbstractNNInterface
    model::T
    params::N
    inp
end


FluxInterface(model) = FluxInterface(model, Flux.params(model), nothing)

function forward!(itf::FluxInterface, inp)
    out = itf.model(inp)
    itf.inp = inp
    out
end

paramvector(itf::FluxInterface) = vcat([vec(x) for x in itf.params]...)

function paramvector!(vec, itf::FluxInterface) 
    vec.= paramvector(itf)
end

nparams(itf::FluxInterface) = sum(length, itf.params)

function setparamvector!(itf::FluxInterface, param)  
    i = 1
    for elem in itf.params
        elem[:] .= param[i:i+length(elem) - 1]
        i += length(elem)
    end
end

function gradinp!(gvec, itf::FluxInterface, inp=itf.inp)
    grad, = Flux.gradient(x -> sum(itf.model(x)), inp)
    gvec .= grad
end

function gradparam!(gvec, itf::FluxInterface, inp=itf.inp)
    grad = Flux.gradient(() -> sum(itf.model(inp)), itf.params)
    i = 1
    for elem in grad.params
        g = grad.grads[elem]
        gvec[i:i+size(g, 2)-1] .= g[:]
        i += length(g)
    end
    gvec
end

function (itf::FluxInterface)(inp)
    itf.model(inp)
end

"No nothing - as the gradients calculated with gradparam! and gradinp!"
function backward!(itf::FluxInterface, args...;kwargs...) end