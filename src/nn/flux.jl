#=
Interface using standard Flux + Zygote
=#
using Flux

mutable struct FluxInterface{T, N} <: AbstractNNInterface
    model::T
    params::N
    inp
    pullback_p
    pullback_inp
end


FluxInterface(model) = FluxInterface(model, Flux.params(model), nothing, nothing, nothing)

function forward!(itf::FluxInterface, inp)
    itf.inp = inp
    itf.model(inp)
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
    if inp != itf.inp
        forward!(itf, inp)
    end
    grad,  = itf.pullback_inp(1)
    gvec .= grad
end

function gradparam!(gvec, itf::FluxInterface, inp=itf.inp)
    if inp != itf.inp
        forward!(itf, inp)
    end
    grad = itf.pullback_p(1)
    i = 1
    for elem in grad.params
        g = grad.grads[elem]
        gvec[i:i+size(g, 2)-1] .= g[:]
        i += length(g)
    end
    gvec
end

function (itf::FluxInterface)(inp)
    forward!(itf, inp)
end

"No nothing - as the gradients calculated with gradparam! and gradinp!"
function backward!(itf::FluxInterface, args...;kwargs...) 
    out, itf.pullback_inp = Flux.pullback(x -> sum(itf.model(x)), itf.inp)
    out, itf.pullback_p = Flux.pullback(() -> sum(itf.model(itf.inp)), itf.params)
    out
end