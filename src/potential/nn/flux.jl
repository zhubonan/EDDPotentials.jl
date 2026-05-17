#=
Interface using standard Flux + Zygote
=#
using Flux

"""
Standard Flux interface wrapping a `model` object.

This interface uses Zygote for AD. 
Performance is not ideal for training using the LM method due to the accumulated overheads of
calling the `gradient` function.
It should be OK to use for inference (but still inferior compared to `ManualFluxBackPropInterface`).
"""
mutable struct FluxInterface{T,N} <: AbstractNNInterface
    model::T
    params::N
    inp::Any
    pullback_p::Any
    pullback_inp::Any
    xt::Any
    yt::Any
    training_mode::Bool
end

function Base.show(io::IO, m::MIME"text/plain", x::FluxInterface)
    println(io, "FluxInterface(")
    Base.show(io, m, x.model)
    print(io, "\n)")
end

get_flux_model(itf::FluxInterface) = itf.model

FluxInterface(model) = FluxInterface(
    model,
    Flux.trainables(model),
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    true,
)


function forward!(itf::FluxInterface, inp)
    itf.inp = inp
    # Reset the pullbacks
    itf.pullback_p = nothing
    itf.pullback_inp = nothing
    out = itf.model(transformed_inp(itf, inp))
    if !isnothing(itf.yt)
        reconstruct!(itf.yt, out)
    end
    out
end

paramvector(itf::FluxInterface) = vcat([vec(x) for x in itf.params]...)

function paramvector!(vec, itf::FluxInterface)
    vec .= paramvector(itf)
end

nparams(itf::FluxInterface) = sum(length, itf.params)

function setparamvector!(itf::FluxInterface, param)
    i = 1
    for elem in itf.params
        elem[:] .= param[i:i+length(elem)-1]
        i += length(elem)
    end
end

function gradinp!(gvec, itf::FluxInterface, inp=itf.inp)
    forward!(itf, inp)
    backward!(itf; computed_grad_inp=true, compute_grad_param=false)
    if itf.yt === nothing
        y_bar = 1
    else
        y_bar = itf.yt.scale[1]
    end
    grad, = itf.pullback_inp(y_bar)
    gvec .= grad
end

function gradparam!(gvec, itf::FluxInterface, inp=itf.inp)
    forward!(itf, inp)
    backward!(itf; computed_grad_inp=false, compute_grad_param=true)
    # Check if the results are standardised
    if itf.yt === nothing
        y_bar = 1
    else
        y_bar = itf.yt.scale[1]
    end
    # Extract flat gradient from the model gradient structure
    grad_model = itf.pullback_p[1]
    grad_flat, _ = Flux.destructure(grad_model)
    gvec .= y_bar .* grad_flat
    gvec
end

function (itf::FluxInterface)(inp)
    forward!(itf, inp)
end

"""
Run the backward step - this creates the pull back functions
"""
function backward!(
    itf::FluxInterface,
    args...;
    compute_grad_param=true,
    computed_grad_inp=true,
    kwargs...,
)
    if (compute_grad_param || itf.training_mode) && (itf.pullback_p === nothing)
        itf.pullback_p = Flux.gradient(m -> sum(m(itf.inp)), itf.model)
    end
    if computed_grad_inp && (itf.pullback_inp === nothing)
        _, itf.pullback_inp = Flux.pullback(x -> sum(itf.model(x)), itf.inp)
    end
end
