#=
Interface for Neutron network implementations
=#
using JLD2
using Random
using Flux
using StatsBase: ZScoreTransform, fit, transform!, reconstruct!

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

"""
    glorot_uniform(T::Type{<:AbstractFloat}=Float64; gain=1)

Return a Glorot uniform initializer function that produces arrays of element type `T`.
The returned function has the signature `dims::Integer... -> AbstractArray{T}`.
"""
glorot_uniform(; gain::Real=1) = glorot_uniform(Float64; gain)
glorot_uniform(T::Type{<:AbstractFloat}; gain::Real=1) =
    (dims::Integer...) -> _glorot_uniform(T, default_rng_value(), dims; gain)
glorot_uniform(rng::AbstractRNG; gain::Real=1) = glorot_uniform(Float64, rng; gain)
glorot_uniform(T::Type{<:AbstractFloat}, rng::AbstractRNG; gain::Real=1) =
    (dims::Integer...) -> _glorot_uniform(T, rng, dims; gain)

function _glorot_uniform(T::Type{<:AbstractFloat}, rng::AbstractRNG, dims; gain::Real=1)
    scale = T(gain) * sqrt(T(24) / T(sum(Flux.nfan(dims...))))
    (rand(rng, T, dims...) .- T(0.5)) .* scale
end

# Legacy alias — now correctly produces Float64
const glorot_uniform_f64 = glorot_uniform(Float64)
const ginit = glorot_uniform_f64

abstract type AbstractNNInterface end

"Return a vector of the gradients of the total energy against the parameters."
function gradparam! end

"Return the gradient of the total energy against the inputs."
function gradinp! end

"Do the forward step"
function forward! end

"Do the backward step"
function backward! end

"Return a vector containing the parameters"
function paramvector end

"Return a vector containing the parameters"
function paramvector! end

"Return the number of parameters"
function nparams end

"Set the parameters with a vector"
function setparamvector! end

## Serialization

"""
    save_as_jld2(str::AbstractString, obj::AbstractNNInterface)

Save an trained interface object as JLD2.
"""
function save_as_jld2(str::AbstractString, obj::AbstractNNInterface)
    jldopen(str, "w") do f
        save_as_jld2(f, obj)
    end
end

"""
    load_from_jld2(str::AbstractString, t::Type{<:AbstractNNInterface})

Load saved interface from JLD2 file.
"""
function load_from_jld2(str::AbstractString, t::Type{<:AbstractNNInterface})
    jldopen(str) do f
        load_from_jld2(f, t)
    end
end


"""
Obtain an baseline MLP model
"""
function flux_mlp_model(
    T::Type{<:AbstractFloat},
    cf::AbstractCellFeatureSpec,
    nodes...;
    init=glorot_uniform(T),
    σ=leaky_tanh,
    embedding=nothing,
    σs=nothing,
)

    if embedding === nothing
        ninp = nfeatures(cf)
    else
        # Adding embedding layer
        fsizes = feature_size(cf)
        ninp = fsizes[1]
        ninp += nfeatures(cf.two_body[1]) * num_embed(embedding.two_body)
        ninp += nfeatures(cf.three_body[1]) * num_embed(embedding.three_body)
    end
    if isnothing(σs)
        input = Dense(ninp => nodes[1], σ; init)
    else
        input = Dense(ninp => nodes[1], σs[1]; init)
    end

    if embedding !== nothing
        # Adding embedding layer
        layers = Any[embedding, input]
    else
        layers = Any[input]
    end
    # Adding dense layers
    i = 1
    while i < length(nodes)
        i += 1
        if isnothing(σs)
            push!(layers, Dense(nodes[i-1] => nodes[i], σ; init))
        else
            push!(layers, Dense(nodes[i-1] => nodes[i], σs[i]; init))
        end
    end
    push!(layers, Dense(nodes[i] => 1; init))
    Chain(layers...)
end

flux_mlp_model(cf::AbstractCellFeatureSpec, nodes...; kwargs...) =
    flux_mlp_model(Float64, cf, nodes...; kwargs...)

"""
Obtain an transformed the input array
"""
function transformed_inp(itf, inp)
    if itf.xt !== nothing
        inptmp = copy(inp)
        transform!(itf.xt, @view(inptmp[end-itf.xt.len+1:end, :]))
        return inptmp
    else
        return inp
    end
end

## Standardisation

include("manual_backprop.jl")
include("linear.jl")
include("ensemble.jl")
include("flux.jl")
include("json_serialization.jl")

"""
    load_from_jld2(f)

Load from JLD2 file/JLD2 group.
"""
function load_from_jld2(f::Union{JLD2.JLDFile,JLD2.Group})
    if "param" in keys(f) || "is_linear_itf" in keys(f)
        load_from_jld2(f, LinearInterface)
    elseif "chain" in keys(f) || "is_manual_flux_itf" in keys(f)
        load_from_jld2(f, ManualFluxBackPropInterface)
    elseif "ensemble" in keys(f)
        load_from_jld2(f, EnsembleNNInterface)
    end
end


"""
    load_from_jld2(str::AbstractString)

Load saved interface from JLD2 file.
"""
function load_from_jld2(str::AbstractString)
    jldopen(str) do f
        load_from_jld2(f)
    end
end
