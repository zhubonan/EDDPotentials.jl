#=
Interface for Neutron network implementations
=#
using JLD2
using StatsBase: ZScoreTransform, fit, transform!, reconstruct!

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
function flux_mlp_model(cf::CellFeature, nodes...;init=glorot_uniform_f64, 
         σ=tanh_fast,
         embedding=nothing,
         σs=nothing)

    if embedding === nothing
        ninp = nfeatures(cf)
    else
        fsizes = feature_size(cf)
        ninp = fsizes[1]
        ninp += nfeatures(cf.two_body[1]) * num_embed(embedding.two_body)
        ninp += nfeatures(cf.three_body[1]) * num_embed(embedding.three_body)
    end
    if isnothing(σs)
        input = Dense(ninp => nodes[1], σ;init)
    else
        input = Dense(ninp => nodes[1], σs[1]; init)
    end

    layers = Any[input]
    if embedding !== nothing
        pushfirst!(layers, embedding)
    end
    i = 1
    while i < length(nodes)
        i += 1
        if isnothing(σs)
            push!(layers, Dense(nodes[i-1]=>nodes[i], σ; init))
        else
            push!(layers, Dense(nodes[i-1]=>nodes[i], σs[i]; init))
        end
    end
    push!(layers, Dense(nodes[i]=>1;init))
    Chain(layers...)
end

## Standardisation

include("manual_backprop.jl")
include("linear.jl")
include("ensemble.jl")
include("flux.jl")