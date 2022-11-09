import Base

"""
    EnsembleNNInterface{T<:Tuple} 

Ensemble of many models
"""
struct EnsembleNNInterface{T<:Tuple} <: AbstractNNInterface
    models::T
    weights::Vector{Float64}
    function EnsembleNNInterface(models, weights)
        @assert length(models) == length(weights)
        @assert all(isa(x, AbstractNNInterface) for x in models)
        new{typeof(models)}(models, weights)
    end
end

function (itf::EnsembleNNInterface)(inp, args...;kwargs...)
    forward!(itf, inp, args...;kwargs...)
end

function forward!(itf::EnsembleNNInterface, inp)
    out = forward!(itf.models[1], inp)
    out .*= itf.weights[1]
    length(itf.models) == 1 && return out
    for im in 2:length(itf.models)
        model, wt = itf.models[im], itf.weights[im]
        out .+= forward!(model, inp) .* wt
    end
    out
end

function backward!(itf::EnsembleNNInterface, args...; kwargs...)
    for model in itf.models
        backward!(model, args...;kwargs...) 
    end
end

function gradinp!(gvec, itf::EnsembleNNInterface;tmpg=copy(gvec))
    fill!(tmpg, 0)
    fill!(gvec, 0)
    for (model, wt) in zip(itf.models, itf.weights)
        gradinp!(tmpg, model)
        gvec .+= tmpg .* wt
        fill!(tmpg, 0)
    end
end

function gradparam!(gvec, itf::EnsembleNNInterface;)
    i = 1
    for model in itf.models
        np = nparams(model)
        gview = @view(gvec[i:i+np-1])
        gradparam!(gview, model)
        i += np
    end
end

function paramvector(itf::EnsembleNNInterface)
    vecs = paramvector(itf.models[1])
    length(itf.models) == 1 && return out

    for im in 2:length(itf.models)
        model = itf.models[im]
        append!(vecs, paramvector(model))
    end
    vecs
end

function paramvector!(v, itf::EnsembleNNInterface)
    i = 1
    for model in itf.models
        np = nparams(model)
        paramvector!(@view(v[i:i+np-1]), model)
        i += np
    end
end

function setparamvector!(itf::EnsembleNNInterface, val)
    i = 1
    for model in itf.models
        np = nparams(model)
        setparamvector!(model, val[i:np+i-1])
        i += np
    end
end

nparams(itf::EnsembleNNInterface) = sum(nparams, itf.models) + length(itf.weights)

function Base.show(io::IO, m::MIME"text/plain", v::EnsembleNNInterface)
    println(io, "EnsembleNNInterface:\nModels:")
    show(io, m, v.models[1])
    println(io, "\n...\nTotal: $(length(v.models)) models.")
    println(io, "Weights: $(v.weights)")
end

function save_as_jld2(f::Union{JLD2.Group, JLD2.JLDFile}, obj::EnsembleNNInterface)
    topgroup = JLD2.Group(f, "ensemble")
    group = JLD2.Group(topgroup, "models")
    for (i, model) in enumerate(obj.models)
        subgroup = JLD2.Group(group, "model-$i")
        save_as_jld2(subgroup, model)
    end
    topgroup["weights"] = obj.weights
    f
end

"""

    load_from_jld2(f::Union{<:JLD2.Group, <:JLD2.JLDFile}, ::Type{<:EnsembleNNInterface})

Load from JLD2 file or JLD2 group. 
"""
function load_from_jld2(f::Union{JLD2.Group, JLD2.JLDFile}, ::Type{<:EnsembleNNInterface})
    topgroup = f["ensemble"]
    group = topgroup["models"] 
    models = []
    ids = Int[]
    for key in keys(group)
        i = parse(Int, split(key, "-")[2])
        push!(ids, i)
        model = load_from_jld2(group[key], ManualFluxBackPropInterface)
        if isa(model, ManualFluxBackPropInterface)
            model.apply_xt = true
        end
        push!(models, model)
    end
    weights = topgroup["weights"]
    # Ensure we load it in the wright order
    models = models[sortperm(ids)]
    EnsembleNNInterface(Tuple(models), weights)
end
