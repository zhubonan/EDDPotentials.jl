
#=
Simply linear implementation, e.g. without NN
=#


mutable struct LinearInterface{T} <: AbstractNNInterface
    param::Matrix{T}
    inp::Any
end

LinearInterface(x) = LinearInterface(x, nothing)

LinearInterface(param::AbstractVector) =
    LinearInterface(collect(transpose(param)), Matrix{eltype(param)}(undef, 0, 0))

function forward!(itf::LinearInterface, inp)
    out = itf.param * inp
    itf.inp = inp
    out
end

paramvector(itf::LinearInterface) = itf.param[:]

function paramvector!(vec, itf::LinearInterface)
    vec .= itf.param[:]
end

nparams(itf::LinearInterface) = length(itf.param)

function setparamvector!(itf::LinearInterface, param)
    vec(itf.param) .= vec(param)
end

function gradinp!(gvec, itf::LinearInterface, inp=itf.inp)
    for i ∈ axes(inp, 2)
        gvec[:, i] = itf.param
    end
    gvec
end

function gradparam!(gvec, itf::LinearInterface, inp=itf.inp)
    gvec .= transpose(sum(inp, dims=2))[:]
end

function (itf::LinearInterface)(inp)
    itf.param * inp
end

function backward!(itf::LinearInterface, args...; kwargs...) end

"""
    save_as_jld2(f, obj::LinearInterface)

Save the interface into an opened JLD2 file/JLD2 group.
"""
function save_as_jld2(f::Union{JLD2.JLDFile,JLD2.Group}, obj::LinearInterface)
    f["param"] = obj.param
    f["is_linear_itf"] = true
end

"""
    load_from_jld2(f, obj::LinearInterface)

Load from JLD2 file/JLD2 group.
"""
function load_from_jld2(f::Union{JLD2.JLDFile,JLD2.Group}, ::Type{LinearInterface})
    LinearInterface(f["param"])
end
