
#=
Simply linear implementation, e.g. without NN
=#


mutable struct LinearInterface{T} <: AbstractNNInterface
    param::Matrix{T}
    inp
end

LinearInterface(x) = LinearInterface(x, nothing)

LinearInterface(param::AbstractVector) = LinearInterface(collect(transpose(param)), Matrix{eltype(param)}(undef, 0, 0))

function forward!(itf::LinearInterface, inp)
    out = itf.param * inp
    itf.inp = inp
    out
end

paramvector(itf::LinearInterface) = itf.param[:]

function paramvector!(vec, itf::LinearInterface) 
    vec.= itf.param[:]
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
    forward!(itf, inp)
end

function backward!(itf::LinearInterface, args...;kwargs...) end