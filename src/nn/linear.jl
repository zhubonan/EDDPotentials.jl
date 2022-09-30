
#=
Simply linear implementation, e.g. without NN
=#


struct LinearInterface{T} <: AbstractNNInterface
    param::Matrix{T}
end

LinearInterface(param::AbstractVector) = LinearInterface(collect(transpose(param)))

forward!(itf::LinearInterface, inp::AbstractVecOrMat) = itf.param * inp

paramvector(itf::LinearInterface) = itf.param[:]

function paramvector!(vec, itf::LinearInterface) 
    vec.= itf.param[:]
end

nparams(itf::LinearInterface) = length(itf.parmas)

function setparamvector!(itf::LinearInterface, vec)  
    itf.parma[:] .= vec
end

function gradinp!(gvec, itf::LinearInterface, inp::AbstractArray)
    for i ∈ axes(inp, 2)
        gvec[:, i] = itf.param
    end
    gvec
end

function gradparam!(gvec, itf::LinearInterface, inp::AbstractArray)
    gvec .= transpose(sum(inp, dims=2))[:]
end

function (itf::LinearInterface)(inp::AbstractArray)
    forward!(itf, inp)
end