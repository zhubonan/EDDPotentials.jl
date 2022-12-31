#=
Manual implementation of back propagation to compute the gradient of the energy

The reason for having an explicity implementation is that the overhead associated with autograd makes
it inefficient for calling many *small* neuron networks.
=#

import Base
using JLD2
using LinearAlgebra
using Flux
import Zygote
import JLD2


"""
Buffer for storing gradients of a dense network

```julia
y = σ.(W * x .+ b)
```
"""
struct DenseGradient{N, T}
    "Gradient of the weight"
    gw::Matrix{T}
    "Gradient of the bias"
    gb::Vector{T}
    "Upstream gradient"
    gu::Matrix{T}
    "Gradient of x -> (K X N)"
    gx::Matrix{T}
    "dimension of the input (K x N)"
    n::Int
    gσ::N
    "Value of W * x + b before activation"
    wx::Matrix{T}
    "Value of x"
    x::Matrix{T}
    "Value of the output"
    out::Matrix{T}
    "Value of the output used as return value"
    out_return::Matrix{T}
end

"""
    DenseGradient(dense::Dense, gσ, n)

Buffer for storing gradients of a dense network

```julia
y = σ.(W * x .+ b)
```

Args:
* `gσ`: the function that computes the local gradient of the activation function. 
   It must take two arguments `x` and `fx`, where `fx = σ(x)`.
* `n`: size of the batch.
* `out`: Matrix of the output after this layer
"""
function DenseGradient(dense::Dense, gσ, n, out)
    m = size(dense.weight, 1)
    k = size(dense.weight, 2)
    gw = similar(dense.weight)
    gb = similar(dense.bias)
    wx = zeros(eltype(dense.weight), m, n)
    gu = zeros(eltype(dense.weight), m, n)
    gx = zeros(eltype(dense.weight), k, n)
    x = zeros(eltype(dense.weight), k, n)
    DenseGradient(gw, gb, gu, gx, n, gσ, wx, x, out, copy(out))
end

function DenseGradient(dense::Dense, gσ, n)
    m = size(dense.weight, 1)
    out = zeros(eltype(dense.weight), m, n)
    DenseGradient(dense, gσ, n, out)
end

weight_gradient(d::DenseGradient) = d.gw
bias_gradient(d::DenseGradient) = d.gb
input_gradient(d::DenseGradient) = d.gx


"""
    backprop!(dg::DenseGradient, d::Dense;weight_and_bias=true)

Compute the gradients of a dense network based on back-propagation.

Args:
* `weight_and_bias`: Update the gradients for weight and bias. Defaults to true.
"""
function backprop!(dg::DenseGradient, d::Dense;weight_and_bias=true)
    # Update the upstream gradient
    for i in eachindex(dg.wx)
        dg.gu[i] *= dg.gσ(dg.wx[i], dg.out[i])
    end
    #dg.gu .*= dg.gσ.(dg.wx) # Downstream of the activation, upstream to the matmul
    if weight_and_bias
        for i = 1:size(dg.gb, 1)
            dg.gb[i] = 0
            for j = 1:size(dg.gu, 2)
                dg.gb[i] += dg.gu[i, j]
            end
        end
    end

    #dg.gb .= sum(dg.gu, dims=2)  # Gradient of the bias
    if weight_and_bias
        mul!(dg.gw, dg.gu, dg.x')   
    end
    mul!(dg.gx, d.weight', dg.gu)
end

struct ChainGradients{T}
    layers::T
    n::Int
end

@inline gtanh_fast(x) = 1 - tanh_fast(x)^2

function ChainGradients(chain::Chain, n::Int)
    gds = []
    nl = length(chain.layers)
    for i = nl:-1:1
        layer = chain.layers[i]
        # gradient of the activation function
        if isa(layer, Dense)
            if layer.σ == identity
                gσ = (x, y) -> one(x)
            elseif  (layer.σ == tanh_fast) || (layer.σ == tanh)
                gσ = (x, y) -> 1 - y^2
            else
                gσ = (x, y) -> layer.σ'(x)
            end
        end

        # Construct and build the gradient propagators
        if i == nl
            if isa(layer, Dense)
                gbuffer = DenseGradient(layer, gσ, n)
            elseif isa(layer, CellEmbedding)
                gbuffer = CellEmbeddingGradient(layer, n)
            else
                throw(ErrorException("Unknown input type: $(layer)."))
            end
        else
            # Output from this layer is the input of the next layer
            if isa(layer, Dense)
                gbuffer = DenseGradient(layer, gσ, n, gds[1].x)
            elseif isa(layer, CellEmbedding)
                gbuffer = CellEmbeddingGradient(layer, n)
            else
                throw(ErrorException("Unknown input type: $(layer)."))
            end
        end
        # Since we build the buffer in the reverse order, push to the front of the Vector
        pushfirst!(gds, gbuffer)
    end
    ChainGradients(tuple(gds...), n)
end

"""
    forward!(chaing::ChainGradients, chain::Chain, x;copy=false)

Do a forward pass compute the intermediate quantities for each layer
"""
function forward!(chaing::ChainGradients, chain::Chain, x)
    nlayers = length(chain.layers)

    for i = 1:length(chain.layers) - 1
        forward!(chaing.layers[i], chain.layers[i], chaing.layers[i+1], x, i, nlayers)
        x = chaing.layers[i].out
    end
    # Last layer
    forward!(chaing.layers[end], chain.layers[end], chaing.layers[end], x, nlayers, nlayers)
    # Copy the output to a new array. The latter may be mutated by down stream, for example reconstructions
    chaing.layers[end].out_return .= chaing.layers[end].out
    chaing.layers[end].out_return
end

"""
Internal per-layer forward pass
"""
function forward!(gradient::DenseGradient, layer::Dense, next_gradient::DenseGradient, x, i, nlayers)
    # Input of the first layer
    i == 1 && (gradient.x .= x)
    # Compute the pre-activation output
    mul!(gradient.wx, layer.weight, x)
    gradient.wx .+= layer.bias
    map!(layer.σ, gradient.out, gradient.wx)
    if i != nlayers
        gradient.out === next_gradient.x || (next_gradient.x .= gradient.out)
    end
end


"""
    backward!(chaing::ChainGradients, chain::Chain;gu=one(eltype(chain.layers[1].weight)))

Back-propagate the gradients after a forward pass.
Args:
* `gu`: is the upstream gradient for the loss of the entire chain. The default is equivalent to:
  `loss = sum(chain(x))`, which is a matrix of `1` in the same shape of the output matrix.
"""
function backward!(chaing::ChainGradients, chain::Chain;gu=1, weight_and_bias=true)
    nlayers = length(chain.layers)
    # Set the upstream gradient
    fill!(chaing.layers[end].gu, gu)
    for i = nlayers:-1:2
        gl = chaing.layers[i]
        l = chain.layers[i]
        backprop!(gl, l; weight_and_bias)
        # The upstream gradient of the next layer is that of the gradient of x of 
        # this layer
        chaing.layers[i-1].gu .= gl.gx
    end
    backprop!(chaing.layers[1], chain.layers[1]; weight_and_bias)
end

"""
    paramvector!(vec, model)

Return a vector containing all parameters of a model concatenated as a vector
"""
function paramvector!(vec::AbstractVector, model::Chain)
    i = 1
    fparam = Flux.params(model)
    # Can be optimized here
    for item in fparam
        l = length(item)
        vec[i: i+l-1] .= item[:]
        i += l
    end
    vec
end

function paramvector(model::Chain)
    fparam = Flux.params(model)
    np = sum(length, fparam)
    out = zeros(eltype(fparam[1]), np)
    paramvector!(out, model)
end

nparams(model::Chain) = sum(length, Flux.params(model))


"""
    update_param!(model, param)

Update the parameters of a model from a given vector
"""
function update_param!(model, param)
    fparams = Flux.params(model)
    @assert sum(length, fparams) == length(param)
    i = 1
    for item in fparams
        l = length(item)
        item[:] .= param[i:i+l-1]
        i += l
    end
    model
end



#=
Interface for DenseGradient implementation 
=#

mutable struct ManualFluxBackPropInterface{T, G, X, Z} <: AbstractNNInterface
    chain::T
    gchains::Vector{ChainGradients{G}}
    last_id::Int
    xt::X
    yt::Z
    apply_xt::Bool
end


get_flux_model(itf::ManualFluxBackPropInterface) = itf.chain

function ManualFluxBackPropInterface(chain::Chain;xt=nothing, yt=nothing, apply_xt=true) 
    g = ChainGradients(chain, 1)
    ManualFluxBackPropInterface(chain, typeof(g)[], 1, xt, yt, apply_xt)
end

function Base.show(io::IO, m::MIME"text/plain", x::ManualFluxBackPropInterface )
    println(io, "ManualFluxBackPropInterface(")
    Base.show(io, m, x.chain)
    print(io, "\n)")
end

"""
Select the ChainGradients with the right size
"""
function _get_or_create_chaingradients(itf, inp)
    bsize = size(inp, 2)
    idx = findfirst(x -> x.n == bsize, itf.gchains)
    # Create the ChainGradients of the desired size on-the-fly
    if isnothing(idx)
        cgtmp = ChainGradients(itf.chain, bsize)
        push!(itf.gchains , cgtmp)
        idx = length(itf.gchains)
    end
    cg = itf.gchains[idx]
    itf.last_id = idx
    cg
end

function clear_transient_gradients!(g::ManualFluxBackPropInterface)
    empty!(g.gchains)
end


function forward!(itf::ManualFluxBackPropInterface, inp::Matrix;make_copy=false)

    gchain = _get_or_create_chaingradients(itf, inp)

    out = forward!(gchain, itf.chain, transformed_inp(itf, inp))

    # Apply y transformation
    if !isnothing(itf.yt)
        reconstruct!(itf.yt, out)
    end
    !make_copy ? out : copy(out)
end

function forward!(itf::ManualFluxBackPropInterface, inp::Matrix, inptmp::Matrix;make_copy=false)
    gchain = _get_or_create_chaingradients(itf, inp)
    # Apply x transformation
    if !isnothing(itf.xt) && itf.apply_xt
        nl = itf.xt.len
        inptmp .= inp
        transform!(itf.xt, @view(inptmp[end-nl+1:end, :]))
        out = forward!(gchain, itf.chain, inptmp)
    else
        out = forward!(gchain, itf.chain, inp)
    end
    # Apply y transformation
    if !isnothing(itf.yt)
        reconstruct!(itf.yt, out)
    end
    !make_copy ? out : copy(out)
end

function (itf::ManualFluxBackPropInterface)(inp;make_copy=false)
    forward!(itf, inp;make_copy)
end

function gradparam!(gvec::AbstractVector, itf::ManualFluxBackPropInterface)
    grad = collect_gradients!(gvec, itf.gchains[itf.last_id])
    if !isnothing(itf.yt)
        grad .*= itf.yt.scale[1]
    end
    grad
end

"""
Return the gradient of the input matrix ``X`` against the sum of the output ``sum(G(X))``.
"""
function gradinp!(gvec::AbstractVecOrMat, itf::ManualFluxBackPropInterface)
    # Collect 
    gvec .= input_gradient(itf.gchains[itf.last_id].layers[1])
    # If transform is applied then we have to scale the gradient
    if !isnothing(itf.xt) && itf.apply_xt
        nl = itf.xt.len
        gvec[end-nl+1:end, :] ./= itf.xt.scale
    end
    if !isnothing(itf.yt)
        gvec .*= itf.yt.scale[1]
    end
    gvec
end

backward!(itf::ManualFluxBackPropInterface;kwargs...) = backward!(itf.gchains[itf.last_id], itf.chain;kwargs...)


paramvector(itf::ManualFluxBackPropInterface) = paramvector(itf.chain)
paramvector!(vec, itf::ManualFluxBackPropInterface) = paramvector!(vec, itf.chain)

function setparamvector!(itf::ManualFluxBackPropInterface, vec::AbstractVector)
    update_param!(itf.chain, vec)
end

nparams(itf::ManualFluxBackPropInterface) = nparams(itf.chain)

function ManualFluxBackPropInterface(cf::CellFeature, 
    nodes...;init=glorot_uniform_f64, xt=nothing, yt=nothing, σ=tanh, apply_xt=true, σs=nothing, embedding=nothing)
    chain = flux_mlp_model(cf, nodes...;init, σ, σs, embedding)
    ManualFluxBackPropInterface(chain; xt, yt, apply_xt)
end

# JLD2 custom serialization - we do not need to store ChainGradients which are 
# We cannot use the official custom serialization interface because we do not want 
# to information about the ChainGradients instans which is in type parameter


"""
    save_as_jld2(f, obj::ManualFluxBackPropInterface)

Save the interface into an opened JLD2 file/JLD2 group.
"""
function save_as_jld2(f::Union{JLD2.JLDFile, JLD2.Group}, obj::ManualFluxBackPropInterface)
    f["chain"] = obj.chain
    f["xt"] = obj.xt
    f["yt"] = obj.yt
    f["apply_xt"] = obj.apply_xt
    f
end


"""
    load_from_jld2(f, obj::ManualFluxBackPropInterface)

Load from JLD2 file/JLD2 group.
"""
function load_from_jld2(f::Union{JLD2.JLDFile, JLD2.Group}, ::Type{ManualFluxBackPropInterface})
    chain = f["chain"]
    xt = f["xt"]
    yt = f["yt"]
    apply_xt=f["apply_xt"]
    ManualFluxBackPropInterface(chain;xt, yt, apply_xt)
end