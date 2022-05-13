#=
Manual implementation of back propagation to compute the gradient of the energy
=#

using LinearAlgebra
using Flux
using Zygote


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
end

function DenseGradient(dense::Dense, gσ, n)
    m = size(dense.weight, 1)
    k = size(dense.weight, 2)
    gw = similar(dense.weight)
    gb = similar(dense.bias)
    wx = zeros(eltype(dense.weight), m, n)
    gu = zeros(eltype(dense.weight), m, n)
    gx = zeros(eltype(dense.weight), k, n)
    x = zeros(eltype(dense.weight), k, n)
    DenseGradient(gw, gb, gu, gx, n, gσ, wx, x)
end

"""
Compute the gradients with backprop
"""
function backprop!(dg::DenseGradient, d::Dense)
    dg.gu .*= dg.gσ.(dg.wx) # Downstream of the activation, upstream to the matmul
    dg.gb .= sum(dg.gu, dims=2)  # Gradient of the bias
    mul!(dg.gw, dg.gu, dg.x')   
    mul!(dg.gx, d.weight', dg.gu)
end

struct ChainGradients{T}
    layers::T
end

@inline gtanh_fast(x) = tanh_fast'(x)

function ChainGradients(chain::Chain, batchsize)
    gds = []
    n = batchsize 
    for layer in chain.layers
        if layer.σ == identity
            gσ = one
        elseif  (layer.σ == tanh_fast) || (layer.σ == tanh)
            gσ = gtanh_fast
        else
            gσ = layer.σ'
        end
        gbuffer = DenseGradient(layer, gσ, n)
        push!(gds, gbuffer)
    end
    ChainGradients(tuple(gds...))
end

"""
Compute the forward pass
"""
function forward!(chaing::ChainGradients, chain::Chain, x)
    output = zeros(size(chain.layers[end].bias, 1), size(x, 2))
    nlayers = length(chain.layers)

    for i = 1:length(chain.layers)
        current_layer = chain.layers[i]
        current_gradient = chaing.layers[i]
        i == 1 && (current_gradient.x .= x)
        current_gradient.wx .= (current_layer.weight * x .+ current_layer.bias)
        if i == nlayers
            # save the final output
            output .= current_layer.σ.(current_gradient.wx)
        else
            # Output of this layer is the input of the next layer
            chaing.layers[i + 1].x .= current_layer.σ.(current_gradient.wx)
            # New input for the next iteration
            x = chaing.layers[i + 1].x
        end
    end
end


"""
    backward!(chaing::ChainGradients, chain::Chain, output;gu=1)

Backward propagation
"""
function backward!(chaing::ChainGradients, chain::Chain;gu=one(eltype(chain.layers[1].weight)))
    nlayers = length(chain.layers)
    for i = nlayers:-1:1
        gl = chaing.layers[i]
        l = chain.layers[i]
        i == nlayers && fill!(gl.gu, gu)
        backprop!(gl, l)
        # The upstream gradient of the next layer is that of the gradient of x of 
        # this layer
        i != 1 && (chaing.layers[i-1].gu .= gl.gx)
    end
end


# ## Validation

# d1 = Dense(5=>10, tanh)
# chain = Chain(d1)
# chaing = ChainGradients(chain, 10)
# x = rand(5, 10)
# @time begin
# forward!(chaing, chain, x)
# backward!(chaing, chain)
# end

# # Now use backprop via Zygote

# f(z) = sum(d1.σ.(z * x .+ d1.bias))
# sum(f(d1.weight) == sum(chain(x)))
# d1gw = Zygote.gradient(f, d1.weight)[1]
# @assert all(d1gw .≈ chaing.layers[1].gw)

# # Two layer
# d1 = Dense(5=>10, tanh)
# d2 = Dense(10=>8)
# chain = Chain(d1, d2)
# chaing = ChainGradients(chain, 10)
# x = rand(5, 10)
# @time begin
# forward!(chaing, chain, x)
# backward!(chaing, chain)
# end

# # Now use backprop via Zygote

# f(z) = sum(d2(d1.σ.(z * x .+ d1.bias)))
# fb(z) = sum(d2(d1.σ.(d1.weight * x .+ z)))
# f2(z) = sum( d2.σ.(z * d1(x) .+ d2.bias))
# @assert all(f(d1.weight) .≈ sum(chain(x)))
# @assert all(f2(d2.weight) .≈ sum(chain(x)))

# d1gw = Zygote.gradient(f, d1.weight)[1]
# @assert all(d1gw .≈ chaing.layers[1].gw)

# d1gb = Zygote.gradient(fb, d1.bias)[1]
# @assert all(d1gb .≈ chaing.layers[1].gb)

# d2gw = Zygote.gradient(f2, d2.weight)[1]
# @assert all(d2gw .≈ chaing.layers[2].gw)


# d2gx = Zygote.gradient(x -> sum(d2(x)), d1(x))[1]
# @assert all(d2gx .≈ chaing.layers[2].gx)

# using BenchmarkTools

# function roundtrip(chaing, chain, x)
#     forward!(chaing, chain, x)
#     backward!(chaing, chain)
# end

# param = Flux.params(chain)
# loss() = sum(chain(x))
# @btime Flux.gradient(loss, param)

# @btime roundtrip(chaing, chain, x)