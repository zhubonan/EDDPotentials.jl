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

"""
    DenseGradient(dense::Dense, gσ, n)

Buffer for storing gradients of a dense network

```julia
y = σ.(W * x .+ b)
```

Args:
* `gσ`: the function that computes the local gradient of the activation function
* `n`: size of the batch.
"""
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
    backprop!(dg::DenseGradient, d::Dense)

Compute the gradients of a dense network based on back-propagation
"""
function backprop!(dg::DenseGradient, d::Dense)
    dg.gu .*= dg.gσ.(dg.wx) # Downstream of the activation, upstream to the matmul
    dg.gb .= sum(dg.gu, dims=2)  # Gradient of the bias
    mul!(dg.gw, dg.gu, dg.x')   
    mul!(dg.gx, d.weight', dg.gu)
end

struct ChainGradients{T}
    layers::T
    n::Int
end

@inline gtanh_fast(x) = tanh_fast'(x)

function ChainGradients(chain::Chain, n::Int)
    gds = []
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
    ChainGradients(tuple(gds...), n)
end

"""
    forward!(chaing::ChainGradients, chain::Chain, x)

Do a forward pass compute the intermediate quantities for each layer
"""
function forward!(chaing::ChainGradients, chain::Chain, x)
    output = zeros(size(chain.layers[end].bias, 1), size(x, 2))
    nlayers = length(chain.layers)

    for i = 1:length(chain.layers)
        current_layer = chain.layers[i]
        current_gradient = chaing.layers[i]
        # Input of the first layer
        i == 1 && (current_gradient.x .= x)
        # Compute the pre-activation output
        current_gradient.wx .= (current_layer.weight * x .+ current_layer.bias)
        if i == nlayers
            # The final output
            return current_layer.σ.(current_gradient.wx)
        else
            # Output of this layer is the input of the next layer
            chaing.layers[i + 1].x .= current_layer.σ.(current_gradient.wx)
            # New input for the next layer
            x = chaing.layers[i + 1].x
        end
    end
end


"""
    backward!(chaing::ChainGradients, chain::Chain;gu=one(eltype(chain.layers[1].weight)))

Back-propagate the gradients after a forward pass.
Args:
* `gu`: is the upstream gradient for the loss of the entire chain. The default is equivalent to:
  `loss = sum(chain(x))`, which is a matrix of `1` in the same shape of the output matrix.
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