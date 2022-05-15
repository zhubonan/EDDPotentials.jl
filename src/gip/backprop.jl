#=
Manual implementation of back propagation to compute the gradient of the energy
=#

using LinearAlgebra
using Flux
import Zygote


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
    DenseGradient(gw, gb, gu, gx, n, gσ, wx, x, out)
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
        if layer.σ == identity
            gσ = (x, y) -> one(x)
        elseif  (layer.σ == tanh_fast) || (layer.σ == tanh)
            gσ = (x, y) -> 1 - y^2
        else
            gσ = (x, y) -> layer.σ'(x)
        end
        if i == nl
            gbuffer = DenseGradient(layer, gσ, n)
        else
            # Output from this layer is the input of the next layer
            gbuffer = DenseGradient(layer, gσ, n, gds[1].x)
        end
        # Since we build the buffer in the reverse order, push to the front of the Vector
        pushfirst!(gds, gbuffer)
    end
    ChainGradients(tuple(gds...), n)
end

"""
    forward!(chaing::ChainGradients, chain::Chain, x)

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
    chaing.layers[end].out
end

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
function backward!(chaing::ChainGradients, chain::Chain;gu=one(eltype(chain.layers[1].weight)), weight_and_bias=true)
    nlayers = length(chain.layers)
    for i = nlayers:-1:1
        gl = chaing.layers[i]
        l = chain.layers[i]
        i == nlayers && fill!(gl.gu, gu)
        backprop!(gl, l; weight_and_bias)
        # The upstream gradient of the next layer is that of the gradient of x of 
        # this layer
        i != 1 && (chaing.layers[i-1].gu .= gl.gx)
    end
end