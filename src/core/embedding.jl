#=
Code for applying embeddings based on an existing feature vector.
The features of the same body-order are required to have the same lengths.

An issue of NN potentials is that the cross-uspec interaction
terms group combinatorially, making multi-element systems much more difficult
to treat beyond binary and ternary compounds.

Embedding can be used to alleviate this issue by applying effectively a specie-aware 
weighted sum of the features.
=#

import Flux

"""
Embedding for a specific body order
"""
struct BodyEmbedding{T}
    weight::Matrix{T}
    flength::Int
end

Flux.@layer BodyEmbedding
Flux.trainable(ce::BodyEmbedding) = (weight=ce.weight,)

function BodyEmbedding(T, features::Union{Vector,Tuple}, n::Int)
    nf = length(features)

    # Check that all features have the same length
    @assert length(unique(nfeatures.(features))) == 1

    # The weight has the dimension (number of feature sets, n)
    weight = rand(T, nf, n)
    BodyEmbedding(weight, nfeatures(features[1]))
end

function Base.show(io::IO, e::BodyEmbedding)
    print(io, "BodyEmbedding(", length_before(e), " => ", length_after(e))
    print(io, ")")
end

BodyEmbedding(features::Union{Vector,Tuple}, n::Int) = BodyEmbedding(Float64, features, n)

"""
Number of embedded vectors
"""
num_embed(x::BodyEmbedding) = size(x.weight, 2)

"""
Number of feature sets to be compressed
"""
num_feat(x::BodyEmbedding) = size(x.weight, 1)

"""
Number of elements for each feature
"""
num_each_feat_elements(x::BodyEmbedding) = x.flength

length_before(x::BodyEmbedding) = num_each_feat_elements(x) * num_feat(x)
length_after(x::BodyEmbedding) = num_each_feat_elements(x) * num_embed(x)


"""
Apply the embedding
"""
function (embed::BodyEmbedding)(vector::AbstractVector)
    nl = num_feat(embed)
    l = embed.flength
    @assert nl * l == length(vector)
    # Construct the output vector
    _apply_embedding(embed.weight, reshape(vector, l, nl))
end

"""
Apply the embedding for an matrix input, the columns are made of vectors 
to be embedded.
"""
function (embed::BodyEmbedding)(matrix::AbstractMatrix)
    _apply_embedding_batch(embed.weight, matrix)
end

"""
Apply the embedding for an matrix input, the columns are made of vectors 
to be embedded.
"""
function _apply_embedding_batch(weight, matrix)
    nl = size(weight, 1)
    @assert size(matrix, 1) % nl == 0
    l = div(size(matrix, 1), nl)
    output = similar(matrix, l * size(weight, 2), size(matrix, 2))
    buffer = similar(matrix, l, nl)
    _apply_embedding_batch!(output, buffer, weight, matrix)
end

"""
Apply the embedding for an matrix input, the columns are made of vectors 
to be embedded.
"""
function _apply_embedding_batch!(output, buffer, weight, matrix)
    for i in axes(matrix, 2)
        buffer[:] .= matrix[:, i]
        output[:, i] .= vec(_apply_embedding(weight, buffer))
    end
    output
end


"""
Apply weighted averages to each column of a matrix, the weights are stored in a matrix as well.

```math
O = F W
```

This is simply right multiply the weight matrix.
"""
function _apply_embedding(weight::AbstractMatrix, mat::AbstractMatrix)
    mat * weight
end


"""
Embedding for a full CellFeature
"""
struct CellEmbedding{C,T}
    cf::C
    two_body::BodyEmbedding{T}
    three_body::BodyEmbedding{T}
    n::Int
    m::Int
end

Flux.@layer CellEmbedding
Flux.trainable(ce::CellEmbedding) = (two_body=ce.two_body, three_body=ce.three_body)

function CellEmbedding(cf::CellFeature, n::Int, m::Int=n)
    CellEmbedding(cf, BodyEmbedding(cf.two_body, n), BodyEmbedding(cf.three_body, m), n, m)
end

length_after(e::CellEmbedding) =
    feature_size(e.cf)[1] + length_after(e.two_body) + length_after(e.three_body)
length_before(e::CellEmbedding) = nfeatures(e.cf)

function Base.show(io::IO, e::CellEmbedding)
    length_in = nfeatures(e.cf)
    length_out =
        feature_size(e.cf)[1] + length_after(e.two_body) + length_after(e.three_body)
    print(io, "CellEmbedding(", length_in, " => ", length_out)
    print(io, ")")
end

function two_body_view(cf, vector)
    n1bd = feature_size(cf)[1]
    n2bd = feature_size(cf)[2]
    view(vector, n1bd+1:n1bd+n2bd)
end

function three_body_view(cf, vector)
    n1bd = feature_size(cf)[1]
    n2bd = feature_size(cf)[2]
    n3bd = feature_size(cf)[3]
    view(vector, n1bd+n2bd+1:n1bd+n2bd+n3bd)
end


"""
Apply CellEmbedding to a full feature vector

This is written in such way to support autograd...
"""
function (ce::CellEmbedding)(vector::AbstractVector)
    fsize = feature_size(ce.cf)
    v2 = two_body_view(ce.cf, vector)
    v3 = three_body_view(ce.cf, vector)
    e2 = ce.two_body(v2)
    e3 = ce.three_body(v3)
    vcat(vector[1:fsize[1]], vec(e2), vec(e3))
end


"""
Apply CellEmbedding to a full feature vector

This is written in such way to support autograd...
"""
function (ce::CellEmbedding)(mat::AbstractMatrix)
    n1bd, n2bd, n3bd = feature_size(ce.cf)
    _apply_embedding_cell(n1bd, n2bd, n3bd, ce.two_body.weight, ce.three_body.weight, mat)
end

function _apply_embedding_cell(n1bd, n2bd, n3bd, w2, w3, mat::AbstractMatrix)
    m2 = mat[n1bd+1:n2bd+n1bd, :]
    m3 = mat[n1bd+n2bd+1:n2bd+n1bd+n3bd, :]
    e2 = _apply_embedding_batch(w2, m2)
    e3 = _apply_embedding_batch(w3, m3)
    vcat(mat[1:n1bd, :], e2, e3)
end



## Explicit gradient computation
"""
Compute the gradient of the embedding layer
      (W)
x -> Embedding -> out


The embeding layer does is 

```math
out = x W
```

Operation is assumed to be in the "batch" mode, taking inputs of multiple columns
"""
struct BodyEmbeddingGradient{T}
    "Gradients of the weight"
    gw::Matrix{T}
    "Upstream gradient"
    gu::Matrix{T}
    "Gradient of the input"
    gx::Matrix{T}
    x::Matrix{T}
    out::Matrix{T}
    out_return::Matrix{T}
    "Number of batch samples"
    n::Int
    nf::Int
end

function BodyEmbeddingGradient(embed::BodyEmbedding, n, out)

    out_i = length_after(embed)
    @assert (out_i, n) == size(out) "Mismatch in output size, expected $((out_i, n)), found $(size(out))."
    BodyEmbeddingGradient(
        similar(embed.weight), # gw
        zeros(eltype(embed.weight), out_i, n), #gu
        zeros(eltype(embed.weight), length_before(embed), n),  #gx
        zeros(eltype(embed.weight), length_before(embed), n),  #x
        out,
        copy(out),
        n,
        num_each_feat_elements(embed),
    )
end

function BodyEmbeddingGradient(embed::BodyEmbedding, n)
    out = zeros(eltype(embed.weight), length_after(embed), n)
    BodyEmbeddingGradient(embed, n, out)
end

weight_gradient(e::BodyEmbeddingGradient) = e.gw
input_gradient(e::BodyEmbeddingGradient) = e.gx


"""
Perform backpropagation - this requires the forward pass to be run an upstream gradients computed
"""
function backprop!(eg::BodyEmbeddingGradient, e::BodyEmbedding; weight_and_bias=true)
    nf = num_each_feat_elements(e)
    neb = num_embed(e)
    gu_tmp = similar(eg.gu, nf, neb)
    x_tmp = similar(eg.x, nf, num_feat(e))

    fill!(eg.gw, 0)
    # Compute for each column - need to pack the inputs
    for i = 1:eg.n
        # Compute for each input column
        gu_tmp[:] .= eg.gu[:, i]
        x_tmp[:] .= eg.x[:, i]
        if weight_and_bias
            # Accumulate the gradients, as the total energy is the sum of that of each atom
            eg.gw .+= x_tmp' * gu_tmp
        end
        eg.gx[:, i] .= vec(gu_tmp * e.weight')
    end
end

"""
Forward pass
"""
function forward!(
    gradient::BodyEmbeddingGradient,
    layer::BodyEmbedding,
    next_input_array,
    x,
    i,
    nlayers,
)
    # Store the input - needed for gradient computation later
    i == 1 && (gradient.x .= x)
    x_reshaped = reshape(x, num_each_feat_elements(layer), num_feat(layer), size(x, 2))
    buff = x_reshaped[:, :, 1]
    for j in axes(x_reshaped, 3)
        buff .= x_reshaped[:, :, j]
        gradient.out[:, j] = vec(buff * layer.weight)
    end
    if i != nlayers
        next_input_array .= gradient.out
    end
    gradient.out
end

"""
Container to operate forward and backward passes
"""
struct CellEmbeddingGradient{T}
    two_body::BodyEmbeddingGradient{T}
    three_body::BodyEmbeddingGradient{T}
    offset::Int
    gu::Matrix{T}
    out::Matrix{T}
    x::Matrix{T}
    gx::Matrix{T}
end

input_gradient(e::CellEmbeddingGradient) = e.gx

function CellEmbeddingGradient(ce::CellEmbedding{C,T}, n::Int, out) where {C,T}
    offset = feature_size(ce.cf)[1]
    #out = zeros(T, length_after(ce.two_body) + length_after(ce.three_body) + offset, n)
    x = zeros(T, length_before(ce.two_body) + length_before(ce.three_body) + offset, n)
    CellEmbeddingGradient(
        BodyEmbeddingGradient(ce.two_body, n),
        BodyEmbeddingGradient(ce.three_body, n),
        offset,
        out,
        similar(out),
        x,
        similar(x),
    )
end

function CellEmbeddingGradient(ce::CellEmbedding{C,T}, n::Int) where {C,T}
    offset = feature_size(ce.cf)[1]
    out = zeros(T, length_after(ce.two_body) + length_after(ce.three_body) + offset, n)
    CellEmbeddingGradient(ce, n, out)
end

"""
Forward pass
"""
function forward!(
    gradient::CellEmbeddingGradient,
    layer::CellEmbedding,
    next_gradient,
    x,
    i,
    nlayers,
)
    n2 = length_before(layer.two_body)
    n3 = length_before(layer.three_body)

    l2 = length_after(layer.two_body)
    l3 = length_after(layer.three_body)

    # Inputs for two-body and three-body features
    offset = gradient.offset
    x_two_body = view(x, offset+1:offset+n2, :)
    x_three_body = view(x, offset+n2+1:offset+n2+n3, :)

    # Next input - again we split it here
    next_two_body = view(next_gradient.x, offset+1:offset+l2, :)
    next_three_body = view(next_gradient.x, offset+l2+1:offset+l2+l3, :)

    forward!(gradient.two_body, layer.two_body, next_two_body, x_two_body, i, nlayers)
    forward!(
        gradient.three_body,
        layer.three_body,
        next_three_body,
        x_three_body,
        i,
        nlayers,
    )

    # Pass through for the one-body
    gradient.out[1:offset, :] = x[1:offset, :]
    # Store the outputs of two and three body
    gradient.out[offset+1:offset+l2, :] .= gradient.two_body.out
    gradient.out[offset+1+l2:offset+l2+l3, :] .= gradient.three_body.out

end


function backprop!(cg::CellEmbeddingGradient, ce::CellEmbedding; weight_and_bias=true)

    n2 = length_before(ce.two_body)
    n3 = length_before(ce.three_body)

    l2 = length_after(ce.two_body)
    l3 = length_after(ce.three_body)

    # Update the upstream gradients
    offset = cg.offset
    cg.two_body.gu .= cg.gu[offset+1:offset+l2, :]
    cg.three_body.gu .= cg.gu[offset+1+l2:offset+l2+l3, :]
    backprop!(cg.two_body, ce.two_body; weight_and_bias)
    backprop!(cg.three_body, ce.three_body; weight_and_bias)

    # Update gx
    # one-body
    cg.gx[1:offset, :] = cg.gu[1:offset, :]
    # two-body
    cg.gx[1+offset:offset+n2, :] .= cg.two_body.gx
    # three-body
    cg.gx[1+n2+offset:offset+n2+n3, :] .= cg.three_body.gx
end
