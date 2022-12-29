#=
Code for applying embeddings based on an existing feature vector.
The features of the same body-order are required to have the same lengths.

An issue of NN potentials is that the cross-uspec interaction
terms group combinatorially, making multi-element systems much more difficult
to treat beyond binary and ternary compounds.

Embedding can be used to alleviate this issue by applying effectively a specie-aware 
weighted sum of the features.
=#

"""
Embedding for a specific body order
"""
struct BodyEmbedding{T}
    weights::Matrix{T}
end

function BodyEmbedding(T, features, n)
    nf = length(features)

    # Check that all features have the same length
    @assert length(unique(nfeatures.(features))) == 1

    # The weight has the dimension (number of feature sets, n)
    weights = rand(T, nf, n)
    BodyEmbedding(weights)
end

BodyEmbedding(features, n) = BodyEmbedding(Float64, features, n)

"""
Number of embedded vectors
"""
num_embed(x::BodyEmbedding) = size(x.weights, 2) 

"""
Number of feature sets to be compressed
"""
num_feat(x::BodyEmbedding) = size(x.weights, 1) 


"""
Apply the embedding
"""
function (embed::BodyEmbedding)(vector)
    nl = num_feat(embed)
    l = div(length(vector), nl)
    @assert nl * l == length(vector) 

    # Construct the output vector
    _apply_embedding(embed.weights, reshape(vector, l, nl))
end

"""
Apply embedding and store the output in an existing array
"""
function _apply_embedding!(out, weight::Matrix, mat)
    @assert size(weight, 1) == size(mat, 2)
    for j in 1:size(weight, 2)
        for (i, col) in enumerate(eachcol(mat))
            for ii in 1:size(mat, 1)
                out[ii, j] += weight[i, j] * col[ii]
            end
        end
    end
    out
end

"""
Apply weighted averages to each column of a matrix, the weights are stored in a matrix as well.

NOTE: this function allocates but can be used for autograd
"""
function _apply_embedding(weights, mat)
    # Use broadcast to computed weighted averages 
    vcat([
        sum(x .* mat', dims=1) for x in eachcol(weights)
        ]...)'
end


"""
Embedding for a full CellFeature
"""
struct CellEmbedding{C, T}
    cf::C
    two_body::BodyEmbedding{T}
    three_body::BodyEmbedding{T}
end

function CellEmbedding(cf::CellFeature, n::Int, m::Int=n)
    CellEmbedding(
        cf, 
        BodyEmbedding(cf.two_body, n),
        BodyEmbedding(cf.three_body,m)
    )
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
"""
function (ce::CellEmbedding)(vector)
    fsize = feature_size(ce.cf)
    v2 = two_body_view(ce.cf, vector)
    v3 = three_body_view(ce.cf, vector)
    e2 = ce.two_body(v2)
    e3 = ce.three_body(v3)
    vcat(vector[1:fsize[1]], vec(e2), vec(e3))
end