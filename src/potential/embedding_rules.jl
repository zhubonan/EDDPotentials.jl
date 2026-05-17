import ChainRulesCore: rrule, NoTangent, unthunk

"""
Custom rule for the _apply_embedding_batch function
"""
function rrule(
    ::typeof(_apply_embedding_batch),
    w::AbstractMatrix,
    x::AbstractMatrix;
    kwargs...,
)

    y = _apply_embedding_batch(w, x)
    nf = div(size(x, 1), size(w, 1)) # Number of elements for each feature
    neb = size(w, 2)   # number of embedding
    nfeat = size(w, 1)  # Number of features 

    function pullback(y1)
        ȳ = unthunk(y1)
        w̄ = similar(w)
        fill!(w̄, 0)
        x̄ = similar(x)
        fill!(x̄, 0)
        x_tmp = similar(x, nf, nfeat)
        ȳ_temp = similar(ȳ, nf, neb)
        for i = 1:size(x, 2)
            ȳ_temp[:] .= ȳ[:, i]
            x_tmp[:] .= x[:, i]
            w̄ .+= x_tmp' * ȳ_temp
            x̄[:, i] .= vec(ȳ_temp * w')
        end
        NoTangent(), w̄, x̄
    end

    return y, pullback
end

"""
Custom rule for the _apply_embedding_cell function
"""
function rrule(
    ::typeof(_apply_embedding_cell),
    n1bd,
    n2bd,
    n3bd,
    w2,
    w3,
    mat::AbstractMatrix;
    kwargs...,
)

    m2 = mat[n1bd+1:n2bd+n1bd, :]
    m3 = mat[n1bd+n2bd+1:n2bd+n1bd+n3bd, :]
    e2 = _apply_embedding_batch(w2, m2)
    e3 = _apply_embedding_batch(w3, m3)
    y = vcat(mat[1:n1bd, :], e2, e3)

    # Size of arrays for two-body and three body embeddings
    @assert n2bd % size(w2, 1) == 0
    @assert n3bd % size(w3, 1) == 0

    nf2 = div(n2bd, size(w2, 1)) # Number of elements for each feature
    nf3 = div(n3bd, size(w3, 1)) # Number of elements for each feature
    neb2 = size(w2, 2)   # number of embedding
    neb3 = size(w3, 2)   # number of embedding
    nfeat2 = size(w2, 1)  # Number of features 
    nfeat3 = size(w3, 1)  # Number of features 

    n2embed = nf2 * neb2
    n3embed = nf3 * neb3

    function pullback(y1)

        # Unthunk ȳ
        ȳ = unthunk(y1)
        w̄2 = similar(w2)
        w̄3 = similar(w3)
        fill!(w̄2, 0)
        fill!(w̄3, 0)
        x̄ = similar(mat)
        fill!(x̄, 0)
        x_tmp2 = similar(mat, nf2, nfeat2)
        x_tmp3 = similar(mat, nf3, nfeat3)
        ȳ_tmp2 = similar(ȳ, nf2, neb2)
        ȳ_tmp3 = similar(ȳ, nf3, neb3)

        for i = 1:size(mat, 2)
            ȳ_tmp2[:] .= ȳ[n1bd+1:n2embed+n1bd, i]
            ȳ_tmp3[:] .= ȳ[n1bd+1+n2embed:n2embed+n1bd+n3embed, i]
            x_tmp2[:] .= mat[n1bd+1:n2bd+n1bd, i]
            x_tmp3[:] .= mat[n1bd+1+n2bd:n2bd+n1bd+n3bd, i]

            # Accumulate the gradient of the weights
            w̄2 .+= x_tmp2' * ȳ_tmp2
            w̄3 .+= x_tmp3' * ȳ_tmp3

            # Compute the gradient of the input matrix
            x̄[1:n1bd, i] .= ȳ[1:n1bd, i]
            x̄[n1bd+1:n2bd+n1bd, i] .= vec(ȳ_tmp2 * w2')
            x̄[n1bd+1+n2bd:n2bd+n1bd+n3bd, i] .= vec(ȳ_tmp3 * w3')
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), w̄2, w̄3, x̄

    end
    return y, pullback
end
