import ChainRulesCore: rrule, NoTangent, unthunk

"""
Custom rule for the _apply_embedding_batch function
"""
function rrule(::typeof(_apply_embedding_batch), w::AbstractMatrix, x::AbstractMatrix;kwargs...)

    y = _apply_embedding_batch(w, x)
    nf = div(size(x, 1), size(w, 1)) # Number of elements for each feature
    neb = size(w, 2)   # number of embedding
    nfeat = size(w, 1)  # Number of features 

    function pullback(ȳ)
        ȳ = unthunk(ȳ)
        w̄ = similar(w)
        fill!(w̄, 0)
        x̄ = similar(x)
        fill!(x̄, 0)
        x_tmp = similar(x, nf, nfeat) 
        ȳ_temp = similar(ȳ, nf, neb)
        for i in 1:size(x, 2)
            ȳ_temp[:] .= ȳ[:, i]
            x_tmp[:] .=  x[:, i]
            w̄ .+= x_tmp' * ȳ_temp
            x̄[:, i] .= vec(ȳ_temp * w')
        end
        NoTangent(), w̄, x̄
    end
    return y, pullback
end