
"""
    glob_allow_abs(path)

Allow path to start with "/" when globbing by converting it to a relative path.
"""
function glob_allow_abs(path)
    if startswith(path, "/")
        path = relpath(path)
    end
    return glob(path)
end

"""
    _split_vector(c, nsplit::Vararg{Real}; shuffle=true, seed=42)

Split a vector by fractions.
"""
function _split_vector(c, nsplit::Vararg{Real}; shuffle=true, seed=42)
    ntot = length(c)
    intsplit = nsplit .* ntot .|> floor .|> Int
    _split_vector(c, intsplit...; shuffle, seed)
end

include("indexvector.jl")