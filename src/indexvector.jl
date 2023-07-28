"""
Storage to store the number-based indices of another array.
"""
struct IndexVector
    index::Vector{Int}
    allow_growth::Bool
end

IndexVector(x::Int;allow_growth=false) = IndexVector(fill(-1, x), allow_growth)

function get_index(arr::IndexVector, j)
    i = 1
    found = false
    # Check for existing entry of j
    while i <= length(arr.index)
        if arr.index[i] == -1
            break
        end
        if arr.index[i] == j
            found = true
            break
        end
        i += 1
    end
    # Not not existing entry - try insert an new one
    if found == false
        if i > length(arr.index)
            if arr.allow_growth
                _increase_size!(arr)
            else
                throw(ErrorException("IndexVector is full!"))
            end
        end
        arr.index[i] = j
    end
    i
end

"""Number of empty slots"""
function nempty(arr::IndexVector)
    sum(x -> x == -1, arr.index)
end

"""
Increase the storage size of the indexing vector
"""
function _increase_size!(arr::IndexVector, n=length(arr))
    l = length(arr)
    resize!(arr.index, n + l)
    arr.index[l+1:end] .= -1
    arr
end

"""Number of filled slots"""
nfilled(arr::IndexVector) = length(arr.index) - nempty(arr)
Base.length(x::IndexVector) = length(x.index)
clear!(arr::IndexVector) = fill!(arr.index, -1)