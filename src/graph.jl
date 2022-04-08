#=
Routines for generating graph related properties and analysis
=#
using LinearAlgebra

"""
Compute the Laplacian matrix
"""
function laplacian_matrix(structure::Cell; bl=3.0)
    dmat = distance_matrix(structure)
    lmat = similar(dmat)
    fill!(lmat, 0.)
    ni = size(lmat)[1]
    for i = 1:ni
        for j = 1:ni
            i == j && continue
            if dmat[i, j] < bl
                lmat[i, j] = -1
            end
        end
    end
    diag = sum(lmat, dims=2)
    for i = 1:ni
        lmat[i, i] -= diag[i]
    end
    lmat
end

"""
Sepecification of a modules in a structure
"""
struct ModuleSpec
    "Number of units"
    nunits::Int
    "Size of each unit"
    unit_sizes::Vector{Int}
    "Indices of each unit in the original structure"
    unit_indices::Matrix{Int}
end

nmodules(m::ModuleSpec;min=2) = count(x -> x >= min, m.unit_sizes)
natoms(m::ModuleSpec) = sum(m.unit_sizes)
nions(m::ModuleSpec) = natoms(m)

Base.getindex(spec::ModuleSpec, i::Int) = spec.unit_indices[1:spec.unit_sizes[i], i]

"""Clip a structure with modules"""
clip(s::Cell, spec::ModuleSpec, i::Int) = clip(s, spec[i])


function find_modules(s::Cell;bl=3.0)
    lmat = laplacian_matrix(s, bl=bl)
    find_modules(lmat)
end

"""
Find the components of the Graph from a given laplacian matrix
"""
function find_modules(lap::AbstractMatrix)
    ns = size(lap)[1]
    labels = fill(-1, ns)

    # Remove the diagonal elements - assuming no self-loop....
    adj = lap .* -1 
    adj .-= diagm(diag(adj))

    labels = dfs(adj)

    unique_labels = unique(labels)
    label_counts = [count(x -> x == l, labels) for l in unique_labels]

    # Sort by size of each component
    tmp = sortperm(label_counts, rev=true)
    label_counts = label_counts[tmp]
    unique_labels = unique_labels[tmp]

    # Array storing the indices of components for each label
    label_indices = zeros(maximum(label_counts), length(unique_labels))
    for (il, l) in enumerate(unique_labels)
        c = 1
        for i = 1:ns
            if labels[i] == l
                label_indices[c, il] = i
                c += 1
            end
        end
    end
    ModuleSpec(length(unique_labels), label_counts, label_indices)
end

"""
    dfs(adj::AbstractMatrix)

Perform Depth First Search for a given adjancy matrix
"""
function dfs(adj::AbstractMatrix)
    ns = size(adj, 1)
    current_label = 1
    current_node = 1
    labels = fill(-1, ns)
    while any(x -> x == -1, labels)
        current_node::Int = findfirst(x -> x == -1, labels)   # Locate the node to search from
        labels[current_node] = current_label   # Label this node
        # Start depath-first-search
        dfs!(labels, current_node, adj)
        current_label += 1   # Increment the label number
    end
    labels
end


"""
    dfs!(labels::AbstractVector, inode::Int, adj::AbstractMatrix)

Recursive Depth First Search. Unlabeled nodes have label `-1` in the 
`labels` vector.
"""
function dfs!(labels::AbstractVector, inode::Int, adj::AbstractMatrix)
    ns = size(adj)[1]
    for i = 1:ns
        if adj[i, inode] == 1 && labels[i] == -1
            labels[i] = labels[inode]
            # God down the depath
            dfs!(labels, i, adj) 
        end
    end
    inode += 1
end


"""
Decorate a structure's `set_indices` using the detected modules
"""
function decorate!(s::Cell, spec::ModuleSpec)
    for i in 1:spec.nunits
        indx = spec[i]
        s.set_indices[indx] .= i
    end
end

"""
Compute the dimensionality
"""
function dimensionality(s::Cell;bl=3.0)
    mspec = find_modules(s; bl)
    s222 = make_supercell(s, 2,2,2)
    mspec2 = find_modules(s222; bl)
    log(8 / (mspec2.nunits/ mspec.nunits)) / log(2), mspec, mspec2
end

"""
Iterator inteface for getting all components of a given minimum size
"""
struct IterModule{T}
    structure::Cell{T}
    spec::ModuleSpec
    min_size::Int
end

"""
Iterate over components of a given minimum size for this structure
"""
eachmodule(s::Cell, spec::ModuleSpec; min_size=2) = IterModule(s, spec, min_size)

##### Overide base methode for the iterator interface ######
function Base.length(x::IterModule)
    lim = 0
    for i = 1:x.spec.nunits
        if x.spec.unit_sizes[i] < x.min_size 
            break
        end
        lim += 1
    end
    lim
end

function Base.iterate(iter::IterModule, state=1)
    state > iter.spec.nunits && (return nothing)
    iter.spec.unit_sizes[state] < iter.min_size && (return nothing)
    clip(iter.structure, iter.spec, state), state + 1
end

Base.eltype(::IterModule{T}) where T = Cell{T}

##### End of the interface ######

"""
Return the unique modules of the structure
"""
function unique_modules(s::Cell, mspec::ModuleSpec;min_size=2, tol=0.1)
    submod = collect(eachmodule(s, mspec, min_size=min_size))
    unique_structures(submod;tol)
end

function unique_structures(structures::Vector{Cell{T}};tol=0.1) where {T}
    dists = [fingerprint(s) for s in structures]
    sym = [sort(species(s)) for s in structures]
    n = length(dists)
    dev_mat = zeros(n, n)
    for j = 1:n
        for i = j+1:n
            if sym[i] != sym[j]
                dev_mat[i, j] = 999.
                dev_mat[j, i] = 999.
                continue
            end
            d = fingerprint_distance(dists[i], dists[j])
            dev_mat[i, j] = d
            dev_mat[j, i] = d
        end
    end
    # Matrix of identical modules
    sim = dev_mat .< tol
    labels = dfs(sim)
    unique_idx = Int[findfirst(x -> x == l, labels) for l in unique(labels)]
    return structures[unique_idx]
end