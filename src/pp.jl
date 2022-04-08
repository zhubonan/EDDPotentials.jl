#=
Pair-potential module
=#
using Optim

"""
Lennard-Jones energy
Note this give half of the actual value since we are doing double counting here.
"""
function lj(σ, ϵ, r2)
    six_term = (σ ^ 2 / r2) ^ 3
    4ϵ * (six_term ^ 2 - six_term)
end

"Shifted energy with: E + α r + β"
function lj(σ, ϵ, r2, α, β)
    lj(σ, ϵ, r2) + α * sqrt(r2) + β
end

"Derivative of the energy"
function ljf(σ, ϵ, r2)
    six_term = (σ ^ 2 / r2) ^ 3
    -24ϵ / r2 * (2 * six_term ^ 2 - six_term)
end

"Shifted force with: F + α"
function ljf(σ, ϵ, r2, α, β)
    ljf(σ, ϵ, r2) + α 
end

"Pair potential specification"
struct PPSpec{T}
    species::Vector{Symbol}  # Unique specie names
    ϵ::Matrix{T}       # ϵ
    σ::Matrix{T}       # σ
    rcut2::Matrix{T}   # Squared cut off
    α::Matrix{T}       # Force shift factor
    β::Matrix{T}       # Energy shift factor
end

"Lookup dictionary map species to indices"
lookup(p::PPSpec) = Dict(symbol=>idx for (idx, symbol) in enumerate(p.species))


"Assign a matrix based on mapping and a index lookup vector"
function assignmat!(mat::Matrix{T}, mapping::Dict{Pair{Symbol, Symbol}, T}, lookup::Dict{Symbol, Int}) where T
  for (pair, value) in mapping 
        i, j = lookup[pair.first], lookup[pair.second]
        mat[i, j] = value
        mat[j, i] = value
    end
    mat
end

function species(mapping::Dict)
    pairs = length(keys(mapping))
    all_species = Symbol[]
    for pair in keys(mapping)
        push!(all_species, pair.first)
        push!(all_species, pair.second)
    end
    all_species
end

function PPSpec(sigma::Dict, epsilon::Dict; range=2.5)
    specie_pairs = keys(sigma)
    all_species = Symbol[]
    for pair in specie_pairs
        push!(all_species, pair.first)
        push!(all_species, pair.second)
    end

    unique_species = sort(unique([species(sigma); species(epsilon)]))
    lookup = Dict(symbol=>idx for (idx, symbol) in enumerate(unique_species))

    nspec = length(unique_species)
    # Initialise matrices
    emat = zeros(Float64, nspec, nspec)
    smat = zeros(Float64, nspec, nspec)
    α = zeros(Float64, nspec, nspec)
    β = zeros(Float64, nspec, nspec)

    # Assign pairwise interaction matrix
    assignmat!(smat, sigma, lookup)
    assignmat!(emat, epsilon, lookup)

    # Assign the cut off matrix based on range scaling factor
    rcut2mat = (smat .* range) .^ 2
    PPSpec(unique_species, emat, smat, rcut2mat, α, β)
end

function compute_shifts!(p::PPSpec)
    for i in eachindex(p.σ)
        p.α[i] = -ljf(p.σ[i], p.ϵ[i], p.rcut2[i])
        p.β[i] = -(lj(p.σ[i], p.ϵ[i], p.rcut2[i]) + sqrt(p.rcut2[i]) * p.α[i])
    end
end

function evalute(s::Cell{T}, p::PPSpec) where T
    svecs = shift_vectors(s.lattice.matrix, sqrt(maximum(p.rcut2)))

    # Construct interge lookup table
    lok = lookup(p)
    spec = Int[lok[site.symbol] for site in s.sites]
    forces = zeros(T, 3, nions(s))

    eng = 0.0
    vtmp = zeros(T, 3)

    any(x->(x != 0), p.α) | any(x->(x !=0 ), p.β) ? has_shift = true : has_shift = false
    for (i, si) in enumerate(s.sites)   # Each site
        for (j, sj) in enumerate(s.sites)    # Each other site
            i == j && continue
            for (k, shift) in enumerate(eachcol(svecs))   # Each shift
                d2 = distance_squared_between(si, sj, shift)
                d2 > p.rcut2[spec[i], spec[j]] && continue
                ci, cj = spec[i], spec[j]
                σ = p.σ[ci, cj]
                ϵ = p.ϵ[ci, cj]
                if has_shift
                    α = p.α[ci, cj]
                    β = p.β[ci, cj]
                    eng += 0.5 * lj(σ, ϵ, d2, α, β)
                    vtmp[:] .= 0
                    forces[:, i] .+= ljf(σ, ϵ, d2, α, β) .* unit_vector_between!(vtmp, si, sj, shift)
                else
                    eng += 0.5 * lj(σ, ϵ, d2)
                    # Force
                    vtmp[:] .= 0
                    forces[:, i] .+= ljf(σ, ϵ, d2) .* unit_vector_between!(vtmp, si, sj, shift)
                end
            end
        end
    end
    eng, forces
end


function opti(s::Cell{T}, pp::PPSpec) where T
    "Avoid recalculations for forces/energy"
    function fg!(F, G, pos::Vector)
        # Assign positions
        for i in 1:length(pos)
            ns = div(i - 1, 3) + 1
            ni = (i - 1) % 3 + 1
            s.sites[ns].position[ni] = pos[i]
        end
        eng, forces = evalute(s, pp)
        if ~isnothing(G)
            copyto!(G, -forces[:])
        end
        if ~isnothing(F)
            return eng
        end
    end
    # Keep the initial positions
    x_init = positions(s)[:]
    optimize(Optim.only_fg!(fg!), x_init)
end