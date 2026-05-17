#=
routine for computing energy and forces with pair potentials
=#

function _repulsion(r, c, p)
    if r < c
        return (2.0 * (c / r - 1.0))^p
    end
    return zero(r)
end


function _grepulsion(r, c, p)
    if r < c
        return -p * (2 * (c / r - 1.0))^(p - 1) / (r * r) * c * 2
    end
    return zero(r)
end



mutable struct RepulsionSpec{F,G,T}
    f::F
    g::G
    radii::Matrix{T}
    symbols::Vector{Symbol}
    p::T
end

function build_repulsion_spec(pairs...; p=12.0)
    all_symbols = sort(unique(vcat(map(x -> collect(first(x)), pairs)...)))
    radii = zeros(length(all_symbols), length(all_symbols))
    for ((a, b), r) in pairs
        ia = findfirst(x -> x == a, all_symbols)
        ib = findfirst(x -> x == b, all_symbols)
        radii[ia, ib] = r
        radii[ib, ia] = r
    end
    RepulsionSpec(_repulsion, _grepulsion, radii, all_symbols, p)
end


mutable struct PairpotWorkspace{T,S}
    forces::Matrix{T}
    energy::T
    stress::Matrix{T}
    spec::S
end

function PairpotWorkspace(cell::Cell{T,D}, spec) where {T,D}
    forces = zeros(T, D, length(cell))
    stress = zeros(T, 3, 3)
    PairpotWorkspace(forces, zero(T), stress, spec)
end

"""
Compute the energy, forces and stress for a repulsion potential
"""
function pairpot_energy_forces(workspace::PairpotWorkspace{T, S}, cell, nl) where {T, S<:RepulsionSpec}
    @assert isa(workspace.spec, RepulsionSpec)
    energy = zero(T)
    forces = workspace.forces
    forces .= zero(T)
    stress = workspace.stress
    stress .= zero(T)
    # Map species to index indices
    idxmap = [findfirst(x -> x == sym, workspace.spec.symbols) for sym in species(cell)]

    for iat = 1:length(cell)
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            idxi, idxj = idxmap[iat], idxmap[jat]
            gcore =
                -workspace.spec.g(rij, workspace.spec.radii[idxi, idxj], workspace.spec.p)
            if iat != jat
                for elm in axes(modvij, 1)
                    forces[elm, iat] -= 2 * gcore * modvid[elm]
                end
            end
            for elm1 in axes(stress, 1)
                for elm2 in axes(stress, 2)
                    stress[elm1, elm2] += vij[elm1] * modvij[elm2] * gcore
                end
            end
            energy +=
                workspace.spec.f(rij, workspace.spec.radii[idxi, idxj], workspace.spec.p)
        end
    end
    workspace.energy = energy
    energy, forces, stress
end


mutable struct PairpotCalc{T,D,S} <: AbstractCalc
    cell::Cell{T,D}
    nl::NeighbourList{T,D}
    workspace::PairpotWorkspace{T,S}
end

get_cell(p::PairpotCalc) = p.cell

PairpotCalc(cell::Cell, workspace::PairpotWorkspace; 
    nl=NeighbourList(cell, 6.0; savevec=true),
) = PairpotCalc(cell, nl, workspace)

set_positions!(calc::PairpotCalc, new) = set_positions!(calc.cell, new)
set_cellmat!(calc::PairpotCalc, new) = set_cellmat!(calc.cell, new)

function calculate!(calc::PairpotCalc;kwargs...)
    rebuild!(calc.nl, get_cell(calc))
    pairpot_energy_forces(calc.workspace, calc.cell, calc.nl)
end

function eddp_forces(calc::PairpotCalc;kwargs...)
    calculate!(calc)
    calc.workspace.forces
end

function eddp_energy(calc::PairpotCalc; kwargs...)
    calculate!(calc)
    calc.workspace.energy
end

function eddp_stress(calc::PairpotCalc; kwargs...)
    calculate!(calc)
    calc.workspace.stress
end

get_positions(calc::PairpotCalc) = get_positions(calc.cell)



mutable struct LJSpec{T}
    sigma::Matrix{T}
    epsilon::Matrix{T}
    symbols::Vector{Symbol}
    cutoff::T
    eshifts::Matrix{T}
    fshifts::Matrix{T}
    p1::Matrix{Int}
    p2::Matrix{Int}
end

function build_lj_spec(pairs...; p1=12, p2=6, epsilon=1.0, cutoff=nothing, cutoff_ratio=3.0)
    all_symbols = sort(unique(vcat(map(x -> collect(first(x)), pairs)...)))
    T = typeof(pairs[1][2])
    sigma_mat = zeros(T, length(all_symbols), length(all_symbols))
    for ((a, b), r) in pairs
        ia = findfirst(x -> x == a, all_symbols)
        ib = findfirst(x -> x == b, all_symbols)
        sigma_mat[ia, ib] = r
        sigma_mat[ib, ia] = r
    end
    # Construct the epsilon and p matrix - they only contains constants for now
    epsilon_mat = fill(epsilon, size(sigma_mat))
    p1_mat = fill(p1, length(all_symbols), length(all_symbols))
    p2_mat = fill(p2, length(all_symbols), length(all_symbols))

    # Determine the cut off radius
    if cutoff == nothing
        cutoff = maximum(sigma_mat) * cutoff_ratio
    end

    # Pre-compute the energy and force shifts
    eshifts = zeros(T, length(all_symbols), length(all_symbols))
    fshifts = zeros(T, length(all_symbols), length(all_symbols))
    for idx in eachindex(sigma_mat)
        eshifts[idx] =
            -e_lj(cutoff, sigma_mat[idx], epsilon_mat[idx], p1=p1_mat[idx], p2=p2_mat[idx])
        fshifts[idx] =
            -f_lj(cutoff, sigma_mat[idx], epsilon_mat[idx], p1=p1_mat[idx], p2=p2_mat[idx])
    end

    LJSpec(sigma_mat, epsilon_mat, all_symbols, cutoff, eshifts, fshifts, p1_mat, p2_mat)
end


function e_lj(r12, sigma, epsilon; p1=12, p2=6)
    return 4epsilon * ((sigma / r12)^p1 - (sigma / r12)^p2)
end

function f_lj(r12, sigma, epsilon; p1=12, p2=6)
    return 4epsilon *
           (p1 * (sigma / r12)^(p1 - 1) * -p2 * (sigma / r12)^(p2 - 1)) *
           (-sigma * r12^(-2))
end

"""Compute lennard-jones energy, forces and stress"""
function pairpot_energy_forces(workspace::PairpotWorkspace{T, S}, cell, nl) where {T, S<:LJSpec}
    @assert isa(workspace.spec, LJSpec)
    energy = zero(T)
    forces = workspace.forces
    forces .= zero(T)
    stress = workspace.stress
    stress .= zero(T)
    (; p1, p2, epsilon, sigma, fshifts, eshifts) = workspace.spec

    # Map species to index indices
    idxmap = [findfirst(x -> x == sym, workspace.spec.symbols) for sym in species(cell)]

    for iat = 1:length(cell)
        for (jat, jextend, rij, vij) in CellBase.eachneighbourvector(nl, iat)
            modvij = vij / rij
            idxi, idxj = idxmap[iat], idxmap[jat]
            # Compute the gradient against the rij (shifted)
            gtmp =
                -f_lj(
                    rij,
                    sigma[idxi, idxj],
                    epsilon[idxi, idxj];
                    p1=p1[idxi, idxj],
                    p2=p2[idxi, idxj],
                ) - rij * fshifts[idxi, idxj]
            if iat != jat
                for elm in axes(modvij, 1)
                    forces[elm, iat] -= 2 * gtmp * modvij[elm]
                end
            end
            for elm1 in axes(stress, 1)
                for elm2 in axes(stress, 2)
                    stress[elm1, elm2] += vij[elm1] * modvij[elm2] * gtmp
                end
            end
            # Compute the energy (shifted)
            energy +=
                e_lj(
                    rij,
                    sigma[idxi, idxj],
                    epsilon[idxi, idxj];
                    p1=p1[idxi, idxj],
                    p2=p2[idxi, idxj],
                ) + eshifts[idxi, idxj]
        end
    end
    workspace.energy = energy
    energy, forces, stress
end

