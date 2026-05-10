import Base
using LinearAlgebra
import CellBase:
    set_cellmat!,
    set_positions!,
    get_cellmat,
    get_positions,
    add_dimensions,
    remove_dimensions,
    distance_squared_between
using StatsBase: ZScoreTransform, transform!
"""
    AbstractCalc

Abstract supertype for all EDDP calculators. Concrete subtypes must implement:
- `get_cell(calc)` — return the `Cell` object
- `calculate!(calc; forces=true, rebuild_nl=true)` — trigger energy/force/stress computation
- `eddp_energy(calc)` — total energy
- `eddp_forces(calc)` — atomic forces matrix
- `eddp_stress(calc)` — 3×3 stress tensor

See [`EDDPCalculator`](@ref), [`MetaCalculator`](@ref).
"""
abstract type AbstractCalc end
const AC = AbstractCalc

## Methods to be implemented for the Abstract type

"""
    get_cell(calc::AbstractCalc) -> Cell

Return the `Cell` object associated with the calculator.
"""
function get_cell end

"""
    eddp_energy(calc::AbstractCalc) -> Number

Return the total energy (eV) of the system.
"""
function eddp_energy end

"""
    calculate!(calc::AbstractCalc; forces=true, rebuild_nl=true)

Compute energy, forces, and stress. Results are cached internally — subsequent
calls with an unchanged cell are no-ops unless `forces=true` is requested after
an energy-only call. Set `rebuild_nl=false` to skip neighbor list rebuild and
instead update incrementally.
"""
function calculate! end

"""
    eddp_forces(calc::AbstractCalc) -> Matrix

Return the atomic forces as a `D×N` matrix where `D` is the number of spatial
dimensions and `N` is the number of atoms. Forces are in eV/Å.
"""
function eddp_forces end

"""
    eddp_stress(calc::AbstractCalc) -> Matrix

Return the 3×3 stress tensor (eV/Å³). The stress is the virial stress divided
by the cell volume.
"""
function eddp_stress end


## Base default implementation 

function get_cellmat(ac::AC)
    get_cellmat(get_cell(ac))
end

get_positions(ac::AC) = get_positions(get_cell(ac))

function set_cellmat!(calc::AC, cellmat; kwargs...)
    set_cellmat!(get_cell(calc), cellmat; kwargs...)
end


function set_positions!(calc::AC, positions)
    set_positions!(get_cell(calc), positions)
end

function Base.show(io::IO, c::AbstractCalc)
    println(io, "$(typeof(c).name.name) for: ")
    Base.show(io, get_cell(c))
end

function Base.show(io::IO, m::MIME"text/plain", cw::AbstractCalc)
    println(io, "$(typeof(cw).name.name) for: ")
    Base.show(io, m, get_cell(cw))
end

### Helper functions

"""
    _cell_unchanged(cell, last_cellmat, last_positions, last_species)

Check if the cell matches the cached state (cell matrix, positions, species).
"""
function _cell_unchanged(cell, last_cellmat, last_positions, last_species)
    all(cellmat(cell) .== last_cellmat) || return false
    all(positions(cell) .== last_positions) || return false
    all(species(cell) .== last_species) || return false
    return true
end

_make_symbol_keys(x) = x
function _make_symbol_keys(dict::Dict{T,K}) where {T,K}
    Dict{Symbol,K}(Symbol(key) => _make_symbol_keys(value) for (key, value) in pairs(dict))
end
get_elemental_energy(cell::Cell, ref_energies::Dict) =
    sum(x -> get(ref_energies, x, 0.0), species(cell))

function eddp_enthalpy(calc::AbstractCalc; kwargs...)
    eddp_energy(calc; kwargs...)
end
    eddp_pressure_gpa(vc::AbstractCalc) 

Return pressure in unit of GPa.
"""
function eddp_pressure_gpa(vc::AbstractCalc)
    eVAngToGPa(tr(eddp_stress(vc)) / 3.0)
end

"""
     check_global_minsep(nl::NeighbourList, threshold)

Check if there are atoms that are too close to each other.
"""
function check_global_minsep(nl::NeighbourList, minsep)
    for i = 1:length(nl.nneigh)
        for (_, _, d) in eachneighbour(nl, i)
            if d < minsep
                return false
            end
        end
    end
    return true
end

"""
    eddp_pressure(calc::AbstractCalc)

Return the total pressure obtained from the trace of the stress tensor.
"""
function eddp_pressure(calc::AbstractCalc)
    stress = eddp_stress(calc)
    return (stress[1, 1] + stress[2, 2] + stress[3, 3]) / 3
end


## Add Convenience methods
for func in [:eddp_energy, :eddp_forces, :eddp_pressure, :eddp_energy_std, :eddp_enthalpy]
    @eval begin
        function $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
            $func(EDDPCalculator(cell, cf, itf); kwargs...)
        end
        @doc """
            $($func)(cell::Cell, cf::CellFeature, itf::AbstractNNInterface;kwargs...)

        Convenient method for calling $($func) with a Calculator constructed ad-hoc.
        """ $func(cell::Cell, cf::CellFeature, itf::AbstractNNInterface; kwargs...)
    end
end

"""
    eddp_energy_per_atom(calc_or_cell, args...; kwargs...) -> Number

Total energy normalized by the number of atoms (eV/atom).
"""
eddp_energy_per_atom(cell::Cell, args...; kwargs...) =
    eddp_energy(cell, args...; kwargs...) / length(cell)
eddp_energy_per_atom(calc::AbstractCalc, args...; kwargs...) =
    eddp_energy(calc, args...; kwargs...) / length(get_cell(calc))
"""
    eddp_energy_std_per_atom(calc_or_cell, args...; kwargs...) -> Number

Ensemble energy standard deviation normalized per atom (eV/atom).
Useful for assessing model confidence independently of system size.
"""
eddp_energy_std_per_atom(cell::Cell, args...; kwargs...) =
    eddp_energy_std(cell, args...; kwargs...) / length(cell)
eddp_energy_std_per_atom(calc::AbstractCalc, args...; kwargs...) =
    eddp_energy_std(calc, args...; kwargs...) / length(get_cell(calc))
