module Simulation

using ..EDDPotentialsCore
using Printf
import CellBase
using CellBase: nions, positions, species, masses, cellmat, remove_dimensions
import ..EDDPotentials: EDDPotentialsOption
import ..EDDPotentialsCore:
    get_cellmat,
    eVAngToGPa,
    get_modifier_parameter,
    set_modifier_parameter!

include("optimizer/abstract.jl")
include("optimizer/utils.jl")
include("optimizer/constraints.jl")
include("optimizer/tpsd.jl")
include("optimizer/fire.jl")
include("opt.jl")
include("relax.jl")
include("md.jl")

export AbstractOptimizer,
    NativeTPSD,
    NativeFIRE,
    AbstractStepConstraint,
    ZeroTranslationConstraint,
    FixAtoms,
    FixCartesian,
    FixScaled,
    FixRelative,
    FixRigidGroup,
    FixLinearIonic,
    IonicConstraint,
    FixCell,
    FixCellConstraint,
    FixVolumeConstraint,
    TrajectoryFrame,
    RelaxOption,
    Relax,
    RelaxResult,
    MDIntegrator,
    MDRunner,
    MDState,
    VelocityVerlet,
    Langevin,
    BerendsenNVT,
    apply_step_constraints!,
    apply_force_constraints!,
    apply_stress_constraints!,
    initialize!,
    step!,
    converged,
    reset!,
    get_iteration,
    optimize!,
    opt_tpsd,
    opt_fire,
    relax!,
    multirelax!,
    external_stress_matrix,
    external_pressure_matrix,
    run!,
    kinetic_energy,
    temperature,
    maxwell_boltzmann_velocities,
    total_energy,
    instantaneous_pressure,
    zero_com_velocity!,
    rescale_velocities!,
    compute_kinetic_energy,
    compute_temperature,
    save_trajectory,
    save_frame,
    KB,
    time_fs,
    time_ps

end # module Simulation
