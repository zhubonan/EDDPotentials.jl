# Experimental Hyperdimensional Relaxation

!!! warning "Experimental"
    Hyperdimensional relaxation is experimental. Use it for exploration and
    method development, and validate resulting structures carefully with
    standard 3D relaxation and reference calculations.

EDDPotentials supports **hyperdimensional relaxation** — an approach where
extra spatial dimensions are temporarily added to a structure, and the penalty
for those extra dimensions is gradually increased during relaxation. This can
help escape local minima in the potential energy landscape.

## What Is Hyperdimensional Relaxation?

In standard 3D relaxation, atom positions are confined to three spatial
dimensions. Hyperdimensional relaxation works by:

1. **Adding extra dimensions** (e.g., 3 extra dimensions → 6D space)
2. **Applying a harmonic penalty** that pushes extra-dimensional coordinates
   back to zero
3. **Gradually increasing the penalty** through a schedule, allowing the
   structure to explore configurations that would be blocked by barriers in
   pure 3D space

The extra dimensions act as a "tunneling" mechanism — atoms can move through
the extra dimensions to find better 3D arrangements.

## Using the MetaCalculator Pipeline

The recommended approach uses `MetaCalculator` with the
`ExtraDimensionPenaltyModifier`.

```julia
using EDDPotentials
using CellBase: add_dimensions, remove_dimensions

# Start with a 3D cell
cell = Cell(Lattice(5.0, 5.0, 5.0), [:C, :C], Float64[
    1.0 2.5
    1.0 2.5
    1.0 2.5
])

# Add 3 extra dimensions
ndim = 3
hcell = add_dimensions(cell, ndim)

# Create source calculator
source = create_calculator(eddp_calc, hcell)

# Build MetaCalculator with penalty modifier
penalty = ExtraDimensionPenaltyModifier(1e-4; start_dim=4)
calc = MetaCalculator(source, [penalty])

# Relax with scheduled penalty
results, dnorm = scheduled_penalty_relax!(calc;
    schedule_every=50,
    max_iterations=500,
    alpha_start=1e-4,
    schedule_function=_get_penalty_schedule(),
    relax_fn=opt_tpsd,
)
```

## Using `hyper_gen_relax`

There are two workflows for calling `hyper_gen_relax`: one using a `Builder`
configuration (for the automated active-learning pipeline), and one using an
`EDDPCalculator` directly (for standalone use with pre-trained models).

### Builder Workflow

For the full automated workflow with a trained ensemble:

```julia
using EDDPotentials

# Load the builder configuration
bu = Builder("link.toml")

# Generate and relax a structure in hyperspace
results, dnorm = hyper_gen_relax(bu;
    ndim=3,
    schedule_every=50,
    target_iterations=500,
    alpha_start=1e-4,
    outfile="relaxed.res",
    label="hyper_struct",
)

# results contains the optimizer returns per schedule block
# dnorm tracks the maximum extra-dimension coordinate norm
```

### EDDPCalculator Workflow

With a standalone trained model (no Builder needed):

```julia
using EDDPotentials

# Load a trained model and structure
calc = load_calculator("model.json")
cell = read_res("input.res")

# Hyperdimensional relaxation directly from the calculator
results, dnorm = hyper_gen_relax(calc, cell;
    ndim=3,
    alpha_start=1e-4,
    schedule_every=50,
    target_iterations=500,
    external_pressure_GPa=0.0,
    relax_fn=opt_tpsd,
)
```

This overload uses `create_calculator` internally to bind the model
to the hyperdimensional cell with the appropriate modifier pipeline
(`ExtraDimensionPenaltyModifier`).

For ordinary `RelaxOption`-based relaxation, use
`external_pressure_gpa` for hydrostatic pressure in GPa:

```julia
opts = RelaxOption(;
    relax_cell = true,
    external_pressure_gpa = 25.0,
)
relax!(Relax(calc, opts))
```

Positive values are compressive and negative values are tensile. The older
`external_pressure` field is deprecated because it uses internal eV/Å³ stress
units; set only one of the two pressure interfaces.

## Lennard-Jones Hyperspace Example

Here is a complete, self-contained example using a Lennard-Jones-like
calculator to demonstrate hyperdimensional relaxation:

```julia
using EDDPotentials
using CellBase: add_dimensions, remove_dimensions, positions

# Build a simple LJ calculator
function lj_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)
    cf = CellFeature([:Ar]; rcut2=rcut, p2_geometry=false, p2=[2.0])
    # Use an lj_baseline calculator
    EDDPotentials.lj_baseline_calc(cell; sigma, epsilon, rcut)
end

# Create a random starting configuration
cell = Cell(Lattice(8.0, 8.0, 8.0), repeat([:Ar], 4), rand(3, 4) .* 4.0)

# ====== Option 1: Scheduled penalty relaxation with MetaCalculator ======
ndim = 3
hcell = add_dimensions(cell, ndim)

source = lj_calc(hcell; sigma=1.0, epsilon=1.0, rcut=3.0)
penalty = ExtraDimensionPenaltyModifier(1e-4; start_dim=4)
metacalc = MetaCalculator(source, [penalty])

results, dnorm = scheduled_penalty_relax!(metacalc;
    schedule_every=50,
    max_iterations=500,
    alpha_start=1e-4,
    schedule_function=_get_penalty_schedule(),
    relax_fn=opt_tpsd,
)

println("Extra-dim norms: ", dnorm)
println("Final energy: ", eddp_energy(metacalc))

# Extract the relaxed 3D structure
relaxed = remove_dimensions(hcell)

# ====== Option 2: Manual step-by-step penalty schedule ======
hcell2 = add_dimensions(cell, ndim)
source2 = lj_calc(hcell2; sigma=1.0, epsilon=1.0, rcut=3.0)
penalty2 = ExtraDimensionPenaltyModifier(1e-4; start_dim=4)
metacalc2 = MetaCalculator(source2, [penalty2])

alphas = [1e-4, 1e-3, 1e-2, 1e-1]
for alpha in alphas
    set_modifier_parameter!(metacalc2, ExtraDimensionPenaltyModifier, :alpha, alpha)
    opt = opt_tpsd(metacalc2; itermax=100, f_tol=1e-3)
    println("Alpha $alpha: iterations=$(opt.iteration), converged=$(converged(opt, metacalc2))")
end

# ====== Option 3: Variable-cell hyperdimensional relaxation ======
hcell3 = add_dimensions(cell, ndim)
source3 = lj_calc(hcell3; sigma=1.0, epsilon=1.0, rcut=3.0)
penalty3 = ExtraDimensionPenaltyModifier(1e-4; start_dim=4)
metacalc3 = MetaCalculator(source3, [penalty3])

results3, dnorm3 = scheduled_penalty_relax!(metacalc3;
    schedule_every=50,
    max_iterations=500,
    alpha_start=1e-4,
    relax_fn=opt_tpsd,
    relax_kwargs=(relax_cell=true,),
)

relaxed3 = remove_dimensions(hcell3)
println("VC-relaxed cell volume: ", volume(relaxed3))
println("VC-relaxed energy: ", eddp_energy(metacalc3))

# Compare with standard 3D relaxation
source_3d = lj_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)
opt_3d = opt_tpsd(source_3d; itermax=500, relax_cell=true)
println("Standard 3D energy: ", eddp_energy(source_3d))
```

## Configuring via link.toml

For `link!` / RSS workflows, hyperspace settings live under `state.relax`:

```toml
[state]
relax_mode = "hyperspace"

    [state.relax]
    hyper_ndim = 3
    hyper_alpha_start = 1e-4
    hyper_schedule_every = 50
    hyper_target_iterations = 500
```

`hyper_gen_relax(bu)` itself is configured through its keyword arguments.

## Penalty Schedule

The schedule function controls how the extra-dimension penalty coefficient
(`alpha`) grows during relaxation:

```
built-in: alpha stays unchanged for first 5 blocks, then alpha = min(n * 0.1, alpha * 2)
```

Custom schedules can be provided:

```julia
function my_schedule(alpha)
    # Exponential ramp
    return alpha * 2.0
end

scheduled_penalty_relax!(metacalc;
    schedule_function=my_schedule,
    ...
)
```

## API Reference

```@docs
EDDPotentials.Workflow.hyper_get_calc
EDDPotentials.Workflow.hyper_gen_relax
EDDPotentials.Search.scheduled_penalty_relax!
EDDPotentials.Potential.get_modifier_parameter
EDDPotentials.Potential.set_modifier_parameter!
```
