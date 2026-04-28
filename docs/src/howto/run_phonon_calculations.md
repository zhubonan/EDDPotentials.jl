# Run Phonon Calculations

Calculate phonon properties (band structures, density of states, thermodynamic properties) using trained EDDP potentials with Phonopy and ASE integration.

## Prerequisites

### Install Python Dependencies

Phonon calculations require Python packages accessible from Julia via PyCall.jl. Set up a conda environment:

```bash
conda create -n eddp-phonon python=3.10 ase phonopy
conda activate eddp-phonon
```

### Configure PyCall.jl

Build PyCall.jl to use your conda environment:

```bash
export PYTHON=$(which python)
julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall")'
unset PYTHON
```

The key step is building PyCall.jl with the `PYTHON` environment variable pointing to your conda environment's Python interpreter.

### Verify Installation

Check that Julia can import Python packages:

```julia
using PyCall

# Test ASE import
ase = pyimport("ase")
println("ASE version: ", ase.__version__)

# Test Phonopy import
phonopy = pyimport("phonopy")
println("Phonopy version: ", phonopy.__version__)
```

If imports fail, rebuild PyCall.jl with the correct Python path.

## Workflow Overview

Phonon calculations require:
1. **Fully relaxed structure** (both atomic positions and cell)
2. **Trained EDDP potential** (loaded as calculator)
3. **Finite displacement calculations** (automated by EDDPotentialsTools)
4. **Post-processing** (using phonopy)

## Step 1: Relax Structure

Start with a structure file (e.g., SHELX `.res` format):

```julia
using EDDPotentials
using CellBase: read_res

# Load trained model
builder = Builder("link.toml")
println("Using iteration: ", builder.state.iteration)

# Read structure
structure = read_res("my_structure.res")

# Create calculator with ensemble
calc = NNCalc(structure, builder.cf, load_ensemble(builder))

# Relax with variable cell optimization (important for phonons!)
vc_calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc))])
converged = opt_tpsd(vc_calc; f_tol=0.01, itermax=200)
@assert converged

# Extract relaxed structure
relaxed_structure = get_cell(vc_calc)
println("Relaxed volume: ", volume(relaxed_structure), " Å³")
```

**Important**: Use variable cell relaxation (`MetaCalculator` with `CellRelaxationModifier`) to ensure the lattice parameters are at equilibrium. Fixed-cell relaxation can give incorrect phonon frequencies.

## Step 2: Run Finite Displacement

Use EDDPotentialsTools to automate finite displacement calculations:

```julia
using EDDPotentialsTools

# Run phonon calculation
EDDPotentialsTools.run_phonon(
    vc_calc;
    outdir="phonon",       # Output directory
    supercell=[2, 2, 2],   # Supercell size (optional)
    distance=0.01          # Displacement distance in Å (optional)
)
```

This creates:
- `phonon/FORCE_SETS` - Forces for each displacement
- `phonon/phonopy_params.yaml` - Phonopy configuration
- `phonon/phonopy_disp.yaml` - Displacement information

## Step 3: Post-Processing with Phonopy

### Band Structure

Calculate phonon band structure using phonopy:

```bash
cd phonon
phonopy --dim="2 2 2" -p band.conf
```

Create `band.conf`:
```
ATOM_NAME = C Si  # Element symbols
DIM = 2 2 2
BAND = 0.0 0.0 0.0  0.5 0.0 0.5  # G-X path
BAND_LABELS = $\Gamma$ X
```

### Density of States

Calculate phonon DOS:

```bash
phonopy --dim="2 2 2" -p mesh.conf
```

Create `mesh.conf`:
```
ATOM_NAME = C Si
DIM = 2 2 2
MP = 20 20 20  # Mesh density
DOS = .TRUE.
```

### Thermodynamic Properties

Calculate thermal properties (heat capacity, entropy, free energy):

```bash
phonopy --dim="2 2 2" -t mesh.conf
```

Outputs `thermal_properties.yaml` with temperature-dependent properties.

## Advanced: Using Sumo

The [sumo](https://smtg-ucl.github.io/sumo/) package provides convenient commands for phonon analysis:

```bash
# Install sumo
pip install sumo

# Automatic band structure with high-symmetry path
cd phonon
sumo-phonon-bandplot --dim="2 2 2"
```

This automatically:
- Determines high-symmetry path from space group
- Calculates band structure
- Generates publication-quality plots

## Tips for Accurate Phonon Calculations

### Force Convergence

Use tight force convergence for relaxation:
- `fmax=0.01` eV/Å for most cases
- `fmax=0.001` eV/Å for high-accuracy phonons
- Check residual forces: `maximum(abs.(eddp_forces(vc_calc)))`

### Supercell Size

Supercell should be large enough to avoid interactions:
- **Minimum**: 2×2×2 (8× unit cell)
- **Recommended**: 3×3×3 for better convergence
- **Large systems**: Use smaller supercells (even 1×1×1)

Rule of thumb: supercell lattice vectors should be > 10 Å

### Displacement Distance

Default (0.01 Å) works for most cases:
- **Smaller** (0.005 Å) for very soft modes
- **Larger** (0.03 Å) for very stiff materials
- Too large: anharmonicity affects results
- Too small: numerical noise dominates

### Imaginary Frequencies

If you see negative (imaginary) frequencies:

**At Γ point only (< 0.1 THz)**:
- Numerical noise - likely acceptable
- Check force convergence was tight enough

**Significant imaginary modes (> 1 THz)**:
- Structure is unstable
- Relax more carefully (tighter fmax)
- Check if EDDP potential is accurate for this structure

## Troubleshooting

### PyCall Cannot Find Python Packages

**Problem**: `pyimport("ase")` fails

**Solutions**:
- Rebuild PyCall.jl with correct Python
- Verify packages installed: `conda list` (from activated environment)
- Check Python path: `PyCall.python` in Julia

### Forces Not Converging

**Problem**: Relaxation doesn't converge to `fmax` threshold

**Solutions**:
- Increase `itermax`
- Use smaller initial structure (closer to equilibrium)
- Check EDDP potential quality on this structure type
- Try `opt_fire(vc_calc; f_tol=...)` instead of `opt_tpsd(...)`

### Phonopy Errors

**Problem**: Phonopy fails to read `FORCE_SETS`

**Solutions**:
- Check `phonon/` directory contains all required files
- Verify `--dim` matches supercell used in `run_phonon`
- Check element symbols in `ATOM_NAME` match structure

## Example: Complete Workflow

Here's a full example for silicon:

```julia
using EDDPotentials
using EDDPotentialsTools
using CellBase: read_res

# 1. Load model and structure
builder = Builder("si_potential/link.toml")
structure = read_res("Si.res")

# 2. Relax structure
calc = NNCalc(structure, builder.cf, load_ensemble(builder))
vc_calc = MetaCalculator(calc, [CellRelaxationModifier(get_cell(calc))])
converged = opt_tpsd(vc_calc; f_tol=0.001, itermax=300)
@assert converged

# 3. Calculate phonons
EDDPotentialsTools.run_phonon(
    vc_calc;
    outdir="phonon_si",
    supercell=[3, 3, 3],
    distance=0.01
)

println("Phonon calculation complete. See phonon_si/ directory")
```

Then post-process:

```bash
cd phonon_si
sumo-phonon-bandplot --dim="3 3 3"
```

This generates phonon band structure plots and data files ready for analysis or publication.
