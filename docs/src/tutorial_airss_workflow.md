# Advanced Tutorial: Automated EDDP Workflow with AIRSS

This tutorial describes the automated iterative workflow for building EDDP potentials using the AIRSS (Ab Initio Random Structure Searching) framework and CASTEP for DFT calculations.

> **Understanding the Workflow**: Before using this tutorial, read [Understanding Iterative Potential Building](concepts_iterative_building.md) to understand how the active learning loop works, what happens during each iteration, and how to configure and monitor the process.

## Overview

The EDDPotentials.jl package includes an automated workflow (`link!`) that:
1. Generates random structures using AIRSS `buildcell`
2. Calculates energies with DFT (CASTEP)
3. Trains an ensemble of neural network potentials
4. Uses the potential to explore the potential energy surface
5. Iterates until convergence

This approach is designed for **crystal structure prediction** across entire chemical spaces, not just fixed compositions.

## Prerequisites

### Required Software

- **Julia 1.9+** with EDDPotentials.jl installed
- **AIRSS** suite (for `buildcell` and structure analysis tools)
- **CASTEP** (or another DFT code with appropriate interface)
- **GNU Parallel** (optional, for parallel structure searches)

### Installation Check

Verify that the required executables are in your PATH:
```bash
which buildcell
which crud.pl
which cryan
which castep
```

## Building Input Files

The basic workflow requires four input files:

### 1. XX-SEARCH.cell (Random Structure Generation)

This file controls how random structures are generated. Create it using the AIRSS `gencell` tool:

```bash
gencell XX
```

**Important modifications:**
- **Disable symmetry**: Comment out or remove `#SYMOPS` to generate structures without symmetry constraints. This ensures comprehensive PES sampling.
- **For multi-component searches**: Delete the `POSITIONS` section and uncomment the atom count section. Use `#FOCUS` to select which compositions to sample (unary, binary, ternary, etc.).

Example for carbon search:
```
#TARGVOL=20-30
#SPECIES=C
#NATOM=4-8
#SYMMOPS=0
#SLACK=0.25
#OVERLAP=0.1
#COMPACT
#MINSEP=1.5 C-C
```

### 2. XX.cell (CASTEP Cell File)

Standard CASTEP `.cell` file for DFT calculations.

**Recommended settings:**
```
kpoints_mp_spacing 0.05  # High k-point sampling accuracy
symmetry_generate        # Let CASTEP find symmetry
```

### 3. XX.param (CASTEP Parameter File)

Standard CASTEP `.param` file.

**Critical setting:**
```
task : singlepoint  # Use single-point calculations, NOT geometry optimization
```

During potential building, we only need energies of generated structures, not relaxed geometries.

### 4. link.toml (EDDP Configuration)

This file controls the potential building workflow. Example configuration:

```toml
[cf]   # Descriptor settings
geometry_sequence = true    # Use geometric sequence for polynomial powers
elements = ["C"]            # Elements for the potential
p2 = [2, 10, 5]            # Two-body powers: from 2 to 10 with 5 polynomials
rcut2 = 6.0                # Two-body cutoff radius (Å)
q3 = [2, 10, 5]            # Three-body powers: from 2 to 10 with 5 polynomials

[trainer]  # Training settings
num_workers = 8     # Number of parallel training workers
nmodels = 256       # Number of models for ensemble (smaller values acceptable)

[scheduler]
script = "job.sh"   # Job script for DFT queue submission
type = "SLURM"      # Queue system (currently only SLURM supported)
njobs = 20          # Number of jobs in array job

[rss]
seedfile = "C-SAMPLING.cell"    # Seed file for RSS with trained potential
ensemble_id = 10                # Ensemble ID for structure searching
packed = true                   # Write output in packed format

[state]
seedfile = "C-SEARCH.cell"     # Seed for random structure generation
per_generation = 100           # Structures per generation
max_iterations = 10            # Maximum number of iterations
rss_nprocs = 16                # Processes for RSS during building
core_size = 1.2                # Hard-core size to prevent overlap (< typical bond length)
rss_pressure_gpa = 1.0         # External pressure for optimization (prevents empty cells)
workdir = "."                  # Working directory
seedfile_calc = "C.cell"       # Seed file for DFT calculation settings

dft_mode = "crud"              # Use crud.pl directly ("crud-queue" uses job.sh)
n_initial = 500                # Number of initial (generation 0) structures
shake_per_minima = 5           # Shaken structures per relaxed structure

    # Elemental energies from pseudoatomic energies
    # These are used to compute formation energies
    [state.elemental_energies]
        C = -147.4951   # From QC5 CASTEP OTFG potential
        # Add other elements as needed

    [state.dft_kwargs]   # Arguments for DFT calculation launcher
    mpinp = 8            # MPI processes per CASTEP calculation
    nparallel = 8        # Number of crud.pl processes in parallel
```

**Important**: Elemental energies should be **pseudo-atomic energies** from the pseudopotentials, NOT atomic energies from isolated atom calculations.

## Iterative Potential Building

### Starting the Workflow

With the four input files prepared, start the iterative process:

```julia
using EDDPotentials
link!(Builder("link.toml"))
```

### Workflow Steps

The iterative EDDP building process works as follows:

#### Generation 0 (Initial)
1. **Generate L random structures** using `buildcell` with XX-SEARCH.cell
   - These are unrelaxed random structures
   - Samples entire chemical space (not fixed composition)
   - Typical L = 500 for binary, 5000 for ternary

2. **Calculate DFT energies** for all structures
   - Uses CASTEP with parameters from XX.cell and XX.param
   - Single-point energy calculations only

3. **Train Generation 0 potential** on these structures

#### Generation N (N ≥ 1)
4. **Generate K new random structures** with buildcell

5. **Relax structures with current potential**
   - Use Generation (N-1) potential for relaxation
   - For each relaxed structure, generate M "shaken" variants by random perturbation
   - Typical K = 100 for binary, 1000 for ternary; M = 5

6. **Calculate DFT energies** for relaxed + shaken structures
   - Single-point calculations on perturbed geometries

7. **Train Generation N potential** on ALL data (Gen 0 to Gen N)
   - Accumulate training data from all generations
   - Monitor quality via RMSE, MAE, Spearman correlation

8. **Check convergence**, if N < max_iterations, increment N and return to step 4

### Why Shake Structures?

Random perturbation ("shaking") serves two purposes:
1. **Sample near local minima** on the potential energy surface
2. **Implicitly include gradient information** without training directly on forces

This allows EDDP to learn forces without explicit force training.

### Monitoring Quality

During training, check:
- **RMSE/MAE**: Energy prediction errors
- **Spearman correlation**: Ranking accuracy (computed per composition)
  - Values > 0.8 indicate good ranking
  - Low data points per composition make this metric unreliable
- **Walk-forward test**: Use Generation N model to predict Generation N+1 data
  - Can be sensitive to outliers

Typical convergence occurs after 5-10 iterations, depending on:
- System complexity
- Structures per generation
- Chemical space size

## Structure Searching with Trained Potential

After potential building, perform extensive structure searches using the trained potential.

### Why Further Searching?

The iterative building process generates only (N-1)×(M+1)×K + L structures. This is insufficient because:
1. **Limited sampling**: Not enough structures to fully explore binary/ternary chemical spaces
2. **Early iteration artifacts**: Structures from early iterations may not represent true PES minima

### Running Parallel Searches

EDDP calculations are fast, and RSS is embarrassingly parallel. Run many single-core searches in parallel:

#### Single Search (Julia REPL)
```julia
using EDDPotentials
run_rss(Builder())
```

#### Parallel Searches with GNU Parallel
```bash
parallel -j 16 julia -e "using EDDPotentials;run_rss(Builder())" ::: {1..1000}
```

This command:
- Runs 1000 independent searches
- Each search generates structures (default: 1000 per search → 1M total)
- GNU Parallel ensures max 16 concurrent searches (16 CPU cores)
- Automatically load-balances and handles completion

**Note**: Omitting `-j` uses all available CPU cores.

## Data Formats

### Packed SHELX Format

Search results are saved in `search/` directory in **packed SHELX format**:
- One `.res` file contains multiple structures
- Reduces filesystem burden (millions of structures)

#### Creating Packed Files
```bash
cat xx/xx-*.res > packed.res
```

#### Splitting Packed Files
```bash
cat packed.res | resplit
```

**Important**: After splitting, filenames come from the SHELX `TITL` line (first field), not the original filename.

### AIRSS Compatibility

This format maintains full compatibility with AIRSS tools:
- `cryan` (structure analysis)
- `ca` (shortcut for `cat *.res | cryan`)
- Other AIRSS utilities

## Preliminary Stability Analysis

### Obtaining Reference Energies

For variable-composition searches, you need elemental reference energies. Two approaches:

1. **Include in potential building**: Search unary systems during EDDP training
2. **Use known structures**: From literature or databases (Materials Project, ICSD)

### Quick DFT Validation

After identifying potentially stable structures/compositions:

1. **Use same DFT settings as training** for initial validation
   - Ensures energy comparability
   - Fast, lower-accuracy calculations

2. **Modify parameters for geometry optimization**:

Create a new `.param` file:
```
task : geometryoptimisation   # Enable structure relaxation
fixed_npw : false             # Use fixed cutoff for variable cell optimization
```

The `fixed_npw : false` setting ensures energy consistency when cell changes significantly.

### Using crud.pl for Batch Calculations

The `crud.pl` tool (**C**ASTEP **ru**n **d**aemon) automates CASTEP calculations:

#### How It Works
1. Monitors `hopper/` directory for SHELX `.res` files
2. When found, moves file to current directory
3. Constructs CASTEP input from `.res` + `XX.cell` + `XX.param`
4. Runs CASTEP optimization
5. Saves results to `good_castep/` directory

**File naming convention**: For `hopper/XX-YYY-ZZZ-AA.res`, uses `XX.cell` and `XX.param`.

#### Running Multiple crud.pl Instances

Process 8 structures simultaneously (8 cores each, 64 total):
```bash
parallel crud.pl -mpinp 8 ::: {1..8}
```

- Each crud.pl processes one structure at a time
- Automatically exits when `hopper/` is empty
- With `-nostop` flag: waits for new files instead of exiting

## Analyzing Results with cryan

The `cryan` (**cry**stal **an**alysis) tool from AIRSS analyzes structures from SHELX files.

### Basic Usage

Reads structures from STDIN:
```bash
cat *.res | cryan -r -t 10          # Rank all, show top 10
gunzip -c lots.res.gz | cryan -f H2O  # Filter specific composition
find . -name "*.res" | xargs cat | cryan -m  # Maxwell construction
```

### Common Options

```bash
# Ranking and filtering
-r, --rank                    # Rank all structures
-s, --summary                 # Summary of most stable per composition
-f, --formula <formula>       # Select specific composition
-t, --top [num]               # Show top N results (default 10)

# Energy and stability
-de, --delta_e <energy>       # Ignore structures above energy/atom
-p, --pressure <pressure>     # Apply external pressure
-m, --maxwell                 # Extract stable compositions (convex hull)

# Structure comparison
-u, --unite <thresh>          # Unite similar structures
-dr, --distance <rmax>        # Distance threshold for comparison

# Output
-el, --elementlist            # Comma-separated element list
-xg, --xmgrace               # Plot with xmgrace
```

### Important: Order Matters

Parameters are processed as a pipeline:
```bash
# Correct: filter THEN rank
cat *.res | cryan -el C,N -r -t 10

# Wrong: rank THEN filter (ranks all first)
cat *.res | cryan -r -el C,N
```

### Shortcut: ca

The `ca` command is shorthand for `cat *.res | cryan`:
```bash
ca -r -t 10   # Equivalent to: cat *.res | cryan -r -t 10
```

## Selecting Structures for High-Accuracy DFT

### Selection Criteria

EDDP energy errors are typically 0.1-0.2 eV/atom. Selection strategy:

1. **Select compositions** within 0.2 eV/atom of convex hull
2. **Within each composition**, select structures within 0.2 eV/atom of lowest

This avoids wasting resources on unstable structures.

### Deduplication

Random searching generates duplicate structures (indicates good PES sampling). Remove duplicates:

```bash
cat selected_res/*.res | cryan -u 0.1 -r
```

The `-u` threshold (typically 0.1 to 0.01) defines similarity:
- Compares sorted bond lengths
- If relative deviation < threshold, structures are considered identical
- Keeps structure with lower energy

### Running High-Accuracy Calculations

After selection and deduplication:
1. Place selected structures in `hopper/`
2. Run crud.pl with optimized settings
3. Analyze with cryan or EDDPotentials.jl tools

### EDDPotentials.jl Analysis Tools

For advanced operations, use Julia scripting:
- Potential accuracy assessment
- Batch phase diagram construction
- Phase diagram-based structure screening
- Custom data analysis and visualization

Example:
```julia
using EDDPotentials

# Load results
results = load_search_results("search/")

# Analyze stability
hull = compute_convex_hull(results)
stable = filter_stable_structures(results, hull; threshold=0.05)

# Generate phase diagram
plot_phase_diagram(hull)
```

## High-Accuracy Production Calculations

For publication-quality results, use standardized high-accuracy DFT:

### Recommended Workflow

1. **VASP/CASTEP/QE/ABACUS + AiiDA** for automated workflows
   - Thermodynamic stability (formation energies)
   - Dynamic stability (phonons)
   - Electronic structure (band structure, DOS)
   - Elastic properties

2. **Consistent methodology**
   - Converged k-point sampling
   - Well-tested pseudopotentials
   - Appropriate exchange-correlation functional
   - Standardized convergence criteria

3. **Quality checks**
   - Phonon calculations for dynamic stability
   - Energy above hull < 25 meV/atom for potential synthesizability
   - Electronic structure analysis for target properties

## Performance Tips

### Optimal Parallelization

For binary systems (typical parameters):
- **Initial generation (L)**: 500 structures
- **Per generation (K)**: 100 structures
- **Shaken structures (M)**: 5 per relaxed structure
- **Total iterations**: 5-10

For ternary systems:
- **Initial generation (L)**: 5000 structures
- **Per generation (K)**: 1000 structures
- **Shaken structures (M)**: 5 per relaxed structure
- **Total iterations**: 10-20

### CPU Core Allocation

Example for 64-core node:
```bash
# Potential building: 8 parallel DFT jobs × 8 cores each
parallel -j 8 crud.pl -mpinp 8 ::: {1..8}

# Final searches: 64 parallel EDDP searches × 1 core each
parallel -j 64 julia -e "using EDDPotentials;run_rss(Builder())" ::: {1..1000}
```

### Storage Considerations

- Use packed SHELX format to reduce inode usage
- Compress intermediate results: `gzip *.res`
- Archive generation data after potential building
- Keep only final ensemble models for production searches

## Troubleshooting

### Common Issues

**Problem**: crud.pl exits immediately
- **Solution**: Check that `hopper/` directory exists and contains `.res` files

**Problem**: CASTEP calculations fail
- **Solution**: Verify `.cell` and `.param` files are valid. Check CASTEP output in calculation directories.

**Problem**: Poor Spearman correlation
- **Solution**: May indicate insufficient training data. Continue iterations or check for outliers. You should also check the calculation parameters and warrnings in the .castep files. Some compounds may need more extra bands than the default given in CASTEP.

**Problem**: Structures cluster around a few compositions
- **Solution**: Adjust `#FOCUS` in SEARCH.cell to sample different composition ranges

**Problem**: Many duplicate structures in final search
- **Solution**: Good sign (PES well-sampled). Increase deduplication threshold or reduce search size.

