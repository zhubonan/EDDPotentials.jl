# AIRSS Workflow

Automate EDDP potential building and crystal structure prediction using AIRSS (Ab Initio Random Structure Searching). This tutorial shows the complete workflow from setup to final structure analysis.

## Prerequisites

### AIRSS Installation

Install AIRSS from [GitHub](https://github.com/CCP-NC/airss):

```bash
git clone https://github.com/CCP-NC/airss.git
cd airss
make
```

Add to your PATH:
```bash
export PATH=/path/to/airss/bin:$PATH
```

Required AIRSS tools: `buildcell`, `cabal`, `cryan`, `resplit`

### CASTEP Setup

CASTEP must be installed and in PATH. Test with:
```bash
castep.serial --version
```

EDDPotentials.jl uses `castep.serial` by default. For parallel runs, set:
```bash
export CASTEP_COMMAND="mpirun -n 8 castep.mpi"
```

## Workflow Overview

The iterative workflow:

1. **Generation 0**: Generate random structures → DFT calculations → Train initial potential
2. **Generation N**: Previous potential relaxes structures → DFT on unique structures → Retrain potential
3. **Convergence**: Stop when potential accuracy stabilizes (typically 5-10 iterations)
4. **Production search**: Use final potential for large-scale structure prediction

Each iteration adds ~100-1000 DFT calculations to the training set.

## Input Files

Create these four files in your working directory:

### XX-SEARCH.cell

Random structure generation specification (used by `buildcell`):

```
#SPECIES=C,N
#NFORM=1-8
#NATOM=8-32
#SLACK=0.25
#OVERLAP=0.1
#MINSEP=1.5 C-C=1.2 N-N=1.2 C-N=1.2

%BLOCK LATTICE_CART
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
#ANGAMP=30
%ENDBLOCK LATTICE_CART
```

Key parameters:
- `#SPECIES`: Elements to search
- `#NFORM`: Formula units per cell
- `#NATOM`: Atom count range
- `#SLACK`: Volume padding (recommended: 0.25)
- `#MINSEP`: Minimum interatomic distances (Å)

### XX.cell

CASTEP cell template (merged with random structures):

```
symmetry_generate
snap_to_symmetry

%BLOCK SPECIES_POT
C C_00PBE.usp
N N_00PBE.usp
%ENDBLOCK SPECIES_POT

kpoints_mp_spacing 0.05
```

Ensure pseudopotentials (`.usp` files) are accessible to CASTEP.

### XX.param

CASTEP calculation parameters:

```
task : singlepoint
xc_functional : PBE
cut_off_energy : 300 eV
opt_strategy : speed
num_dump_cycles : 0
write_cell_structure : true
```

For potential building, use `task : singlepoint`. For final validation, use `task : geometryoptimisation`.

### link.toml

EDDPotentials workflow configuration:

```toml
[state]
workdir = "CN-link"
seedfile_calc = "CN.cell"
max_iterations = 5
n_initial = 500
per_generation = 100
shake_per_minima = 5
buildcell_nprocs = 8
dft_mode = "crud"

[state.rss]
seedfile = "CN-SEARCH.cell"

[state.dft_kwargs]
mpinp = 8
nparallel = 8

[cf]
elements = ["C", "N"]
p2 = [2, 6, 5]
p3 = [2, 5, 4]
q3 = [1, 3, 3]
rcut2 = 4.0
rcut3 = 3.5

[trainer]
n_nodes = [20]
nmodels = 10
max_iter = 300
earlystop = 30
num_workers = 8

[rss]
max = 1000
subfolder_name = "search"
# Defaults: unpacked `.res` output, pressure_gpa = 0.01,
# ensemble_std_min = -1.0, ensemble_std_max = -1.0.
```

Adjust parallelization based on available resources.

## Running the Workflow

### Initialize Builder

```julia
using EDDPotentials

builder = Builder("link.toml")
```

This creates the directory structure:
```
CN-link/
├── gen0/            # Generation 0 inputs
├── gen0-dft/        # Generation 0 DFT outputs
├── training/        # Datasets and individual models
├── ensemble-gen0.json
└── link.toml        # Configuration
```

### Run Iterative Building

```julia
# Run complete workflow (all generations)
link!(builder)

# Or run single generation
step!(builder)
```

### Monitor Progress

```julia
# Check current status
println("Iteration: ", builder.state.iteration)

# Load current ensemble
ensemble = load_ensemble(builder, max(builder.state.iteration - 1, 0))
```

## Understanding the Process

**Generation 0** (cold start):
1. Generate L random structures with `buildcell`
2. Run DFT single-point calculations
3. Train initial potential on DFT data
4. Relax structures with potential
5. Generate M shaken structures per relaxed structure
6. Calculate DFT energies for shaken structures
7. Retrain potential on expanded dataset

**Generation N** (N > 0):
1. Use previous generation's potential
2. Relax K new random structures
3. Generate M shaken structures per relaxed structure
4. Calculate DFT for unique structures only (deduplication)
5. Add to training set and retrain potential

**Convergence criteria**:
- Spearman correlation > 0.98 on test set
- RMSE stabilizes across generations
- Typically 5-10 generations for binary systems, 10-20 for ternary systems

## Production Structure Search

After building the potential:

```julia
# Configure and run large-scale search
builder.rss.max = 10_000
builder.rss.subfolder_name = "production"
builder.rss.ensemble_id = max(builder.state.iteration - 1, 0)
run_rss(builder)
```

## Data Management: Packed SHELX Format

AIRSS outputs individual `.res` files by default. For large searches (10k+ structures), enable packed output:

```toml
[rss]
packed = true
```

### Packing Multiple Files

```bash
cat *.res | respack > all_structures.res
```

Benefits:
- Single file vs thousands of inodes
- Faster I/O operations
- Easier data transfer

### Reading Packed Files

```julia
using EDDPotentials
using CellBase: read_res

# Read packed file (returns array of structures)
structures = read_res("all_structures.res")
println("Loaded $(length(structures)) structures")
```

### Splitting Packed Files

```bash
cat packed.res | resplit
```

Filenames come from SHELX `TITL` line, not original names.

## Analyzing Results with cryan

The `cryan` tool analyzes structures from SHELX files.

### Basic Usage

```bash
# Rank all structures, show top 10
cat *.res | cryan -r -t 10

# Filter specific composition
gunzip -c lots.res.gz | cryan -f H2O -r

# Extract convex hull
cat *.res | cryan -m -r
```

### Essential Options

```bash
-r, --rank                  # Rank structures by energy
-t, --top N                 # Show top N (default 10)
-f, --formula <formula>     # Filter by composition
-m, --maxwell               # Convex hull analysis
-u, --unite <thresh>        # Deduplicate structures
-s, --summary               # Summary per composition
```

**Order matters**: Parameters process as pipeline. Filter BEFORE ranking:
```bash
cat *.res | cryan -f C2N -r    # Correct
cat *.res | cryan -r -f C2N    # Wrong (ranks all first)
```

### Shortcut

```bash
ca -r -t 10   # Equivalent to: cat *.res | cryan -r -t 10
```

## Selecting Structures for High-Accuracy DFT

EDDP errors are typically 0.1-0.2 eV/atom. Selection strategy:

1. **Select compositions** within 0.2 eV/atom of convex hull
2. **Within composition**, select structures within 0.2 eV/atom of lowest energy

### Deduplication

```bash
# Remove duplicates (0.1 threshold)
cat selected/*.res | cryan -u 0.1 -r > unique.res
```

Threshold (0.01-0.1) defines similarity by comparing sorted bond lengths.

For single-composition EDDP searches, the same idea can be applied during RSS
instead of only as a post-processing step. Enable `[rss.prune]` for production
`run_rss`, or `[state.rss.prune]` for structures generated during `link!`:

```toml
[rss.prune]
enabled = true
pool_size = 100
keep_fraction = 0.10
dedup_tol = 0.10
```

Each pruning pool is ranked by predicted enthalpy per atom and deduplicated with
the in-process CellBase fingerprint distance. In multi-process `link!` runs,
each worker emits pruning diagnostics to its own `rss-process-*` stderr log.
Pruning is automatically disabled for formula-sampling/random-composition
searches.

### Running High-Accuracy Calculations

Use `crud.pl` (**C**ASTEP **ru**n **d**aemon) for batch processing:

```bash
# Process 8 structures in parallel (8 cores each)
parallel crud.pl -mpinp 8 ::: {1..8}
```

Workflow:
1. Place structures in `hopper/` directory
2. crud.pl monitors hopper/, runs CASTEP on each file
3. Results saved to `good_castep/`

For `hopper/XX-YYY-ZZZ.res`, uses `XX.cell` and `XX.param`.

### Reference Energies

For variable-composition searches, obtain elemental reference energies:
- Include unary systems in EDDP training, OR
- Use known structures (Materials Project, ICSD)

### DFT Parameters for Validation

Modify `XX.param` for geometry optimization:
```
task : geometryoptimisation
fixed_npw : false            # Consistent energy for variable cell
```

## High-Accuracy Production Calculations

For publication-quality results:

**Recommended tools**: VASP/CASTEP/Quantum Espresso + AiiDA

**Analyses**:
- Formation energies (thermodynamic stability)
- Phonons (dynamic stability)
- Electronic structure (band structure, DOS)
- Elastic properties

**Quality criteria**:
- Converged k-point sampling
- Energy above hull < 25 meV/atom (synthesizability threshold)
- Positive phonon frequencies (stable structure)

## Performance Tips

### Parallelization Guidelines

**Binary systems**:
- Initial generation: 500 structures
- Per generation: 100 structures
- Shaken structures: 5 per relaxed structure
- Total iterations: 5-10

**Ternary systems**:
- Initial generation: 5000 structures
- Per generation: 1000 structures
- Shaken structures: 5 per relaxed structure
- Total iterations: 10-20

### CPU Allocation (64-core node example)

```bash
# Potential building: 8 DFT jobs × 8 cores
parallel -j 8 crud.pl -mpinp 8 ::: {1..8}

# Production searches: 64 EDDP searches × 1 core
parallel -j 64 julia -e "run_search()" ::: {1..1000}
```

### Storage Management

- Use packed SHELX format (`[rss] packed = true`) to reduce inodes
- Compress intermediate results: `gzip *.res`
- Archive generation data after convergence
- Keep only final ensemble for production

## Troubleshooting

**crud.pl exits immediately**
- Check `hopper/` directory exists and contains `.res` files

**CASTEP calculations fail**
- Verify `.cell` and `.param` syntax
- Check CASTEP output in calculation directories
- Ensure pseudopotentials accessible

**Poor Spearman correlation**
- Insufficient training data: continue iterations
- Check for outliers in training set
- Verify CASTEP warnings (e.g., insufficient bands for some compounds)

**Structures cluster around few compositions**
- Adjust `#FOCUS` in SEARCH.cell to sample different composition ranges

**Many duplicate structures**
- Good sign (PES well-sampled)
- Increase deduplication threshold or reduce search size

## Next Steps

- **Phonon calculations**: [Run Phonon Calculations](../howto/run_phonon_calculations.md)
- **GPU acceleration**: [Enable GPU Acceleration](../howto/enable_gpu_acceleration.md)
- **Iterative training theory**: [Iterative Training](../concepts/iterative_training.md)
