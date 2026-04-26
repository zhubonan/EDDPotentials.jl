![](assets/logo_small.png)
# EDDPotentials.jl

A [Julia](https://julialang.org/) package for [Ephemeral Data Derived Potentials](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.014102) (EDDP) - machine learning interatomic potentials designed for crystal structure prediction and rapid structural screening.

## Quick Links

**New users:** Start with [Getting Started](getting_started.md)

**Learn concepts:** [What is EDDP?](concepts/what_is_eddp.md) | [Understanding Features](concepts/features.md) | [Iterative Training](concepts/iterative_training.md)

**Step-by-step tutorial:** [AIRSS Workflow](tutorials/airss_workflow.md)

**Common tasks:** [Customize Features](howto/customize_features.md) | [Structure Prediction](howto/structure_prediction.md) | [Phonon Calculations](howto/run_phonon_calculations.md)

## What is EDDP?

EDDP is a machine learning potential optimized for **crystal structure prediction**. Key features:

- **Energy-only training:** No expensive DFT force calculations required
- **Fast training:** Minutes to hours, not days
- **Physically motivated:** Based on Lennard-Jones-like potentials
- **Iterative building:** Automated data generation guided by model uncertainty
- **High throughput:** Screen thousands of candidate structures rapidly

EDDP trades ultimate accuracy for speed and simplicity, making it ideal for ranking candidate structures before high-accuracy DFT validation.

## Key Capabilities

**Feature Generation**
- Two-body and three-body atomic environment descriptors
- Rotation/translation invariant, periodic boundary conditions
- Flexible cutoff radii and power series

**Training**
- Shallow neural networks (fast, good generalization)
- Ensemble models with NNLS weighting
- GPU acceleration support (optional)
- Energy-only or energy+force training

**Structure Prediction**
- Integration with AIRSS (Ab Initio Random Structure Searching)
- Automated iterative potential building (`link!()`)
- Parallel random structure search (RSS)
- CASTEP DFT integration for training data

**Analysis**
- Uncertainty quantification (ensemble standard deviation)
- Walk-forward testing for convergence validation
- Phonon calculations via Phonopy interface

**Analytic baselines**
- Fixed-weight force-shifted Lennard-Jones helper via `EDDPotentials.lj_baseline_calc`

## Installation

```julia
using Pkg
Pkg.add("EDDPotentials")
```

**Optional dependencies:**
```julia
# GPU acceleration
Pkg.add("CUDA")

# Python integration (Phonopy, ASE)
Pkg.add("PyCall")

# Visualization
Pkg.add("Plots")
Pkg.add("PlotlyJS")
```

## Typical Workflow

```julia
using EDDPotentials

# 1. Train a model — species auto-detected, defaults for features
tm = train_model("carbon_data/*.res";
    nnodes=[20, 20], max_epochs=300, earlystop=30)

# Or: train_model("data/*.res", CellFeature([:C]; rcut2=5.0); ...) for explicit control

# 2. Predict energy/forces on a new structure
result = predict(tm, my_cell)
result.energy   # total energy (eV)
result.forces   # atomic forces (matrix)
result.stress   # stress tensor (3×3)

# 3. Save the model for later use
save_model_artifact("model.json", tm)

# 4. Use for geometry optimization
calc = EDDPCalculator(tm)
set_cell!(calc, my_cell)
relax!(calc; method="fire", f_tol=0.01, itermax=500)
```

The `train_model` function is the recommended path for single-potential training.
It handles structure loading, feature computation, data splitting, standardisation,
model creation, and training in one call. By default (`verbose=true`) it prints
a configuration summary and final RMSE/MAE statistics.

For users who need fine-grained control over the training pipeline, the
step-by-step API (`StructureContainer` → `FeatureContainer` → `split` → `train!`)
is still fully available. See [Getting Started](getting_started.md) for both
approaches.

For automated potential building with iterative training, see [AIRSS Workflow](tutorials/airss_workflow.md).

## Analytic Lennard-Jones baseline

For quick debugging, optimizer checks, or pair-potential baselines, you can create a
force-shifted Lennard-Jones calculator directly in the regular `NNCalc` stack:

```julia
using EDDPotentials
using CellBase

cell = Cell(Lattice(12.0, 12.0, 12.0), [:Ar, :Ar], [0.0 1.4; 0.0 0.0; 0.0 0.0])
calc = EDDPotentials.lj_baseline_calc(cell; sigma=1.0, epsilon=1.0, rcut=3.0)

energy = eddp_energy(calc)
forces = eddp_forces(calc)
```

For mixed species, supply per-pair `sigma` and `epsilon` dictionaries.

## Documentation Structure

This documentation follows the [Diátaxis](https://diataxis.fr/) framework:

**[Getting Started](getting_started.md)** - Quick introduction and first example

**Tutorials** - Learning-oriented guides
- [AIRSS Workflow](tutorials/airss_workflow.md): Complete structure prediction workflow

**How-To Guides** - Task-oriented recipes
- [Customize Features](howto/customize_features.md): Advanced feature engineering
- [Crystal Structure Prediction](howto/structure_prediction.md): Large-scale RSS workflows
- [Run Phonon Calculations](howto/run_phonon_calculations.md): Vibrational properties
- [Enable GPU Acceleration](howto/enable_gpu_acceleration.md): Training speedup
- [Batched LM Training](howto/batched_lm_training.md): Stochastic LM for large datasets
- [SGD/Adam Training](howto/sgd_training.md): Mini-batch SGD and Adam optimizers

**Concepts** - Understanding-oriented explanations
- [What is EDDP?](concepts/what_is_eddp.md): Design philosophy and use cases
- [Understanding Features](concepts/features.md): Feature theory and selection
- [Iterative Training](concepts/iterative_training.md): Automated potential building

**[API Reference](functions.md)** - Technical function documentation

## Related Tools

- **[AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS):** *Ab initio* random structure searching toolkit
- **[CASTEP](http://www.castep.org):** Plane-wave DFT code for training data generation
- **[DISP](https://zhubonan.github.io/disp):** Distributed structure prediction on computing clusters
- **[Phonopy](https://phonopy.github.io/phonopy/):** Phonon calculations (Python interface)

## Citation

If you use EDDPotentials.jl, please cite:

> C. J. Pickard, *Ephemeral Data Derived Potentials for Random Structure Search*, [Phys. Rev. B **106**, 014102 (2022)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.014102)

```bibtex
@article{pickard2022ephemeral,
  title={Ephemeral data derived potentials for random structure search},
  author={Pickard, Chris J},
  journal={Physical Review B},
  volume={106},
  number={1},
  pages={014102},
  year={2022},
  publisher={APS}
}
```

## Contributing

Contributions welcome! Report issues or submit pull requests on [GitHub](https://github.com/zhubonan/EDDPotentials.jl).

```@meta
CurrentModule = EDDPotentials
```

## Index

```@index
```
