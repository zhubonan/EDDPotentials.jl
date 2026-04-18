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
using EDDPotentials, ExtXYZ
using Flux: Chain, Dense
import EDDPotentials: FluxInterface, train!, train_val_split, get_fit_data, create_ensemble

# 1. Load training data (DFT-calculated structures)
structures = ExtXYZ.load("carbon_data.xyz")
sc = StructureContainer(structures)

# 2. Define features (two-body + three-body)
cf = CellFeature([:C])

# 3. Compute feature vectors
fc = FeatureContainer(cf, sc)
fc_train, fc_val = train_val_split(fc)

# 4. Train ensemble of neural networks
models = []
for i in 1:10
    chain = Chain(Dense(nfeatures(cf), 20, tanh), Dense(20, 1))
    model = FluxInterface(chain)
    train!(model, fc_train, fc_val; maxIter=300)
    push!(models, model)
end

# 5. Create ensemble predictor
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)

# 6. Use for predictions and structure optimization
structure = first(structures)
calc = NNCalc(structure, cf, ensemble)
vc_calc = VariableCellCalc(calc)
opt = opt_fire(vc_calc; f_tol=0.01, itermax=500)
@assert converged(opt, vc_calc)
```

For automated potential building with iterative training, see [AIRSS Workflow](tutorials/airss_workflow.md).

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
