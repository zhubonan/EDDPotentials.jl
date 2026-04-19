# What is EDDP?

Ephemeral Data Derived Potentials (EDDP) are machine learning interatomic potentials designed for **crystal structure prediction**. They trade ultimate accuracy for speed and simplicity, making them ideal for rapidly screening thousands of candidate crystal structures.

## The Core Idea

EDDP trains simple N-body models to reproduce DFT total energies. Unlike most ML potentials (GAP, NequIP, MACE), EDDP:

- **Trains on energies only** — no expensive DFT force calculations needed
- **Uses physically motivated features** — based on Lennard-Jones-like pairwise and triplet descriptors
- **Trains in minutes** — shallow networks with the Levenberg-Marquardt algorithm
- **Is designed for ranking, not precision** — reliably orders structures by energy to identify the most promising candidates for DFT validation

The name "ephemeral" reflects that these potentials are built for a specific search and discarded afterward. They are not meant to be general-purpose force fields.

## When to Use EDDP

**EDDP is well-suited for:**

- Crystal structure prediction (CSP) — ranking candidate structures by energy
- High-throughput screening of thousands of randomly generated structures
- Exploring unknown regions of configuration space
- Systems where DFT is too expensive for large-scale searching
- Identifying which compositions and stoichiometries are worth investigating

**EDDP is not well-suited for:**

- Producing publication-quality energies or forces (use DFT for final validation)
- Long molecular dynamics simulations (accuracy is typically 0.1–0.2 eV/atom)
- Systems requiring sub-meV/atom accuracy
- Replacing a general-purpose interatomic potential for production MD

## How EDDP Differs from Other ML Potentials

| Property | EDDP | GAP / SNAP | NequIP / MACE |
|---|---|---|---|
| Training data | Energies only | Energies + forces | Energies + forces |
| Training time | Minutes | Hours | Hours to days |
| Typical accuracy | 0.1–0.2 eV/atom | 1–10 meV/atom | < 1 meV/atom |
| Primary use | Structure ranking | General-purpose MD | General-purpose MD |
| Force prediction | Approximate | Accurate | Accurate |
| Model complexity | Shallow NN (10–50 params) | Sparse GP (1000s of params) | Deep NN / equivariant (1000s of params) |
| GPU required | Optional | No | Recommended |

The key trade-off: EDDP sacrifices accuracy for speed. It reliably distinguishes low-energy from high-energy structures, which is what matters for structure prediction.

## How It Works

### Feature Generation

Atomic environments are described by N-body features:

- **Two-body features**: `f(r)^p` — sum over neighbors, where `f(r) = sqrt(2(1 - r/rc))` smoothly decays to zero at the cutoff
- **Three-body features**: `f(r_ij)^p * f(r_ik)^p * f(r_jk)^q` — captures angular information through triplet distances
- **One-body features**: one-hot encoding of element species

See [Understanding Features](features.md) for the mathematical details.

### Model Training

A shallow neural network (typically 1–2 hidden layers) maps feature vectors to atomic energies. Training uses the Levenberg-Marquardt algorithm, which is well-suited for small networks:

```julia
chain = Chain(Dense(nfeatures, 20, tanh), Dense(20, 1))
model = FluxInterface(chain)
train!(model, train_data, val_data; maxIter=300)
```

### Ensemble Prediction

Multiple models with different initializations are combined using non-negative least squares (NNLS) weighting. The ensemble provides:
- **Better accuracy** than any single model
- **Uncertainty estimates** via the standard deviation across models
- **Guidance for active learning** — structures with high uncertainty need more DFT data

## The Iterative Workflow

EDDP's power comes from its iterative training loop:

```
Generation 0:  Random structures → DFT → Train ensemble
Generation 1:  RSS with ensemble → Filter uncertain → DFT → Retrain on all data
Generation 2:  RSS with ensemble → Filter uncertain → DFT → Retrain on all data
...
```

Each iteration adds DFT data where the model is most uncertain, systematically improving the potential in under-explored regions. See [Iterative Training](iterative_training.md) for details.

## Design Philosophy

1. **Ephemeral**: Build for a specific search, discard afterward. No need for a general-purpose model.
2. **Energy-only**: Forces are expensive to compute with DFT and not needed for ranking structures by energy.
3. **Physically motivated**: Lennard-Jones-like features ensure the model respects basic physics (smoothness, invariance, cutoff behavior).
4. **Uncertainty-aware**: Ensemble standard deviation tells you where the model is unreliable, guiding efficient data collection.
5. **Fast iteration**: Minutes to train, not days. This enables the active learning loop where the model guides its own training.

## Citation

If you use EDDPotentials.jl, please cite:

> C. J. Pickard, *Ephemeral Data Derived Potentials for Random Structure Search*, Phys. Rev. B **106**, 014102 (2022)
