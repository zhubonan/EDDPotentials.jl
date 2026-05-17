# SGD/Adam Training

EDDPotentials supports stochastic gradient descent (SGD) training as an alternative to the default Levenberg-Marquardt (LM) optimizer. SGD training uses mini-batches and implements Adam, plain SGD, and SGD with momentum directly on a flat parameter vector via `Flux.destructure`/`rebuild` and `Zygote.gradient`.

## When to Use SGD vs LM

**Use SGD/Adam training when:**

- Your dataset is large (> 5,000 structures) and LM becomes slow or runs out of memory
- You want to experiment with deeper network architectures that have many parameters
- You need fine-grained control over the learning rate schedule
- You are training models that will later be fine-tuned incrementally

**Stick with LM when:**

- Dataset is small to medium (< 3,000 structures) -- LM converges faster and more reliably
- You need the best possible accuracy -- LM uses exact second-order information
- You are training the shallow networks (1-2 hidden layers) that are standard for EDDP

## Quick Start

### Via `train!` dispatch (recommended)

Pass `train_method="sgd"` to the standard `train!` function. All SGD keyword arguments are forwarded automatically.

```julia
using EDDPotentials
using EDDPotentials.Potential: ManualFluxBackPropInterface

# Load and prepare data
sc = StructureContainer(["data/*.res"])
cf = CellFeature([:C]; p2=2:8, rcut2=5.0)
fc = FeatureStore(sc, cf)
fc_train, fc_val = split(fc, 450, 50; standardize=true)

# Create model with standardization transforms
itf = ManualFluxBackPropInterface(
    cf, 30, 30;
    xt=fc_train.xt, yt=fc_train.yt
)

# Train with Adam optimizer
train!(itf, fc_train, fc_val;
    train_method="sgd",
    optimizer=:adam,
    lr=1e-3,
    max_epochs=500,
    earlystop=50,
    batch_size=32,
    show_progress=true,
)
```

### Direct function call

For more control, call `train_sgd!` directly with raw feature matrices:

```julia
using EDDPotentials: train_sgd!

x_train, y_train = get_fit_data(fc_train)
x_val, y_val = get_fit_data(fc_val)

opt_res, params, history, funcs = train_sgd!(
    itf, x_train, y_train;
    x_val, y_val,
    optimizer=:adam,
    lr=1e-3,
    max_epochs=300,
)
```

## Optimizer Options

Three optimizers are available via the `optimizer` keyword:

| Optimizer | Keyword | Description |
|---|---|---|
| Adam | `:adam` (default) | Adaptive moment estimation. Good default for most cases. Converges reliably with minimal tuning. |
| SGD with momentum | `:sgd_momentum` | Classical momentum SGD. Works well with larger learning rates and cosine annealing schedules. |
| Plain SGD | `:sgd` | Vanilla SGD without momentum. Simplest option; requires careful learning rate tuning. |

```julia
# Adam (default)
train!(itf, fc_train, fc_val; train_method="sgd", optimizer=:adam, lr=1e-3)

# SGD with momentum
train!(itf, fc_train, fc_val; train_method="sgd", optimizer=:sgd_momentum,
       lr=1e-2, momentum=0.9)

# Plain SGD
train!(itf, fc_train, fc_val; train_method="sgd", optimizer=:sgd, lr=1e-2)
```

## Learning Rate Schedules

The `lr_schedule` keyword accepts a closure `(epoch) -> lr`. Three schedule constructors are provided:

### Constant learning rate

The default. Used automatically when `lr_schedule` is not specified.

```julia
using EDDPotentials: constant_lr

schedule = constant_lr(1e-3)
# schedule(1) == 1e-3, schedule(100) == 1e-3, ...
```

### Step decay

Multiplies the learning rate by `factor` every `step_size` epochs.

```julia
using EDDPotentials: step_decay_lr

# Halve the learning rate every 100 epochs
schedule = step_decay_lr(1e-2; step_size=100, factor=0.5)
# schedule(1)   == 0.01
# schedule(99)  == 0.01
# schedule(100) == 0.005
# schedule(200) == 0.0025

train!(itf, fc_train, fc_val;
    train_method="sgd", optimizer=:adam,
    lr=1e-2, lr_schedule=schedule, max_epochs=400,
)
```

### Cosine annealing

Smoothly decreases the learning rate from `initial_lr` to `min_lr` over `max_epochs` using a cosine curve.

```julia
using EDDPotentials: cosine_annealing_lr

schedule = cosine_annealing_lr(1e-2, 1e-4, 300)
# schedule(0)   == 0.01
# schedule(150) == ~0.005
# schedule(300) == ~0.001

train!(itf, fc_train, fc_val;
    train_method="sgd", optimizer=:sgd_momentum,
    lr=1e-2, lr_schedule=schedule, momentum=0.9, max_epochs=300,
)
```

You can also write custom schedules -- any function `epoch -> Float64` works:

```julia
# Custom exponential decay
my_schedule = epoch -> 1e-2 * exp(-0.01 * epoch)
```

## Usage Examples

### Basic Adam training

The simplest configuration. Adam with default hyperparameters is a good starting point for most EDDP training tasks.

```julia
using EDDPotentials
using EDDPotentials.Potential: ManualFluxBackPropInterface

sc = StructureContainer(["data/*.res"])
cf = CellFeature([:Si, :O])
fc = FeatureStore(sc, cf)
fc_train, fc_val = split(fc, 400, 100; standardize=true)

itf = ManualFluxBackPropInterface(
    cf, 30, 30;
    xt=fc_train.xt, yt=fc_train.yt
)

train!(itf, fc_train, fc_val;
    train_method="sgd",
    optimizer=:adam,
    lr=1e-3,
    max_epochs=500,
    earlystop=50,
    batch_size=32,
    show_progress=true,
)
```

### SGD with momentum and learning rate schedule

For finer control over convergence. Cosine annealing combined with momentum-based SGD can achieve better final accuracy than Adam on some datasets.

```julia
using EDDPotentials: cosine_annealing_lr

schedule = cosine_annealing_lr(1e-2, 1e-4, 300)

train!(itf, fc_train, fc_val;
    train_method="sgd",
    optimizer=:sgd_momentum,
    lr=1e-2,
    lr_schedule=schedule,
    momentum=0.9,
    batch_size=64,
    max_epochs=300,
    earlystop=50,
    weight_decay=1e-4,
    show_progress=true,
)
```

### Integration with FeatureStore workflow

SGD training fits into the standard EDDP training and ensemble pipeline:

```julia
using EDDPotentials
using EDDPotentials.Potential: ManualFluxBackPropInterface, nparams
using EDDPotentials: create_ensemble

# Prepare data
sc = StructureContainer(["data/*.res"])
cf = CellFeature([:C]; p2=2:8, rcut2=5.0)
fc = FeatureStore(sc, cf)
fc_train, fc_val = split(fc, 450, 50; standardize=true)

# Train an ensemble of models
models = []
for i in 1:10
    itf = ManualFluxBackPropInterface(
        cf, 30, 30;
        xt=fc_train.xt, yt=fc_train.yt
    )
    train!(itf, fc_train, fc_val;
        train_method="sgd",
        optimizer=:adam,
        lr=1e-3,
        max_epochs=300,
        earlystop=30,
        batch_size=32,
    )
    push!(models, itf)
end

# Create ensemble predictor
x_train, y_train = get_fit_data(fc_train)
ensemble = create_ensemble(models, x_train, y_train)
```

### Writing training logs to file

Set `log_file` to append epoch-by-epoch RMSE to a file for later analysis:

```julia
train!(itf, fc_train, fc_val;
    train_method="sgd",
    optimizer=:adam,
    lr=1e-3,
    max_epochs=500,
    log_file="training_log.txt",
)
```

Each line contains: epoch number, loop time, elapsed time, train RMSE (eV/atom), validation RMSE (eV/atom), optimizer name, batch size, and current learning rate.

## Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `optimizer` | `:adam` | Optimizer: `:adam`, `:sgd`, or `:sgd_momentum` |
| `lr` | `1e-3` | Initial learning rate |
| `lr_schedule` | `nothing` | Closure `(epoch) -> lr`. Defaults to `constant_lr(lr)` |
| `momentum` | `0.9` | Momentum coefficient for `:sgd_momentum` |
| `batch_size` | `32` | Number of structures per mini-batch |
| `max_epochs` | `300` | Maximum training epochs |
| `earlystop` | `50` | Stop if validation RMSE hasn't improved for this many epochs. Set to `0` to disable |
| `keep_best` | `true` | Restore best parameters (by validation RMSE) at the end |
| `show_progress` | `false` | Print epoch-by-epoch progress to stdout |
| `log_file` | `""` | Append progress to this file |
| `shuffle_seed` | `42` | RNG seed for reproducible data shuffling |
| `pow` | `2` | Loss power: `2` for MSE, `1` for MAE |
| `weights` | `nothing` | Per-structure loss weights |
| `beta` | `(0.9, 0.999)` | Adam beta1 and beta2 |
| `eps` | `1e-8` | Adam epsilon for numerical stability |
| `weight_decay` | `0.0` | L2 regularization coefficient |
| `x_val`, `y_val` | same as `x`, `y` | Validation data (defaults to training data if not provided) |

## Implementation Notes

**Destructure/rebuild pattern:** The implementation uses `Flux.destructure` to flatten all model parameters into a single vector, and `rebuild` to reconstruct the Flux `Chain` after each update. This avoids Flux's internal optimizer API and gives direct control over the parameter vector. Gradients are computed via `Zygote.gradient` on the flat vector.

**Per-structure gradient accumulation:** Each mini-batch computes gradients structure-by-structure using Zygote, accumulating into a shared gradient buffer. The gradients are averaged over the batch before the optimizer update. This means memory usage scales with the number of atoms in a single structure, not the full batch.

**Weight decay:** L2 regularization is implemented by adding `weight_decay * flat` to the gradient before the optimizer step. This is equivalent to decoupled weight decay (AdamW-style) rather than L2-penalized loss, which matters for Adam because it avoids the interaction between the regularization term and the adaptive learning rate.

**Non-finite gradient handling:** If any gradient element is `NaN` or `Inf`, the update for that batch is skipped and a warning is printed. This prevents the parameter vector from being corrupted by divergent gradients.

**Return value:** `train_sgd!` returns `(nothing, params, rmse_history, nothing)` where `rmse_history` is an Nx2 matrix with train and validation RMSE per epoch. This matches the return shape of `train_lm_batched!` for compatibility.
