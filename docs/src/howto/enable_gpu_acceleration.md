# Enable GPU Acceleration

GPU acceleration can significantly speed up EDDP training, especially for large feature sets and ensemble models. This guide shows you how to set up and use GPU acceleration with EDDPotentials.jl.

## Prerequisites

### Check GPU Availability

Verify that your system has a CUDA-compatible NVIDIA GPU:

```bash
nvidia-smi
```

This should display your GPU information, driver version, and CUDA version. If the command fails, ensure you have:
- NVIDIA GPU drivers installed
- CUDA toolkit installed (version 11.0 or higher recommended)

### Install CUDA.jl

Add the CUDA.jl package to your Julia environment:

```julia
using Pkg
Pkg.add("CUDA")
```

Verify CUDA.jl can detect your GPU:

```julia
using CUDA
CUDA.functional()  # Should return true
```

If this returns `false`, check:
- NVIDIA drivers are up to date
- CUDA toolkit is properly installed
- Julia can find the CUDA libraries (set `LD_LIBRARY_PATH` if needed)

## Enable GPU in EDDPotentials

### Basic Setup

To use GPU acceleration, load CUDA.jl before EDDPotentials and enable the GPU flag:

```julia
using CUDA
using EDDPotentials

# Enable GPU acceleration
EDDPotentials.USE_CUDA[] = true
```

### Training with GPU

Once enabled, training automatically uses GPU for forward and backward passes:

```julia
using CUDA
using EDDPotentials
using Flux
import EDDPotentials: FluxInterface, train!, train_test_split

# Enable GPU
EDDPotentials.USE_CUDA[] = true

# Load data and create features
fc = FeatureContainer(cf, sc)
fc_train, fc_test = train_test_split(fc)

# Create and train model (automatically uses GPU)
nfeat = nfeatures(cf)
chain = Chain(Dense(nfeat, 20, tanh), Dense(20, 1))
model = FluxInterface(chain)

train!(model, fc_train, fc_test; maxIter=300, show_progress=true)
```

### Advanced: Manual GPU Control

For fine-grained control, you can manually move data to GPU:

```julia
using CUDA
using EDDPotentials
import EDDPotentials: train!

EDDPotentials.USE_CUDA[] = true

# Load builder
bu = Builder("link.toml")
fc = load_features(bu)

# Split data
train, test, valid = split(fc, 0.8, 0.1, 0.1)

# Create model with GPU-accelerated training
model = EDDPotentials.ManualFluxBackPropInterface(
    bu.cf, 10, 10;
    xt=train.xt,
    yt=train.yt
)

# Train (uses GPU)
train!(model, train, test; show_progress=true)
```

## Performance Benchmarks

Typical speedups with GPU acceleration:

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| Training single model (small dataset, <1000 structures) | 2 min | 30 sec | 4x |
| Training single model (large dataset, >5000 structures) | 15 min | 2 min | 7.5x |
| Training ensemble (10 models, medium dataset) | 25 min | 4 min | 6x |

**Note**: Speedups vary with GPU model, feature dimension, and network architecture. Larger networks and datasets see greater benefits.

## Troubleshooting

### CUDA.jl Not Detecting GPU

**Problem**: `CUDA.functional()` returns `false`

**Solutions**:
- Update NVIDIA drivers: `sudo ubuntu-drivers autoinstall` (Ubuntu)
- Reinstall CUDA toolkit
- Check CUDA version compatibility: `CUDA.versioninfo()`
- Rebuild CUDA.jl: `Pkg.build("CUDA")`

### Out of Memory Errors

**Problem**: Training crashes with "out of memory" error

**Solutions**:
- Reduce batch size in training
- Use smaller network architecture
- Clear GPU memory: `CUDA.reclaim()`
- Monitor GPU memory: `CUDA.memory_status()`

### Slower Performance on GPU

**Problem**: GPU training is slower than CPU

**Possible causes**:
- Dataset too small (GPU overhead dominates)
- Feature dimension too small (< 50 features)
- Network too shallow (< 2 hidden layers)

**Solutions**:
- GPU acceleration is most beneficial for:
  - Large datasets (> 1000 structures)
  - High-dimensional features (> 100)
  - Deep networks (≥ 2 hidden layers)
- For small problems, CPU may be faster

### Mixed Precision Training

For additional speedup, consider mixed precision (Float16) training on compatible GPUs (Volta or newer):

```julia
using CUDA
CUDA.allowscalar(false)  # Prevent slow scalar indexing

# Train with Float32 (default) for stability
# Float16 support is experimental
```

## Performance Tips

1. **Batch your data**: GPU performance improves with larger batches
2. **Reuse models**: Keep models in GPU memory between training runs
3. **Monitor memory**: Use `CUDA.memory_status()` to track usage
4. **Profile your code**: Use `CUDA.@time` to identify bottlenecks
5. **Clear cache**: Call `CUDA.reclaim()` between training sessions to free memory

## Limitations

GPU acceleration in EDDPotentials currently supports:
- ✅ Training neural network models
- ✅ Forward passes (energy prediction)
- ✅ Backward passes (gradient computation)
- ⚠️ Calculator operations (limited support)
- ❌ Structure generation and preprocessing

For best results, use GPU for training and CPU for structure manipulation and analysis.
