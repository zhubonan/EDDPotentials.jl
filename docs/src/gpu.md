## GPU Acceleration

GPU can be used to accelerate the training process of EDDPotentials.jl using the Levenberg-Marquardt algorithm when the model size is large. This is often beneficial for ternary or quaternary searches.

To enable GPU acceleration, you need to have a CUDA-enabled GPU and CUDA toolkit installed on your system. You can check if your system has CUDA support by running the following command in the terminal:

```
nvidia-smi
```

If you have CUDA support, you can enable GPU acceleration by setting the `EDDPotential.USE_CUDA[]` parameter to `true`. In addition, the `CUDA.jl` should be installed and loaded before running the code. Here is an example:

```julia
using CUDA, EDDPotentials

EDDPotential.USE_CUDA[] = true

bu = Builder()
fc = load_features(bu)
train, test, valid = split(fc, 0.8, 0.1, 0.1);
model = EDDPotentials.ManualFluxBackPropInterface(bu.cf, 10, 10;xt=train.xt, yt=train.yt)

train!(model, training_data, test_data; show_progress=true)
```

For system with multiple GPUs you can set the id of the device to be used via the `EDDP_CUDA_DEVICE` environment variable.
