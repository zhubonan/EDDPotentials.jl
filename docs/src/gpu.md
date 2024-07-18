## GPU Acceleration

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
