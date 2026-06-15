# MatterSim Carbon Distributed Example

This example builds a tiny carbon EDDPotential below 100 GPa with one
file-backed distributed server and two DFT/ML clients. The reference data path
is EDDPotentials `dft_mode = "acrud"` routed to airsspy:

```bash
ap run crud --code ml --calculator ase:mattersim_ase:EnvMatterSimCalculator --singlepoint
```

The direct `mattersim:/path/to/checkpoint` torch-sim route is intentionally not
used here because the installed environment currently fails on that path with a
MatterSim import mismatch.

## Run

```bash
cd examples/mattersim-c-100gpa-distributed
./run-local-demo.sh
```

The script starts two clients, stages `C.cell` into `work/`, runs the
distributed server, then runs a small 100 GPa verification search. It recreates
the local `work/` directory on each run.

## Environment

The checked-in defaults match the local environment used to verify this example:

```bash
export MATTERSIM_CHECKPOINT=/home/bonan/appdir/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth
export MATTERSIM_DEVICE=cpu
```

`link.toml` uses `/home/bonan/appdir/airsspy/.venv/bin/ap`, because that airsspy
environment has MatterSim installed.

## Expected Outputs

- `work/crud-work/gen*/good_castep/*.res`: MatterSim reference structures.
- `work/gen*-dft/*.res`: EDDPotentials training data collected by the server.
- `work/*.jld2`: trained model and ensemble artifacts.
- `work/verify-search-100gpa/*.res`: verification search structures.
- `run-logs/*.out` and `run-logs/*.err`: server, client, and search logs.
