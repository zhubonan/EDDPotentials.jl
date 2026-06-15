#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
JULIA_CMD="${JULIA_CMD:-/home/bonan/.julia/juliaup/julia-1.11.7+0.x64.linux.gnu/bin/julia}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-/home/bonan/appdir/mattersim/pretrained_models/mattersim-v1.0.0-1M.pth}"
export MATTERSIM_DEVICE="${MATTERSIM_DEVICE:-cpu}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/eddp-mattersim-mpl}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "$MPLCONFIGDIR" "$ROOT/run-logs"
cd "$ROOT"

rm -rf work
rm -f train-log-*
mkdir -p work
cp C.cell work/C.cell

"$JULIA_CMD" --project="$REPO_ROOT" client.jl client-1 > run-logs/client-1.out 2> run-logs/client-1.err &
client1=$!
"$JULIA_CMD" --project="$REPO_ROOT" client.jl client-2 > run-logs/client-2.out 2> run-logs/client-2.err &
client2=$!

cleanup() {
  if kill -0 "$client1" 2>/dev/null; then kill "$client1" 2>/dev/null || true; fi
  if kill -0 "$client2" 2>/dev/null; then kill "$client2" 2>/dev/null || true; fi
}
trap cleanup EXIT

"$JULIA_CMD" --project="$REPO_ROOT" server.jl > run-logs/server.out 2> run-logs/server.err

if kill -0 "$client1" 2>/dev/null || kill -0 "$client2" 2>/dev/null; then
  "$JULIA_CMD" --project="$REPO_ROOT" -e 'using EDDPotentials; EDDPotentials.Workflow.server_request_shutdown!(Builder("link.toml"))' || true
fi
wait "$client1" || true
wait "$client2" || true

"$JULIA_CMD" --project="$REPO_ROOT" run-search.jl > run-logs/search.out 2> run-logs/search.err
cat run-logs/search.out
