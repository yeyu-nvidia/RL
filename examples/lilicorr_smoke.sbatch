#!/bin/bash
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
# Whole node: this partition allocates all 4 GPUs per node, and asking for
# one is rejected as a size-limit violation. The smoke only uses GPU 0.
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --job-name=lilicorr_smoke
#SBATCH --output=/lustre/fsw/portfolios/cosmos/users/yeyu/logs/lilicorr_smoke_%j.log
# One GPU, no Ray: exercise the drafter path while the 16-node pipeclean queues.
set -x
RL_DIR=${RL_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/RL}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/cosmos/users/yeyu/containers/nemo-rl-v0.8.0.sqsh}
srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch \
     --container-workdir="$RL_DIR" \
     bash -lc '
       set -eux
       export PYTHONPATH="'"$RL_DIR"'${PYTHONPATH:+:$PYTHONPATH}"
       # The image venv ships no megatron -- it comes from resolving this branch's lock,
       # which is why the pipeclean needs `uv run` too. Same uv repair as there: an image
       # older than the branch cannot parse pyproject.toml.
       UV=uv
       if ! $UV lock --check >/dev/null 2>&1 && ! $UV tree --quiet >/dev/null 2>&1; then
         UVH=/lustre/fsw/portfolios/cosmos/users/yeyu/.smoke-uv
         mkdir -p "$UVH"
         curl -LsSf https://astral.sh/uv/0.11.28/install.sh | env UV_UNMANAGED_INSTALL="$UVH" sh
         UV="$UVH/uv"
       fi
       $UV --version
       $UV run python -c "import megatron, nemo_rl; print(\"[smoke] megatron ok\")"
       $UV run python examples/lilicorr_smoke.py --workdir /lustre/fsw/portfolios/cosmos/users/yeyu/lilicorr_smoke
     '
