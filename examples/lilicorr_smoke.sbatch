#!/bin/bash
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
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
       python examples/lilicorr_smoke.py --workdir /lustre/fsw/portfolios/cosmos/users/yeyu/lilicorr_smoke
     '
