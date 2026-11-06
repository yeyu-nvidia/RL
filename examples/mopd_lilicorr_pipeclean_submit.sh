#!/bin/bash
# Submit the QA-MOPD + LiLiCorr pipeclean on OCI-JHB.
#
# Deliberately a short run: the point is to prove the path end to end -- drafter loads,
# capture feeds it the layers its checkpoint names, draft loss joins the GRPO loss, a
# step completes -- not to produce an RL result. Raise max_num_steps once it is green.
#
# CP is pinned to 1 in the recipe because DFlashBody raises on context_parallel_size > 1
# and LiLiCorr reuses that backbone. The parent runs CP=2, so this roughly doubles
# per-rank activation memory on a 120B; if it OOMs, raise TP/PP rather than re-enabling
# CP, which would just raise at startup.
set -eux

# 16 nodes x 4 GPUs: the recipe's cluster block is 16n4g and JHB nodes carry 4 GPUs.
# Asking sbatch for 8 would either not allocate or leave Ray's view disagreeing with
# cluster.gpus_per_node, which fails after the whole cluster is up.

RL_DIR=${RL_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/RL}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/cosmos/users/yeyu/containers/nemo-rl-v0.7.0.sqsh}
STEPS=${STEPS:-3}

[ -f "$CONTAINER" ] || { echo "container missing: $CONTAINER" >&2; exit 2; }

cd "$RL_DIR"

export CONTAINER
# /lustre carries the target, the drafter and the results dir; hf-local carries the
# TextTower weights the recipe pins.
export MOUNTS="/lustre:/lustre"
export COMMAND="examples/mopd_lilicorr_pipeclean.sh grpo.max_num_steps=${STEPS} logger.wandb_enabled=True"
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/cosmos/users/yeyu/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HF_HOME"

sbatch \
  --account=coreai_dlalgo_modelopt \
  --partition=batch \
  --job-name=mopd_lilicorr_pipeclean \
  --nodes=16 \
  --gpus-per-node=4 \
  --time=2:00:00 \
  --output=/lustre/fsw/portfolios/cosmos/users/yeyu/logs/mopd_lilicorr_%j.log \
  ray.sub
