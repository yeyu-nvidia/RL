#!/bin/bash
# COMMAND script for ray.sub: QA-MOPD with a co-trained LiLiCorr drafter.
#
# ray.sub runs this inside the container with the Ray cluster already up, from
# SLURM_SUBMIT_DIR. Extra args are forwarded as config overrides, which is how the
# pipeclean shortens the run without editing the recipe.
set -eux

CONFIG=${CONFIG:-examples/configs/recipes/llm/mopd-lilicorr-super35-nvfp4-realquant.yaml}

uv run python examples/run_grpo.py --config "$CONFIG" "$@"
