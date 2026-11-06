#!/bin/bash
# COMMAND script for ray.sub: QA-MOPD with a co-trained LiLiCorr drafter.
#
# ray.sub runs this inside the container with the Ray cluster already up, from
# SLURM_SUBMIT_DIR. Extra args are forwarded as config overrides, which is how the
# pipeclean shortens the run without editing the recipe.
set -eux

CONFIG=${CONFIG:-examples/configs/recipes/llm/mopd-lilicorr-super35-nvfp4-realquant.yaml}

# The container's uv must be able to parse this repo's pyproject.toml. A container
# older than the branch fails on `override-dependencies`' map form with a TOML parse
# error that names a line number and not the real problem, after Ray has already come
# up on every node. Check it up front and say what is actually wrong.
UV_BIN=${UV_BIN:-uv}
if ! "$UV_BIN" tree --quiet >/dev/null 2>&1 && ! "$UV_BIN" lock --check >/dev/null 2>&1; then
  if ! "$UV_BIN" run python -c "" >/dev/null 2>&1; then
    echo "[pipeclean] ${UV_BIN} ($("$UV_BIN" --version 2>&1 | head -1)) cannot use this" \
         "repo's pyproject.toml." >&2
    echo "[pipeclean] The container predates the branch; docker/Dockerfile pins" \
         "UV_VERSION=0.11.28. Use a newer nemo-rl image, or set UV_BIN to a newer uv." >&2
    exit 3
  fi
fi

"$UV_BIN" run python examples/run_grpo.py --config "$CONFIG" "$@"
