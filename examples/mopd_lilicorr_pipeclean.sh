#!/bin/bash
# COMMAND script for ray.sub: QA-MOPD with a co-trained LiLiCorr drafter.
#
# ray.sub runs this inside the container with the Ray cluster already up, from
# SLURM_SUBMIT_DIR. Extra args are forwarded as config overrides, which is how the
# pipeclean shortens the run without editing the recipe.
set -eux

CONFIG=${CONFIG:-examples/configs/recipes/llm/mopd-lilicorr-super35-nvfp4-realquant.yaml}
# Matches docker/Dockerfile's pin, i.e. the uv this repo's pyproject.toml is written for.
PINNED_UV_VERSION=${PINNED_UV_VERSION:-0.11.28}

# The container's uv must be able to parse this repo's pyproject.toml. An image older
# than the branch dies on `override-dependencies`' map form with a TOML parse error
# naming a line number rather than the problem -- and only after Ray has come up on
# every node. Check first, and repair it rather than wasting the allocation.
UV_BIN=${UV_BIN:-uv}
if ! "$UV_BIN" lock --check >/dev/null 2>&1 && ! "$UV_BIN" tree --quiet >/dev/null 2>&1; then
  echo "[pipeclean] container uv ($("$UV_BIN" --version 2>&1 | head -1)) cannot parse" \
       "this branch's pyproject.toml; installing uv ${PINNED_UV_VERSION}" >&2
  UV_HOME=${UV_HOME:-${SLURM_SUBMIT_DIR:-$PWD}/.pipeclean-uv}
  mkdir -p "$UV_HOME"
  # Compute nodes on this cluster do reach astral.sh (probed), but say so plainly if
  # that changes rather than failing later inside a dependency resolve.
  if ! curl -LsSf "https://astral.sh/uv/${PINNED_UV_VERSION}/install.sh" \
       | env UV_UNMANAGED_INSTALL="$UV_HOME" sh >&2; then
    echo "[pipeclean] could not install uv ${PINNED_UV_VERSION}. Use a newer nemo-rl" \
         "image (v0.8.0+) or set UV_BIN to a uv that can read this pyproject." >&2
    exit 3
  fi
  UV_BIN="$UV_HOME/uv"
  "$UV_BIN" --version >&2
fi

"$UV_BIN" run python examples/run_grpo.py --config "$CONFIG" "$@"
