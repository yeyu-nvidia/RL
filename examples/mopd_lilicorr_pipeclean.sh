#!/bin/bash
# COMMAND script for ray.sub: QA-MOPD with a co-trained LiLiCorr drafter.
#
# ray.sub runs this inside the container with the Ray cluster already up, from
# SLURM_SUBMIT_DIR. Extra args are forwarded as config overrides, which is how the
# pipeclean shortens the run without editing the recipe.
set -eux

CONFIG=${CONFIG:-examples/configs/recipes/llm/mopd-lilicorr-super35-nvfp4-realquant.yaml}

# Run with the container's interpreter, NOT `uv run`.
#
# ray.sub starts the Ray head and workers from the container's prebuilt venv. `uv run`
# tears that venv down and rebuilds it from this branch's lockfile, which resolves a
# different Ray and Python:
#
#   cluster:  Ray 2.55.1 / Python 3.13.13      (container)
#   driver:   Ray 2.58.0 / Python 3.13.14      (uv-resolved)
#
# and the driver is then refused by the cluster it just started. The driver has to share
# the environment the cluster runs, so the container's interpreter is the correct one --
# rebuilding the venv is only safe if the cluster is rebuilt with it too.
#
# This works because the LiLiCorr co-training changes add no new dependencies: they are
# pure Python inside nemo_rl, built on modules the image already carries. Set
# PIPECLEAN_UV_RUN=1 to force the uv path if that stops being true, and expect to need a
# container whose Ray matches this branch's lock.
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ "${PIPECLEAN_UV_RUN:-0}" == "1" ]]; then
  echo "[pipeclean] PIPECLEAN_UV_RUN=1: rebuilding the venv. The Ray version must match" \
       "the one ray.sub started the cluster with, or the driver cannot join." >&2
  exec uv run python examples/run_grpo.py --config "$CONFIG" "$@"
fi

# Fail early and legibly if the image cannot import the branch, rather than part-way
# through cluster setup.
"$PYTHON_BIN" -c "import ray, nemo_rl; print('[pipeclean] ray', ray.__version__)" >&2

exec "$PYTHON_BIN" examples/run_grpo.py --config "$CONFIG" "$@"
