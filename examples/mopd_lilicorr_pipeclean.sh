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

# The image ships its own nemo_rl at /opt/nemo-rl, which shadows this checkout. Running
# the branch's examples/run_grpo.py against the image's older package fails with
# ImportError on names the branch added (make_policy_factory). Put the checkout first.
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}${PYTHONPATH:+:$PYTHONPATH}"

# Verify the BRANCH's nemo_rl is what imports -- not merely that some nemo_rl does.
# An earlier version of this check only did `import nemo_rl`, which the image satisfies
# from /opt/nemo-rl, so it passed and the run still died on a missing symbol.
"$PYTHON_BIN" - <<'PREFLIGHT' >&2
import os, sys
import ray, nemo_rl
repo = os.path.realpath(os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd())
pkg = os.path.realpath(os.path.dirname(nemo_rl.__file__))
print(f"[pipeclean] ray {ray.__version__}")
print(f"[pipeclean] nemo_rl {pkg}")
if not pkg.startswith(repo):
    sys.exit(
        f"[pipeclean] nemo_rl resolves to {pkg}, not the checkout at {repo}. "
        "The image's copy would shadow this branch and fail on symbols it adds."
    )
from nemo_rl.data_plane.factory import make_policy_factory  # noqa: F401
from nemo_rl.models.megatron.draft.training import LiLiCorrSpeculator  # noqa: F401
print("[pipeclean] branch nemo_rl imports, lilicorr speculator present")
PREFLIGHT

exec "$PYTHON_BIN" examples/run_grpo.py --config "$CONFIG" "$@"
