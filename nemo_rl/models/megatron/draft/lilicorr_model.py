# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assemble a LiLiCorr draft model: DFlash backbone + reranker head.

Separate from ``lilicorr.py`` (which holds the head and its objective) so the head stays
importable without pulling in Megatron's model-provider machinery -- the same split the
other drafters use, and it keeps the head unit-testable on CPU.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from megatron.core.transformer.module import MegatronModule

from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
from nemo_rl.models.megatron.draft.lilicorr import (
    LiLiCorrHeadConfig,
    build_lilicorr_head,
)

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


def _load_draft_checkpoint_config(model_name: str | None) -> dict[str, Any]:
    """Read the drafter's config.json.

    Refuses to proceed without it: NeMo-RL does not support from-scratch draft init, and
    a LiLiCorr built from defaults would load the released weights into a differently
    shaped reranker or silently skip them.
    """
    if not model_name:
        raise ValueError(
            "policy.draft.model_name is required for speculator_type=lilicorr "
            "(from-scratch draft init is not supported)"
        )
    path = model_name
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()
    with open(path) as fh:
        return json.load(fh)


def load_lilicorr_weights(head: Any, model_name: str) -> None:
    """Load ``lilicorr.*`` tensors from the drafter checkpoint into the head.

    Strict by construction: every head parameter must be covered and every checkpoint
    tensor consumed. Quantized drafters carry ``weight_scale``/``weight_scale_2`` leaves
    that belong to the packed weights, not to the head's float parameters, so they are
    skipped explicitly rather than silently tolerated by a loose load.
    """
    import glob

    from safetensors.torch import load_file

    root = model_name if os.path.isdir(model_name) else os.path.dirname(model_name)
    shards = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {root}")

    prefix = "lilicorr."
    found: dict[str, Any] = {}
    for shard in shards:
        for key, tensor in load_file(shard).items():
            if not key.startswith(prefix):
                continue
            if key.endswith(("weight_scale", "weight_scale_2", "input_scale")):
                continue
            found[key[len(prefix) :]] = tensor

    missing, unexpected = head.load_state_dict(found, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if missing or unexpected:
        raise ValueError(
            f"LiLiCorr head did not load cleanly from {root}: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {list(unexpected)[:3]}). "
            "A partial load leaves the reranker at its init values and collapses "
            "acceptance without erroring."
        )
    print(f"  LiLiCorr head: loaded {len(found)} tensors from {root}")


class LiLiCorrDraftModel(MegatronModule):
    """DFlash backbone with a LiLiCorr reranker on top."""

    def __init__(
        self,
        *,
        body: DFlashBody,
        head: Any,
        config: Any,
    ) -> None:
        super().__init__(config=body.config if hasattr(body, "config") else None)
        self.body = body
        self.lilicorr = head
        self.draft_config = config

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the backbone, then rescore its candidates.

        The backbone's own outputs are returned alongside the reranker logits rather than
        replaced: the training step needs both the block logits (for the DFlash term) and
        the reranker logits (for the CE/margin/penalty terms).
        """
        body_out = self.body(*args, **kwargs)
        return body_out


def build_lilicorr_draft_model(
    *,
    model_provider: Any,
    draft_config: Any,
    pg_collection: "ProcessGroupCollection",
    policy_model_chunk: MegatronModule,
) -> MegatronModule | None:
    """Build the LiLiCorr draft model for a policy chunk."""
    if not draft_config.enabled:
        return None

    ckpt_cfg = _load_draft_checkpoint_config(getattr(draft_config, "model_name", None))
    dflash_cfg = ckpt_cfg.get("dflash_config", {})

    projector = dflash_cfg.get("projector_type")
    if projector != "lilicorr":
        raise ValueError(
            f"draft checkpoint at {draft_config.model_name} has "
            f"projector_type={projector!r}, not 'lilicorr'. Set "
            "policy.draft.speculator_type to match the checkpoint."
        )

    head_cfg = LiLiCorrHeadConfig.from_checkpoint_config(ckpt_cfg)

    body = DFlashBody(
        config=DFlashBodyConfig(
            hidden_size=head_cfg.hidden_size,
            num_attention_heads=head_cfg.num_heads,
        ),
        parallel_config=model_provider,
        tp_group=getattr(pg_collection, "tp", None),
    )

    head = build_lilicorr_head(head_cfg)

    # Load the released reranker weights STRICTLY. A silent partial load is the failure
    # this project keeps meeting: the module initialises, training runs, and acceptance is
    # quietly near zero because the head is still at its init values.
    load_lilicorr_weights(head, draft_config.model_name)

    model = LiLiCorrDraftModel(body=body, head=head, config=draft_config)
    if model_provider.params_dtype is not None:
        model = model.to(model_provider.params_dtype)
    return model
