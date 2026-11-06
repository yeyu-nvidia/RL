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

"""LiLiCorr reranker head for Megatron draft co-training.

LiLiCorr is a DFlash backbone plus a candidate-lattice reranker: the backbone proposes a
block of tokens, and the reranker rescores a small candidate set per slot before the
target verifies. That is the same relationship DSpark has to DFlash, so this module holds
only the head and reuses :class:`~nemo_rl.models.megatron.draft.dflash.DFlashBody` for the
backbone rather than reimplementing a tensor-parallel transformer.

Geometry is read from the draft checkpoint's ``config.json`` (``lilicorr_hidden_size``,
``lilicorr_num_layers``, ``lilicorr_num_heads``, ``lilicorr_mlp_ratio``,
``lilicorr_candidate_topk``) rather than configured here: a mismatch against trained
weights loses accuracy silently instead of raising.

Weight names follow the serving layout that NVIDIA's released drafters use
(``lilicorr.feature_mlp.up_proj``), not the ``nn.Sequential`` positional names
(``lilicorr.feature_mlp.1``) that ModelOpt's exporter emitted before
NVIDIA/Model-Optimizer#3061.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class LiLiCorrHeadConfig:
    """Every constructor argument ModelOpt's LiLiCorrHead takes, read from a checkpoint.

    Each field maps to a key the released drafters carry in ``dflash_config``. None of it
    is defaulted: a reranker built with guessed geometry loads cleanly and scores a
    different function, surfacing much later as a lower acceptance length rather than as
    an error.
    """

    model_hidden_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    block_size: int
    candidate_topk: int
    factor_dim: int
    rms_norm_eps: float
    vector_eps: float
    logit_scale: float

    _KEYS = {
        "hidden_size": "lilicorr_hidden_size",
        "num_layers": "lilicorr_num_layers",
        "num_heads": "lilicorr_num_heads",
        "mlp_ratio": "lilicorr_mlp_ratio",
        "candidate_topk": "lilicorr_candidate_topk",
        "factor_dim": "lilicorr_factor_dim",
        "vector_eps": "lilicorr_vector_eps",
        "logit_scale": "lilicorr_logit_scale",
    }

    @classmethod
    def from_checkpoint_config(cls, cfg: dict) -> "LiLiCorrHeadConfig":
        """Build from a drafter ``config.json``."""
        d = cfg.get("dflash_config", cfg)
        missing = [src for src in cls._KEYS.values() if src not in d]
        if "block_size" not in d and "block_size" not in cfg:
            missing.append("block_size")
        if missing:
            raise ValueError(
                f"draft checkpoint config is missing LiLiCorr geometry {sorted(missing)}; "
                "refusing to default it, since a guessed reranker scores a different "
                "function without erroring"
            )
        kwargs = {dst: d[src] for dst, src in cls._KEYS.items()}
        return cls(
            model_hidden_size=int(cfg.get("hidden_size", 4096)),
            block_size=int(d.get("block_size", cfg.get("block_size"))),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            hidden_size=int(kwargs["hidden_size"]),
            num_layers=int(kwargs["num_layers"]),
            num_heads=int(kwargs["num_heads"]),
            mlp_ratio=float(kwargs["mlp_ratio"]),
            candidate_topk=int(kwargs["candidate_topk"]),
            factor_dim=int(kwargs["factor_dim"]),
            vector_eps=float(kwargs["vector_eps"]),
            logit_scale=float(kwargs["logit_scale"]),
        )


def build_lilicorr_head(config: LiLiCorrHeadConfig) -> nn.Module:
    """Instantiate ModelOpt's LiLiCorrHead.

    Imported rather than reimplemented. The released drafters were trained with exactly
    this module, so a local copy would be a second definition free to drift from the
    weights it has to load -- and an earlier hand-written version here was wrong in ways
    the parameter *names* hid: ``feature_norm`` is [5], a learned weighting across the
    five captured target layers, not a projection of the 4096-dim hidden, and the three
    factor heads emit ``factor_dim`` vectors for cosine scoring rather than a scalar.

    ModelOpt is already a NeMo-RL dependency (see ``nemo_rl/modelopt/``), so this adds no
    new one.
    """
    try:
        from modelopt.torch.speculative.plugins.modeling_lilicorr import LiLiCorrHead
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "speculator_type=lilicorr needs modelopt's LiLiCorrHead "
            "(modelopt.torch.speculative.plugins.modeling_lilicorr). Install "
            "nvidia-modelopt in the training image."
        ) from exc

    return LiLiCorrHead(
        model_hidden_size=config.model_hidden_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        block_size=config.block_size,
        candidate_topk=config.candidate_topk,
        factor_dim=config.factor_dim,
        rms_norm_eps=config.rms_norm_eps,
        vector_eps=config.vector_eps,
        logit_scale=config.logit_scale,
    )


def lilicorr_reranker_loss(
    logits: Tensor,
    gold_index: Tensor,
    *,
    ce_alpha: float = 1.0,
    margin_alpha: float = 1.0,
    penalty_alpha: float = 1.0,
) -> Tensor:
    """Three-term reranker objective: CE + margin + penalty.

    This is the part of LiLiCorr with no analogue in DFlash's projected soft CE or
    DSpark's hard-CE/TVD/confidence terms. ``gold_index`` is ``-100`` where the target's
    token is not among the candidates, which is a normal outcome rather than an error --
    those slots contribute nothing instead of being scored against an arbitrary candidate.
    """
    valid = gold_index.ne(-100)
    if not bool(valid.any()):
        return logits.sum() * 0.0

    flat_logits = logits[valid]  # [N, K]
    flat_gold = gold_index[valid]  # [N]

    ce = torch.nn.functional.cross_entropy(flat_logits, flat_gold)

    gold_score = flat_logits.gather(1, flat_gold.unsqueeze(1))
    others = flat_logits.masked_fill(
        torch.nn.functional.one_hot(flat_gold, flat_logits.size(1)).bool(),
        float("-inf"),
    )
    best_other = others.max(dim=1).values
    margin = torch.nn.functional.relu(1.0 - (gold_score.squeeze(1) - best_other)).mean()

    # Keep scores from drifting far from zero; unbounded logits make the margin term
    # satisfiable by scale alone rather than by ranking.
    penalty = flat_logits.pow(2).mean()

    return ce_alpha * ce + margin_alpha * margin + penalty_alpha * penalty


@dataclass
class LiLiCorrDraftOutput:
    """What a LiLiCorr draft forward must hand the loss.

    The backbone and the reranker are trained by different terms over different shapes, so
    both have to survive the forward rather than one replacing the other:

    * ``student_logits`` -- ``[B, S, V]`` next-token logits from the DFlash backbone, which
      the existing :class:`DraftCrossEntropyLossFn` consumes against the teacher.
    * ``reranker_logits`` -- ``[B, S, K]`` per-candidate scores.
    * ``gold_index`` -- ``[B, S]`` index of the target's token within each slot's
      candidates, ``-100`` where it is absent.
    """

    student_logits: Tensor
    reranker_logits: Tensor
    candidate_ids: Tensor

    def gold_index(self, target_ids: Tensor) -> Tensor:
        """Resolve the target's token within each slot's candidates."""
        return gold_index_from_candidates(self.candidate_ids, target_ids)


def gold_index_from_candidates(candidate_ids: Tensor, target_ids: Tensor) -> Tensor:
    """Locate the target's token among each slot's candidates.

    Returns ``-100`` where the target token was not proposed. That is a normal outcome --
    the backbone's topk missed -- and those slots must contribute nothing rather than be
    scored against an arbitrary candidate, which would train the reranker toward a wrong
    answer on exactly the positions it already failed.
    """
    match = candidate_ids.eq(target_ids.unsqueeze(-1))
    found = match.any(dim=-1)
    # argmax returns 0 for an all-false row; the `found` mask is what makes that safe.
    return torch.where(
        found, match.float().argmax(dim=-1), torch.full_like(target_ids, -100)
    )


def lilicorr_draft_loss(
    output: LiLiCorrDraftOutput,
    target_ids: Tensor,
    *,
    backbone_loss: Tensor,
    ce_alpha: float = 1.0,
    margin_alpha: float = 1.0,
    penalty_alpha: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combine the backbone's draft loss with the reranker's three terms.

    ``backbone_loss`` comes from the existing draft cross-entropy path, so the DFlash
    objective is unchanged and only the reranker terms are added. Returned metrics are
    detached floats, so logging them cannot hold the graph alive.
    """
    gold = output.gold_index(target_ids)
    rerank = lilicorr_reranker_loss(
        output.reranker_logits,
        gold,
        ce_alpha=ce_alpha,
        margin_alpha=margin_alpha,
        penalty_alpha=penalty_alpha,
    )
    total = backbone_loss + rerank
    scored = int(gold.ne(-100).sum())
    return total, {
        "draft_backbone_loss": float(backbone_loss.detach()),
        "draft_reranker_loss": float(rerank.detach()),
        # Share of slots whose target token the backbone actually proposed. A collapse here
        # means the reranker is being trained on almost nothing, which a falling total
        # loss would hide.
        "draft_candidate_hit_rate": scored / max(gold.numel(), 1),
    }
