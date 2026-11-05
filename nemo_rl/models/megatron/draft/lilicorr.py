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
    """Reranker geometry, mirrored from the draft checkpoint's dflash_config."""

    hidden_size: int = 1024
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 2.0
    candidate_topk: int = 8
    target_hidden_size: int = 4096
    vocab_size: int = 131072

    @classmethod
    def from_checkpoint_config(cls, cfg: dict) -> "LiLiCorrHeadConfig":
        """Build from a drafter ``config.json``'s ``dflash_config`` block.

        Raises rather than defaulting on a missing field: a reranker built with guessed
        geometry loads cleanly and scores a different function, which shows up only as a
        lower acceptance length much later.
        """
        d = cfg.get("dflash_config", cfg)
        required = (
            "lilicorr_hidden_size",
            "lilicorr_num_layers",
            "lilicorr_num_heads",
            "lilicorr_mlp_ratio",
            "lilicorr_candidate_topk",
        )
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(
                f"draft checkpoint config is missing LiLiCorr geometry {missing}; "
                "refusing to default it, since a guessed reranker scores a different "
                "function without erroring"
            )
        return cls(
            hidden_size=int(d["lilicorr_hidden_size"]),
            num_layers=int(d["lilicorr_num_layers"]),
            num_heads=int(d["lilicorr_num_heads"]),
            mlp_ratio=float(d["lilicorr_mlp_ratio"]),
            candidate_topk=int(d["lilicorr_candidate_topk"]),
            target_hidden_size=int(cfg.get("hidden_size", 4096)),
            vocab_size=int(cfg.get("vocab_size", 131072)),
        )


class _FeatureMLP(nn.Module):
    """LayerNorm + up/down projection, named to match the released checkpoints."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(in_dim)
        self.feature_mlp = nn.ModuleDict(
            {
                "up_proj": nn.Linear(in_dim, hidden),
                "down_proj": nn.Linear(hidden, hidden),
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.feature_mlp["down_proj"](
            torch.nn.functional.silu(self.feature_mlp["up_proj"](self.feature_norm(x)))
        )


class LiLiCorrHead(nn.Module):
    """Candidate-lattice reranker over a DFlash backbone's per-slot hidden states.

    Scores ``candidate_topk`` candidates per block slot by combining the backbone's slot
    state with each candidate's token embedding, and returns per-candidate logits the
    training objective turns into its CE / margin / penalty terms.
    """

    def __init__(self, config: LiLiCorrHeadConfig, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        # Kept as a reference, not a submodule: the token embedding belongs to the policy,
        # and registering it here would duplicate it in the draft's state dict and in the
        # draft optimizer's param group.
        object.__setattr__(self, "_embed_tokens", embed_tokens)

        self.feature = _FeatureMLP(config.target_hidden_size, h)
        self.token_proj = nn.Linear(config.target_hidden_size, h)
        self.context_proj = nn.Linear(h * 2, h)
        self.pass_hidden_proj = nn.Linear(h * 2, h)
        self.factor_input_proj = nn.Linear(h * 3, h)
        self.anchor_norm = nn.LayerNorm(h)
        self.anchor_out_head = nn.Linear(h, h)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "attn": nn.MultiheadAttention(
                        h, config.num_heads, batch_first=True
                    ),
                    "mlp": nn.ModuleDict(
                        {
                            "up_proj": nn.Linear(h, int(h * config.mlp_ratio)),
                            "down_proj": nn.Linear(int(h * config.mlp_ratio), h),
                        }
                    ),
                }
            )
            for _ in range(config.num_layers)
        )
        self.out_head = nn.Linear(h, 1)

    def forward(self, slot_hidden: Tensor, candidate_ids: Tensor) -> Tensor:
        """Score candidates per slot.

        Args:
            slot_hidden: ``[B, S, target_hidden]`` backbone states, one per block slot.
            candidate_ids: ``[B, S, K]`` candidate token ids from the backbone's topk.

        Returns:
            ``[B, S, K]`` reranker logits.
        """
        b, s, k = candidate_ids.shape
        feat = self.feature(slot_hidden)  # [B, S, h]

        # Detach the embedding lookup: the candidates are fixed ids feeding the reranker,
        # so nothing downstream needs a gradient to the policy's embedding table, and
        # leaving it attached lets the draft loss update policy weights.
        cand_embed = self._embed_tokens(candidate_ids).detach()
        cand = self.token_proj(cand_embed)  # [B, S, K, h]

        ctx = self.context_proj(
            torch.cat([feat, self.anchor_out_head(self.anchor_norm(feat))], dim=-1)
        )
        x = ctx.unsqueeze(2).expand(b, s, k, -1)
        x = self.factor_input_proj(torch.cat([x, cand, x * cand], dim=-1))

        x = x.reshape(b * s, k, -1)
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, need_weights=False)
            x = x + attn_out
            x = x + layer["mlp"]["down_proj"](
                torch.nn.functional.silu(layer["mlp"]["up_proj"](x))
            )
        return self.out_head(x).reshape(b, s, k)


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
