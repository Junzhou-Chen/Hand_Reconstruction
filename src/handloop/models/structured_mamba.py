"""Spatial hand attention followed by dense, causal per-joint Mamba.

This is a continuous-token adaptation of WestWorld's spatial/temporal
factorization, not its scalar-bin, action-conditioned Sys-MoE implementation.
Each block uses pre-norm spatial self-attention + FFN, then pre-norm Mamba,
with a residual connection around each sublayer. There are no action tokens,
experts, image features, or categorical geometry heads in this module.

Token order is MANO's INTERNAL 16-rotation order, before the commonly used
thumb-first, 21-joint visualization reorder:
    wrist, index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3.
The first token may also encode wrist translation. The caller provides time,
joint identity, shape, handedness, availability, and quality information in
the input features. This module adds parent/finger/depth structure embeddings.

Source checks for the skeleton (2026-10-14):
  https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py
    lev1_idxs / lev2_idxs / lev3_idxs and the subsequent FK composition.
  https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/mano.py
    MANO_RIGHT_KEYPOINTS, explicitly the original MANO J_regressor order.

Padding is removed BEFORE any encoder block, rather than merely zeroed before
the SSM. Valid frames remain in chronological order. Each forward starts a new
Mamba state; it does not maintain a cache across calls. Missing real frames or
joints should remain valid time slots with their availability encoded by the
caller. Only nonexistent/padded time slots belong in history_valid=False.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


MANO_TOKEN_NAMES = (
    "wrist", "index1", "index2", "index3", "middle1", "middle2", "middle3",
    "pinky1", "pinky2", "pinky3", "ring1", "ring2", "ring3",
    "thumb1", "thumb2", "thumb3",
)
MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
# Finger categories: wrist=0, index=1, middle=2, pinky=3, ring=4, thumb=5.
MANO_FINGER_IDS = (0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5)
MANO_DEPTHS = (0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3)
NUM_HAND_TOKENS = len(MANO_TOKEN_NAMES)


class _StructuredMambaBlock(nn.Module):
    """Operate on fully valid [B,T,16,D] sequences only."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
        layer_idx: int,
        mamba_class: type[nn.Module],
    ) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.spatial_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal = mamba_class(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            layer_idx=layer_idx,
        )
        self.temporal_dropout = nn.Dropout(dropout)
        # Preserve Mamba's specialized A/D/dt initializations; do not apply a
        # generic initialization recursively to this block.

    def forward(self, tokens: Tensor) -> Tensor:
        batch, time, joints, width = tokens.shape
        spatial = tokens.reshape(batch * time, joints, width)
        normalized = self.spatial_norm(spatial)
        attended, _ = self.spatial_attention(
            normalized, normalized, normalized, need_weights=False,
        )
        spatial = spatial + self.spatial_dropout(attended)
        spatial = spatial + self.ffn_dropout(self.ffn(self.ffn_norm(spatial)))

        # The batch dimension, not the sequence dimension, contains joints.
        # Mamba's causal convolution and scan therefore only mix past/current
        # features of the same joint. Spatial mixing happened above.
        temporal = spatial.reshape(batch, time, joints, width)
        temporal = temporal.permute(0, 2, 1, 3).reshape(batch * joints, time, width)
        mixed = self.temporal(self.temporal_norm(temporal).contiguous())
        temporal = temporal + self.temporal_dropout(mixed)
        return temporal.reshape(batch, joints, time, width).permute(0, 2, 1, 3)


class StructuredMambaEncoder(nn.Module):
    """Encode numerical hand tokens while completely skipping padded frames.

    Args mirror the TWM configuration but do not depend on TWMConfig. Default
    capacity references WestWorld Appendix D.1: six blocks, width 256, four
    spatial heads, FFN 512, and Mamba state/conv/expansion 64/4/2. Those values
    are a starting configuration, not validated hand-restoration optima.

    Requires the actual ``mamba_ssm.Mamba`` implementation and a compatible
    runtime. Dependency/load errors are explicit; no alternate temporal model
    is silently substituted. Device/kernel support must be checked remotely.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if min(num_layers, dim_feedforward, d_state, d_conv, expand) < 1:
            raise ValueError("layers, FFN width, Mamba state/conv/expand must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise ImportError(
                "StructuredMambaEncoder requires mamba_ssm.Mamba with compiled "
                "kernels compatible with the remote PyTorch/CUDA environment. "
                "No fallback backbone is enabled."
            ) from exc

        self.d_model = d_model
        # Shift parents by one: index 0 means no parent (the wrist), while
        # indices 1..16 refer to MANO joint IDs 0..15.
        self.register_buffer("parent_indices", torch.tensor(MANO_PARENTS) + 1)
        self.register_buffer("finger_indices", torch.tensor(MANO_FINGER_IDS))
        self.register_buffer("depth_indices", torch.tensor(MANO_DEPTHS))
        self.parent_embedding = nn.Embedding(NUM_HAND_TOKENS + 1, d_model)
        self.finger_embedding = nn.Embedding(6, d_model)
        self.depth_embedding = nn.Embedding(4, d_model)
        for embedding in (self.parent_embedding, self.finger_embedding, self.depth_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([
            _StructuredMambaBlock(
                d_model, nhead, dim_feedforward, dropout, d_state, d_conv,
                expand, layer_idx, Mamba,
            )
            for layer_idx in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def _encode_valid(self, tokens: Tensor) -> Tensor:
        structure = (
            self.parent_embedding(self.parent_indices)
            + self.finger_embedding(self.finger_indices)
            + self.depth_embedding(self.depth_indices)
        )
        tokens = tokens + structure[None, None].to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)

    def forward(self, tokens: Tensor, history_valid: Tensor) -> Tensor:
        """Return [B,T,16,D]; invalid time slots are exactly zero.

        ``history_valid`` is a bool [B,T] mask INCLUDING the final current
        frame, which must be True in every sample. It marks real time slots,
        not observation reliability or joint availability. All past slots may
        be padding; the current slot alone then forms a length-one sequence.

        Rows with the same number of valid frames share a Mamba call, even if
        their mask patterns differ. Each row is gathered/scattered using its
        own ordered indices, so padding never changes convolution/SSM state.
        If a caller skips actual time instants, its timestamp features must
        still describe elapsed physical time; compaction does not invent dt.
        """
        if tokens.ndim != 4 or tokens.shape[-2:] != (NUM_HAND_TOKENS, self.d_model):
            raise ValueError(f"tokens must be [B,T,16,{self.d_model}]")
        batch, time, joints, width = tokens.shape
        if batch < 1 or time < 1:
            raise ValueError("tokens require at least one sample and the current frame")
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if history_valid.shape != (batch, time) or history_valid.dtype != torch.bool:
            raise ValueError("history_valid must be bool [B,T], including current")
        if history_valid.device != tokens.device:
            raise ValueError("history_valid and tokens must be on the same device")
        if not bool(history_valid[:, -1].all()):
            raise ValueError("the final current frame must be valid in every sample")

        # The training case with no padding needs no gather/scatter.
        if bool(history_valid.all()):
            return self._encode_valid(tokens)

        lengths = history_valid.sum(dim=1)
        output = tokens.new_zeros(batch * time, joints, width)
        for length in torch.unique(lengths).tolist():
            rows = torch.nonzero(lengths == length, as_tuple=False).flatten()
            row_masks = history_valid.index_select(0, rows)
            # nonzero enumerates row-major, giving chronological time within
            # each row; all rows in this group have exactly `length` entries.
            times = row_masks.nonzero(as_tuple=False)[:, 1].reshape(rows.numel(), length)
            compact = tokens[rows[:, None], times]
            encoded = self._encode_valid(compact)
            flat_indices = (rows[:, None] * time + times).reshape(-1)
            output = output.index_copy(
                0, flat_indices, encoded.reshape(-1, joints, width),
            )
        return output.reshape(batch, time, joints, width)
