"""Mask-spatial injection for TemporalDisMoMotionEncoder.

Injects the spatial position information of the first-frame target mask into
motion_4d, so each motion token carries "which region of the frame my object is
in". In multi-object scenes this means:
    - each object goes through encoder.forward(ref_n, mask_lat=mask_n) independently
    - each object's output motion_4d is already spatially anchored to its own mask region
    - cat along the K_pf dimension -> [1, T_lat, N*K_pf, D]
    - the injection-side cross-attention naturally routes each object's motion to the latent token at the matching spatial position

Design references path A's wan/modules/motiontoken_processor.MaskSpatialEncoder +
inject_target_mask, but adapted to the Temporal path's 4D motion_4d (T_lat, K_pf dims).

Compatibility with existing ckpt:
    - The whole module is newly added; old ckpts do not contain these parameters,
      so load_state_dict(strict=False) will skip them.
    - The second cross-attn layer's attn.out_proj is zero-init -> at step-0 the
      forward is identity (motion_4d unchanged). Continuing training from
      ckpt-120000 will not break existing learning.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


__all__ = [
    "MaskSpatialEncoder",
    "MaskSpatialInject",
    "MaskSpatialPreCrossInject",
]


# ============================================================================
# Mask spatial encoder: mask -> 64 spatial tokens
# ============================================================================

class MaskSpatialEncoder(nn.Module):
    """Encode a [B, 1, H_lat, W_lat] mask into [B, S, D] spatial tokens.

    Three Conv2d /8 layers + AdaptiveAvgPool to a fixed (8, 8) -> 64 tokens, then
    a Linear projection up to D. Same design as path A
    (motiontoken_processor.MaskSpatialEncoder), but internal_dim=D instead of 768
    (the Temporal path uses the DiT hidden dim directly, with no up-projection Linear).

    Args:
        out_dim:    output D (= DiT hidden, e.g.: 3072).
        pool_size:  AdaptiveAvgPool2d target size; (8, 8) -> 64 spatial tokens.
        conv_dim:   output channels of the last Conv layer; default 128.
    """

    def __init__(
        self,
        out_dim: int,
        pool_size=(8, 8),
        conv_dim: int = 128,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.pool_size = tuple(pool_size)
        self.conv_dim = int(conv_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),     # /2
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # /4
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, self.conv_dim, kernel_size=3, stride=2, padding=1),  # /8
            nn.GroupNorm(8, self.conv_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(self.pool_size),
        )
        self.proj = nn.Linear(self.conv_dim, out_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: [B, 1, H_lat, W_lat] in [0, 1]
        Returns:
            [B, S=pool_h*pool_w, out_dim]
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        if mask.dim() != 4:
            raise ValueError(f"mask must be [B,1,H,W]; got shape {tuple(mask.shape)}")
        if mask.shape[1] != 1:
            raise ValueError(f"mask must have 1 channel; got {mask.shape[1]}")

        x = self.conv(mask.float())           # [B, conv_dim, ph, pw]
        x = x.flatten(2).transpose(1, 2)      # [B, S=ph*pw, conv_dim]
        x = self.proj(x)                      # [B, S, out_dim]
        return x


# ============================================================================
# Cross-attention layer (pre-norm, zero-init out_proj)
# ============================================================================

class _MaskCrossAttnLayer(nn.Module):
    """Pre-norm cross-attention + residual.

    Q  comes from motion_4d (flattened to [B, T_lat * K_pf, D])
    KV comes from mask_feats [B, S, D]

    out_proj zero-init -> at step-0 the output = 0, x unchanged (identity).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        zero_init_out: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.norm_q = nn.LayerNorm(dim, eps=eps)
        self.norm_kv = nn.LayerNorm(dim, eps=eps)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        if zero_init_out:
            nn.init.zeros_(self.attn.out_proj.weight)
            if self.attn.out_proj.bias is not None:
                nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  [B, N_q, D]  motion query (flattened motion_4d)
            kv: [B, S, D]    mask spatial tokens
        Returns:
            [B, N_q, D]
        """
        q = self.norm_q(x)
        kv_n = self.norm_kv(kv)
        h, _ = self.attn(q, kv_n, kv_n, need_weights=False)
        return x + h


# ============================================================================
# MaskSpatialInject: inject mask spatial information into motion_4d
# ============================================================================

class MaskSpatialInject(nn.Module):
    """Let motion_4d read spatial information from mask spatial tokens via cross-attention.

    Args:
        dim:           D (DiT hidden, e.g.: 3072).
        num_heads:     number of cross-attn heads; requires dim % num_heads == 0.
        num_layers:    number of cross-attn layers (default 2, matching path A).
        mask_pool_size: pool size of MaskSpatialEncoder; (8, 8) -> 64 spatial tokens.
        mode:          "full" keeps the original cross-attn residual; "weak" adds only
                       a weak residual and removes the common direction shared by all motion tokens.
        weak_scale:    scaling coefficient of the mask residual when mode="weak".
        eps:           LayerNorm eps.

    Forward:
        motion_4d: [B, T_lat, K_pf, D]
        mask_lat:  [B, 1, H_lat, W_lat] in [0, 1]  (pass-through when None, degenerating to identity)

    Returns:
        motion_4d_with_position: [B, T_lat, K_pf, D]

    Step-0 behavior:
        The last cross-attn layer's out_proj is zero-init -> output is approx identity,
        motion_4d unchanged. This way, when loading from an existing ckpt, the early
        training does not break the existing motion signal.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_layers: int = 2,
        mask_pool_size=(8, 8),
        mode: str = "full",
        weak_scale: float = 0.02,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if mode not in {"full", "weak"}:
            raise ValueError(f"mode must be 'full' or 'weak', got {mode!r}")
        if weak_scale < 0:
            raise ValueError(f"weak_scale must be >= 0, got {weak_scale}")
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mode = mode
        self.weak_scale = float(weak_scale)

        self.mask_encoder = MaskSpatialEncoder(
            out_dim=dim, pool_size=mask_pool_size,
        )
        # Multi-layer cross-attn; only the last layer strictly needs zero-init
        # (other layers do not, because as long as the last layer = 0, the overall
        # output = pass-through).
        # But for extra stability, zero-init out_proj on all layers, giving a fully
        # identity output at the start of training.
        self.layers = nn.ModuleList([
            _MaskCrossAttnLayer(
                dim=dim, num_heads=num_heads, eps=eps, zero_init_out=True,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        motion_4d: torch.Tensor,
        mask_lat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            motion_4d: [B, T_lat, K_pf, D]
            mask_lat:  [B, 1, H_lat, W_lat] or None (returns motion_4d directly when None).
        Returns:
            [B, T_lat, K_pf, D]
        """
        if mask_lat is None:
            return motion_4d

        if motion_4d.dim() != 4:
            raise ValueError(
                f"motion_4d must be [B, T_lat, K_pf, D]; got {tuple(motion_4d.shape)}"
            )
        B, T_lat, K_pf, D = motion_4d.shape
        if D != self.dim:
            raise ValueError(
                f"motion_4d dim {D} != MaskSpatialInject.dim {self.dim}"
            )

        # mask -> spatial tokens (same dtype as motion, reduces mixed-precision overhead)
        mask_feats = self.mask_encoder(mask_lat)                  # [B, S, D]
        mask_feats = mask_feats.to(dtype=motion_4d.dtype)

        # Flatten motion_4d to [B, T_lat * K_pf, D] so all tokens attend the mask in parallel
        x = motion_4d.reshape(B, T_lat * K_pf, D)
        x_in = x
        for layer in self.layers:
            x = layer(x, mask_feats)
        if self.mode == "weak":
            delta = x - x_in
            delta = delta - delta.mean(dim=1, keepdim=True)
            x = x_in + self.weak_scale * delta
        return x.reshape(B, T_lat, K_pf, D)


class MaskSpatialPreCrossInject(nn.Module):
    """Pre-fold mask conditioning: do a motion<-mask cross-attn before each self-attn layer.

    Differences from MaskSpatialInject:
      - Input/output are both a motion sequence [B, T_dismo, D]; the final motion_4d is untouched.
      - forward_layer(i) is called once before each fold block, forming an
        interleaved mask cross-attn -> motion self-attn structure.
      - gate is fixed at 1; cross-attn out_proj is zero-init, still approx identity initially.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_layers: int = 2,
        mask_pool_size=(8, 8),
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mask_encoder = MaskSpatialEncoder(
            out_dim=dim, pool_size=mask_pool_size,
        )
        self.layers = nn.ModuleList([
            _MaskCrossAttnLayer(
                dim=dim, num_heads=num_heads, eps=eps, zero_init_out=True,
            )
            for _ in range(num_layers)
        ])

    def encode_mask(
        self,
        mask_lat: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask_feats = self.mask_encoder(mask_lat)
        return mask_feats.to(dtype=dtype)

    def forward_layer(
        self,
        x: torch.Tensor,
        mask_feats: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,T,D]; got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"x dim {x.shape[-1]} != {self.dim}")
        return self.layers[layer_idx](x, mask_feats)
