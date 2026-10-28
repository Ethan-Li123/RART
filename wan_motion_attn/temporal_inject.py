"""Temporal motion cross-attention injection.

By default, a reshape implements the strong inductive bias "latent t only
attends motion t": fold batch/time into the batch dimension so each (sample, t)
does cross-attn independently. Ablation experiments can switch to full mode,
letting all latent queries attend the motion tokens of all timesteps.

Shapes:
    x:              [B, L_padded, D]   L_padded = seq_len (max_seq); the actual effective length is determined by grid_sizes
    motion_4d:      [B, T_lat, K_pf, D]
    grid_sizes:     [B, 3]   (T_i, H_i, W_i) after patch_embedding (Wan ti2v_5B: patch_size=(1,2,2))

Requires (T_i, H_i, W_i) to be identical across the batch (naturally satisfied
in training with expand_video_clips + the same max_frames + the same resolution).
Raises when inconsistent; inference batch_size is usually = 1, so it is unaffected.

Injection flow (inside each WanAttentionBlock.forward, after cross_attn, before ffn):
    x_lat = x[:, :T*H*W].view(B, T, H*W, D)             # un-pad
    Q <- x_lat   reshape -> [B*T, H*W, D]
    K,V <- motion_4d  reshape -> [B*T, K_pf, D]
    out = flash_attention(Q, K, V)                      # [B*T, H*W, D]
    x_out = scatter back to [B, L_padded, D] (pad region stays 0)

Parameter count (dim=3072): 4 Linear (q/k/v/o) ≈ 37.7M / block, matching the fully-connected version.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.modules.attention import flash_attention
from wan.modules.model import WanLayerNorm, WanRMSNorm

_MOTION_RECOMPUTING = ContextVar("rart_motion_recomputing", default=False)


# ============================================================================
# DataProvider (parallel to wan_motion_attn.modules.MotionDataProvider, but passes 4D)
# ============================================================================

class TemporalMotionDataProvider:
    """Set before forward, get inside the block. 4D motion version."""

    def __init__(self):
        self.data: Optional[dict] = None
        self._snapshot_override = ContextVar("rart_motion_snapshot", default=())

    def _active_data(self):
        override = self._snapshot_override.get()
        return override[0] if override else self.data

    def snapshot(self) -> Optional[dict]:
        """Keep each forward's conditions, including their original autograd graph."""
        data = self._active_data()
        return None if data is None else dict(data)

    @contextmanager
    def use_snapshot(self, snapshot: Optional[dict], *, recompute: bool = False):
        """Restore forward conditions during checkpoint recomputation.

        Tensor values are deliberately neither cloned nor detached. Recomputed
        diagnostics must not replace the original forward's routing-loss graph.
        """
        token = self._snapshot_override.set(
            (None if snapshot is None else dict(snapshot),)
        )
        recompute_token = _MOTION_RECOMPUTING.set(
            _MOTION_RECOMPUTING.get() or bool(recompute)
        )
        try:
            yield
        finally:
            _MOTION_RECOMPUTING.reset(recompute_token)
            self._snapshot_override.reset(token)

    def set(
        self,
        motion_4d: Optional[torch.Tensor],
        mask_first: Optional[torch.Tensor] = None,
        object_counts: Optional[torch.Tensor] = None,
        tokens_per_object: Optional[int] = None,
        mask_bias_scale: float = 0.0,
        mask_neg_bias_scale: float = 0.0,
        mask_bias_frames: str = "first",
        attn_route_mask: Optional[torch.Tensor] = None,
        attn_route_loss_enabled: bool = False,
        attn_route_own_weight: float = 0.0,
        attn_route_own_target: float = 0.0,
    ):
        """motion_4d: [B, T_lat, K_pf, D].

        Optional mask routing fields are used only when a mask bias scale > 0:
            mask_first:        [B, N, H_lat, W_lat] first-frame object masks,
                               or [B, N, T, H_lat, W_lat] dynamic masks
            object_counts:     [B] valid object count per sample
            tokens_per_object: K tokens allocated to each object in K_pf
        """
        self.data = {
            "motion_4d": motion_4d,
            "mask_first": mask_first,
            "object_counts": object_counts,
            "tokens_per_object": tokens_per_object,
            "mask_bias_scale": float(mask_bias_scale),
            "mask_neg_bias_scale": float(mask_neg_bias_scale),
            "mask_bias_frames": str(mask_bias_frames),
            "attn_route_mask": attn_route_mask,
            "attn_route_loss_enabled": bool(attn_route_loss_enabled),
            "attn_route_own_weight": float(attn_route_own_weight),
            "attn_route_own_target": float(attn_route_own_target),
        }

    def get(self) -> Optional[torch.Tensor]:
        data = self._active_data()
        if data is None:
            return None
        return data.get("motion_4d")

    def get_mask_bias_kwargs(self) -> dict:
        data = self._active_data()
        if data is None:
            return {}
        return {
            "mask_first": data.get("mask_first"),
            "object_counts": data.get("object_counts"),
            "tokens_per_object": data.get("tokens_per_object"),
            "mask_bias_scale": data.get("mask_bias_scale", 0.0),
            "mask_neg_bias_scale": data.get("mask_neg_bias_scale", 0.0),
            "mask_bias_frames": data.get("mask_bias_frames", "first"),
            "attn_route_mask": data.get("attn_route_mask"),
            "attn_route_loss_enabled": data.get(
                "attn_route_loss_enabled", False,
            ),
            "attn_route_own_weight": data.get(
                "attn_route_own_weight", 0.0,
            ),
            "attn_route_own_target": data.get(
                "attn_route_own_target", 0.0,
            ),
        }

    def reset(self):
        self.data = None


# ============================================================================
# Cross-attention with configurable temporal visibility
# ============================================================================

class TemporalMotionCrossAttention(nn.Module):
    """Frame-wise or full-temporal motion cross-attention.

    Inputs:
        x:              [B, L_padded, D]
        motion_4d:      [B, T_lat, K_pf, D]
        grid_sizes:     [B, 3]  (T, H, W) sizes after patch_embedding

    Output:
        out:            [B, L_padded, D]   injection signal (pad region is 0, caller applies the alpha gate)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        zero_init_o: bool = True,
        temporal_mode: str = "framewise",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        temporal_mode = str(temporal_mode).lower()
        if temporal_mode not in {"framewise", "full"}:
            raise ValueError(
                "temporal_mode must be 'framewise' or 'full', got "
                f"{temporal_mode!r}"
            )
        self.temporal_mode = temporal_mode

        self.norm_q = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm_kv = WanLayerNorm(dim, eps, elementwise_affine=True)

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        self.qk_norm = qk_norm
        self.norm_q_head = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_head = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self._last_mask_bias_diag = {
            "active": 0.0,
            "mean": 0.0,
            "raw_abs_mean": 0.0,
            "logit_ratio": 0.0,
            "logit_share": 0.0,
            "boost_frac": 0.0,
            "neg_mean": 0.0,
            "neg_raw_abs_mean": 0.0,
            "neg_logit_ratio": 0.0,
            "neg_logit_share": 0.0,
            "neg_boost_frac": 0.0,
            "all_logit_ratio": 0.0,
        }
        self._last_attn_route_loss = None
        self._last_attn_route_diag = {
            "active": 0.0,
            "loss": 0.0,
            "leak_attn": 0.0,
            "own_attn": 0.0,
            "leak_share": 0.0,
            "objects": 0.0,
        }

        if zero_init_o:
            nn.init.zeros_(self.o.weight)

    def forward(
        self,
        x: torch.Tensor,                      # [B, L_padded, D]
        motion_4d: torch.Tensor,              # [B, T, K_pf, D]
        grid_sizes: torch.Tensor,             # [B, 3]
        mask_first: Optional[torch.Tensor] = None,
        object_counts: Optional[torch.Tensor] = None,
        tokens_per_object: Optional[int] = None,
        mask_bias_scale: float = 0.0,
        mask_neg_bias_scale: float = 0.0,
        mask_bias_frames: str = "first",
        attn_route_mask: Optional[torch.Tensor] = None,
        attn_route_loss_enabled: bool = False,
        attn_route_own_weight: float = 0.0,
        attn_route_own_target: float = 0.0,
    ) -> torch.Tensor:
        if not _MOTION_RECOMPUTING.get():
            self._last_attn_route_loss = None
        if not _MOTION_RECOMPUTING.get():
            self._last_attn_route_diag = {
                "active": 0.0,
                "loss": 0.0,
                "leak_attn": 0.0,
                "own_attn": 0.0,
                "leak_share": 0.0,
                "objects": 0.0,
            }
        if motion_4d is None or motion_4d.shape[1] == 0:
            return torch.zeros_like(x)

        B, L_padded, D = x.shape
        T_m, K_pf = motion_4d.shape[1], motion_4d.shape[2]
        n, d = self.num_heads, self.head_dim

        # ---- Verify (T,H,W) is identical across the batch ----
        gs = grid_sizes
        T = int(gs[0, 0].item())
        H = int(gs[0, 1].item())
        W = int(gs[0, 2].item())
        if B > 1:
            for i in range(1, B):
                if (int(gs[i, 0].item()), int(gs[i, 1].item()), int(gs[i, 2].item())) != (T, H, W):
                    raise ValueError(
                        "TemporalMotionCrossAttention requires identical (T,H,W) "
                        f"across batch; got {gs.tolist()}"
                    )
        if T_m != T:
            raise ValueError(
                f"motion T_lat={T_m} != latent T={T} (after patch_embedding). "
                "Make sure --motion_embs_per_latent matches vae_stride[0] * patch_size_t."
            )

        L_eff = T * H * W

        # ---- Extract the valid region, reshape ----
        # x_lat: [B, L_eff, D]
        x_lat = x[:, :L_eff, :]
        x_lat = x_lat.view(B, T, H * W, D)                 # [B, T, S, D]   S=H*W
        S = H * W

        # Q ← x_lat
        q = self.norm_q_head(self.q(self.norm_q(x_lat)))   # [B, T, S, D]
        # K, V ← motion_4d
        kv_n = self.norm_kv(motion_4d)                     # [B, T, K_pf, D]
        k = self.norm_k_head(self.k(kv_n))                 # [B, T, K_pf, D]
        v = self.v(kv_n)                                   # [B, T, K_pf, D]

        frame_attn_bias = self._build_mask_attn_bias(
            mask_first=mask_first,
            object_counts=object_counts,
            tokens_per_object=tokens_per_object,
            B=B, T=T, S=S, K_pf=K_pf, H=H, W=W,
            device=q.device, dtype=q.dtype,
            scale=mask_bias_scale,
            neg_scale=mask_neg_bias_scale,
            frames=mask_bias_frames,
        )

        if self.temporal_mode == "framewise":
            # Fold batch+time so each latent frame can only see its aligned
            # motion tokens.
            BT = B * T
            q_attn = q.reshape(BT, S, n, d)
            k_attn = k.reshape(BT, K_pf, n, d)
            v_attn = v.reshape(BT, K_pf, n, d)
            attn_bias = frame_attn_bias
        else:
            # Full-temporal ablation: every latent query can see every motion
            # time. Parameters are unchanged; only token visibility differs.
            q_attn = q.reshape(B, T * S, n, d)
            k_attn = k.reshape(B, T * K_pf, n, d)
            v_attn = v.reshape(B, T * K_pf, n, d)
            attn_bias = self._expand_framewise_bias_to_full(
                frame_attn_bias, B=B, T=T, S=S, K_pf=K_pf,
            )

        out = self._run_attention(
            q_attn, k_attn, v_attn, attn_bias=attn_bias,
            head_dim=d,
            key_validity=self._motion_key_validity(
                object_counts, tokens_per_object,
                B=B, T=T, K_pf=K_pf, device=q.device,
                temporal_mode=self.temporal_mode,
            ),
        )

        if attn_route_loss_enabled:
            route_loss, route_diag = self._compute_attn_route_loss(
                q=q_attn,
                k=k_attn,
                attn_bias=attn_bias,
                route_mask=attn_route_mask,
                object_counts=object_counts,
                tokens_per_object=tokens_per_object,
                B=B, T=T, S=S, K_pf=K_pf, H=H, W=W,
                head_dim=d,
                own_weight=attn_route_own_weight,
                own_target=attn_route_own_target,
                temporal_mode=self.temporal_mode,
            )
            if not _MOTION_RECOMPUTING.get():
                self._last_attn_route_loss = route_loss
            if not _MOTION_RECOMPUTING.get():
                self._last_attn_route_diag = route_diag

        out = out.reshape(B, T, S, D)
        out = self.o(out.reshape(B, L_eff, D))             # [B, L_eff, D]

        # ---- Restore to [B, L_padded, D], pad region stays 0 ----
        if L_padded > L_eff:
            out = torch.cat([
                out, out.new_zeros(B, L_padded - L_eff, D)
            ], dim=1)
        return out

    @staticmethod
    def _expand_framewise_bias_to_full(
        attn_bias: Optional[torch.Tensor],
        B: int,
        T: int,
        S: int,
        K_pf: int,
    ) -> Optional[torch.Tensor]:
        if attn_bias is None:
            return None
        # [B,T,S,K_pf] -> [B,1,T*S,T*K_pf]. A spatial object bias applies
        # equally to every source-time token owned by that object.
        bias = attn_bias.reshape(B, T, S, K_pf)
        bias = bias.unsqueeze(3).expand(B, T, S, T, K_pf)
        return bias.reshape(B, 1, T * S, T * K_pf)

    @staticmethod
    def _empty_mask_bias_diag() -> dict:
        return {
            "active": 0.0,
            "mean": 0.0,
            "raw_abs_mean": 0.0,
            "logit_ratio": 0.0,
            "logit_share": 0.0,
            "boost_frac": 0.0,
            "neg_mean": 0.0,
            "neg_raw_abs_mean": 0.0,
            "neg_logit_ratio": 0.0,
            "neg_logit_share": 0.0,
            "neg_boost_frac": 0.0,
            "all_logit_ratio": 0.0,
        }

    @staticmethod
    def _motion_key_validity(
        object_counts, tokens_per_object, *, B, T, K_pf, device,
        temporal_mode,
    ):
        if object_counts is None or tokens_per_object is None:
            return None
        counts = object_counts.to(device=device).long().view(-1)
        if counts.numel() != B or int(tokens_per_object) <= 0:
            raise ValueError("Invalid motion object-count / tokens-per-object metadata")
        lengths = counts * int(tokens_per_object)
        if bool(((lengths <= 0) | (lengths > K_pf)).any().item()):
            raise ValueError("Motion object counts exceed the available motion tokens")
        if bool((lengths == K_pf).all().item()):
            return None
        valid = torch.arange(K_pf, device=device)[None, :] < lengths[:, None]
        valid = valid[:, None, :].expand(B, T, K_pf)
        if temporal_mode == "framewise":
            return valid.reshape(B * T, K_pf)
        return valid.reshape(B, T * K_pf)

    def _run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        head_dim: int,
        key_validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _MOTION_RECOMPUTING.get():
            self._last_mask_bias_diag = self._empty_mask_bias_diag()
        if attn_bias is None:
            if key_validity is None:
                return flash_attention(q, k, v)
            key_lengths = key_validity.sum(dim=-1, dtype=torch.int32)
            if self.temporal_mode == "full":
                # Full-temporal layout interleaves each frame's padding.
                # Pack real keys first so FlashAttention's lengths are valid.
                def pack(tensor):
                    return torch.stack([
                        torch.cat([
                            sample[valid],
                            sample.new_zeros(
                                sample.shape[0] - int(valid.sum().item()),
                                *sample.shape[1:],
                            ),
                        ], dim=0)
                        for sample, valid in zip(tensor, key_validity)
                    ])
                k, v = pack(k), pack(v)
            return flash_attention(q, k, v, k_lens=key_lengths)

        q_s = q.permute(0, 2, 1, 3).contiguous()
        k_s = k.permute(0, 2, 1, 3).contiguous()
        v_s = v.permute(0, 2, 1, 3).contiguous()
        if not _MOTION_RECOMPUTING.get():
            self._last_mask_bias_diag = self._compute_mask_bias_diag(
                q_s=q_s,
                k_s=k_s,
                attn_bias=attn_bias,
                head_dim=head_dim,
            )
        if key_validity is not None:
            attn_bias = attn_bias.masked_fill(
                ~key_validity[:, None, None, :], float("-inf"),
            )
        out = F.scaled_dot_product_attention(
            q_s, k_s, v_s, attn_mask=attn_bias,
            dropout_p=0.0, is_causal=False,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _compute_mask_bias_diag(
        q_s: torch.Tensor,
        k_s: torch.Tensor,
        attn_bias: torch.Tensor,
        head_dim: int,
    ) -> dict:
        """Measure bias/logit scale without materializing full-temporal logits."""
        with torch.no_grad():
            batch, n_heads, q_len, _ = q_s.shape
            k_len = k_s.shape[2]
            max_logits = 4_000_000
            q_chunk = max(
                1,
                min(
                    q_len,
                    max_logits // max(1, batch * n_heads * k_len),
                ),
            )
            k_t = k_s.detach().float().transpose(-2, -1)

            pos_count = q_s.new_zeros((), dtype=torch.float32)
            neg_count = q_s.new_zeros((), dtype=torch.float32)
            raw_pos_sum = q_s.new_zeros((), dtype=torch.float32)
            raw_neg_sum = q_s.new_zeros((), dtype=torch.float32)
            bias_pos_sum = q_s.new_zeros((), dtype=torch.float32)
            bias_neg_sum = q_s.new_zeros((), dtype=torch.float32)
            raw_all_sum = q_s.new_zeros((), dtype=torch.float32)
            bias_all_sum = q_s.new_zeros((), dtype=torch.float32)
            bias_count = 0
            raw_count = 0

            for q0 in range(0, q_len, q_chunk):
                q1 = min(q0 + q_chunk, q_len)
                raw = torch.matmul(
                    q_s[:, :, q0:q1].detach().float(), k_t,
                ) * (float(head_dim) ** -0.5)
                bias_f = attn_bias[:, :, q0:q1].detach().float()
                pos_w = (bias_f > 1e-6).float()
                neg_w = (bias_f < -1e-6).float()
                raw_abs = raw.abs()

                pos_count = pos_count + pos_w.sum()
                neg_count = neg_count + neg_w.sum()
                raw_pos_sum = raw_pos_sum + (raw_abs * pos_w).sum()
                raw_neg_sum = raw_neg_sum + (raw_abs * neg_w).sum()
                bias_pos_sum = bias_pos_sum + (
                    bias_f.clamp_min(0.0) * pos_w
                ).sum()
                bias_neg_sum = bias_neg_sum + (
                    (-bias_f).clamp_min(0.0) * neg_w
                ).sum()
                raw_all_sum = raw_all_sum + raw_abs.sum()
                bias_all_sum = bias_all_sum + bias_f.abs().sum()
                bias_count += bias_f.numel()
                raw_count += raw_abs.numel()

            pos_denom = pos_count.clamp_min(1.0)
            neg_denom = neg_count.clamp_min(1.0)
            raw_pos = raw_pos_sum / (pos_denom * float(n_heads))
            raw_neg = raw_neg_sum / (neg_denom * float(n_heads))
            bias_pos = bias_pos_sum / pos_denom
            bias_neg = bias_neg_sum / neg_denom
            raw_all = raw_all_sum / max(1, raw_count)
            bias_all = bias_all_sum / max(1, bias_count)
            pos_ratio = bias_pos / raw_pos.clamp_min(1e-6)
            neg_ratio = bias_neg / raw_neg.clamp_min(1e-6)

            return {
                "active": 1.0,
                "mean": float(bias_pos.item()),
                "raw_abs_mean": float(raw_pos.item()),
                "logit_ratio": float(pos_ratio.item()),
                "logit_share": float(
                    (bias_pos / (bias_pos + raw_pos).clamp_min(1e-6)).item()
                ),
                "boost_frac": float(
                    (pos_count / max(1, bias_count)).item()
                ),
                "neg_mean": float(bias_neg.item()),
                "neg_raw_abs_mean": float(raw_neg.item()),
                "neg_logit_ratio": float(neg_ratio.item()),
                "neg_logit_share": float(
                    (bias_neg / (bias_neg + raw_neg).clamp_min(1e-6)).item()
                ),
                "neg_boost_frac": float(
                    (neg_count / max(1, bias_count)).item()
                ),
                "all_logit_ratio": float(
                    (bias_all / raw_all.clamp_min(1e-6)).item()
                ),
            }

    @staticmethod
    def _build_mask_attn_bias(
        mask_first: Optional[torch.Tensor],
        object_counts: Optional[torch.Tensor],
        tokens_per_object: Optional[int],
        B: int,
        T: int,
        S: int,
        K_pf: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        scale: float,
        neg_scale: float,
        frames: str,
    ) -> Optional[torch.Tensor]:
        pos_scale = float(scale)
        neg_scale = float(neg_scale)
        if (
            mask_first is None
            or object_counts is None
            or tokens_per_object is None
            or (pos_scale <= 0.0 and neg_scale <= 0.0)
            or int(tokens_per_object) <= 0
        ):
            return None

        masks = mask_first
        if masks.dim() not in (4, 5) or masks.shape[0] != B:
            return None

        masks = masks.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        dynamic_masks = masks.dim() == 5
        if dynamic_masks:
            Bm, N, Tm, Hm, Wm = masks.shape
            masks = F.interpolate(
                masks.reshape(Bm * N * Tm, 1, Hm, Wm),
                size=(H, W),
                mode="area",
            ).reshape(Bm, N, Tm, S).clamp(0.0, 1.0)
            if Tm != T:
                t_idx = torch.linspace(
                    0, Tm - 1, T, device=device,
                ).round().long().clamp(0, Tm - 1)
                masks = masks.index_select(2, t_idx)
        else:
            Bm, N, Hm, Wm = masks.shape
            masks = F.interpolate(
                masks.reshape(Bm * N, 1, Hm, Wm),
                size=(H, W),
                mode="area",
            ).reshape(Bm, N, S).clamp(0.0, 1.0)

        # Only normalize overlaps; do not amplify soft mask boundaries where
        # a single object has fractional occupancy after area downsampling.
        denom = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        masks = masks / denom

        counts = object_counts.to(device=device).long().view(-1)
        k_obj = int(tokens_per_object)
        bias = torch.zeros(B, T, S, K_pf, device=device, dtype=torch.float32)
        frames_mode = str(frames).lower()
        use_dynamic = dynamic_masks and frames_mode == "dynamic"
        first_only = frames_mode != "all" and not use_dynamic
        for b in range(B):
            n_obj = int(counts[b].item()) if b < counts.numel() else 0
            n_obj = max(0, min(n_obj, N))
            for obj_idx in range(n_obj):
                k0 = obj_idx * k_obj
                k1 = min(k0 + k_obj, K_pf)
                if k0 >= k1:
                    break
                if use_dynamic:
                    obj_mask = masks[b, obj_idx].view(T, S, 1)
                    if pos_scale > 0.0:
                        bias[b, :, :, k0:k1] += (
                            pos_scale * obj_mask
                        ).expand(T, S, k1 - k0)
                    if neg_scale > 0.0 and n_obj > 1:
                        neg_bias = -neg_scale * obj_mask
                        for other_idx in range(n_obj):
                            if other_idx == obj_idx:
                                continue
                            ok0 = other_idx * k_obj
                            ok1 = min(ok0 + k_obj, K_pf)
                            if ok0 < ok1:
                                bias[b, :, :, ok0:ok1] += neg_bias.expand(
                                    T, S, ok1 - ok0,
                                )
                    continue

                if dynamic_masks:
                    obj_mask = masks[b, obj_idx, 0].view(S, 1)
                else:
                    obj_mask = masks[b, obj_idx].view(S, 1)
                if first_only:
                    if pos_scale > 0.0:
                        obj_bias = pos_scale * obj_mask
                        bias[b, 0, :, k0:k1] += obj_bias.expand(S, k1 - k0)
                    if neg_scale > 0.0 and n_obj > 1:
                        neg_bias = -neg_scale * obj_mask
                        for other_idx in range(n_obj):
                            if other_idx == obj_idx:
                                continue
                            ok0 = other_idx * k_obj
                            ok1 = min(ok0 + k_obj, K_pf)
                            if ok0 < ok1:
                                bias[b, 0, :, ok0:ok1] += neg_bias.expand(
                                    S, ok1 - ok0,
                                )
                else:
                    if pos_scale > 0.0:
                        obj_bias = pos_scale * obj_mask
                        bias[b, :, :, k0:k1] += obj_bias.view(
                            1, S, 1,
                        ).expand(T, S, k1 - k0)
                    if neg_scale > 0.0 and n_obj > 1:
                        neg_bias = -neg_scale * obj_mask
                        for other_idx in range(n_obj):
                            if other_idx == obj_idx:
                                continue
                            ok0 = other_idx * k_obj
                            ok1 = min(ok0 + k_obj, K_pf)
                            if ok0 < ok1:
                                bias[b, :, :, ok0:ok1] += neg_bias.view(
                                    1, S, 1,
                                ).expand(T, S, ok1 - ok0)

        return bias.reshape(B * T, 1, S, K_pf).to(dtype=dtype)

    @staticmethod
    def _resize_object_masks(
        masks: Optional[torch.Tensor],
        B: int,
        T: int,
        H: int,
        W: int,
        S: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if masks is None or masks.dim() not in (4, 5) or masks.shape[0] != B:
            return None

        masks = masks.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        if masks.dim() == 5:
            Bm, N, Tm, Hm, Wm = masks.shape
            masks = F.interpolate(
                masks.reshape(Bm * N * Tm, 1, Hm, Wm),
                size=(H, W),
                mode="area",
            ).reshape(Bm, N, Tm, S).clamp(0.0, 1.0)
            if Tm != T:
                t_idx = torch.linspace(
                    0, Tm - 1, T, device=device,
                ).round().long().clamp(0, Tm - 1)
                masks = masks.index_select(2, t_idx)
        else:
            Bm, N, Hm, Wm = masks.shape
            masks = F.interpolate(
                masks.reshape(Bm * N, 1, Hm, Wm),
                size=(H, W),
                mode="area",
            ).reshape(Bm, N, 1, S).clamp(0.0, 1.0)
            masks = masks.expand(Bm, N, T, S).contiguous()

        denom = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        return masks / denom

    def _compute_attn_route_loss(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        route_mask: Optional[torch.Tensor],
        object_counts: Optional[torch.Tensor],
        tokens_per_object: Optional[int],
        B: int,
        T: int,
        S: int,
        K_pf: int,
        H: int,
        W: int,
        head_dim: int,
        own_weight: float,
        own_target: float,
        temporal_mode: str,
    ):
        diag = {
            "active": 0.0,
            "loss": 0.0,
            "leak_attn": 0.0,
            "own_attn": 0.0,
            "leak_share": 0.0,
            "objects": 0.0,
        }
        if (
            route_mask is None
            or object_counts is None
            or tokens_per_object is None
            or int(tokens_per_object) <= 0
        ):
            return None, diag

        if temporal_mode == "full":
            return self._compute_attn_route_loss_full(
                q=q,
                k=k,
                attn_bias=attn_bias,
                route_mask=route_mask,
                object_counts=object_counts,
                tokens_per_object=tokens_per_object,
                B=B,
                T=T,
                S=S,
                K_pf=K_pf,
                H=H,
                W=W,
                head_dim=head_dim,
                own_weight=own_weight,
                own_target=own_target,
            )

        masks = self._resize_object_masks(
            route_mask, B=B, T=T, H=H, W=W, S=S, device=q.device,
        )
        if masks is None:
            return None, diag

        counts = object_counts.to(device=q.device).long().view(-1)
        k_obj = int(tokens_per_object)
        N = masks.shape[1]
        n_heads = int(self.num_heads)
        q_s = q.permute(0, 2, 1, 3).contiguous()
        k_s = k.permute(0, 2, 1, 3).contiguous()
        logits = torch.matmul(q_s, k_s.transpose(-2, -1))
        logits = logits * (float(head_dim) ** -0.5)
        if attn_bias is not None:
            logits = logits + attn_bias.to(dtype=logits.dtype)
        key_validity = self._motion_key_validity(
            object_counts, tokens_per_object,
            B=B, T=T, K_pf=K_pf, device=q.device,
            temporal_mode="framewise",
        )
        if key_validity is not None:
            logits = logits.masked_fill(
                ~key_validity[:, None, None, :], float("-inf"),
            )
        probs = torch.softmax(logits, dim=-1).view(B, T, n_heads, S, K_pf)

        if T > 0:
            masks = masks.clone()
            masks[:, :, 0] = 0.0

        losses = []
        own_vals = []
        leak_vals = []
        share_vals = []
        object_terms = 0
        own_weight = float(own_weight)
        own_target = float(own_target)
        for b in range(B):
            n_obj = int(counts[b].item()) if b < counts.numel() else 0
            n_obj = max(0, min(n_obj, N))
            if n_obj < 2:
                continue
            union = masks[b, :n_obj].sum(dim=0).clamp(0.0, 1.0)
            for obj_idx in range(n_obj):
                k0 = obj_idx * k_obj
                k1 = min(k0 + k_obj, K_pf)
                if k0 >= k1:
                    break
                own_mask = masks[b, obj_idx]
                other_mask = (union - own_mask).clamp_min(0.0)
                own_area = own_mask.sum()
                other_area = other_mask.sum()
                if (
                    float(own_area.detach().item()) <= 1e-4
                    or float(other_area.detach().item()) <= 1e-4
                ):
                    continue

                obj_prob = probs[b, :, :, :, k0:k1].sum(dim=-1).float()
                own_attn = (
                    obj_prob * own_mask[:, None, :]
                ).sum() / (own_area * n_heads).clamp_min(1e-6)
                leak_attn = (
                    obj_prob * other_mask[:, None, :]
                ).sum() / (other_area * n_heads).clamp_min(1e-6)
                leak_loss = leak_attn / (
                    own_attn.detach() + leak_attn
                ).clamp_min(1e-6)
                if own_weight > 0.0 and own_target > 0.0:
                    own_loss = F.relu(
                        own_attn.new_tensor(own_target) - own_attn
                    ).pow(2)
                    leak_loss = leak_loss + own_weight * own_loss
                losses.append(leak_loss)
                own_vals.append(own_attn.detach())
                leak_vals.append(leak_attn.detach())
                share_vals.append(
                    (leak_attn.detach() / (
                        own_attn.detach() + leak_attn.detach()
                    ).clamp_min(1e-6))
                )
                object_terms += 1

        if not losses:
            return None, diag

        loss = torch.stack(losses).mean()
        diag = {
            "active": 1.0,
            "loss": float(loss.detach().float().item()),
            "leak_attn": float(torch.stack(leak_vals).mean().float().item()),
            "own_attn": float(torch.stack(own_vals).mean().float().item()),
            "leak_share": float(torch.stack(share_vals).mean().float().item()),
            "objects": float(object_terms),
        }
        return loss, diag

    def _compute_attn_route_loss_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        route_mask: torch.Tensor,
        object_counts: torch.Tensor,
        tokens_per_object: int,
        B: int,
        T: int,
        S: int,
        K_pf: int,
        H: int,
        W: int,
        head_dim: int,
        own_weight: float,
        own_target: float,
    ):
        """Route loss for full attention, grouping all times of one object."""
        diag = {
            "active": 0.0,
            "loss": 0.0,
            "leak_attn": 0.0,
            "own_attn": 0.0,
            "leak_share": 0.0,
            "objects": 0.0,
        }
        masks = self._resize_object_masks(
            route_mask, B=B, T=T, H=H, W=W, S=S, device=q.device,
        )
        if masks is None:
            return None, diag
        if q.shape[0] != B or q.shape[1] != T * S:
            raise ValueError(
                "full temporal route loss expected q [B,T*S,n,d], got "
                f"{tuple(q.shape)}"
            )
        if k.shape[0] != B or k.shape[1] != T * K_pf:
            raise ValueError(
                "full temporal route loss expected k [B,T*K_pf,n,d], got "
                f"{tuple(k.shape)}"
            )

        if T > 0:
            masks = masks.clone()
            masks[:, :, 0] = 0.0

        counts = object_counts.to(device=q.device).long().view(-1)
        k_obj = int(tokens_per_object)
        n_heads = int(self.num_heads)
        N = masks.shape[1]
        q_len = T * S
        k_len = T * K_pf
        max_logits = 8_000_000
        q_chunk = max(
            1,
            min(q_len, max_logits // max(1, n_heads * k_len)),
        )
        n_slots = (K_pf + k_obj - 1) // k_obj

        losses = []
        own_vals = []
        leak_vals = []
        share_vals = []
        object_terms = 0
        own_weight = float(own_weight)
        own_target = float(own_target)

        for b in range(B):
            n_obj = int(counts[b].item()) if b < counts.numel() else 0
            n_obj = max(0, min(n_obj, N))
            if n_obj < 2:
                continue

            union = masks[b, :n_obj].sum(dim=0).clamp(0.0, 1.0)
            specs = []
            for obj_idx in range(n_obj):
                k0 = obj_idx * k_obj
                k1 = min(k0 + k_obj, K_pf)
                if k0 >= k1:
                    break
                own_mask = masks[b, obj_idx].reshape(q_len)
                other_mask = (union - masks[b, obj_idx]).clamp_min(
                    0.0
                ).reshape(q_len)
                own_area = own_mask.sum()
                other_area = other_mask.sum()
                if (
                    float(own_area.detach().item()) <= 1e-4
                    or float(other_area.detach().item()) <= 1e-4
                ):
                    continue
                specs.append({
                    "slot_idx": obj_idx,
                    "k0": k0,
                    "k1": k1,
                    "own_mask": own_mask,
                    "other_mask": other_mask,
                    "own_area": own_area,
                    "other_area": other_area,
                    "own_num": q.new_zeros((), dtype=torch.float32),
                    "leak_num": q.new_zeros((), dtype=torch.float32),
                })
            if not specs:
                continue

            q_b = q[b].permute(1, 0, 2).contiguous()
            k_grouped = k[b].reshape(T, K_pf, n_heads, head_dim)
            k_grouped = k_grouped.permute(2, 0, 1, 3).contiguous()

            def _group_logsumexp(q_part, k_part, bias_part):
                k_flat = k_part.reshape(n_heads, k_len, head_dim)
                logits = torch.matmul(q_part, k_flat.transpose(-2, -1))
                logits = logits * (float(head_dim) ** -0.5)
                if bias_part.numel() > 0:
                    logits = logits + bias_part.to(dtype=logits.dtype)
                logits = logits.view(
                    n_heads, q_part.shape[1], T, K_pf,
                )
                slot_lse = []
                for slot_idx in range(n_slots):
                    sk0 = slot_idx * k_obj
                    sk1 = min(sk0 + k_obj, K_pf)
                    slot_logits = logits[:, :, :, sk0:sk1].reshape(
                        n_heads, q_part.shape[1], -1,
                    )
                    slot_lse.append(torch.logsumexp(slot_logits, dim=-1))
                return torch.stack(slot_lse, dim=-1)

            from torch.utils.checkpoint import checkpoint as checkpoint_fn
            for q0 in range(0, q_len, q_chunk):
                q1 = min(q0 + q_chunk, q_len)
                if attn_bias is None:
                    bias_part = q.new_empty(0)
                else:
                    bias_part = attn_bias[b, :, q0:q1, :]
                group_lse = checkpoint_fn(
                    _group_logsumexp,
                    q_b[:, q0:q1],
                    k_grouped,
                    bias_part,
                    use_reentrant=False,
                )
                if n_obj < n_slots:
                    group_lse = group_lse[..., :n_obj]
                group_probs = torch.softmax(group_lse, dim=-1)

                for spec in specs:
                    # Group softmax is exactly the probability mass of all
                    # source-time tokens owned by this object slot.
                    obj_prob = group_probs[
                        :, :, spec["slot_idx"]
                    ].float()
                    own_w = spec["own_mask"][q0:q1].view(1, -1)
                    other_w = spec["other_mask"][q0:q1].view(1, -1)
                    spec["own_num"] = spec["own_num"] + (
                        obj_prob * own_w
                    ).sum()
                    spec["leak_num"] = spec["leak_num"] + (
                        obj_prob * other_w
                    ).sum()

            for spec in specs:
                own_attn = spec["own_num"] / (
                    spec["own_area"] * n_heads
                ).clamp_min(1e-6)
                leak_attn = spec["leak_num"] / (
                    spec["other_area"] * n_heads
                ).clamp_min(1e-6)
                leak_loss = leak_attn / (
                    own_attn.detach() + leak_attn
                ).clamp_min(1e-6)
                if own_weight > 0.0 and own_target > 0.0:
                    own_loss = F.relu(
                        own_attn.new_tensor(own_target) - own_attn
                    ).pow(2)
                    leak_loss = leak_loss + own_weight * own_loss
                losses.append(leak_loss)
                own_vals.append(own_attn.detach())
                leak_vals.append(leak_attn.detach())
                share_vals.append(
                    leak_attn.detach() / (
                        own_attn.detach() + leak_attn.detach()
                    ).clamp_min(1e-6)
                )
                object_terms += 1

        if not losses:
            return None, diag

        loss = torch.stack(losses).mean()
        diag = {
            "active": 1.0,
            "loss": float(loss.detach().float().item()),
            "leak_attn": float(torch.stack(leak_vals).mean().float().item()),
            "own_attn": float(torch.stack(own_vals).mean().float().item()),
            "leak_share": float(torch.stack(share_vals).mean().float().item()),
            "objects": float(object_terms),
        }
        return loss, diag


class TemporalMotionCrossAttentionAdapter(nn.Module):
    """Attached to each WanAttentionBlock: TemporalMotionCrossAttention + alpha gate."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        alpha_init: float = 0.0,
        temporal_mode: str = "framewise",
        data_provider: Optional[TemporalMotionDataProvider] = None,
    ):
        super().__init__()
        self.attn = TemporalMotionCrossAttention(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, eps=eps,
            zero_init_o=True, temporal_mode=temporal_mode,
        )
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self._last_diag = {
            "inject_ratio": 0.0,
            "raw_ratio": 0.0,
            "alpha": float(alpha_init),
            "attn_route_active": 0.0,
            "attn_route_loss": 0.0,
            "attn_route_leak_attn": 0.0,
            "attn_route_own_attn": 0.0,
            "attn_route_leak_share": 0.0,
        }
        object.__setattr__(self, "_data_provider", data_provider)

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        provider = getattr(self, "_data_provider", None)
        if provider is None:
            if not _MOTION_RECOMPUTING.get():
                self._last_diag = {
                    "inject_ratio": 0.0,
                    "raw_ratio": 0.0,
                    "alpha": float(self.alpha.detach().float().mean().item()),
                    "attn_route_active": 0.0,
                    "attn_route_loss": 0.0,
                    "attn_route_leak_attn": 0.0,
                    "attn_route_own_attn": 0.0,
                    "attn_route_leak_share": 0.0,
                }
            return torch.zeros_like(x)
        motion_4d = provider.get()
        if motion_4d is None or motion_4d.shape[1] == 0:
            if not _MOTION_RECOMPUTING.get():
                self._last_diag = {
                    "inject_ratio": 0.0,
                    "raw_ratio": 0.0,
                    "alpha": float(self.alpha.detach().float().mean().item()),
                    "attn_route_active": 0.0,
                    "attn_route_loss": 0.0,
                    "attn_route_leak_attn": 0.0,
                    "attn_route_own_attn": 0.0,
                    "attn_route_leak_share": 0.0,
                }
            return torch.zeros_like(x)
        mask_bias_kwargs = provider.get_mask_bias_kwargs()
        out = self.attn(x, motion_4d, grid_sizes, **mask_bias_kwargs)
        injected = self.alpha * out
        with torch.no_grad():
            x_rms = x.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)
            out_rms = out.detach().float().pow(2).mean().sqrt()
            inj_rms = injected.detach().float().pow(2).mean().sqrt()
            bias_diag = getattr(self.attn, "_last_mask_bias_diag", {}) or {}
            route_diag = getattr(self.attn, "_last_attn_route_diag", {}) or {}
            mask_bias_active = float(bias_diag.get("active", 0.0))
            if not _MOTION_RECOMPUTING.get():
                self._last_diag = {
                    "inject_ratio": float((inj_rms / x_rms).item()),
                    "raw_ratio": float((out_rms / x_rms).item()),
                    "alpha": float(self.alpha.detach().float().mean().item()),
                    "attn_route_active": float(route_diag.get("active", 0.0)),
                    "attn_route_loss": float(route_diag.get("loss", 0.0)),
                    "attn_route_leak_attn": float(
                        route_diag.get("leak_attn", 0.0)
                    ),
                    "attn_route_own_attn": float(
                        route_diag.get("own_attn", 0.0)
                    ),
                    "attn_route_leak_share": float(
                        route_diag.get("leak_share", 0.0)
                    ),
                    "mask_bias_active": mask_bias_active,
                    "mask_bias_mean": float(bias_diag.get("mean", 0.0)),
                    "mask_bias_raw_abs_mean": float(
                        bias_diag.get("raw_abs_mean", 0.0)
                    ),
                    "mask_bias_logit_ratio": float(
                        bias_diag.get("logit_ratio", 0.0)
                    ),
                    "mask_bias_logit_share": float(
                        bias_diag.get("logit_share", 0.0)
                    ),
                    "mask_bias_boost_frac": float(
                        bias_diag.get("boost_frac", 0.0)
                    ),
                    "mask_neg_bias_mean": float(
                        bias_diag.get("neg_mean", 0.0)
                    ),
                    "mask_neg_bias_raw_abs_mean": float(
                        bias_diag.get("neg_raw_abs_mean", 0.0)
                    ),
                    "mask_neg_bias_logit_ratio": float(
                        bias_diag.get("neg_logit_ratio", 0.0)
                    ),
                    "mask_neg_bias_logit_share": float(
                        bias_diag.get("neg_logit_share", 0.0)
                    ),
                    "mask_neg_bias_boost_frac": float(
                        bias_diag.get("neg_boost_frac", 0.0)
                    ),
                    "mask_bias_all_logit_ratio": float(
                        bias_diag.get("all_logit_ratio", 0.0)
                    ),
                }
        return injected


# ============================================================================
# Patching utilities (parallel to wan_motion_attn/modules.py)
# ============================================================================

def _patch_block_forward_temporal(
    block: nn.Module, adapter: TemporalMotionCrossAttentionAdapter,
):
    """Monkey-patch WanAttentionBlock.forward to insert temporal motion-attn after cross_attn.

    Almost identical to wan_motion_attn/modules.py:_patch_block_forward, except the
    adapter call passes one extra grid_sizes.
    """
    if hasattr(block, "_motion_attn_orig_forward"):
        return
    block._motion_attn_orig_forward = block.forward

    def new_forward(
        x, e, seq_lens, grid_sizes, freqs, context, context_lens,
        return_pre_ffn=False,
    ):
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_tup = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attention
        y = block.self_attn(
            block.norm1(x).float() * (1 + e_tup[1].squeeze(2)) + e_tup[0].squeeze(2),
            seq_lens, grid_sizes, freqs,
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_tup[2].squeeze(2)

        # cross-attention (frozen, text)
        attn_delta = block.cross_attn(block.norm3(x), context, context_lens)
        x = x + attn_delta

        # === temporal motion cross-attention (alpha-gated) ===
        x = x + adapter(x, grid_sizes)

        # ffn
        y = block.ffn(
            block.norm2(x).float() * (1 + e_tup[4].squeeze(2)) + e_tup[3].squeeze(2),
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_tup[5].squeeze(2)
        if return_pre_ffn:
            return x, attn_delta
        return x

    block.forward = new_forward


def inject_temporal_motion_cross_attention(
    model: nn.Module,
    dim: int,
    num_heads: int,
    qk_norm: bool = True,
    eps: float = 1e-6,
    alpha_init: float = 0.0,
    layer_indices: Optional[Iterable[int]] = None,
    temporal_mode: str = "framewise",
) -> Tuple[List[Optional[TemporalMotionCrossAttentionAdapter]], TemporalMotionDataProvider]:
    """Inject TemporalMotionCrossAttentionAdapter into WanModel.blocks.

    Args / Returns are parallel to wan_motion_attn.modules.inject_motion_cross_attention,
    except data_provider is a TemporalMotionDataProvider and .set accepts 4D motion.
    temporal_mode=framewise keeps a strict 1-to-1; full is for a full-temporal-visibility ablation.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError(f"model {type(model).__name__} has no .blocks")

    if layer_indices is None:
        layer_indices = set(range(len(blocks)))
    else:
        layer_indices = set(int(i) for i in layer_indices)

    data_provider = TemporalMotionDataProvider()
    adapters: List[Optional[TemporalMotionCrossAttentionAdapter]] = []
    n_injected = 0
    for i, blk in enumerate(blocks):
        if i not in layer_indices:
            adapters.append(None)
            continue
        adapter = TemporalMotionCrossAttentionAdapter(
            dim=dim, num_heads=num_heads, qk_norm=qk_norm, eps=eps,
            alpha_init=alpha_init, temporal_mode=temporal_mode,
            data_provider=data_provider,
        )
        blk.add_module("_temporal_motion_attn_adapter", adapter)
        _patch_block_forward_temporal(blk, adapter)
        adapters.append(adapter)
        n_injected += 1

    n_params = count_temporal_motion_attn_trainable(model)
    print(
        f"[TemporalMotionAttn] injected into {n_injected}/{len(blocks)} blocks "
        f"(dim={dim}, heads={num_heads}, temporal_mode={temporal_mode}, "
        f"alpha_init={alpha_init}); "
        f"trainable params: {n_params / 1e6:.1f}M"
    )
    return adapters, data_provider


def count_temporal_motion_attn_trainable(model: nn.Module) -> int:
    total = 0
    for mod in model.modules():
        if isinstance(mod, TemporalMotionCrossAttentionAdapter):
            total += sum(p.numel() for p in mod.parameters())
    return total


def get_temporal_motion_attn_state_dict(model: nn.Module) -> dict:
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TemporalMotionCrossAttentionAdapter):
            for sub_name, p in mod.named_parameters():
                out[f"{name}.{sub_name}"] = p.detach().cpu()
    return out


def load_temporal_motion_attn_state_dict(
    model: nn.Module, state_dict: dict, strict: bool = False,
) -> Tuple[int, int]:
    name_to_p = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TemporalMotionCrossAttentionAdapter):
            for sub_name, p in mod.named_parameters():
                name_to_p[f"{name}.{sub_name}"] = p

    matched, missing = 0, 0
    for k, v in state_dict.items():
        if k in name_to_p and name_to_p[k].shape == v.shape:
            with torch.no_grad():
                name_to_p[k].copy_(
                    v.to(name_to_p[k].device, dtype=name_to_p[k].dtype),
                )
            matched += 1
        else:
            missing += 1
    return matched, missing


def get_temporal_motion_attn_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect all trainable TemporalMotionCrossAttentionAdapter parameters.

    Note: does not forcibly modify requires_grad. If the caller has already frozen
    some parameters (e.g. alpha) beforehand, this respects that setting and only
    collects parameters with requires_grad=True.
    """
    params = []
    for mod in model.modules():
        if isinstance(mod, TemporalMotionCrossAttentionAdapter):
            for p in mod.parameters():
                if p.requires_grad:
                    params.append(p)
    return params
