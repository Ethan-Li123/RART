"""Temporal-aligned DisMo motion encoder.

Folding side: DisMo per-frame motion embs -> split into chunks by latent-frame
-> shared self-attn folding -> [B, T_lat, K_pf, D]. Semantically aligned with
the original DisMo CogVideoX "GRU 4-step aggregation", but each latent frame
outputs K_pf=p tokens (instead of 1), leaving capacity for spatial selectivity
in the subsequent cross-attn injection.

Chunk alignment (reuses the boundaries from DisMo CogVideoX cogvideox.py:210-220):
    DisMo forward_sliding(lookahead=p) -> embs [1, T_dismo, 128]
    p = motion_embs_per_latent = vae_stride[0] * patch_size_t = 4 (Wan2.2 ti2v_5B)
    T_lat = 1 + n_mid + 1, n_mid = (T_dismo - 1) // p
        t = 0:        chunk = [embs[0]] repeated p times  (single-emb expansion)
        t = 1..n_mid: chunk = embs[1+(t-1)*p : 1+t*p]  (p consecutive embs)
        t = T_lat-1:  chunk = learnable [p, D] padding (no input)

Folding (all chunks share one set of self-attn weights, translation-invariant):
    chunk + intra_chunk_PE  -> SelfAttn (1-2 layers) -> chunk_refined [p, D]

Output: [B, T_lat, K_pf=p, D]   (training script flattens it into [B, T_lat*p, D])

Note: the folding attn does not need frame_PE internally; the inter-frame identity
is carried by the chunk splitting itself, left to the injection side
(grid_sizes implies T_lat).
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from wan.modules.model import WanRMSNorm


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    """ALiBi per-head slopes (original paper: Press et al. 2021).

    slope_i = 2^(-8/n * (i+1)),  i = 0, 1, ..., n-1
    n = num_heads

    Larger slope -> attention focuses more on recent positions;
    different heads use different slopes -> some heads look near (local motion),
    some look far (global trend).
    """
    ratio = 2.0 ** (-8.0 / num_heads)
    slopes = torch.tensor(
        [ratio ** (i + 1) for i in range(num_heads)], dtype=torch.float32,
    )
    return slopes


def _build_alibi_causal_bias(
    seq_len: int,
    num_heads: int,
    batch_size: int = 1,
    window: Optional[int] = None,
    chunk_size: Optional[int] = None,
    attention_mode: str = "causal",
) -> torch.Tensor:
    """Build the fold self-attention bias [B*num_heads, seq_len, seq_len].

    causal:
      - upper triangle (future): -inf  (not allowed to see the future)
      - chunk-aware ALiBi: no decay within the same chunk; history decays by
                            token distance measured from the current chunk start.
      - window=0:          only see causal-visible tokens within the same chunk.
      - window>0:          same chunk does not consume the window; only allow
                            looking at the past `window` history tokens.

    bidirectional:
      - only allow tokens within the same latent chunk to attend bidirectionally;
        no access to history chunks.
      - the distance within the same chunk is always 0, so no ALiBi decay is applied.

    Returns a float tensor, passed directly to nn.MultiheadAttention(attn_mask=...).
    """
    if attention_mode not in {"causal", "bidirectional"}:
        raise ValueError(
            "attention_mode must be 'causal' or 'bidirectional', "
            f"got {attention_mode!r}"
        )
    slopes = _alibi_slopes(num_heads)                          # [num_heads]
    positions = torch.arange(seq_len)
    # dist[i][j] = i - j  (positive=past, negative=future)
    raw_dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # [L, L]

    if chunk_size is not None and int(chunk_size) > 1:
        chunk_size = int(chunk_size)
        # DisMo/Wan alignment boundary: token 0 is the first-frame singleton;
        # afterwards every K_pf tokens starting from token 1 correspond to one
        # latent chunk.
        chunk_start = positions.clone()
        non_first = positions > 0
        chunk_start[non_first] = (
            1 + ((positions[non_first] - 1) // chunk_size) * chunk_size
        )
        q_chunk_start = chunk_start.unsqueeze(1)
        key_pos = positions.unsqueeze(0)
        # same chunk: 0; history before current chunk: chunk_start_q - key_pos.
        # future positions are masked out later uniformly by raw_dist < 0.
        alibi_dist = (q_chunk_start - key_pos).clamp_min(0)
    else:
        alibi_dist = raw_dist.clamp_min(0)

    if attention_mode == "bidirectional":
        if chunk_size is not None and int(chunk_size) > 1:
            # Tokens within the same chunk are mutually visible. Token 0 is the
            # first-frame singleton, consistent with the Wan/DisMo aligned
            # splitting in _fold_chunks.
            same_chunk = (
                chunk_start.unsqueeze(1) == chunk_start.unsqueeze(0)
            )
        else:
            same_chunk = torch.ones(
                seq_len, seq_len, dtype=torch.bool,
            )
        bias = torch.zeros(
            num_heads, seq_len, seq_len, dtype=torch.float32,
        )
        bias = bias.masked_fill(~same_chunk.unsqueeze(0), float('-inf'))
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return bias.reshape(batch_size * num_heads, seq_len, seq_len)

    raw_dist = raw_dist.float()
    alibi_dist = alibi_dist.float()

    # ALiBi bias: -slope * effective distance.
    # [num_heads, L, L]
    bias = -slopes.unsqueeze(1).unsqueeze(2) * alibi_dist.unsqueeze(0)

    # causal: future positions (raw_dist < 0) set to -inf
    future_mask = raw_dist < 0                                  # [L, L]
    bias = bias.masked_fill(future_mask.unsqueeze(0), float('-inf'))
    if window is not None:
        past_too_far = alibi_dist > int(window)
        bias = bias.masked_fill(
            past_too_far.unsqueeze(0), float('-inf'),
        )

    # expand to [B * num_heads, L, L]
    bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)    # [B, H, L, L]
    bias = bias.reshape(batch_size * num_heads, seq_len, seq_len)
    return bias


def _sinusoidal_pe(length: int, dim: int) -> torch.Tensor:
    """Generate a fixed sinusoidal positional encoding [length, dim].

    Consistent with the original Transformer:
      PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Does not participate in gradients; stored via register_buffer, moves with the
    model device/dtype.
    """
    position = torch.arange(length).unsqueeze(1).float()          # [length, 1]
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim)
    )                                                              # [dim/2]
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


_DISMO_HF_CKPT_URL = (
    "https://huggingface.co/CompVis/DisMo/resolve/main/motion_extractor_large.pt"
)


@contextlib.contextmanager
def _patched_dino_hub(dino_ckpt_path: Optional[str]):
    """Redirect torch.hub.load(facebookresearch/dinov2) to a local .pt.

    Fully equivalent to the function of the same name in
    wan/modules/dismo_motion_encoder.py; copied here to avoid cross-package
    dependency.
    """
    if dino_ckpt_path is None:
        yield
        return
    if not os.path.isfile(dino_ckpt_path):
        raise FileNotFoundError(f"dino_ckpt_path not found: {dino_ckpt_path}")
    obj = torch.load(dino_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj.get("model", obj))
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        raise ValueError(f"Unrecognised DINOv2 ckpt format at {dino_ckpt_path}")

    class _DinoShim:
        def __init__(self, _sd): self._sd = _sd
        def state_dict(self): return self._sd

    original_load = torch.hub.load

    def _patched(repo_or_dir, model, *args, **kwargs):
        if isinstance(repo_or_dir, str) and "facebookresearch/dinov2" in repo_or_dir:
            return _DinoShim(sd)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = _patched
    try:
        yield
    finally:
        torch.hub.load = original_load


def _load_motion_extractor_class(dismo_repo_path: Optional[str]):
    if dismo_repo_path is not None:
        repo = os.path.abspath(dismo_repo_path)
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"dismo_repo_path not found: {repo}")
        if repo not in sys.path:
            sys.path.insert(0, repo)
    from dismo.model import MotionExtractor_Large
    return MotionExtractor_Large


class _SelfAttnFoldBlock(nn.Module):
    """Pre-norm self-attention + FFN.

    Supports two modes:
    - chunk mode (old): [N_chunks, p, D], attn_mask=None -> fully connected within chunk
    - causal mode (new): [B, T_seq, D], attn_mask=causal -> full-sequence causal attention
    """

    def __init__(self, dim: int, num_heads: int, ff_mult: float = 4.0,
                 zero_init_out: bool = False):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        d_ff = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_ff, dim),
        )
        if zero_init_out:
            nn.init.zeros_(self.attn.out_proj.weight)
            if self.attn.out_proj.bias is not None:
                nn.init.zeros_(self.attn.out_proj.bias)
            nn.init.zeros_(self.ff[-1].weight)
            nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                return_attn_weights: bool = False):
        # x: [B, L, D]
        h = self.norm_attn(x)
        # attn_mask dtype must match h (LayerNorm output),
        # otherwise nn.MultiheadAttention will raise a dtype mismatch.
        _mask = attn_mask
        if _mask is not None and _mask.dtype != h.dtype:
            _mask = _mask.to(dtype=h.dtype)
        if return_attn_weights:
            h, attn_weights = self.attn(
                h, h, h, attn_mask=_mask, need_weights=True,
            )
        else:
            h, attn_weights = self.attn(
                h, h, h, attn_mask=_mask, need_weights=False,
            ), None
            h = h[0] if isinstance(h, tuple) else h
        x = x + h
        x = x + self.ff(self.norm_ff(x))
        if return_attn_weights:
            return x, attn_weights
        return x


class _DismoProjAdapter(nn.Module):
    """LoRA-style low-rank residual after DisMo's DINO->width projection."""

    def __init__(self, dim: int, hidden_dim: int = 64, scale: float = 0.1):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.scale = float(scale)
        self.rank = hidden_dim
        self.lora_a = nn.Linear(dim, hidden_dim, bias=False)
        self.lora_b = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.lora_b(self.lora_a(x))


class TemporalDisMoMotionEncoder(nn.Module):
    """Per-latent-frame self-attn folding encoder.

    Args:
        out_dim:                  Output dimension D (= DiT hidden, e.g. 3072).
        motion_embs_per_latent:   p, number of motion embs per latent frame.
                                  Wan2.2 ti2v_5B: vae_stride[0]=4 + patch_size[0]=1 -> 4.
        num_layers:               Number of self-attn fold blocks (default 2).
        num_heads:                Number of self-attn heads (needs out_dim % num_heads == 0).
        ff_mult:                  FFN expansion factor (default 4).
        max_frames:               Upper bound on ref video frames (uniform-stride subsample); None uses all.
        spatial_size:             Spatial size fed into DisMo (DisMo train_resolution=256).
        sliding_lookahead:        DisMo max_delta_time, default 4. Must == p.
        fold_attn_window:         fold self-attn local causal window. 0/None=only see
                                  causal tokens within the same chunk; N=same chunk +
                                  past N history DisMo motion tokens.
        fold_attention_mode:      "causal"=original chunk-causal attention;
                                  "bidirectional"=bidirectional attention within chunk only.
        disable_folding:          Bypass fold self-attn/FFN, but keep input projection,
                                  positional encoding, out_norm and the pre-interleave
                                  target-mask cross-attention; used for structural ablation.
        build_mask_spatial_when_disabled:
                                  Build the mask module even when mask anchoring is not
                                  performed, only to keep the random initialization of
                                  subsequent modules aligned during ablation.
        use_mask_spatial:         Whether to enable MaskSpatialInject to inject the
                                  first-frame mask spatial position into motion_4d.
                                  Default False (backward compatible with old ckpt).
        mask_spatial_layers:      Number of cross-attn layers inside MaskSpatialInject; default 2.
        mask_pool_size:           MaskSpatialEncoder pool size (h, w); default (8, 8) -> 64 tokens.
        mask_spatial_stage:       "post"=inject after fold/out_norm; "pre_interleave"=
                                  do motion<-mask cross-attn before each fold self-attn layer.
        mask_spatial_mode:        "full" keeps original injection; "weak" weak injection and removes common direction.
        mask_spatial_weak_scale:  residual scaling when mask_spatial_mode="weak".
        dismo_feature_source:      "head128"=original DisMo 128-d motion head;
                                  "query1024"=take the RMSNorm(1024) after query hidden,
                                  skipping the subsequent FFN->128 head.
        dismo_unfreeze_query_tokens:
                                  Whether to train the DisMo query token when using query1024.
                                  Default False, only tuning the readout edge is more stable.
        dismo_use_proj_adapter:   Whether to add a residual adapter after dismo.proj, for
                                  masked-video input domain adaptation.
        dismo_proj_adapter_hidden_dim:
                                  LoRA rank of the proj adapter.
        dismo_proj_adapter_scale: Fixed scaling of the proj adapter residual.
        dismo_post_dino_mask_bg:   Background retention ratio of the post-DINO feature gate.
                                  1.0=disabled; 0.1=background feature retained at 10%.
        dismo_checkpoint_transformer:
                                  Gradient-checkpoint the frozen transformer layers in the
                                  custom DisMo forward, mainly to save memory when
                                  proj_adapter needs to backprop through the DisMo
                                  transformer.
    """

    def __init__(
        self,
        out_dim: int,
        motion_embs_per_latent: int = 4,
        num_layers: int = 2,
        num_heads: int = 12,
        ff_mult: float = 4.0,
        dismo_ckpt_path: Optional[str] = None,
        dismo_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        spatial_size: int = 256,
        d_motion: int = 128,
        sliding_lookahead: int = 4,
        fold_attn_window: Optional[int] = None,
        fold_attention_mode: str = "causal",
        disable_folding: bool = False,
        use_mask_spatial: bool = False,
        build_mask_spatial_when_disabled: bool = False,
        mask_spatial_layers: int = 2,
        mask_pool_size=(8, 8),
        mask_spatial_stage: str = "post",
        mask_spatial_mode: str = "full",
        mask_spatial_weak_scale: float = 0.02,
        dismo_feature_source: str = "head128",
        dismo_unfreeze_query_tokens: bool = False,
        dismo_use_proj_adapter: bool = False,
        dismo_proj_adapter_hidden_dim: int = 64,
        dismo_proj_adapter_scale: float = 0.1,
        dismo_post_dino_mask_bg: float = 1.0,
        dismo_checkpoint_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.motion_embs_per_latent = int(motion_embs_per_latent)
        self.K_pf = self.motion_embs_per_latent
        self.max_frames = max_frames
        self.dtype = dtype
        self.spatial_size = spatial_size
        self.d_motion = d_motion
        if dismo_feature_source not in {"head128", "query1024"}:
            raise ValueError(
                "dismo_feature_source must be 'head128' or 'query1024', "
                f"got {dismo_feature_source!r}"
        )
        self.dismo_feature_source = dismo_feature_source
        self.dismo_unfreeze_query_tokens = bool(dismo_unfreeze_query_tokens)
        self.dismo_use_proj_adapter = bool(dismo_use_proj_adapter)
        self.dismo_proj_adapter_hidden_dim = int(dismo_proj_adapter_hidden_dim)
        self.dismo_proj_adapter_scale = float(dismo_proj_adapter_scale)
        self.dismo_post_dino_mask_bg = float(dismo_post_dino_mask_bg)
        self.dismo_checkpoint_transformer = bool(dismo_checkpoint_transformer)
        if not (0.0 <= self.dismo_post_dino_mask_bg <= 1.0):
            raise ValueError("dismo_post_dino_mask_bg must be in [0, 1]")
        self.fold_num_heads = int(num_heads)
        self.sliding_lookahead = sliding_lookahead
        self.fold_attn_window = 0 if fold_attn_window is None else int(fold_attn_window)
        if self.fold_attn_window < 0:
            raise ValueError("fold_attn_window must be >= 0")
        if fold_attention_mode not in {"causal", "bidirectional"}:
            raise ValueError(
                "fold_attention_mode must be 'causal' or 'bidirectional', "
                f"got {fold_attention_mode!r}"
            )
        if fold_attention_mode == "bidirectional" and self.fold_attn_window != 0:
            raise ValueError(
                "bidirectional fold attention requires fold_attn_window=0; "
                "it is defined as within-chunk attention only"
            )
        self.fold_attention_mode = fold_attention_mode
        self.disable_folding = bool(disable_folding)
        if mask_spatial_stage not in {"post", "pre_interleave"}:
            raise ValueError(
                "mask_spatial_stage must be 'post' or 'pre_interleave', "
                f"got {mask_spatial_stage!r}"
            )
        self.mask_spatial_stage = mask_spatial_stage

        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim {out_dim} not divisible by num_heads {num_heads}"
            )

        # ---------- DisMo backbone (frozen, eval, bf16) ----------
        MotionExtractor_Large = _load_motion_extractor_class(dismo_repo_path)
        with _patched_dino_hub(dino_ckpt_path):
            self.dismo = MotionExtractor_Large()
        if dismo_ckpt_path is not None:
            sd = torch.load(dismo_ckpt_path, map_location="cpu", weights_only=True)
            self.dismo.load_state_dict(sd)
        else:
            sd = torch.hub.load_state_dict_from_url(
                _DISMO_HF_CKPT_URL, map_location="cpu",
            )
            self.dismo.load_state_dict(sd)
        if getattr(self.dismo, "d_motion", d_motion) != d_motion:
            raise ValueError(
                f"DisMo d_motion={self.dismo.d_motion} != expected {d_motion}"
            )
        if getattr(self.dismo, "max_delta_time", sliding_lookahead) != sliding_lookahead:
            raise ValueError(
                f"DisMo max_delta_time={self.dismo.max_delta_time} != "
                f"sliding_lookahead={sliding_lookahead}"
            )
        self.dismo.requires_grad_(False)
        self.dismo.eval()
        self.dismo.to(dtype=dtype)
        dismo_width = int(getattr(self.dismo, "proj").out_features)
        self.dismo_proj_adapter = None
        if self.dismo_use_proj_adapter:
            self.dismo_proj_adapter = _DismoProjAdapter(
                dim=dismo_width,
                hidden_dim=self.dismo_proj_adapter_hidden_dim,
                scale=self.dismo_proj_adapter_scale,
            ).to(dtype=dtype)
        self._configure_dismo_trainable_params()

        # ---------- trainable folding modules ----------
        # input projection: DisMo feature -> D
        self.motion_feature_dim = (
            dismo_width
            if self.dismo_feature_source == "query1024"
            else int(d_motion)
        )
        self.input_proj = nn.Linear(self.motion_feature_dim, out_dim)
        # Full-sequence sinusoidal PE: positional encoding over the whole DisMo
        # output (T_dismo frames). Uses causal self-attn (not chunk-local), so
        # each token can see the motion history of all preceding frames and gain
        # temporal context. Sinusoidal naturally supports variable-length
        # extrapolation. Preallocate a sufficient length (256), sliced to
        # T_dismo at forward time.
        _max_seq_len = 256
        self.register_buffer(
            "full_seq_pe",
            _sinusoidal_pe(_max_seq_len, out_dim),
            persistent=True,
        )
        # Keep intra_chunk_pe for ckpt compatibility (no unexpected key on load_state_dict)
        self.register_buffer(
            "intra_chunk_pe",
            _sinusoidal_pe(self.K_pf, out_dim),
            persistent=True,
        )

        # Shared self-attn fold blocks (translation-invariant)
        # Zero-init on the last layer makes chunk_refined ~= chunk_in initially
        # (info passthrough, no motion injection difference).
        # Note: this only guarantees the folding side is near-identity at step 0;
        # the alpha gate / o_proj zero-init on the injection side is the real
        # safety lock for true "zero injection".
        blocks: List[nn.Module] = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            blocks.append(_SelfAttnFoldBlock(
                dim=out_dim, num_heads=num_heads, ff_mult=ff_mult,
                zero_init_out=is_last,
            ))
        self.fold_blocks = nn.ModuleList(blocks)
        # Final RMSNorm: consistent in style with the original DisMo (RMSNorm at
        # the end of head + MappingNetwork.out_norm); re-normalizes the
        # accumulated residual stream of the pre-norm fold_blocks.
        # Use RMSNorm instead of LayerNorm:
        #   * does not subtract the mean, preserving the full-dim same-sign DC
        #     component of the motion vector
        #   * gentler than LayerNorm, preserving more of the relative difference
        #     in motion magnitude
        #   * consistent with the whole DisMo backbone design (head/Mapping/TokenSplit)
        # Note: greatly reduces late-stage gradient precision degradation under
        #     bf16 (without this norm the RMS can grow to ~20, and with bf16
        #     ULP ~= 0.156 small gradients get truncated to 0).
        self.out_norm = WanRMSNorm(out_dim, eps=1e-6)

        # Final-frame padding: the T_lat-1 frame has no motion input, use a purely learnable constant
        self.final_frame_padding = nn.Parameter(torch.zeros(self.K_pf, out_dim))
        nn.init.trunc_normal_(self.final_frame_padding, std=0.02)

        # ---------- optional: MaskSpatialInject (multi-object mask spatial anchoring) ----------
        # Via cross-attention, let motion_4d carry the spatial position info of the mask region.
        # For single-object training N=1, this module still takes effect (learns "motion should stick to the mask region");
        # for multi-object inference N>=2, each object is forwarded independently then cat along the K_pf dim, and the injection-side attention
        # naturally routes each object's motion to the corresponding spatial position.
        # Zero-init the last cross-attn out_proj -> output = identity at step 0 (compatible with old ckpt).
        self.use_mask_spatial = bool(use_mask_spatial)
        self.build_mask_spatial_when_disabled = bool(
            build_mask_spatial_when_disabled
        )
        self.mask_spatial_inject = None
        self.mask_spatial_pre_inject = None
        if self.use_mask_spatial or self.build_mask_spatial_when_disabled:
            if self.mask_spatial_stage == "post":
                from .mask_spatial_inject import MaskSpatialInject
                self.mask_spatial_inject = MaskSpatialInject(
                    dim=out_dim,
                    num_heads=num_heads,
                    num_layers=int(mask_spatial_layers),
                    mask_pool_size=tuple(mask_pool_size),
                    mode=mask_spatial_mode,
                    weak_scale=float(mask_spatial_weak_scale),
                    eps=1e-6,
                )
            elif self.mask_spatial_stage == "pre_interleave":
                from .mask_spatial_inject import MaskSpatialPreCrossInject
                self.mask_spatial_pre_inject = MaskSpatialPreCrossInject(
                    dim=out_dim,
                    num_heads=num_heads,
                    num_layers=int(mask_spatial_layers),
                    mask_pool_size=tuple(mask_pool_size),
                    eps=1e-6,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.dismo.parameters()).device

    def _configure_dismo_trainable_params(self) -> None:
        """Freeze DisMo base, then expose only the requested adaptation params."""
        self.dismo.requires_grad_(False)
        self.dismo.eval()
        if self.dismo_feature_source == "query1024":
            # Minimal low-risk adaptation for the hidden path. Keep DINO and the
            # DisMo transformer frozen; only adjust the readout edge by default.
            if self.dismo_unfreeze_query_tokens:
                self.dismo.query_tokens.requires_grad_(True)
            self.dismo.head[0].requires_grad_(True)  # RMSNorm(1024)

    def describe_dismo_adaptation(self) -> dict:
        trainable = [
            n for n, p in self.dismo.named_parameters() if p.requires_grad
        ]
        return {
            "feature_source": self.dismo_feature_source,
            "feature_dim": self.motion_feature_dim,
            "unfreeze_query_tokens": self.dismo_unfreeze_query_tokens,
            "use_proj_adapter": self.dismo_use_proj_adapter,
            "proj_adapter_hidden_dim": self.dismo_proj_adapter_hidden_dim,
            "proj_adapter_scale": self.dismo_proj_adapter_scale,
            "post_dino_mask_bg": self.dismo_post_dino_mask_bg,
            "checkpoint_transformer": self.dismo_checkpoint_transformer,
            "trainable": trainable,
        }

    def _normalise_video(
        self, ref_video: Union[torch.Tensor, Sequence],
    ) -> torch.Tensor:
        """Aligned with wan/modules/dismo_motion_encoder.py:_normalise_video.
        Input tolerates [3,T,H,W] / [T,3,H,W] / [T,H,W,3] (uint8 or float in [0,1] or [-1,1]).
        Output: [T', H, W, 3] in [-1,1] dtype=self.dtype on self.device.
        """
        from PIL import Image

        if isinstance(ref_video, (list, tuple)) and len(ref_video) > 0 and \
                isinstance(ref_video[0], Image.Image):
            arr = np.stack([np.asarray(im.convert("RGB")) for im in ref_video])
            x = torch.from_numpy(arr)
        elif isinstance(ref_video, np.ndarray):
            x = torch.from_numpy(ref_video)
        elif isinstance(ref_video, torch.Tensor):
            x = ref_video
        else:
            raise TypeError(f"Unsupported ref_video type: {type(ref_video).__name__}")

        if x.dim() != 4:
            raise ValueError(f"ref_video must be 4D, got {tuple(x.shape)}")
        if x.shape[-1] == 3 and x.shape[0] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.shape[0] == 3 and x.shape[-1] != 3:
            x = x.permute(1, 0, 2, 3).contiguous()
        elif x.shape[1] == 3:
            x = x.contiguous()
        else:
            raise ValueError(f"Cannot infer channel from {tuple(x.shape)}")

        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div_(127.5).sub_(1.0)
        else:
            x = x.to(dtype=torch.float32)
            mn = float(x.amin().item()) if x.numel() > 0 else 0.0
            if mn >= -1e-3 and x.amax().item() <= 1.0 + 1e-3:
                x = x.mul_(2.0).sub_(1.0)

        T = x.shape[0]
        if self.max_frames is not None and T > self.max_frames:
            idx = torch.linspace(0, T - 1, self.max_frames).round().long()
            x = x.index_select(0, idx)

        x = x.to(device=self.device)
        x = F.interpolate(
            x, size=(self.spatial_size, self.spatial_size),
            mode="bilinear", antialias=True, align_corners=False,
        )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.to(dtype=self.dtype)
        return x

    def _normalise_ref_mask(
        self,
        ref_mask_video: Optional[torch.Tensor],
        target_frames: int,
    ) -> Optional[torch.Tensor]:
        """Normalize ref-side per-frame mask to [1, T, H, W] in [0, 1]."""
        if ref_mask_video is None:
            return None
        m = ref_mask_video
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m)
        if m.dim() == 4:
            if m.shape[0] == 1:
                m = m[0]                  # [T, H, W]
            elif m.shape[1] == 1:
                m = m[:, 0]               # [T, H, W]
            else:
                raise ValueError(
                    f"Cannot infer mask channel from {tuple(m.shape)}"
                )
        elif m.dim() != 3:
            raise ValueError(
                f"ref_mask_video must be [1,T,H,W], [T,1,H,W] or [T,H,W], "
                f"got {tuple(m.shape)}"
            )

        m = m.to(device=self.device, dtype=torch.float32).clamp_(0.0, 1.0)
        T = m.shape[0]
        if self.max_frames is not None and T > self.max_frames:
            idx = torch.linspace(0, T - 1, self.max_frames, device=m.device)
            idx = idx.round().long().clamp(0, T - 1)
            m = m.index_select(0, idx)
            T = m.shape[0]
        if T != target_frames:
            idx = torch.linspace(0, T - 1, target_frames, device=m.device)
            idx = idx.round().long().clamp(0, T - 1)
            m = m.index_select(0, idx)
        return m.unsqueeze(0).contiguous()

    def _dismo_transformer_forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Run the inherited DisMo Transformer without the MotionExtractor head."""
        B, *dims, D = x.shape
        x = x.reshape(B, -1, D)
        pos = pos.reshape(B, -1, pos.shape[-1])
        theta = self.dismo.pos_emb(pos).movedim(-2, -3)
        for layer in self.dismo.layers:
            if (
                self.dismo_checkpoint_transformer
                and torch.is_grad_enabled()
                and x.requires_grad
            ):
                def _layer_forward(h, th, module=layer):
                    return module(h, theta=th)

                x = torch_checkpoint(
                    _layer_forward, x, theta, use_reentrant=False,
                )
            else:
                x = layer(x, theta=theta)
        return x.reshape(B, *dims, D)

    def _dismo_forward_sliding_features(
        self,
        x: torch.Tensor,
        ref_mask_video: Optional[torch.Tensor] = None,
        lookahead: Optional[int] = None,
    ) -> torch.Tensor:
        """DisMo forward_sliding with selectable output feature.

        head128:    original DisMo output [B, T_dismo, 128].
        query1024:  query hidden after head[0] RMSNorm, before FFN->128.
        """
        if lookahead is None:
            lookahead = self.sliding_lookahead

        from dismo.model import make_axial_pos_3d

        B, T, H, W, _ = x.shape
        if T < lookahead:
            raise ValueError(f"Input video length {T} must be at least {lookahead}.")
        ref_mask = self._normalise_ref_mask(ref_mask_video, T)

        T_max = min(self.dismo.train_resolution[0], T)

        if self.dismo.frame_encoder is not None:
            # Keep the huge DINO backbone frozen for the DisMo adaptation branch.
            with torch.no_grad():
                feat = rearrange(x, "b t h w c -> (b t) h w c")
                feat = self.dismo.frame_encoder(feat)
                feat = rearrange(feat, "(b t) h w c -> b t h w c", b=B)
            if (
                ref_mask is not None
                and self.dismo_post_dino_mask_bg < 1.0
            ):
                _, _, H_feat, W_feat, _ = feat.shape
                mask_flat = rearrange(ref_mask, "b t h w -> (b t) 1 h w")
                mask_feat = F.interpolate(
                    mask_flat,
                    size=(H_feat, W_feat),
                    mode="bilinear",
                    align_corners=False,
                ).clamp_(0.0, 1.0)
                mask_feat = rearrange(
                    mask_feat, "(b t) 1 h w -> b t h w 1", b=B,
                ).to(device=feat.device, dtype=feat.dtype)
                bg = float(self.dismo_post_dino_mask_bg)
                gate = bg + (1.0 - bg) * mask_feat
                feat = feat * gate
            x = feat
            B, T, H, W, _ = x.shape

        x = self.dismo.proj(x)
        self._last_dismo_proj_pre_adapter = x.detach()
        if self.dismo_proj_adapter is not None:
            x_before_adapter = x
            x = self.dismo_proj_adapter(x)
            with torch.no_grad():
                self._last_dismo_proj_adapter_ratio = (
                    (x - x_before_adapter).detach().float().norm()
                    / x_before_adapter.detach().float().norm().clamp_min(1e-6)
                ).detach()
        else:
            self._last_dismo_proj_adapter_ratio = None
        pos = make_axial_pos_3d(T_max, H, W, device=x.device).expand(B, -1, 3)

        query_tokens = self.dismo.query_tokens.repeat(B, T_max - lookahead, 1)
        query_pos = make_axial_pos_3d(
            T_max - lookahead, 1, 1, device=x.device,
        ).expand(B, -1, 3)

        window_offsets = list(range(max(T - self.dismo.train_resolution[0], 0) + 1))
        x_b = torch.cat([
            torch.cat([query_tokens, x[:, t:(t + T_max)].flatten(1, 3)], dim=1)
            for t in window_offsets
        ])
        pos_b = torch.cat([query_pos, pos], dim=1).repeat(len(window_offsets), 1, 1)
        embs_b = self._dismo_transformer_forward(x=x_b, pos=pos_b)
        embs_b = embs_b[:, :query_tokens.shape[1]]
        if self.dismo_feature_source == "query1024":
            embs_b = self.dismo.head[0](embs_b)
        else:
            embs_b = self.dismo.head(embs_b)
        embs_b = rearrange(embs_b, "(a b) t d -> a b t d", a=len(window_offsets))

        return torch.cat([embs_b[0]] + list(embs_b[1:, :, -1:].unbind(0)), dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fold_chunks(
        self,
        embs_proj: torch.Tensor,
        mask_lat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-sequence causal self-attention, then reshape to the chunk structure.

        Causal structure: token t can only attend to tokens 0..t, letting later
        frames see the motion history of all preceding frames (similar to the
        accumulation effect of a GRU), while earlier frames are not affected by
        later ones (preserving causality).

        Args:
            embs_proj: [1, T_dismo, D]   motion features after input_proj (no PE).
            mask_lat:  [1, 1, H_lat, W_lat] first-frame target mask; used only
                       when mask_spatial_stage="pre_interleave".

        Returns:
            [1, 1 + n_mid, K_pf, D]      K_pf tokens for the first frame + n_mid middle frames.
        """
        B, T_dismo, D = embs_proj.shape
        p = self.K_pf

        # ---- full-sequence sinusoidal PE ----
        pe = self.full_seq_pe[:T_dismo].to(
            dtype=embs_proj.dtype, device=embs_proj.device,
        )                                                         # [T_dismo, D]
        x = embs_proj + pe.unsqueeze(0)                           # [1, T_dismo, D]
        mask_feats = None
        pre_mask_delta_sum = None
        pre_mask_delta_ratios = []
        if self.use_mask_spatial and self.mask_spatial_stage == "pre_interleave" \
                and self.mask_spatial_pre_inject is not None \
                and mask_lat is not None:
            mask_feats = self.mask_spatial_pre_inject.encode_mask(
                mask_lat, dtype=x.dtype,
            ).to(device=x.device)

        # ---- chunk-aware fold attention bias ----
        # causal is the original ALiBi + chunk-causal structure; bidirectional
        # only opens up future tokens within the same chunk, without introducing
        # cross-chunk history, ensuring this ablation only changes directionality.
        alibi_bias = _build_alibi_causal_bias(
            T_dismo,
            self.fold_num_heads,
            batch_size=B,
            window=self.fold_attn_window,
            chunk_size=self.K_pf,
            attention_mode=self.fold_attention_mode,
        ).to(device=x.device, dtype=x.dtype)                     # [B*H, T, T]

        def _apply_pre_mask_layer(layer_idx: int) -> None:
            nonlocal x, pre_mask_delta_sum
            if mask_feats is not None and \
                    layer_idx < self.mask_spatial_pre_inject.num_layers:
                x_before_mask = x
                x = self.mask_spatial_pre_inject.forward_layer(
                    x, mask_feats, layer_idx=layer_idx,
                )
                with torch.no_grad():
                    delta = (x - x_before_mask).detach()
                    pre_mask_delta_sum = delta if pre_mask_delta_sum is None \
                        else pre_mask_delta_sum + delta
                    ratio = (
                        delta.float().norm()
                        / x_before_mask.detach().float().norm().clamp_min(1e-6)
                    )
                    pre_mask_delta_ratios.append(ratio.detach())

        if self.disable_folding:
            # No-folding ablation: target-mask cross-attention is still executed
            # layer-by-layer, but no temporal self-attention or FFN mixes tokens.
            if mask_feats is not None:
                for i in range(self.mask_spatial_pre_inject.num_layers):
                    _apply_pre_mask_layer(i)
        else:
            # ---- fold self-attn with the selected chunk mask ----
            for i, blk in enumerate(self.fold_blocks):
                _apply_pre_mask_layer(i)
                x = blk(x, attn_mask=alibi_bias)                 # [1, T_dismo, D]
        if pre_mask_delta_ratios:
            ratios = torch.stack(pre_mask_delta_ratios)
            self._last_pre_mask_delta_seq = pre_mask_delta_sum.detach()
            self._last_pre_mask_delta_ratio = ratios.mean().detach()
            self._last_pre_mask_delta_ratio_max = ratios.max().detach()
        else:
            self._last_pre_mask_delta_seq = None
            self._last_pre_mask_delta_ratio = None
            self._last_pre_mask_delta_ratio_max = None

        # ---- reshape to the chunk structure (aligned to latent frames) ----
        # First frame: x[0] repeated p times (latent frame 0 = image condition frame,
        #   weak motion; but after causal attn x[0] contains only its own info,
        #   no longer "4 identical copies")
        first = x[:, 0:1].expand(-1, p, -1)                      # [1, p, D]

        # Middle frames: grouped every p
        T_rest = T_dismo - 1
        n_mid = T_rest // p
        if n_mid <= 0:
            chunks = first.unsqueeze(1)                           # [1, 1, p, D]
        else:
            mid = x[:, 1: 1 + n_mid * p]                          # [1, n_mid*p, D]
            mid = rearrange(mid, "b (n p) d -> b n p d", p=p)     # [1, n_mid, p, D]
            chunks = torch.cat([first.unsqueeze(1), mid], dim=1)  # [1, 1+n_mid, p, D]

        return chunks                                             # [1, 1+n_mid, p, D]

    @torch.no_grad()
    def debug_fold_attn_stats(self, embs_proj: torch.Tensor) -> dict:
        """Debug: return statistics of the fold causal attn (called periodically during training).

        The returned dict contains:
          - qk_logit_mean/std: magnitude of the Q·K scaled logits
          - attn_weight_near_mean: mean attention weight at distance 1-4 tokens
          - attn_weight_far_mean:  mean attention weight at distance 20+ tokens
          - attn_weight_self_mean: mean weight of the diagonal (self)
        For the last fold block, the last token (best reflects "how far it looks").
        """
        if self.disable_folding:
            return {}

        B, T_dismo, D = embs_proj.shape
        # Training forward usually runs under bf16 autocast, so _last_embs_proj
        # may be bf16 while fold block weights remain fp32. This debug pass runs
        # outside autocast; align inputs/masks to module dtype to avoid MHA dtype
        # mismatches such as "expected scalar type BFloat16 but found Float".
        ref_param = next(self.fold_blocks.parameters(), None)
        dbg_dtype = ref_param.dtype if ref_param is not None else embs_proj.dtype
        dbg_device = ref_param.device if ref_param is not None else embs_proj.device
        embs_proj = embs_proj.to(device=dbg_device, dtype=dbg_dtype)
        pe = self.full_seq_pe[:T_dismo].to(
            dtype=embs_proj.dtype, device=embs_proj.device,
        )
        x = embs_proj + pe.unsqueeze(0)

        alibi_bias = _build_alibi_causal_bias(
            T_dismo,
            self.fold_num_heads,
            batch_size=B,
            window=self.fold_attn_window,
            chunk_size=self.K_pf,
            attention_mode=self.fold_attention_mode,
        ).to(device=x.device, dtype=x.dtype)

        # Run all fold blocks, take attn weights from the last layer
        for i, blk in enumerate(self.fold_blocks):
            if i == len(self.fold_blocks) - 1:
                x, attn_w = blk(x, attn_mask=alibi_bias, return_attn_weights=True)
                # attn_w: [B, T, T] (already averaged over heads by nn.MHA? No, need_weights=True returns the averaged [B, T, T])
                # In practice nn.MultiheadAttention returns average_attn_weights: [B, T, T]
            else:
                x = blk(x, attn_mask=alibi_bias)

        if attn_w is None:
            return {}

        # Take the attention distribution of the last token (token T-1): [B, T]
        last_attn = attn_w[0, -1, :]   # [T] attention weight of the last token over all predecessors
        T = last_attn.shape[0]
        positions = torch.arange(T, device=last_attn.device)
        q_pos = T - 1
        if self.K_pf > 1 and q_pos > 0:
            q_chunk_start = 1 + ((q_pos - 1) // self.K_pf) * self.K_pf
            eff_dist = (q_chunk_start - positions).clamp_min(0)
        else:
            eff_dist = (q_pos - positions).clamp_min(0)
        if self.fold_attention_mode == "bidirectional":
            visible = (positions >= q_chunk_start) & (
                positions < q_chunk_start + self.K_pf
            ) if self.K_pf > 1 and q_pos > 0 else positions == q_pos
        else:
            visible = positions <= q_pos

        stats = {}
        # self weight (diagonal, i.e. token T-1's attention to itself)
        stats["fold_attn/self_weight"] = float(last_attn[-1].item())
        same_chunk = visible & (eff_dist == 0)
        if same_chunk.any():
            stats["fold_attn/same_chunk_mean"] = float(
                last_attn[same_chunk].mean().item()
            )
        if self.fold_attention_mode == "bidirectional":
            # Use queries across the whole sequence to measure weights toward
            # future tokens within a chunk. Looking only at the last token there
            # is no future key, so bidirectional cannot be distinguished from causal.
            future_weights = []
            for q in range(T):
                if q == 0:
                    continue
                q_start = 1 + ((q - 1) // self.K_pf) * self.K_pf
                q_end = min(q_start + self.K_pf, T)
                if q + 1 < q_end:
                    future_weights.append(attn_w[0, q, q + 1:q_end])
            if future_weights:
                stats["fold_attn/future_same_chunk_mean"] = float(
                    torch.cat(future_weights).mean().item()
                )
        # near/mid/far are all measured by chunk-aware effective distance:
        # the same chunk does not count as near; history is counted by token
        # going back from the current chunk start.
        near_mask = visible & (eff_dist >= 1) & (eff_dist <= 4)
        if near_mask.any():
            stats["fold_attn/near_1_4_mean"] = float(
                last_attn[near_mask].mean().item()
            )
        mid_mask = visible & (eff_dist >= 5) & (eff_dist <= 15)
        if mid_mask.any():
            stats["fold_attn/mid_5_15_mean"] = float(
                last_attn[mid_mask].mean().item()
            )
        # far: use distance 20+ for full-history; for local-window use the
        # farther half within the window instead.
        # Keep reusing the far_20plus_mean key so the old training-side logging
        # logic stays compatible.
        if self.fold_attn_window is not None:
            max_dist = min(int(self.fold_attn_window), T - 1)
            if max_dist >= 5:
                far_start = max(5, max_dist // 2 + 1)
                far_mask = (
                    visible & (eff_dist >= far_start) & (eff_dist <= max_dist)
                )
                if far_mask.any():
                    far_vals = last_attn[far_mask]
                    far_mean = float(far_vals.mean().item())
                    stats["fold_attn/far_allowed_mean"] = far_mean
                    stats["fold_attn/far_20plus_mean"] = far_mean
        else:
            far_mask = visible & (eff_dist >= 20)
            if far_mask.any():
                stats["fold_attn/far_20plus_mean"] = float(
                    last_attn[far_mask].mean().item()
                )
        # entropy (higher = more uniform)
        eps = 1e-8
        entropy = -float((last_attn * (last_attn + eps).log()).sum().item())
        allowed = visible
        if self.fold_attn_window is not None:
            allowed = allowed & (eff_dist <= int(self.fold_attn_window))
        allowed_len = int(allowed.sum().item())
        max_entropy = float(np.log(allowed_len))
        stats["fold_attn/entropy_ratio"] = entropy / max_entropy if max_entropy > 0 else 0

        return stats

    def forward(
        self,
        ref_video: Union[torch.Tensor, Sequence],
        mask_lat: Optional[torch.Tensor] = None,
        ref_mask_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ref_video -> motion tokens [B=1, T_lat, K_pf, D].

        T_lat = 1 + n_mid + 1, n_mid = (T_dismo - 1) // K_pf

        Args:
            ref_video: Reference video (existing format).
            mask_lat:  Optional [1, 1, H_lat, W_lat] in [0, 1] first-frame target mask.
                       Effective only when self.use_mask_spatial=True; for stage="post"
                       it is injected into motion_4d via MaskSpatialInject; for
                       stage="pre_interleave" it does interleaved mask cross-attn
                       before the fold self-attn.
                       Passing None degrades to the existing behavior (motion is spatially agnostic).
            ref_mask_video:
                       Optional ref-side per-frame mask. When dismo_post_dino_mask_bg < 1,
                       does a soft feature gate after the DINO feature and before dismo.proj.
        """
        x = self._normalise_video(ref_video)                     # [T', 256, 256, 3]
        x = x.unsqueeze(0)                                       # [1, T', H, W, 3]

        T = x.shape[1]
        if T < self.sliding_lookahead + 1:
            raise ValueError(
                f"need at least {self.sliding_lookahead + 1} frames, got {T}"
            )

        dismo_has_trainable = any(p.requires_grad for p in self.dismo.parameters())
        adapter_has_trainable = (
            self.dismo_proj_adapter is not None
            and any(p.requires_grad for p in self.dismo_proj_adapter.parameters())
        )
        needs_custom_dismo = (
            self.dismo_feature_source == "query1024"
            or dismo_has_trainable
            or adapter_has_trainable
            or (
                ref_mask_video is not None
                and self.dismo_post_dino_mask_bg < 1.0
            )
        )
        needs_grad = dismo_has_trainable or adapter_has_trainable
        grad_ctx = contextlib.nullcontext() if needs_grad else torch.no_grad()
        with grad_ctx:
            if not needs_custom_dismo:
                embs = self.dismo.forward_sliding(
                    x, lookahead=self.sliding_lookahead,
                )                                                # [1, T_dismo, 128]
            else:
                embs = self._dismo_forward_sliding_features(
                    x,
                    ref_mask_video=ref_mask_video,
                    lookahead=self.sliding_lookahead,
                )
        self._last_dismo_embs = embs.detach()

        embs_proj = self.input_proj(embs.to(dtype=next(self.input_proj.parameters()).dtype))
        # Store the most recent embs_proj for debug_fold_attn_stats (detached, does not affect the graph)
        self._last_embs_proj = embs_proj.detach()
        # [1, T_dismo, D]

        folded = self._fold_chunks(
            embs_proj, mask_lat=mask_lat,
        )                                                        # [1, 1+n_mid, p, D]

        # Final-frame padding (no motion input, purely learnable constant)
        pad = self.final_frame_padding.to(
            dtype=folded.dtype, device=folded.device,
        ).unsqueeze(0).unsqueeze(0)                              # [1, 1, p, D]
        out = torch.cat([folded, pad], dim=1)                    # [1, T_lat, p, D]
        # Final RMSNorm: re-normalize the accumulated pre-norm residual stream,
        # to prevent bf16 precision degradation + align with the original DisMo style.
        out = self.out_norm(out)
        self._last_pre_mask_motion_4d = out.detach()

        # MaskSpatialInject (optional): inject the first-frame mask spatial position into motion_4d.
        # Degrades to the existing behavior for old training (use_mask_spatial=False) or when mask_lat is not passed.
        if self.use_mask_spatial and self.mask_spatial_inject is not None \
                and self.mask_spatial_stage == "post" and mask_lat is not None:
            out = self.mask_spatial_inject(out, mask_lat)
        self._last_post_mask_motion_4d = out.detach()

        return out.to(dtype=self.dtype)

    def forward_batch(
        self,
        ref_videos: Sequence[Union[torch.Tensor, Sequence]],
        masks_lat: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Forward per sample (T may differ) then concatenate along the batch dim.

        Args:
            ref_videos: List of reference videos.
            masks_lat:  Optional list of [1, 1, H_lat, W_lat] masks, same length as ref_videos.
                        When None no mask is passed for any sample (backward compatible).

        Returns: [B, T_lat, K_pf, D]

        Requires all samples to have the same T_lat (naturally satisfied when
        --max_frames is fixed during training); otherwise the cat will fail, and
        a clear error is raised here.
        """
        if masks_lat is not None and len(masks_lat) != len(ref_videos):
            raise ValueError(
                f"masks_lat length {len(masks_lat)} != ref_videos {len(ref_videos)}"
            )
        outs = []
        for i, rv in enumerate(ref_videos):
            m = masks_lat[i] if masks_lat is not None else None
            outs.append(self.forward(rv, mask_lat=m))
        shapes = [tuple(o.shape) for o in outs]
        if any(s != shapes[0] for s in shapes):
            raise ValueError(
                f"forward_batch: all samples must have the same motion_4d shape, got {shapes}. "
                "On the training side, ensure ref videos within a batch have the same number of frames (e.g. fixed --max_frames)."
            )
        return torch.cat(outs, dim=0)
