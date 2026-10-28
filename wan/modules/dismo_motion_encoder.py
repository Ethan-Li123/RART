"""DisMo-based motion encoder (drop-in replacement for Qwen3VLMLLMEncoder).

Pipeline:
    ref RGB video [3, T, H, W] in [0, 1]
        -> _normalise_video        -> [1, T', 256, 256, 3] in [-1, 1] (bf16)
        -> dismo.forward_sliding   -> [1, T'-lookahead, 128]   (frozen, no_grad)
        -> input_proj              -> [1, T'-lookahead, internal_dim]
        -> resampler (Perceiver)   -> [1, K, internal_dim]
        -> out_norm                -> [1, K, internal_dim]

Design notes
------------
* DisMo (`MotionExtractor_Large`) outputs per-frame motion embeddings with the
  layout-/appearance-disentangled training objective. This is exactly what the
  multi-object motion-transfer task needs: the ref-side spatial structure is
  intentionally discarded, and the target_mask cross-attn (downstream in
  ``MotionTokenEncoder``) re-spatialises the abstract motion onto the target.
* The resampler is BLIP-2 / Q-Former style (pure cross-attn + FFN, no self-attn
  among queries). Two layers; only the LAST layer's cross-attn ``out_proj`` and
  FFN ``out`` are zero-initialised so motion contribution is exactly zero at
  step 0 (combined with the existing ``output_proj`` zero-init in
  ``MotionTokenEncoder``).
* DisMo backbone is run in bf16 (matches upstream: DisMo train.py uses bf16
  autocast; the official CogVideoX integration uses bf16 weights). Critical
  numerics inside DisMo (RMSNorm, RoPE) auto-promote to fp32 internally.
* DisMo backbone is frozen (``requires_grad_(False)``) and called inside
  ``torch.no_grad()`` to avoid building a backward graph through DINOv2.
"""

from __future__ import annotations

import contextlib
import os
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sinusoidal_pos_emb(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding, returns [length, dim]."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal pos emb requires even dim, got {dim}")
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    omega = torch.arange(dim // 2, device=device, dtype=torch.float32)            # [D/2]
    omega = 1.0 / (10000.0 ** (omega / (dim // 2)))                               # [D/2]
    angle = pos * omega.unsqueeze(0)                                              # [L, D/2]
    pe = torch.cat([angle.sin(), angle.cos()], dim=-1)                            # [L, D]
    return pe.to(dtype=dtype)


# HuggingFace URL for the released DisMo motion extractor checkpoint (see
# DisMo/hubconf.py:42). We reuse it for the no-ckpt-path fallback.
_DISMO_HF_CKPT_URL = (
    "https://huggingface.co/CompVis/DisMo/resolve/main/motion_extractor_large.pt"
)


@contextlib.contextmanager
def _patched_dino_hub(dino_ckpt_path: Optional[str]):
    """Temporarily replace ``torch.hub.load("facebookresearch/dinov2", ...)``
    so that DisMo's ``DinoFeatureExtractor`` can pick up DINOv2 weights from a
    local ``.pt`` file instead of hitting the network.

    DisMo's loader only consumes ``original_model.state_dict()`` and then
    copies it into a re-implemented backbone (see DisMo/dismo/dinov2.py:399-410).
    We therefore return a tiny shim object that exposes ``.state_dict()``
    backed by the local checkpoint dict; the rest of DisMo's loading code is
    untouched.

    The patch is a no-op when ``dino_ckpt_path`` is ``None`` — DisMo will then
    fall back to its original ``torch.hub.load`` behaviour (network download).

    Accepted local file shapes:
        * raw ``state_dict`` (mapping[str, Tensor]) -- typical official release
        * dict with key ``"state_dict"`` / ``"model"``
        * already-instantiated ``nn.Module``
    """
    if dino_ckpt_path is None:
        yield
        return

    if not os.path.isfile(dino_ckpt_path):
        raise FileNotFoundError(
            f"dino_ckpt_path={dino_ckpt_path!r} does not exist."
        )

    obj = torch.load(dino_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            sd = obj
    elif hasattr(obj, "state_dict"):
        sd = obj.state_dict()
    else:
        raise ValueError(
            f"Unrecognised DINOv2 checkpoint format at {dino_ckpt_path!r}; "
            "expected a state_dict mapping, a dict with 'state_dict'/'model' "
            "key, or an nn.Module."
        )

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
    """Resolve ``dismo.model.MotionExtractor_Large``, optionally injecting a
    local repo path onto ``sys.path`` first."""
    if dismo_repo_path is not None:
        repo = os.path.abspath(dismo_repo_path)
        if not os.path.isdir(repo):
            raise FileNotFoundError(
                f"dismo_repo_path={repo!r} does not exist or is not a directory."
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)
    try:
        from dismo.model import MotionExtractor_Large  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Could not import dismo.model.MotionExtractor_Large. Either pass "
            "dismo_repo_path=<path/to/CompVis/DisMo> to DisMoMotionEncoder, "
            "or install DisMo on the PYTHONPATH."
        ) from e
    return MotionExtractor_Large


class _ResamplerBlock(nn.Module):
    """Cross-attention + self-attention + FFN block (Perceiver / Q-Former style).

    Self-attention among queries lets them coordinate / specialise (avoid
    redundancy when K is large). Order: cross-attn (absorb info from KV) ->
    self-attn (queries协调) -> FFN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        zero_init_out: bool = False,
    ):
        super().__init__()
        # Cross-attention (queries attend to DisMo features)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True,
        )
        # Self-attention among queries
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True,
        )
        # FFN
        self.norm_ff = nn.LayerNorm(dim)
        d_ff = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_ff, dim),
        )
        if zero_init_out:
            # zero-init cross_attn / self_attn out_proj and FFN out -> the
            # block is an identity at init -> no motion contribution at step 0.
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            if self.cross_attn.out_proj.bias is not None:
                nn.init.zeros_(self.cross_attn.out_proj.bias)
            nn.init.zeros_(self.self_attn.out_proj.weight)
            if self.self_attn.out_proj.bias is not None:
                nn.init.zeros_(self.self_attn.out_proj.bias)
            nn.init.zeros_(self.ff[-1].weight)
            nn.init.zeros_(self.ff[-1].bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # 1. Cross-attention: queries吸收 DisMo feature 序列信息
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        q = q + attn_out
        # 2. Self-attention: queries 互相协调, 减少冗余
        q_s = self.norm_self(q)
        self_out, _ = self.self_attn(q_s, q_s, q_s, need_weights=False)
        q = q + self_out
        # 3. FFN
        q = q + self.ff(self.norm_ff(q))
        return q


class DisMoMotionEncoder(nn.Module):
    """Drop-in replacement for ``Qwen3VLMLLMEncoder`` using DisMo's
    ``MotionExtractor_Large`` backbone.

    Public surface (matches the Qwen3-VL contract called from
    ``MotionTokenEncoder`` / ``MotionTokenProcessor``):
        * ``forward(ref_video, instruction=None) -> [1, K, out_dim]``
        * ``forward_batch(ref_videos, instruction=None) -> [B, K, out_dim]``
        * ``enable_lm_lora(...)`` no-op stub returning 0
        * ``lm_lora_*`` constant attributes (always disabled)
    """

    def __init__(
        self,
        out_dim: int,                                # adapter internal_dim (== 768)
        num_video_queries: int = 512,                # K
        dismo_ckpt_path: Optional[str] = None,
        dismo_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,        # local DINOv2 weights (.pt)
        max_frames: Optional[int] = None,            # subsample temporal dim if set
        dtype: torch.dtype = torch.bfloat16,
        resampler_num_layers: int = 2,
        resampler_num_heads: int = 12,
        resampler_ff_mult: float = 4.0,
        gradient_checkpointing: bool = False,
        spatial_size: int = 256,                     # DisMo train_resolution[1:]
        dismo_motion_dim: int = 128,                 # MotionExtractor_Large d_motion
        sliding_lookahead: int = 4,                  # DisMo max_delta_time
    ) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.num_video_queries = num_video_queries
        self.max_frames = max_frames
        self.dtype = dtype
        self.resampler_num_layers = resampler_num_layers
        self.resampler_num_heads = resampler_num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.spatial_size = spatial_size
        self.dismo_motion_dim = dismo_motion_dim
        self.sliding_lookahead = sliding_lookahead
        self.dismo_ckpt_path = dismo_ckpt_path
        self.dino_ckpt_path = dino_ckpt_path

        # ---------- Build DisMo backbone (frozen) ----------
        MotionExtractor_Large = _load_motion_extractor_class(dismo_repo_path)
        # While instantiating MotionExtractor_Large, DisMo internally calls
        # torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg") to seed
        # DINOv2 weights. _patched_dino_hub redirects that to a local .pt when
        # dino_ckpt_path is provided, so offline runs don't hit the network.
        with _patched_dino_hub(dino_ckpt_path):
            self.dismo = MotionExtractor_Large()
        # Load weights: prefer explicit local ckpt, otherwise pull from HF via
        # torch.hub helper (no git clone needed -- direct URL download).
        if dismo_ckpt_path is not None:
            sd = torch.load(
                dismo_ckpt_path, map_location="cpu", weights_only=True,
            )
            self.dismo.load_state_dict(sd)
        else:
            sd = torch.hub.load_state_dict_from_url(
                _DISMO_HF_CKPT_URL, map_location="cpu",
            )
            self.dismo.load_state_dict(sd)
        # Sanity checks against the loaded module (protects against ckpt/code
        # version drift inside the DisMo repo).
        if getattr(self.dismo, "d_motion", dismo_motion_dim) != dismo_motion_dim:
            raise ValueError(
                f"DisMo d_motion={self.dismo.d_motion} != expected "
                f"{dismo_motion_dim}; pass dismo_motion_dim accordingly."
            )
        if getattr(self.dismo, "max_delta_time", sliding_lookahead) != sliding_lookahead:
            raise ValueError(
                f"DisMo max_delta_time={self.dismo.max_delta_time} != expected "
                f"sliding_lookahead={sliding_lookahead}; pass sliding_lookahead "
                "accordingly."
            )
        # Freeze and cast.
        self.dismo.requires_grad_(False)
        self.dismo.eval()
        self.dismo.to(dtype=dtype)

        # ---------- Trainable resampler stack ----------
        self.input_proj = nn.Linear(dismo_motion_dim, out_dim, dtype=dtype)
        self.queries = nn.Parameter(
            torch.randn(num_video_queries, out_dim, dtype=dtype) * 0.02
        )
        blocks: List[nn.Module] = []
        for i in range(resampler_num_layers):
            is_last = (i == resampler_num_layers - 1)
            blk = _ResamplerBlock(
                dim=out_dim,
                num_heads=resampler_num_heads,
                ff_mult=resampler_ff_mult,
                zero_init_out=is_last,
            ).to(dtype=dtype)
            blocks.append(blk)
        self.resampler = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(out_dim, dtype=dtype)

        # ---------- LoRA stub state (always disabled for DisMo) ----------
        # DisMo has no language model; the surface exists only so the
        # MotionTokenProcessor LoRA helpers can call through and get no-ops.
        self.lm_lora_enabled: bool = False
        self.lm_lora_rank: int = 0
        self.lm_lora_alpha: float = 0.0
        self.lm_lora_target_modules: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.dismo.parameters()).device

    def _assert_input_temporal(self, T: int) -> None:
        # forward_sliding requires T >= lookahead (asserted upstream too) and
        # we want at least one motion token, i.e. T_max - lookahead >= 1.
        # Since T_max = min(train_resolution[0]=8, T), a single motion token
        # requires T >= sliding_lookahead + 1.
        min_T = self.sliding_lookahead + 1
        if T < min_T:
            raise ValueError(
                f"DisMo requires at least T >= {min_T} frames "
                f"(sliding_lookahead={self.sliding_lookahead} + 1); got T={T}. "
                "Either supply a longer ref video or increase --dismo_max_frames."
            )

    def _normalise_video(
        self, ref_video: Union[torch.Tensor, Sequence],
    ) -> torch.Tensor:
        """Normalise heterogeneous ref-video formats to ``[T', H, W, 3]`` in
        ``[-1, 1]`` (bf16) on ``self.device``.

        Accepted inputs:
            * ``list[PIL.Image]``
            * ``np.ndarray`` ``[T, H, W, 3]`` (uint8 or float in [0,1])
            * ``torch.Tensor`` ``[3, T, H, W]`` or ``[T, 3, H, W]`` (uint8 or
              float in [0,1])
        """
        from PIL import Image
        import numpy as np

        # ---- 1. Coerce to torch tensor of shape [T, 3, H, W] (or [T, H, W, 3]
        # which we'll permute later).
        if isinstance(ref_video, (list, tuple)) and len(ref_video) > 0 and \
                isinstance(ref_video[0], Image.Image):
            arr = np.stack([np.asarray(im.convert("RGB")) for im in ref_video])
            x = torch.from_numpy(arr)                           # [T, H, W, 3] uint8
        elif isinstance(ref_video, np.ndarray):
            x = torch.from_numpy(ref_video)
        elif isinstance(ref_video, torch.Tensor):
            x = ref_video
        else:
            raise TypeError(
                f"Unsupported ref_video type: {type(ref_video).__name__}. "
                "Expected list[PIL.Image] / np.ndarray / torch.Tensor."
            )

        # ---- 2. Get tensor into [T, 3, H, W] layout (channels-first).
        if x.dim() != 4:
            raise ValueError(f"ref_video must be 4D, got shape {tuple(x.shape)}")
        if x.shape[-1] == 3 and x.shape[0] != 3:
            # [T, H, W, 3] -> [T, 3, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.shape[0] == 3 and x.shape[-1] != 3:
            # [3, T, H, W] -> [T, 3, H, W]
            x = x.permute(1, 0, 2, 3).contiguous()
        elif x.shape[1] == 3:
            # already [T, 3, H, W]
            x = x.contiguous()
        else:
            raise ValueError(
                f"Could not infer channel axis from shape {tuple(x.shape)}; "
                "expected one of [T,H,W,3] / [3,T,H,W] / [T,3,H,W]."
            )

        # ---- 3. Range conversion -> [-1, 1] float.
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div_(127.5).sub_(1.0)
        else:
            x = x.to(dtype=torch.float32)
            mx = float(x.amax().item()) if x.numel() > 0 else 0.0
            mn = float(x.amin().item()) if x.numel() > 0 else 0.0
            if mn < -1.0 - 1e-3 or mx > 1.0 + 1e-3:
                raise ValueError(
                    f"ref_video float values out of expected range; got "
                    f"min={mn:.4f} max={mx:.4f}. Pass values in [0,1] (preferred) "
                    "or [-1,1]; see scripts/train_motiontoken.py:to_motion_rgb."
                )
            if mn >= -1e-3 and mx <= 1.0 + 1e-3:
                # [0, 1] -> [-1, 1]
                x = x.mul_(2.0).sub_(1.0)
            # else: already in ~[-1,1], leave as-is.

        # ---- 4. Optional uniform-stride temporal subsample.
        T = x.shape[0]
        if self.max_frames is not None and T > self.max_frames:
            idx = torch.linspace(0, T - 1, self.max_frames).round().long()
            x = x.index_select(0, idx)

        # ---- 5. Move to device, resize, cast to compute dtype.
        x = x.to(device=self.device)
        x = F.interpolate(
            x, size=(self.spatial_size, self.spatial_size),
            mode="bilinear", antialias=True, align_corners=False,
        )                                                       # [T, 3, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()                  # [T, H, W, 3]
        x = x.to(dtype=self.dtype)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        ref_video: Union[torch.Tensor, Sequence],
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode one reference video into ``[1, K, out_dim]`` motion queries.

        ``instruction`` is accepted for API compatibility with the Qwen3-VL
        encoder and is ignored (DisMo is purely vision-driven).
        """
        del instruction

        x = self._normalise_video(ref_video)            # [T', 256, 256, 3] bf16
        x = x.unsqueeze(0)                              # [1, T', 256, 256, 3]
        self._assert_input_temporal(x.shape[1])

        with torch.no_grad():
            feats = self.dismo.forward_sliding(
                x, lookahead=self.sliding_lookahead,
            )                                           # [1, T'-lookahead, 128]
        feats = self.input_proj(feats)                  # [1, T'-lookahead, D]

        # 加 sinusoidal 时序位置编码: DisMo features 是按时间顺序排列的,
        # cross-attention 本身是 permutation-invariant, 没有 PE 会丢运动顺序.
        T_feat = feats.shape[1]
        pe = _build_sinusoidal_pos_emb(
            T_feat, feats.shape[-1],
            device=feats.device, dtype=feats.dtype,
        )                                               # [T_feat, D]
        feats = feats + pe.unsqueeze(0)                 # [1, T_feat, D]

        q = self.queries.unsqueeze(0)                   # [1, K, D]
        if self.gradient_checkpointing and self.training:
            for blk in self.resampler:
                q = torch.utils.checkpoint.checkpoint(
                    blk, q, feats, use_reentrant=False,
                )
        else:
            for blk in self.resampler:
                q = blk(q, feats)
        return self.out_norm(q)                         # [1, K, D]

    def forward_batch(
        self,
        ref_videos: Sequence[Union[torch.Tensor, Sequence]],
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """Loop wrapper for batched calls. Returns ``[B, K, out_dim]``.

        Each sample may have different ``T`` after subsampling, which makes a
        single batched forward awkward. We follow the Qwen3-VL implementation
        and process samples one-by-one.
        """
        outs = [self.forward(rv, instruction=instruction) for rv in ref_videos]
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # LoRA stub contract (no-ops for DisMo)
    # ------------------------------------------------------------------

    def enable_lm_lora(
        self,
        rank: int,
        alpha: Optional[float] = None,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> int:
        """No-op: DisMo has no language model. Always returns 0.

        The ``MotionTokenProcessor.enable_lm_lora`` API still calls into this
        for parity with the Qwen3-VL path, but no LoRA layers are attached.
        """
        del rank, alpha, target_modules
        if rank and rank > 0:
            warnings.warn(
                "enable_lm_lora called on DisMoMotionEncoder; DisMo has no "
                "language model so this is a no-op (rank ignored).",
                RuntimeWarning, stacklevel=2,
            )
        return 0

    def lm_lora_named_parameters(self):
        return iter(())

    def lm_lora_state_dict(self) -> dict:
        return {}

    def load_lm_lora_state_dict(
        self, state: dict, strict: bool = False,
    ) -> Tuple[int, int]:
        del state, strict
        return (0, 0)
