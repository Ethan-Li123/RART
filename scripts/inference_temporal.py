"""RART inference for released motion-transfer checkpoints.

Unified input semantics: one independent ref video per object (multi-ref paradigm).
  - Single object: ref_videos provides 1 clip + 1 tgt mask (mask optional, all-1 fallback)
  - Multi-object (N>=2): ref_videos provides N clips + N tgt masks (required)

ref_mask_paths (one mask per ref clip, masking out the ref video background) is always optional;
applied if given (behavior consistent with training's motion_ref_mask), otherwise the whole ref is used.

Usage:
  # single command
  python scripts/inference_temporal.py \\
      --motion_ckpt outputs/checkpoint-200000 \\
      --checkpoint_dir /path/to/Wan2.2-TI2V-5B \\
      --dismo_repo_path ... --dismo_ckpt_path ... --dino_ckpt_path ... \\
      --ref_videos /path/r1.mp4 \\
      --target_image /path/tgt.jpg \\
      --persons "woman" --actions "dancing" \\
      --output_path out.mp4 \\
      --temporal_use_mask_spatial

  # multi-object single command
  python scripts/inference_temporal.py \\
      ... \\
      --ref_videos "/path/r1.mp4,/path/r2.mp4" \\
      --target_image /path/tgt.jpg \\
      --target_mask_paths "/path/tm1.mp4+/path/tm2.mp4" \\
      --persons "woman|man" --actions "dancing|running" \\
      --output_path out.mp4

  # batch CSV
  python scripts/inference_temporal.py \\
      ... \\
      --inference_csv test.csv \\
      --output_dir outputs/inference \\
      --concat_ref

CSV schema:
  required (all rows):
    ref_videos          ',' separated; 1 clip -> single object, N clips -> multi-object
    target_image
    persons             '|' separated, count == len(ref_videos)
    actions             '|' separated, count == len(ref_videos), order aligned with ref_videos
    output_name
  required (multi-object N>=2):
    target_mask_paths   '+' separated N tgt masks, count == len(ref_videos)
  optional:
    target_mask_paths   omittable for single object (uses all-1 mask when omitted, global motion)
    ref_videos_clip_starts  ',' separated N start frames (default 0)
    ref_mask_paths      ',' separated N ref masks (mask out ref background, consistent with training's motion_ref_mask)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import time
import argparse
from typing import List, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.modules.model import WanModel
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.configs import WAN_CONFIGS
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="RART multi-object motion transfer inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Model paths")
    g.add_argument("--checkpoint_dir", type=str,
                   default=os.environ.get("CHECKPOINT_DIR") or "weights/Wan2.2-TI2V-5B",
                   help="Wan2.2-TI2V-5B directory")
    g.add_argument("--motion_ckpt", type=str,
                   default=os.environ.get("MOTION_CKPT") or "outputs/checkpoint-200000",
                   help="training checkpoint directory (contains temporal_motion.pt)")
    g.add_argument("--config_name", type=str, default="ti2v-5B")
    g.add_argument("--dismo_repo_path", type=str,
                   default=os.environ.get("DISMO_REPO_PATH") or "DisMo")
    g.add_argument("--dismo_ckpt_path", type=str,
                   default=os.environ.get("DISMO_CKPT_PATH") or "weights/motion_extractor_large.pt")
    g.add_argument("--dino_ckpt_path", type=str,
                   default=os.environ.get("DINO_CKPT_PATH") or "weights/dinov2_vitl14_reg.pth")

    g = p.add_argument_group("Encoder config (consistent with training checkpoint)")
    g.add_argument("--temporal_motion_embs_per_latent", type=int, default=4)
    g.add_argument("--temporal_fold_num_layers", type=int, default=2)
    g.add_argument("--temporal_fold_num_heads", type=int, default=12)
    g.add_argument("--temporal_fold_attn_window", type=int, default=0,
                   help="fold self-attn local causal window. 0=only attend within the same chunk; "
                        "N=same chunk + past N history tokens.")
    g.add_argument("--temporal_fold_attention_mode", type=str, default="causal",
                   choices=["causal", "bidirectional"],
                   help="fold chunk attention direction; automatically adopted when the training checkpoint contains this config.")
    g.add_argument("--temporal_disable_folding", action="store_true", default=True,
                   help="bypass temporal fold self-attn/FFN; automatically adopted when the training checkpoint contains this config.")
    g.add_argument("--temporal_enable_folding", dest="temporal_disable_folding", action="store_false",
                   help="enable folding when the training checkpoint does not specify it.")
    g.add_argument("--temporal_motion_attn_layer_indices", type=str,
                   default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29",
                   help="Wan block indices into which motion-attn is injected; defaults to all 30 layers of Wan2.2-TI2V-5B.")
    g.add_argument("--temporal_motion_attn_mode", type=str,
                   default="framewise", choices=["framewise", "full"],
                   help="motion cross-attn temporal visibility; automatically adopted when the training checkpoint contains this config.")
    g.add_argument("--temporal_motion_attn_alpha_init", type=float, default=1.0)
    g.add_argument("--temporal_motion_attn_mask_bias", action="store_true",
                   help="add a query-side bias from the first-frame object mask "
                        "onto the DiT temporal motion cross-attn logits.")
    g.add_argument("--temporal_motion_attn_mask_bias_scale", type=float, default=1.0,
                   help="positive mask bias strength for the correct object motion token.")
    g.add_argument("--temporal_motion_attn_neg_bias_scale",
                   "--temporal_motion_attn_mask_neg_bias_scale",
                   dest="temporal_motion_attn_neg_bias_scale",
                   type=float, default=0.0,
                   help="negative bias strength applied by in-mask queries toward other objects' motion tokens. "
                        "0=disabled.")
    g.add_argument("--temporal_motion_attn_mask_bias_frames", type=str,
                   default="first", choices=["first", "all", "dynamic"],
                   help="frames on which the mask routing bias acts. first=only the first latent frame; "
                        "all=all latent frames reuse the first-frame mask; "
                        "dynamic=use the per-frame masks produced by propagation.")
    g.add_argument("--temporal_motion_attn_mask_propagate", action="store_true",
                   help="at inference, use the DiT text-only hidden state to propagate the first-frame mask "
                        "to all latent frames, then use it for motion-attn routing bias.")
    g.add_argument("--temporal_motion_attn_mask_propagate_steps", type=int,
                   default=8,
                   help="number of initial sampling steps during which the dynamic mask is updated online; frozen at the last update afterwards.")
    g.add_argument("--temporal_motion_attn_mask_propagate_layers", type=str,
                   default="4,8,12",
                   help="DiT block indices used for mask propagation, comma-separated; empty string means the last layer.")
    g.add_argument("--temporal_motion_attn_mask_propagate_feat_dim", type=int,
                   default=256,
                   help="how many dimensions to uniformly sample from the DiT hidden dim for affinity during propagation.")
    g.add_argument("--temporal_motion_attn_mask_propagate_topk", type=int,
                   default=15)
    g.add_argument("--temporal_motion_attn_mask_propagate_temp", type=float,
                   default=0.05)
    g.add_argument("--temporal_motion_attn_mask_propagate_prev", type=int,
                   default=2,
                   help="for frames other than the first, how many recently propagated frames each frame additionally references during propagation.")
    g.add_argument("--temporal_motion_attn_mask_propagate_save",
                   action="store_true",
                   help="save the propagated dynamic routing mask video next to the output directory.")
    g.add_argument("--temporal_motion_attn_mask_propagate_save_every", type=int,
                   default=1,
                   help="when saving is enabled, save every this many propagation steps; 1=save every time.")
    g.add_argument("--temporal_use_mask_spatial", action="store_true", default=True)
    g.add_argument("--no_temporal_use_mask_spatial", dest="temporal_use_mask_spatial", action="store_false",
                   help="disable spatial masks unless required by the training checkpoint.")
    g.add_argument("--temporal_mask_spatial_layers", type=int, default=2)
    g.add_argument("--temporal_mask_spatial_stage", type=str, default="pre_interleave",
                   choices=["post", "pre_interleave"])
    g.add_argument("--temporal_mask_spatial_mode", type=str, default="full",
                   choices=["full", "weak"])
    g.add_argument("--temporal_mask_spatial_weak_scale", type=float, default=0.02)
    g.add_argument("--dismo_max_frames", type=int, default=49)
    g.add_argument("--dismo_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    g = p.add_argument_group("Ref-mask masking (optional, consistent with training's motion_ref_mask)")
    g.add_argument("--motion_ref_mask_bg", type=float, default=0.0,
                   help="background retention ratio; 0=fully masked out, 1=not masked. Enabled when ref_mask_paths "
                        "is given, otherwise not applied.")
    g.add_argument("--motion_ref_mask_fill", type=float, default=0.0,
                   help="background fill value (in [-1, 1] space).")

    g = p.add_argument_group("Single-command input (mutually exclusive with --inference_csv)")
    g.add_argument("--ref_videos", type=str, default=None,
                   help="',' separated N single-object ref videos; 1 clip=single object, N clips=multi-object")
    g.add_argument("--ref_videos_clip_starts", type=str, default=None,
                   help="',' separated N clip_start values, default all 0")
    g.add_argument("--ref_mask_paths", type=str, default=None,
                   help="',' separated N ref-masks (optional, mask out ref background)")
    g.add_argument("--target_image", type=str, default=None,
                   help="can be an image (.jpg/.png/...) or a video; for videos use "
                        "--target_image_frame_idx to extract that frame.")
    g.add_argument("--target_image_frame_idx", type=int, default=0,
                   help="which frame to take when target_image is a video (default 0). Ignored for images.")
    g.add_argument("--target_mask_paths", type=str, default=None,
                   help="'+' separated N tgt masks. Omittable for single object, required for multi-object. "
                        "For videos, aligned with target_image_frame_idx (i.e. take that frame's mask).")
    g.add_argument(
        "--target_mask_all_ones",
        action="store_true",
        help="inference ablation: keep the target-mask entries and mask-conditioning structure, "
             "but replace each object's target mask with an all-white mask before the model forward, "
             "thereby removing the target's spatial anchor information.",
    )
    g.add_argument("--persons", type=str, default="",
                   help="'|' separated, count consistent with ref_videos")
    g.add_argument("--actions", type=str, default="",
                   help="'|' separated, count consistent with ref_videos, order aligned")
    g.add_argument("--output_path", type=str, default=None)

    g = p.add_argument_group("Batch CSV")
    g.add_argument("--inference_csv", type=str, default=None,
                   help="batch CSV; defaults to data/RART_test_data/inference.csv when no single-command input is given.")
    g.add_argument("--output_dir", type=str,
                   default=os.environ.get("OUTPUT_DIR") or "outputs/inference")
    g.add_argument("--csv_start", type=int, default=os.environ.get("CSV_START") or 0)
    g.add_argument("--csv_limit", type=int, default=os.environ.get("CSV_LIMIT") or -1)

    g = p.add_argument_group("Sampling parameters")
    g.add_argument("--num_frames", type=int, default=49,
                   help="number of generated frames, usually equal to training max_frames (=49). "
                        "ref/mask are also sliced accordingly. Requires (N-1) %% vae_temporal_stride == 0; "
                        "Wan2.2-TI2V-5B (stride=4) valid values: 1, 5, ..., 49, ...")
    g.add_argument("--max_area", type=int, default=480 * 832,
                   help="generated video area (h*w)")
    g.add_argument("--resolution", type=str, default="480,736",
                   help="fixed generation resolution 'H,W'. Takes priority over --max_area when set, "
                        "used to exactly align with the training resolution. Pass an empty string to use --max_area.")
    g.add_argument("--sampling_steps", type=int, default=50)
    g.add_argument("--guide_scale", type=float, default=3.0)
    g.add_argument("--motion_guide_scale", type=float, default=5.0,
                   help="extra motion guidance strength. 1=keep the old inference path; when not 1, "
                        "use p_u0 + guide*(p_c0-p_u0) + "
                        "motion_guide*(p_cm-p_c0), one extra DiT forward per step.")
    g.add_argument("--shift", type=float, default=5.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--fps", type=int, default=16)
    g.add_argument("--use_caption", action="store_true", default=True)
    g.add_argument("--no_use_caption", dest="use_caption", action="store_false")
    g.add_argument("--neg_prompt", type=str, default=None)

    g = p.add_argument_group("Output")
    g.add_argument("--concat_ref", action="store_true",
                   help="output the ref videos concatenated (N ref clips on the left / generation on the right).")
    g.add_argument("--vae_offload", action="store_true", default=True)
    g.add_argument("--no_vae_offload", dest="vae_offload", action="store_false")
    g.add_argument("--t5_cpu_offload", action="store_true",
                   default=(os.environ.get("T5_CPU_OFFLOAD") or "1") == "1",
                   help="Precompute text embeddings before loading Wan, then move T5 to CPU.")
    g.add_argument("--no_t5_cpu_offload", dest="t5_cpu_offload", action="store_false")

    args = p.parse_args()
    # Use the bundled examples only when no batch or single-command input was selected.
    if (args.inference_csv is None and args.target_image is None
            and args.ref_videos is None and args.output_path is None):
        args.inference_csv = os.environ.get("INFERENCE_CSV") or "data/RART_test_data/inference.csv"
    args.dismo_dtype_torch = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dismo_dtype]

    if args.inference_csv is None:
        if args.target_image is None:
            p.error("Either --inference_csv or --target_image must be given.")
        if args.ref_videos is None:
            p.error("--ref_videos is required (1 path = single-obj, "
                    "',' separated paths = multi-obj).")
        if args.output_path is None:
            p.error("--output_path is required when --inference_csv is not set.")
    if args.temporal_motion_attn_mask_bias_scale < 0:
        p.error("--temporal_motion_attn_mask_bias_scale must be >= 0")
    if args.temporal_motion_attn_neg_bias_scale < 0:
        p.error("--temporal_motion_attn_neg_bias_scale must be >= 0")
    if args.temporal_motion_attn_mask_propagate:
        args.temporal_motion_attn_mask_bias = True
        if args.temporal_motion_attn_mask_propagate_steps < 0:
            p.error("--temporal_motion_attn_mask_propagate_steps must be >= 0")
        if args.temporal_motion_attn_mask_propagate_feat_dim <= 0:
            p.error("--temporal_motion_attn_mask_propagate_feat_dim must be > 0")
        if args.temporal_motion_attn_mask_propagate_topk <= 0:
            p.error("--temporal_motion_attn_mask_propagate_topk must be > 0")
        if args.temporal_motion_attn_mask_propagate_temp <= 0:
            p.error("--temporal_motion_attn_mask_propagate_temp must be > 0")
        if args.temporal_motion_attn_mask_propagate_prev < 0:
            p.error("--temporal_motion_attn_mask_propagate_prev must be >= 0")
        if args.temporal_motion_attn_mask_propagate_save_every <= 0:
            p.error("--temporal_motion_attn_mask_propagate_save_every must be > 0")
    if args.motion_guide_scale < 0:
        p.error("--motion_guide_scale must be >= 0")
    args.resolution_hw = None
    if args.resolution:
        try:
            parts = [int(x.strip()) for x in str(args.resolution).split(',')]
        except Exception:
            parts = []
        if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0:
            p.error("--resolution must be formatted as positive 'H,W'")
        args.resolution_hw = (parts[0], parts[1])
    return args


# ============================================================================
# Helpers
# ============================================================================

def _is_nan(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _split_pipe(s) -> List[str]:
    if _is_nan(s):
        return []
    return [x.strip() for x in str(s).split('|') if x.strip()]


def _split_plus(s) -> List[str]:
    if _is_nan(s):
        return []
    return [x.strip() for x in str(s).split('+') if x.strip()]


def _split_comma(s) -> List[str]:
    if _is_nan(s):
        return []
    return [x.strip() for x in str(s).split(',') if x.strip()]


def _parse_int_list(s) -> List[int]:
    if _is_nan(s):
        return []
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]


def _parse_clip_starts(value):
    """Read optional clip indices, keeping the existing zero fallback for bad entries."""
    starts = []
    for item in _split_comma(value):
        try:
            starts.append(int(item))
        except Exception:
            starts.append(0)
    return starts


def format_persons_actions(persons: str, actions: str) -> str:
    """Consistent with the training-side format_persons_actions.
    For each of N: join persons[i]+' '+actions[i] with ','.
    """
    p_list = _split_pipe(persons)
    a_list = _split_pipe(actions)
    n = min(len(p_list), len(a_list))
    if n == 0:
        if p_list:
            return ', '.join(p_list)
        if a_list:
            return ', '.join(a_list)
        return ''
    pairs = [f"{p} {a}" for p, a in zip(p_list[:n], a_list[:n])]
    return ', '.join(pairs)


def apply_motion_ref_mask_to_rgb(ref_video, ref_mask, bg_ratio,
                                 fill_value: float = 0.0):
    """Consistent with the training side."""
    T_v, H_v, W_v = ref_video.shape[1:]
    mask = ref_mask.float().to(ref_video.device)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[1] != T_v:
        t_idx = torch.linspace(
            0, mask.shape[1] - 1, T_v, device=mask.device,
        ).round().long().clamp(0, mask.shape[1] - 1)
        mask = mask[:, t_idx]
    if mask.shape[2] != H_v or mask.shape[3] != W_v:
        mask = F.interpolate(
            mask.unsqueeze(0), size=(T_v, H_v, W_v), mode='nearest',
        ).squeeze(0)
    alpha = bg_ratio + (1.0 - bg_ratio) * mask
    alpha = alpha.to(dtype=ref_video.dtype)
    return ref_video * alpha + fill_value * (1.0 - alpha)


def to_motion_rgb(ref_video):
    return ((ref_video.clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()


def build_motion_attn_mask_first(mask_lat_per_obj, device):
    """Pack per-object first-frame latent masks to [1, N, H_lat, W_lat]."""
    if not mask_lat_per_obj:
        return None
    first_shape = None
    for m in mask_lat_per_obj:
        if m is None:
            continue
        first_shape = tuple(m.shape[-2:])
        break
    if first_shape is None:
        return None

    N = len(mask_lat_per_obj)
    H, W = first_shape
    out = torch.zeros(1, N, H, W, device=device, dtype=torch.float32)
    for obj_idx, m in enumerate(mask_lat_per_obj):
        if m is None:
            continue
        m = m.to(device=device, dtype=torch.float32)
        if m.dim() == 4:
            m = m[0]
        if m.dim() == 3:
            m = m[0]
        out[0, obj_idx] = m.clamp(0.0, 1.0)
    return out


def propagate_motion_attn_masks(
    first_masks: torch.Tensor,
    features: torch.Tensor,
    T: int,
    H: int,
    W: int,
    feat_dim: int = 256,
    topk: int = 15,
    temperature: float = 0.05,
    prev_frames: int = 2,
) -> torch.Tensor:
    """Propagate first-frame object masks with DiT hidden-state affinity.

    Args:
        first_masks: [B, N, H0, W0] in [0, 1].
        features:    [B, >=T*H*W, C] hidden states from text-only Wan branch.
    Returns:
        [B, N, T, H, W] soft masks.
    """
    if first_masks is None or features is None:
        return first_masks
    B, N = first_masks.shape[:2]
    L = T * H * W
    if features.shape[0] != B or features.shape[1] < L:
        return first_masks

    masks0 = first_masks.to(device=features.device, dtype=torch.float32)
    if masks0.dim() != 4:
        return first_masks
    masks0 = F.interpolate(
        masks0.reshape(B * N, 1, masks0.shape[-2], masks0.shape[-1]),
        size=(H, W),
        mode="area",
    ).reshape(B, N, H, W).clamp(0.0, 1.0)

    C = features.shape[-1]
    use_dim = min(int(feat_dim), C)
    if use_dim < C:
        feat_idx = torch.linspace(
            0, C - 1, use_dim, device=features.device,
        ).round().long()
        features = features.index_select(-1, feat_idx)

    out_all = []
    with torch.amp.autocast(features.device.type, enabled=False):
        feat = features[:, :L].float().reshape(B, T, H * W, use_dim)
        feat = F.normalize(feat, dim=-1, p=2)
        k_top = max(1, int(topk))
        temp = float(temperature)
        prev = max(0, int(prev_frames))
        for b in range(B):
            obj_masks = []
            fb = feat[b]
            for n in range(N):
                cur = [masks0[b, n].reshape(H * W)]
                for ti in range(1, T):
                    src_ids = [0]
                    if prev > 0:
                        start = max(1, ti - prev)
                        src_ids.extend(range(start, ti))
                    src_feats = torch.cat([fb[j] for j in src_ids], dim=0)
                    src_masks = torch.cat([cur[j] for j in src_ids], dim=0)

                    # Keep foreground correspondences dominant while leaving a
                    # tiny background signal to avoid all-zero affinities.
                    src_weight = 0.01 + 0.99 * src_masks.clamp(0.0, 1.0)
                    src_feats = F.normalize(
                        src_feats * src_weight.unsqueeze(-1),
                        dim=-1, p=2,
                    )
                    logits = fb[ti].matmul(src_feats.transpose(0, 1)) / temp
                    k = min(k_top, logits.shape[-1])
                    vals, idx = torch.topk(logits, k=k, dim=-1)
                    probs = torch.softmax(vals, dim=-1)
                    picked = src_masks.index_select(0, idx.reshape(-1))
                    picked = picked.reshape(H * W, k)
                    cur.append((probs * picked).sum(dim=-1).clamp(0.0, 1.0))
                obj_masks.append(torch.stack(cur, dim=0).reshape(T, H, W))
            out_all.append(torch.stack(obj_masks, dim=0))
    return torch.stack(out_all, dim=0).to(
        device=first_masks.device, dtype=torch.float32,
    )


def save_dynamic_routing_masks(
    masks: torch.Tensor,
    out_dir: str,
    tag: str,
    step_idx: int,
    timestep: torch.Tensor,
    fps: int,
    resolution: Tuple[int, int],
):
    """Save [B,N,T,H,W] masks as debug videos."""
    if masks is None:
        return
    try:
        m = masks.detach().float().clamp(0.0, 1.0)
        if m.dim() != 5 or m.shape[0] < 1:
            return
        H_out, W_out = resolution
        B, N, T, Hm, Wm = m.shape
        m0 = m[0]  # [N, T, H, W]; inference is per task / batch=1
        m0 = F.interpolate(
            m0.reshape(N * T, 1, Hm, Wm),
            size=(H_out, W_out),
            mode="nearest",
        ).reshape(N, T, H_out, W_out)
        rgb = (m0[:, None].expand(N, 3, T, H_out, W_out) * 2.0 - 1.0)

        t_val = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        save_dir = os.path.join(out_dir, "dynamic_masks", tag)
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"step_{step_idx:03d}_t{t_val}"
        cache_video(
            rgb, save_file=os.path.join(save_dir, f"{prefix}_grid.mp4"),
            fps=fps, nrow=max(1, min(N, 4)),
            normalize=True, value_range=(-1, 1),
        )
        for obj_idx in range(N):
            cache_video(
                rgb[obj_idx:obj_idx + 1],
                save_file=os.path.join(save_dir, f"{prefix}_obj{obj_idx}.mp4"),
                fps=fps, nrow=1, normalize=True, value_range=(-1, 1),
            )
    except Exception as e:
        print(f"[warn] [{tag}] save dynamic routing mask failed: {e}")


@contextmanager
def vae_on_gpu(vae, device, enabled=True):
    if not enabled:
        yield
        return
    vae.model.to(device)
    try:
        yield
    finally:
        vae.model.cpu()
        torch.cuda.empty_cache()


# ============================================================================
# IO
# ============================================================================

def load_video_decord(
    video_path: str, start_frame: int = 0, num_frames: int = 49,
    resolution: Tuple[int, int] = (480, 832),
) -> torch.Tensor:
    """[3, T, H, W] in [-1, 1]."""
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"empty video: {video_path}")

    if total <= num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        end = min(start_frame + num_frames, total)
        indices = list(range(start_frame, end))
        if len(indices) < num_frames:
            indices = indices + [indices[-1]] * (num_frames - len(indices))

    frames = vr.get_batch(indices)
    frames = frames.permute(3, 0, 1, 2).float()    # [C, T, H, W]
    frames = frames.permute(1, 0, 2, 3)             # [T, C, H, W]
    frames = F.interpolate(
        frames, size=resolution, mode='bilinear', align_corners=False,
    )
    frames = frames.permute(1, 0, 2, 3)             # [C, T, H, W]
    return frames / 127.5 - 1.0


def load_mask_decord(
    mask_path: str, start_frame: int = 0, num_frames: int = 49,
    resolution: Tuple[int, int] = (480, 832),
) -> torch.Tensor:
    """[1, T, H, W] in [0, 1]."""
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(mask_path)
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"empty mask video: {mask_path}")

    # start_frame out-of-bounds guard: fallback to 0 + warn (mask video may differ in length from the original video)
    if start_frame >= total:
        print(
            f"[warn] mask {os.path.basename(mask_path)}: start_frame={start_frame} "
            f">= total {total}; fallback to 0"
        )
        start_frame = 0

    if total <= num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        end = min(start_frame + num_frames, total)
        indices = list(range(start_frame, end))
        if len(indices) < num_frames:
            indices = indices + [indices[-1]] * (num_frames - len(indices))

    frames = vr.get_batch(indices)
    frames = frames[..., 0:1].permute(3, 0, 1, 2).float()
    frames = F.interpolate(
        frames.permute(1, 0, 2, 3),
        size=resolution, mode='nearest',
    ).permute(1, 0, 2, 3)
    return frames / 255.0


# static image extensions (everything else is treated as video)
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def _is_image_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def load_target_image(path: str, frame_idx: int = 0) -> torch.Tensor:
    """Load target image -> [3, H, W] in [-1, 1].

    - image extension (.jpg/.png/...): read directly with PIL, frame_idx ignored.
    - other extensions: treated as video, decord takes frame frame_idx (auto fallback to 0 if out of bounds).
    """
    if _is_image_path(path):
        img = Image.open(path).convert('RGB')
        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t.sub_(0.5).div_(0.5)

    # video path
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(path)
    total = len(vr)
    if total <= 0:
        raise RuntimeError(f"empty video: {path}")
    fi = int(frame_idx)
    if fi < 0 or fi >= total:
        print(
            f"[warn] target_image_frame_idx={fi} out of range (video has {total} frames); "
            f"fallback to 0"
        )
        fi = 0
    frame = vr[fi]                                  # [H, W, C] uint8
    frame = frame.permute(2, 0, 1).float() / 127.5 - 1.0  # [3, H, W] in [-1, 1]
    return frame.contiguous()


# ============================================================================
# Task
# ============================================================================

class InferTask:
    """A single inference task (unified multi-ref paradigm).

    - ref_videos: List[str] (1 = single object, N = multi-object)
    - ref_mask_paths: List[str] (optional; applied if given, count must == len(ref_videos))
    - target_mask_paths: List[str] (omittable for single object, required for multi-object with count == len(ref_videos))
    - persons / actions: '|' separated, count == len(ref_videos)
    """
    def __init__(
        self,
        ref_videos: List[str],
        target_image: str,
        persons: str = "",
        actions: str = "",
        target_mask_paths: Optional[List[str]] = None,
        ref_mask_paths: Optional[List[str]] = None,
        ref_videos_clip_starts: Optional[List[int]] = None,
        target_image_frame_idx: int = 0,
        output_path: str = "out.mp4",
        tag: str = "",
    ):
        self.ref_videos = ref_videos
        self.target_image = target_image
        self.target_image_frame_idx = int(target_image_frame_idx)
        self.persons = persons
        self.actions = actions
        self.target_mask_paths = target_mask_paths or []
        self.ref_mask_paths = ref_mask_paths or []
        self.ref_videos_clip_starts = ref_videos_clip_starts or [0] * len(ref_videos)
        self.output_path = output_path
        self.tag = tag or os.path.splitext(os.path.basename(output_path))[0]

    @property
    def n_obj(self) -> int:
        return len(self.ref_videos)

    @property
    def is_multi(self) -> bool:
        return self.n_obj >= 2

    def validate(self):
        N = self.n_obj
        if N == 0:
            raise ValueError(f"[{self.tag}] ref_videos is empty")

        # ref_videos_clip_starts
        if len(self.ref_videos_clip_starts) != N:
            self.ref_videos_clip_starts = (
                self.ref_videos_clip_starts +
                [0] * (N - len(self.ref_videos_clip_starts))
            )[:N]

        # multi-object: target_mask_paths must be given and count-aligned
        if self.is_multi:
            if len(self.target_mask_paths) != N:
                raise ValueError(
                    f"[{self.tag}] multi-object (N={N}): target_mask_paths must provide N entries "
                    f"(got {len(self.target_mask_paths)})"
                )
        else:
            # single object: target_mask_paths is omittable, but if given must be exactly 1
            if len(self.target_mask_paths) > 1:
                raise ValueError(
                    f"[{self.tag}] single-object: target_mask_paths must be 0 or 1 entry "
                    f"(got {len(self.target_mask_paths)})"
                )

        # ref_mask_paths, if given, must == N
        if self.ref_mask_paths and len(self.ref_mask_paths) != N:
            raise ValueError(
                f"[{self.tag}] ref_mask_paths provided {len(self.ref_mask_paths)} entries, "
                f"must == N={N} (or be omitted)"
            )

        # persons / actions warn (mismatched counts still run, format_persons_actions takes min)
        n_p = len(_split_pipe(self.persons))
        n_a = len(_split_pipe(self.actions))
        if n_p > 0 and n_p != N:
            print(f"[warn] [{self.tag}] persons count ({n_p}) != N ({N})")
        if n_a > 0 and n_a != N:
            print(f"[warn] [{self.tag}] actions count ({n_a}) != N ({N})")

    def build_caption(self) -> str:
        return format_persons_actions(self.persons, self.actions)


def _csv_row_to_task(row: dict, output_dir: str) -> InferTask:
    def _get(k, default=''):
        v = row.get(k, default)
        if _is_nan(v):
            return default
        return str(v).strip() if isinstance(v, str) else v

    ref_videos = _split_comma(_get('ref_videos'))
    if len(ref_videos) == 0:
        raise ValueError("CSV row missing 'ref_videos'")

    target_image = _get('target_image')
    if not target_image:
        raise ValueError("CSV row missing 'target_image'")

    target_mask_paths = _split_plus(_get('target_mask_paths'))
    ref_mask_paths = _split_comma(_get('ref_mask_paths'))

    starts = _parse_clip_starts(_get('ref_videos_clip_starts'))

    output_name = _get('output_name')
    if not output_name:
        output_name = f"sample_{abs(hash(target_image)) & 0xffff:04x}.mp4"
    output_path = os.path.join(output_dir, output_name)

    # target_image_frame_idx: CSV column, default 0
    tif = 0
    try:
        _v = _get('target_image_frame_idx', 0)
        tif = int(_v) if _v != '' else 0
    except Exception:
        tif = 0

    return InferTask(
        ref_videos=ref_videos,
        ref_videos_clip_starts=starts,
        ref_mask_paths=ref_mask_paths,
        target_image=target_image,
        target_image_frame_idx=tif,
        target_mask_paths=target_mask_paths,
        persons=_get('persons'),
        actions=_get('actions'),
        output_path=output_path,
        tag=output_name,
    )


def _args_to_single_task(args) -> InferTask:
    ref_videos = _split_comma(args.ref_videos)
    starts = (
        _parse_clip_starts(args.ref_videos_clip_starts)
        if args.ref_videos_clip_starts else []
    )
    return InferTask(
        ref_videos=ref_videos,
        ref_videos_clip_starts=starts,
        ref_mask_paths=_split_comma(args.ref_mask_paths or ''),
        target_image=args.target_image,
        target_image_frame_idx=args.target_image_frame_idx,
        target_mask_paths=_split_plus(args.target_mask_paths or ''),
        persons=args.persons,
        actions=args.actions,
        output_path=args.output_path,
    )


# ============================================================================
# Model loading
# ============================================================================

def _create_text_encoder(args, config, device):
    """Construct T5; the caller controls freezing, residency and lifetime."""
    return T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(args.checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(args.checkpoint_dir, config.t5_tokenizer),
    )


class _CachedT5Encoder:
    """Reuse CPU-cached embeddings without running the offloaded T5 again."""

    def __init__(self, encoder, embeddings):
        self.encoder = encoder
        self.embeddings = embeddings
        self.device = encoder.device

    def __call__(self, texts, device):
        outputs = []
        for text in texts:
            if text not in self.embeddings:
                raise KeyError(f"T5 embedding was not precomputed for prompt: {text!r}")
            outputs.append(self.embeddings[text].to(device=device))
        return outputs


def _precompute_t5_cpu_cache(args, config, device, tasks):
    neg = args.neg_prompt or config.sample_neg_prompt
    prompts = ["", neg]
    if args.use_caption:
        prompts.extend(task.build_caption() for task in tasks)
    prompts = list(dict.fromkeys(prompts))
    print(f"Precomputing {len(prompts)} T5 text embeddings on {device} before Wan loading...")
    encoder = _create_text_encoder(args, config, device)
    encoder.model.requires_grad_(False)
    encoder.model.eval()
    embeddings = {}
    try:
        with torch.no_grad():
            for prompt in prompts:
                embeddings[prompt] = encoder([prompt], device)[0].detach().cpu()
    finally:
        encoder.model.cpu()
        encoder.device = torch.device("cpu")
        torch.cuda.empty_cache()
    print("T5 moved to CPU; sampling will reuse the precomputed text embeddings.")
    cached_encoder = _CachedT5Encoder(encoder, embeddings) if args.use_caption else None
    return (
        cached_encoder,
        embeddings[""].to(device=device),
        embeddings[neg].to(device=device),
    )


def _apply_checkpoint_config(args, saved_config):
    """Restore trained architecture while retaining explicit mask-bias options."""
    if "fold_attention_mode" in saved_config:
        args.temporal_fold_attention_mode = str(
            saved_config["fold_attention_mode"]
        )
    if "disable_folding" in saved_config:
        args.temporal_disable_folding = bool(saved_config["disable_folding"])
    if "motion_attn_mode" in saved_config:
        args.temporal_motion_attn_mode = str(
            saved_config["motion_attn_mode"]
        )
    build_mask_spatial_for_init = bool(
        saved_config.get("build_mask_spatial_for_init", False)
    )
    saved_window = int(saved_config.get("fold_attn_window", 0) or 0)
    if args.temporal_fold_attn_window <= 0 and saved_window > 0:
        args.temporal_fold_attn_window = saved_window
        print(f"  using fold_attn_window from ckpt: {saved_window}")
    if "fold_attention_mode" in saved_config or "disable_folding" in saved_config:
        print(
            "  using fold config from ckpt: "
            f"mode={args.temporal_fold_attention_mode}, "
            f"disable_folding={args.temporal_disable_folding}"
        )
    print(
        "  using motion-attn temporal mode: "
        f"{args.temporal_motion_attn_mode}"
    )
    if bool(saved_config.get("use_mask_spatial", False)):
        args.temporal_use_mask_spatial = True
        args.temporal_mask_spatial_layers = int(
            saved_config.get(
                "mask_spatial_layers", args.temporal_mask_spatial_layers,
            )
        )
        args.temporal_mask_spatial_stage = str(
            saved_config.get(
                "mask_spatial_stage", args.temporal_mask_spatial_stage,
            )
        )
        args.temporal_mask_spatial_mode = str(
            saved_config.get(
                "mask_spatial_mode", args.temporal_mask_spatial_mode,
            )
        )
        args.temporal_mask_spatial_weak_scale = float(
            saved_config.get(
                "mask_spatial_weak_scale",
                args.temporal_mask_spatial_weak_scale,
            )
        )
        print(
            "  using mask spatial config from ckpt: "
            f"stage={args.temporal_mask_spatial_stage}, "
            f"layers={args.temporal_mask_spatial_layers}, "
            f"mode={args.temporal_mask_spatial_mode}"
        )

    if (
        bool(saved_config.get("motion_attn_mask_bias", False))
        and not args.temporal_motion_attn_mask_bias
    ):
        args.temporal_motion_attn_mask_bias = True
        args.temporal_motion_attn_mask_bias_scale = float(
            saved_config.get(
                "motion_attn_mask_bias_scale",
                args.temporal_motion_attn_mask_bias_scale,
            )
        )
        args.temporal_motion_attn_neg_bias_scale = float(
            saved_config.get(
                "motion_attn_neg_bias_scale",
                args.temporal_motion_attn_neg_bias_scale,
            )
        )
        args.temporal_motion_attn_mask_bias_frames = str(
            saved_config.get(
                "motion_attn_mask_bias_frames",
                args.temporal_motion_attn_mask_bias_frames,
            )
        )
        print(
            "  using motion-attn mask bias config from ckpt: "
            f"pos={args.temporal_motion_attn_mask_bias_scale}, "
            f"neg={args.temporal_motion_attn_neg_bias_scale}, "
            f"frames={args.temporal_motion_attn_mask_bias_frames}"
        )
    return build_mask_spatial_for_init


def load_model_and_encoder(args, config, device, tasks=None):
    tmp_path = os.path.join(args.motion_ckpt, "temporal_motion.pt")
    if not os.path.isfile(tmp_path):
        raise FileNotFoundError(
            f"motion ckpt missing: {tmp_path}. Expected: "
            f"<motion_ckpt>/temporal_motion.pt"
        )
    print(f"Loading motion ckpt: {tmp_path}")
    _ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
    _saved_cfg = _ckpt.get("config", {}) or {}
    build_mask_spatial_for_init = _apply_checkpoint_config(args, _saved_cfg)

    # dismo feature source / proj-adapter config is read from the ckpt (saved at training time),
    # otherwise the default head128(128d) would mismatch the input_proj size trained with query1024(1024d).
    _dismo_feature_source = str(
        _saved_cfg.get("dismo_feature_source", "head128")
    )
    _dismo_unfreeze_query_tokens = bool(
        _saved_cfg.get("dismo_unfreeze_query_tokens", False)
    )
    _dismo_use_proj_adapter = bool(
        _saved_cfg.get("dismo_use_proj_adapter", False)
    )
    _dismo_proj_adapter_hidden_dim = int(
        _saved_cfg.get("dismo_proj_adapter_hidden_dim", 64)
    )
    _dismo_proj_adapter_scale = float(
        _saved_cfg.get("dismo_proj_adapter_scale", 0.1)
    )
    _dismo_post_dino_mask_bg = float(
        _saved_cfg.get("dismo_post_dino_mask_bg", 1.0)
    )
    print(
        "  using dismo config from ckpt: "
        f"feature_source={_dismo_feature_source}, "
        f"use_proj_adapter={_dismo_use_proj_adapter}, "
        f"adapt_mode={_saved_cfg.get('dismo_adapt_mode', 'unknown')}"
    )

    cached_t5 = None
    if args.t5_cpu_offload:
        if tasks is None:
            raise ValueError("--t5_cpu_offload requires the inference task list before model loading.")
        cached_t5 = _precompute_t5_cpu_cache(args, config, device, tasks)

    print("Loading Wan2.2 model...")
    model = WanModel.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()
    model.requires_grad_(False)

    if args.temporal_motion_attn_layer_indices:
        t_layer_indices = [
            int(x) for x in args.temporal_motion_attn_layer_indices.split(",")
            if x.strip()
        ]
    else:
        t_layer_indices = None

    print("Building Temporal motion encoder + cross-attn adapters...")
    from wan_motion_attn import (
        TemporalDisMoMotionEncoder,
        inject_temporal_motion_cross_attention,
        load_temporal_motion_attn_state_dict,
    )

    temporal_encoder = TemporalDisMoMotionEncoder(
        out_dim=config.dim,
        motion_embs_per_latent=args.temporal_motion_embs_per_latent,
        num_layers=args.temporal_fold_num_layers,
        num_heads=args.temporal_fold_num_heads,
        dismo_ckpt_path=args.dismo_ckpt_path,
        dismo_repo_path=args.dismo_repo_path,
        dino_ckpt_path=args.dino_ckpt_path,
        max_frames=args.dismo_max_frames,
        dtype=args.dismo_dtype_torch,
        sliding_lookahead=args.temporal_motion_embs_per_latent,
        fold_attn_window=args.temporal_fold_attn_window,
        fold_attention_mode=args.temporal_fold_attention_mode,
        disable_folding=args.temporal_disable_folding,
        use_mask_spatial=args.temporal_use_mask_spatial,
        build_mask_spatial_when_disabled=build_mask_spatial_for_init,
        mask_spatial_layers=args.temporal_mask_spatial_layers,
        mask_spatial_stage=args.temporal_mask_spatial_stage,
        mask_spatial_mode=args.temporal_mask_spatial_mode,
        mask_spatial_weak_scale=args.temporal_mask_spatial_weak_scale,
        dismo_feature_source=_dismo_feature_source,
        dismo_unfreeze_query_tokens=_dismo_unfreeze_query_tokens,
        dismo_use_proj_adapter=_dismo_use_proj_adapter,
        dismo_proj_adapter_hidden_dim=_dismo_proj_adapter_hidden_dim,
        dismo_proj_adapter_scale=_dismo_proj_adapter_scale,
        dismo_post_dino_mask_bg=_dismo_post_dino_mask_bg,
    ).to(device)
    temporal_encoder.eval()
    temporal_encoder.requires_grad_(False)

    temporal_motion_attn_adapters, temporal_motion_attn_data_provider = (
        inject_temporal_motion_cross_attention(
            model,
            dim=config.dim,
            num_heads=config.num_heads,
            qk_norm=config.qk_norm,
            eps=config.eps,
            alpha_init=args.temporal_motion_attn_alpha_init,
            layer_indices=t_layer_indices,
            temporal_mode=args.temporal_motion_attn_mode,
        )
    )
    object.__setattr__(
        model, "_temporal_motion_attn_data_provider",
        temporal_motion_attn_data_provider,
    )
    # the injected adapter is a newly created nn.Module, on cpu by default;
    # must call model.to(device) again to move the newly added adapter to GPU too.
    # (on the training side accelerator.prepare(model) handles this automatically.)
    model = model.to(device)

    for adapter in temporal_motion_attn_adapters:
        if adapter is None:
            continue
        adapter.requires_grad_(False)

    _enc_sd = _ckpt.get("encoder_state_dict", {})
    _attn_sd = _ckpt.get("attn_state_dict", {})

    missing, unexpected = temporal_encoder.load_state_dict(_enc_sd, strict=False)
    missing_real = [k for k in missing if not k.startswith("dismo.")]
    unexpected_real = [k for k in unexpected if not k.startswith("dismo.")]
    print(
        f"  temporal encoder: loaded {len(_enc_sd)} keys, "
        f"missing={len(missing_real)}, unexpected={len(unexpected_real)}"
    )
    if missing_real:
        print(f"    missing (first 5): {missing_real[:5]}")
    if unexpected_real:
        print(f"    unexpected (first 5): {unexpected_real[:5]}")

    m, miss = load_temporal_motion_attn_state_dict(
        model, _attn_sd, strict=False,
    )
    print(f"  temporal motion-attn: matched {m}, missing {miss}")

    # force PE to use sinusoidal (overrides any learned values that may be stored in the ckpt)
    from wan_motion_attn.temporal_encoder import _sinusoidal_pe
    _D = temporal_encoder.full_seq_pe.shape[-1]
    _dev = temporal_encoder.full_seq_pe.device
    _dt = temporal_encoder.full_seq_pe.dtype
    temporal_encoder.full_seq_pe.copy_(
        _sinusoidal_pe(
            temporal_encoder.full_seq_pe.shape[0], _D,
        ).to(device=_dev, dtype=_dt)
    )
    temporal_encoder.intra_chunk_pe.copy_(
        _sinusoidal_pe(temporal_encoder.K_pf, _D).to(device=_dev, dtype=_dt)
    )
    print(f"    -> PE forced to sinusoidal (full_seq_pe + intra_chunk_pe)")

    print("Loading VAE...")
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.checkpoint_dir, config.vae_checkpoint),
        device=device,
    )
    if args.vae_offload:
        vae.model.cpu()
        torch.cuda.empty_cache()

    text_encoder = None
    empty_embed = None
    neg_embed = None
    if cached_t5 is not None:
        text_encoder, empty_embed, neg_embed = cached_t5
    elif args.use_caption:
        print("Loading T5 (kept on GPU for caption encoding)...")
        text_encoder = _create_text_encoder(args, config, device)
        text_encoder.model.requires_grad_(False)
        text_encoder.model.eval()
        with torch.no_grad():
            empty_embed = text_encoder([""], device)[0]
            neg = args.neg_prompt or config.sample_neg_prompt
            neg_embed = text_encoder([neg], device)[0]
    else:
        print("Loading T5 (one-shot for empty/neg embed)...")
        _t5 = _create_text_encoder(args, config, device)
        with torch.no_grad():
            empty_embed = _t5([""], device)[0]
            neg = args.neg_prompt or config.sample_neg_prompt
            neg_embed = _t5([neg], device)[0]
        del _t5
        torch.cuda.empty_cache()

    return model, vae, text_encoder, empty_embed, neg_embed, temporal_encoder


# ============================================================================
# Single-task inference
# ============================================================================

@torch.no_grad()
def run_one_task(
    task: InferTask, args, config, device,
    model, vae, text_encoder, empty_embed, neg_embed, temporal_encoder,
):
    task.validate()
    N = task.n_obj

    # 1. tgt image -> determine (h, w)
    #    target_image can be an image or a video; for video take the target_image_frame_idx frame.
    img = load_target_image(
        task.target_image, frame_idx=task.target_image_frame_idx,
    ).to(device)
    h_in, w_in = img.shape[1], img.shape[2]
    if args.resolution_hw is not None:
        h, w = args.resolution_hw
        if h % config.vae_stride[1] != 0 or w % config.vae_stride[2] != 0:
            raise ValueError(
                f"--resolution {h},{w} must be divisible by spatial "
                f"vae_stride={config.vae_stride[1:]}"
            )
        lat_h = h // config.vae_stride[1]
        lat_w = w // config.vae_stride[2]
        if lat_h % config.patch_size[1] != 0 or lat_w % config.patch_size[2] != 0:
            raise ValueError(
                f"--resolution {h},{w} gives latent {(lat_h, lat_w)}, "
                f"which must be divisible by patch_size={config.patch_size[1:]}"
            )
    else:
        aspect = h_in / w_in
        lat_h = round(
            np.sqrt(args.max_area * aspect) // config.vae_stride[1]
            // config.patch_size[1] * config.patch_size[1]
        )
        lat_w = round(
            np.sqrt(args.max_area / aspect) // config.vae_stride[2]
            // config.patch_size[2] * config.patch_size[2]
        )
        h = lat_h * config.vae_stride[1]
        w = lat_w * config.vae_stride[2]

    # 2. frame-count validation
    F_v = args.num_frames
    vae_t_stride = config.vae_stride[0]
    if (F_v - 1) % vae_t_stride != 0:
        adj = ((F_v - 1) // vae_t_stride) * vae_t_stride + 1
        print(
            f"[{task.tag}] num_frames={F_v} does not satisfy (F-1)%%{vae_t_stride}==0; "
            f"auto-adjusted to {adj}"
        )
        F_v = adj
    T_lat = (F_v - 1) // vae_t_stride + 1
    max_seq_len = T_lat * (lat_h // config.patch_size[1]) \
        * (lat_w // config.patch_size[2])

    # 3. tgt_first_masks: omittable for single object (-> all 1), read first frame for multi-object.
    # Note: tgt mask's start_frame is aligned with target_image_frame_idx,
    # i.e. when target_image is the K-th frame of a video, the mask also takes the K-th frame (mask video usually corresponds frame-by-frame to the original video).
    if args.target_mask_all_ones:
        tgt_first_masks = [
            torch.ones(1, 1, h, w, device=device) for _ in range(N)
        ]
        print(
            f"[{task.tag}] target-mask ablation: replaced {N} mask(s) "
            "with all-ones masks"
        )
    elif N == 1 and len(task.target_mask_paths) == 0:
        tgt_first_masks = [torch.ones(1, 1, h, w, device=device)]
    else:
        tgt_first_masks = []
        for mp in task.target_mask_paths:
            m = load_mask_decord(
                mp, start_frame=task.target_image_frame_idx, num_frames=1,
                resolution=(h, w),
            ).to(device)
            tgt_first_masks.append(m)

    # 4. load N ref videos + optional ref masks
    ref_videos_per_obj = []
    for rv_path, rv_start in zip(task.ref_videos, task.ref_videos_clip_starts):
        rv = load_video_decord(
            rv_path, start_frame=rv_start, num_frames=F_v, resolution=(h, w),
        ).to(device)
        ref_videos_per_obj.append(rv)

    ref_masks_per_obj = None
    if task.ref_mask_paths:
        ref_masks_per_obj = []
        for rmp, rv_start in zip(
            task.ref_mask_paths, task.ref_videos_clip_starts,
        ):
            rm = load_mask_decord(
                rmp, start_frame=rv_start, num_frames=F_v, resolution=(h, w),
            ).to(device)
            ref_masks_per_obj.append(rm)

    # 5. caption
    if args.use_caption and text_encoder is not None:
        prompt = task.build_caption()
        if prompt:
            cond_embed = text_encoder([prompt], device)[0]
        else:
            cond_embed = empty_embed
        print(f"[{task.tag}] prompt: {prompt[:120]}")
    else:
        prompt = ""
        cond_embed = empty_embed
    context = [cond_embed]
    context_null = [neg_embed]

    # 6. VAE encode tgt image -> z_image
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)
    z_dim = vae.model.z_dim
    noise = torch.randn(
        z_dim, T_lat, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device,
    )
    with vae_on_gpu(vae, device, enabled=args.vae_offload):
        img_resized = F.interpolate(
            img[None].cpu(), size=(h, w), mode='bicubic',
        ).to(device)
        img_for_vae = img_resized[0].unsqueeze(1)
        z_image = vae.encode([img_for_vae])[0]

    # 7. Motion encoder per-object forward
    per_obj_4d = []
    motion_attn_mask_lat_per_obj = []
    for obj_idx in range(N):
        ref_for_motion = ref_videos_per_obj[obj_idx]
        ref_mask_for_dismo = None
        if ref_masks_per_obj is not None:
            ref_mask_for_dismo = ref_masks_per_obj[obj_idx]
            ref_for_motion = apply_motion_ref_mask_to_rgb(
                ref_for_motion, ref_mask_for_dismo,
                args.motion_ref_mask_bg, fill_value=args.motion_ref_mask_fill,
            )

        tgt_mask_obj = tgt_first_masks[obj_idx]
        if tgt_mask_obj.dim() == 3:
            tgt_mask_obj = tgt_mask_obj[None]
        mask_lat = F.interpolate(
            tgt_mask_obj.float(),
            size=(lat_h, lat_w), mode='nearest',
        ).to(device=device)
        motion_attn_mask_lat_per_obj.append(mask_lat)

        mask_in = None
        if args.temporal_use_mask_spatial:
            mask_in = mask_lat.to(dtype=args.dismo_dtype_torch)

        with torch.cuda.amp.autocast(dtype=config.param_dtype):
            m4d = temporal_encoder.forward(
                to_motion_rgb(ref_for_motion),
                mask_lat=mask_in,
                ref_mask_video=ref_mask_for_dismo,
            )
        per_obj_4d.append(m4d)

    motion_4d = torch.cat(per_obj_4d, dim=2)  # [1, T_lat, N*K_pf, D]
    motion_attn_mask_first = None
    motion_attn_object_counts = None
    if args.temporal_motion_attn_mask_bias:
        motion_attn_mask_first = build_motion_attn_mask_first(
            motion_attn_mask_lat_per_obj,
            device=device,
        )
        if motion_attn_mask_first is not None:
            motion_attn_object_counts = torch.tensor(
                [N], device=device, dtype=torch.long,
            )
    motion_attn_mask_current = motion_attn_mask_first
    propagate_mask = (
        bool(args.temporal_motion_attn_mask_propagate)
        and motion_attn_mask_first is not None
    )
    propagate_layers = _parse_int_list(
        args.temporal_motion_attn_mask_propagate_layers,
    )
    if propagate_mask:
        args.temporal_motion_attn_mask_bias_frames = "dynamic"
        print(
            f"[{task.tag}] dynamic routing mask: "
            f"steps={args.temporal_motion_attn_mask_propagate_steps}, "
            f"layers={propagate_layers or 'last'}, "
            f"feat_dim={args.temporal_motion_attn_mask_propagate_feat_dim}"
        )

    # 8. UniPC sampling
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=1, use_dynamic_shifting=False,
    )
    sample_scheduler.set_timesteps(
        args.sampling_steps, device=device, shift=args.shift,
    )
    timesteps = sample_scheduler.timesteps

    mask2 = torch.ones_like(noise)
    mask2[:, 0] = 0
    latent = (1.0 - mask2) * z_image + mask2 * noise

    mask_bias_scale = (
        args.temporal_motion_attn_mask_bias_scale
        if motion_attn_mask_first is not None else 0.0
    )
    mask_neg_bias_scale = (
        args.temporal_motion_attn_neg_bias_scale
        if motion_attn_mask_first is not None else 0.0
    )

    def _set_motion_provider():
        model._temporal_motion_attn_data_provider.set(
            motion_4d=motion_4d,
            mask_first=motion_attn_mask_current,
            object_counts=motion_attn_object_counts,
            tokens_per_object=args.temporal_motion_embs_per_latent,
            mask_bias_scale=mask_bias_scale,
            mask_neg_bias_scale=mask_neg_bias_scale,
            mask_bias_frames=args.temporal_motion_attn_mask_bias_frames,
        )

    def _update_routing_masks(features, step_idx, timestep):
        """Propagate and optionally save masks for either guidance branch."""
        updated_mask = propagate_motion_attn_masks(
            motion_attn_mask_first,
            features,
            T=T_lat,
            H=lat_h // config.patch_size[1],
            W=lat_w // config.patch_size[2],
            feat_dim=args.temporal_motion_attn_mask_propagate_feat_dim,
            topk=args.temporal_motion_attn_mask_propagate_topk,
            temperature=args.temporal_motion_attn_mask_propagate_temp,
            prev_frames=args.temporal_motion_attn_mask_propagate_prev,
        )
        if (
            args.temporal_motion_attn_mask_propagate_save
            and step_idx % args.temporal_motion_attn_mask_propagate_save_every == 0
        ):
            save_dynamic_routing_masks(
                updated_mask,
                out_dir=os.path.dirname(os.path.abspath(task.output_path)),
                tag=task.tag,
                step_idx=step_idx,
                timestep=timestep,
                fps=args.fps,
                resolution=(h, w),
            )
        return updated_mask

    try:
        empty_motion_ctx = torch.zeros(
            motion_4d.shape[0], 0, motion_4d.shape[-1],
            device=motion_4d.device, dtype=motion_4d.dtype,
        )
        arg_c = {
            'context': context, 'seq_len': max_seq_len,
            'motion_context': empty_motion_ctx,
        }
        arg_null = {
            'context': context_null, 'seq_len': max_seq_len,
            'motion_context': empty_motion_ctx,
        }
        motion_guidance_scale = float(args.motion_guide_scale)
        use_motion_guidance = abs(motion_guidance_scale - 1.0) > 1e-6
        if not use_motion_guidance and not propagate_mask:
            _set_motion_provider()

        with torch.cuda.amp.autocast(dtype=config.param_dtype):
            for step_idx, t in enumerate(
                tqdm(timesteps, desc=f"[{task.tag}] sampling", leave=False)
            ):
                lat_in = [latent.to(device)]
                timestep = t.unsqueeze(0).to(device)
                temp_ts = (
                    mask2[0][:, ::config.patch_size[1],
                             ::config.patch_size[2]] * timestep
                ).flatten()
                if temp_ts.size(0) < max_seq_len:
                    pad = temp_ts.new_ones(
                        max_seq_len - temp_ts.size(0)
                    ) * timestep
                    temp_ts = torch.cat([temp_ts, pad])
                ts = temp_ts.unsqueeze(0)
                need_propagate = (
                    propagate_mask
                    and step_idx < args.temporal_motion_attn_mask_propagate_steps
                )

                if use_motion_guidance:
                    model._temporal_motion_attn_data_provider.reset()
                    p_u0 = model(lat_in, t=ts, **arg_null)[0]
                    if need_propagate:
                        c0_out, c0_feat = model(
                            lat_in, t=ts, **arg_c,
                            return_features=True,
                            feature_layer_indices=propagate_layers,
                        )
                        p_c0 = c0_out[0]
                        motion_attn_mask_current = _update_routing_masks(
                            c0_feat, step_idx, t,
                        )
                    else:
                        p_c0 = model(lat_in, t=ts, **arg_c)[0]
                    _set_motion_provider()
                    p_cm = model(lat_in, t=ts, **arg_c)[0]
                    npred = (
                        p_u0
                        + args.guide_scale * (p_c0 - p_u0)
                        + motion_guidance_scale * (p_cm - p_c0)
                    )
                else:
                    if need_propagate:
                        model._temporal_motion_attn_data_provider.reset()
                        _, c0_feat = model(
                            lat_in, t=ts, **arg_c,
                            return_features=True,
                            feature_layer_indices=propagate_layers,
                        )
                        motion_attn_mask_current = _update_routing_masks(
                            c0_feat, step_idx, t,
                        )
                    if propagate_mask:
                        _set_motion_provider()
                    nc = model(lat_in, t=ts, **arg_c)[0]
                    nu = model(lat_in, t=ts, **arg_null)[0]
                    npred = nu + args.guide_scale * (nc - nu)
                latent = sample_scheduler.step(
                    npred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g,
                )[0].squeeze(0)
                latent = (1.0 - mask2) * z_image + mask2 * latent

            with vae_on_gpu(vae, device, enabled=args.vae_offload):
                video = vae.decode([latent])[0]
    finally:
        model._temporal_motion_attn_data_provider.reset()

    # 9. concat ref (optional): concatenate ref videos / ref masks / tgt masks together
    if args.concat_ref:
        try:
            has_tgt_mask = bool(
                args.target_mask_all_ones or len(task.target_mask_paths) > 0
            )
            has_ref_masks = len(task.ref_mask_paths) > 0
            video = _concat_ref_video(
                video,
                ref_videos_per_obj=ref_videos_per_obj,
                ref_masks_per_obj=ref_masks_per_obj if has_ref_masks else None,
                tgt_first_masks=tgt_first_masks if has_tgt_mask else None,
                is_multi=task.is_multi,
            )
        except Exception as _e:
            print(f"[warn] [{task.tag}] concat ref failed: {_e}")
            import traceback
            traceback.print_exc()

    # 10. save
    os.makedirs(os.path.dirname(os.path.abspath(task.output_path)), exist_ok=True)
    cache_video(
        video[None], save_file=task.output_path,
        fps=args.fps, nrow=1, normalize=True, value_range=(-1, 1),
    )
    print(f"[{task.tag}] -> {task.output_path}")


def _concat_ref_video(
    video, ref_videos_per_obj, is_multi: bool,
    ref_masks_per_obj=None, tgt_first_masks=None,
):
    """Display references, optional masks, then the generated video side by side.

    Each reference may have its own grayscale mask panel. Target masks repeat
    across all frames; shorter inputs repeat their last frame. Panels use the
    generated video's size and [-1, 1] range, with width padded to a multiple of 16.
    """
    C_v, F_v, H_v, W_v = video.shape
    device = video.device
    dtype = video.dtype

    def _resize_video(rv):
        """[3, T, H, W] in [-1,1] -> [3, F_v, H_v, W_v]."""
        ref_T = rv.shape[1]
        if ref_T == F_v:
            ref_sel = rv
        elif ref_T > F_v:
            ref_sel = rv[:, :F_v]
        else:
            pad_t = rv[:, -1:].expand(-1, F_v - ref_T, -1, -1)
            ref_sel = torch.cat([rv, pad_t], dim=1)
        ref_sel = F.interpolate(
            ref_sel.permute(1, 0, 2, 3),
            size=(H_v, W_v), mode='bilinear', align_corners=False,
        ).permute(1, 0, 2, 3).to(device, dtype=dtype)
        return ref_sel

    def _mask_to_rgb_video(mask, target_T):
        """Single-channel mask [1, T_in, H_in, W_in] -> [3, target_T, H_v, W_v] in [-1, 1].

        - mask values [0, 1] mapped to grayscale: 0 -> -1 (black), 1 -> 1 (white).
        - copied to 3 channels.
        - T dimension: repeat the last frame if too few, truncate if too many (aligned with ref video).
        - spatial dimension: nearest interpolate to (H_v, W_v).
        """
        m = mask.float().clamp(0, 1)         # [1, T_in, H_in, W_in]
        T_in = m.shape[1]
        if T_in == target_T:
            m_t = m
        elif T_in > target_T:
            m_t = m[:, :target_T]
        else:
            pad_t = m[:, -1:].expand(-1, target_T - T_in, -1, -1)
            m_t = torch.cat([m, pad_t], dim=1)
        # spatial resize to (H_v, W_v); F.interpolate 4D input: [N, C, H, W],
        # treat T as the batch dimension.
        m_t = F.interpolate(
            m_t.permute(1, 0, 2, 3),         # [T, 1, H, W]
            size=(H_v, W_v), mode='nearest',
        ).permute(1, 0, 2, 3)                # [1, T, H, W]
        # mask [0, 1] -> RGB [-1, 1] grayscale
        m_rgb = m_t.expand(3, -1, -1, -1) * 2.0 - 1.0   # [3, T, H, W]
        return m_rgb.to(device, dtype=dtype).contiguous()

    sep_w_per = 4
    sep_per = torch.zeros(
        C_v, F_v, H_v, sep_w_per, device=device, dtype=dtype,
    )

    # build the ref area (per object: ref_video [+ ref_mask right next to it])
    ref_parts = []
    N = len(ref_videos_per_obj)
    for k in range(N):
        if k > 0:
            ref_parts.append(sep_per.clone())
        ref_parts.append(_resize_video(ref_videos_per_obj[k]))
        if ref_masks_per_obj is not None and k < len(ref_masks_per_obj):
            ref_parts.append(_mask_to_rgb_video(
                ref_masks_per_obj[k], target_T=F_v,
            ))

    # build the tgt mask area (each object's mask side by side horizontally, separated by sep)
    tgt_parts = []
    if tgt_first_masks is not None and len(tgt_first_masks) > 0:
        for k, m in enumerate(tgt_first_masks):
            if k > 0:
                tgt_parts.append(sep_per.clone())
            tgt_parts.append(_mask_to_rgb_video(m, target_T=F_v))

    # sep_big width: align the total width to 16 (cache_video encoding requirement).
    # total width = sum(ref_parts) + sep_big + sum(tgt_parts) + sep_big? + W_v
    # simplification: insert sep_big between the ref/tgt/gen segments, each width adapts so total width % 16 == 0.
    sep_w_big_min = 8

    def _make_sep_big(extra_pad):
        sep_w_big = sep_w_big_min + extra_pad
        return torch.zeros(
            C_v, F_v, H_v, sep_w_big, device=device, dtype=dtype,
        )

    # estimate: first sum all non-sep_big parts, then decide the number of sep_big and padding.
    cat_parts = list(ref_parts)
    use_tgt_sep = len(tgt_parts) > 0
    if use_tgt_sep:
        # placeholder sep_big (actual width filled in later)
        cat_parts.append(None)   # placeholder #1
        cat_parts.extend(tgt_parts)
    cat_parts.append(None)       # placeholder #2 (before gen)
    cat_parts.append(video)

    # compute the fixed width (total width excluding placeholders)
    fixed_w = sum(
        p.shape[-1] for p in cat_parts if p is not None
    )
    n_sep_big = sum(1 for p in cat_parts if p is None)
    # each sep_big is at least sep_w_big_min, and total width must be % 16 == 0
    base_w = fixed_w + n_sep_big * sep_w_big_min
    pad_total = (-base_w) % 16
    # add the pad to the first sep_big (others stay at min)
    sep_big_widths = [sep_w_big_min] * n_sep_big
    if n_sep_big > 0 and pad_total > 0:
        sep_big_widths[0] += pad_total

    final_parts = []
    sep_idx = 0
    for p in cat_parts:
        if p is None:
            final_parts.append(_make_sep_big(
                sep_big_widths[sep_idx] - sep_w_big_min
            ))
            sep_idx += 1
        else:
            final_parts.append(p)

    return torch.cat(final_parts, dim=-1)


# ============================================================================
# Main
# ============================================================================

def _setup_distributed():
    """Setup multi-GPU via accelerate.

    Returns: (rank, world_size, device, is_distributed, accelerator).
    Falls back to single-process when on a single GPU / not launched via accelerate launch.
    """
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        device = accelerator.device
        is_dist = world_size > 1
        return rank, world_size, device, is_dist, accelerator
    except Exception:
        # not launched via accelerate launch -> single process
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return 0, 1, device, False, None


def main():
    args = parse_args()
    rank, world_size, device, is_dist, accelerator = _setup_distributed()
    config = WAN_CONFIGS[args.config_name]

    def _print(msg):
        """All ranks print, but with a [rank/world] prefix."""
        if is_dist:
            print(f"[rank {rank}/{world_size}] {msg}", flush=True)
        else:
            print(msg, flush=True)

    def _print_main(msg):
        """Only rank 0 prints (for summary information)."""
        if rank == 0:
            print(msg, flush=True)

    if is_dist:
        _print_main(
            f"=== Distributed inference: {world_size} processes ==="
        )

    # ---- collect tasks ----
    tasks = []
    parse_failed = 0
    if args.inference_csv:
        df = pd.read_csv(args.inference_csv)
        end = len(df) if args.csv_limit < 0 else min(
            len(df), args.csv_start + args.csv_limit,
        )
        df = df.iloc[args.csv_start:end].reset_index(drop=True)
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
        if is_dist and accelerator is not None:
            accelerator.wait_for_everyone()
        for i, row in df.iterrows():
            try:
                task = _csv_row_to_task(row.to_dict(), args.output_dir)
                tasks.append(task)
            except Exception as _e:
                parse_failed += 1
                _print(f"[warn] csv row {i} parse failed: {_e}; skip")
        _print_main(
            f"Collected {len(tasks)} tasks from {args.inference_csv}"
        )
        if not tasks:
            _print_main(
                f"[err] No valid inference tasks were selected "
                f"({parse_failed} CSV row(s) failed to parse)."
            )
            return 1

        # ---- task assignment: stride split (rank=0 takes 0,K,2K,...; rank=1 takes 1,K+1,...) ----
        my_tasks = tasks[rank::world_size]
        _print(f"Assigned {len(my_tasks)} / {len(tasks)} tasks")
    else:
        # single-command mode: only rank 0 runs, other ranks idle
        if rank == 0:
            tasks = [_args_to_single_task(args)]
            my_tasks = tasks
        else:
            _print(
                f"single-command mode: rank {rank} idle "
                f"(only rank 0 runs the single task)"
            )
            my_tasks = []

    # Each rank loads a model and T5 cache for its assigned tasks.
    t0 = time.time()
    _print(f"Loading model on {device}...")
    model, vae, text_encoder, empty_embed, neg_embed, temporal_encoder = (
        load_model_and_encoder(args, config, device, tasks=my_tasks)
    )
    _print(f"Model loaded in {time.time() - t0:.1f}s")

    # wait for all ranks to finish loading before starting (avoid faster ranks grabbing GPU resources)
    if is_dist and accelerator is not None:
        accelerator.wait_for_everyone()

    # ---- run tasks ----
    failed = 0
    pbar = tqdm(
        my_tasks,
        desc=f"rank{rank} inference",
        position=rank if is_dist else 0,
        leave=False,
    ) if my_tasks else my_tasks
    for task in pbar:
        try:
            run_one_task(
                task, args, config, device,
                model, vae, text_encoder, empty_embed, neg_embed, temporal_encoder,
            )
        except Exception as _e:
            failed += 1
            import traceback
            traceback.print_exc()
            _print(f"[err] [{task.tag}] failed: {_e}")

    # ---- sync + summarize ----
    n_my = len(my_tasks)
    n_my_ok = n_my - failed
    if is_dist and accelerator is not None:
        # use reduce to sum all ranks' stats onto rank 0
        stats = torch.tensor([n_my_ok, failed], device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_ok = int(stats[0].item())
        total_failed = int(stats[1].item())
        accelerator.wait_for_everyone()
    else:
        total_ok, total_failed = n_my_ok, failed

    # Every rank parsed the same CSV; count invalid rows once, after reduction.
    total_failed += parse_failed
    _print(f"Local: {n_my_ok}/{n_my} succeeded, {failed} failed")
    if rank == 0:
        total = total_ok + total_failed
        if total_failed:
            _print_main(
                f"\n=== Done: {total_ok}/{total} succeeded; "
                f"{total_failed} failed across {world_size} ranks ==="
            )
        else:
            _print_main(f"\n=== Done: all {total} tasks succeeded ===")
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
