"""Motion Token Processor for Wan2.2 (DisMo motion encoder edition).

结构:
- 运动路径: DisMo MotionExtractor (frozen, bf16, DINOv2-L14 帧编码器 + 解耦
            motion transformer)
            -> Linear(128 -> internal_dim) 升维
            -> 2 层 Perceiver/Q-Former 风格 resampler cross-attn
               (K=num_video_queries 个 learnable query, 最后一层 zero-init)
            -> LayerNorm                                        -> [B, K, internal_dim]
- 位置路径: 2D Conv 从 target mask 编码空间位置, 输出 64 个 token (8x8 pool)
- 融合:    Position Cross-Attention ×2 layers 将目标位置注入运动表示
- 输出:    LayerNorm + Linear (kaiming_uniform_ 初始化, 不再 zero-init,
            避免与 LoRA-B zero-init / branch alpha 叠加形成冷启动死锁) -> DiT hidden_dim

输入语义:
    motion_input 是 *reference RGB video* (PIL list / numpy / torch RGB,
    [3,T,H,W] in [0,1]). DisMoMotionEncoder 内部自动 resize 到 256x256 并
    shift 到 [-1, 1].

历史命名:
    内部属性 ``self.mllm_encoder`` 之所以保留 mllm_ 名字, 是为了减少 ckpt 命名
    迁移成本; 实际指向 DisMoMotionEncoder. (旧的 Qwen3-VL MLLM 路径已彻底移除.)
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .dismo_motion_encoder import DisMoMotionEncoder


class MaskSpatialEncoder(nn.Module):
    """位置路径: 从 target mask 编码空间位置特征.

    输入: target_mask [B, 1, H, W] (latent 分辨率)
    输出: mask_features [B, num_tokens, internal_dim]
    """

    def __init__(self, internal_dim: int = 768, pool_size: Tuple[int, int] = (8, 8)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(pool_size),
        )
        self.proj = nn.Linear(128, internal_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        x = self.conv(mask)
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class MotionTokenEncoder(nn.Module):
    """Motion Token Encoder = DisMoMotionEncoder (frozen DisMo backbone +
    learnable resampler -> [B, K, internal_dim]) + position cross-attn over
    target mask + 标准 kaiming_uniform 输出投影 (不再 zero-init).

    Pipeline:
        ref_video (RGB frames)
            -> DisMoMotionEncoder (frozen DisMo, learnable resampler queries
               + zero-init last block out_proj)                 -> [B, K, internal_dim]
            -> + (mask cross-attn over MaskSpatialEncoder)      -> [B, K, internal_dim]
            -> output_proj (kaiming-init, weight 标准初始化)     -> [B, K, hidden_dim]

    其中 K = ``num_video_queries``. 不依赖输入分辨率, motion token 数量恒定.

    注: 本模块的 "初始 motion 信号近零" 安全保护改由 resampler 末层的
    ``zero_init_out=True`` 提供 (cross_attn.out_proj + ff[-1] 都 zero-init);
    output_proj 改用标准 kaiming 初始化, 避免多重 zero-init 叠加形成
    梯度死锁.
    """

    def __init__(
        self,
        output_dim: int,                          # = DiT hidden_dim
        internal_dim: int = 768,
        adapter_num_heads: int = 12,              # mask cross-attn heads
        mask_pool_size: Tuple[int, int] = (8, 8),
        use_mask: bool = True,                    # False: 跳过 mask cross-attn 路径
        # ---- DisMo encoder 参数 ----
        num_video_queries: int = 512,
        dismo_ckpt_path: Optional[str] = None,
        dismo_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        dismo_max_frames: Optional[int] = None,
        dismo_dtype: torch.dtype = torch.bfloat16,
        dismo_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.internal_dim = internal_dim
        self.use_mask = bool(use_mask)
        self.num_video_queries = num_video_queries
        self.dismo_ckpt_path = dismo_ckpt_path
        self.dismo_repo_path = dismo_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.dismo_max_frames = dismo_max_frames
        self.dismo_dtype = dismo_dtype
        self.dismo_gradient_checkpointing = dismo_gradient_checkpointing

        # === 路径 1: DisMo motion encoder ===
        # 内部属性名仍为 mllm_encoder (历史命名), 实际是 DisMoMotionEncoder.
        # 输出维度 = internal_dim, 对齐 mask cross-attn / output_proj.
        self.mllm_encoder = DisMoMotionEncoder(
            out_dim=internal_dim,
            num_video_queries=num_video_queries,
            dismo_ckpt_path=dismo_ckpt_path,
            dismo_repo_path=dismo_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            max_frames=dismo_max_frames,
            dtype=dismo_dtype,
            resampler_num_layers=2,
            resampler_num_heads=adapter_num_heads,
            gradient_checkpointing=dismo_gradient_checkpointing,
        )

        # === 路径 2: 位置编码 (target mask) ===
        # use_mask=False 时不构造 mask_encoder / pos_cross_attn, 直接跳过整个 mask 路径
        if self.use_mask:
            self.mask_encoder = MaskSpatialEncoder(internal_dim, mask_pool_size)

            # === 融合: Position Cross-Attention (2 层堆叠, 带残差) ===
            # 多层 cross-attn 让 motion query 反复对照 mask 来精炼空间归属,
            # 多物体场景下能更好地区分各物体的空间区域.
            self.pos_cross_num_layers = 2
            self.pos_cross_norms_q = nn.ModuleList([
                nn.LayerNorm(internal_dim) for _ in range(self.pos_cross_num_layers)
            ])
            self.pos_cross_norms_kv = nn.ModuleList([
                nn.LayerNorm(internal_dim) for _ in range(self.pos_cross_num_layers)
            ])
            self.pos_cross_attns = nn.ModuleList([
                nn.MultiheadAttention(internal_dim, adapter_num_heads, batch_first=True)
                for _ in range(self.pos_cross_num_layers)
            ])
        else:
            self.mask_encoder = None
            self.pos_cross_num_layers = 0
            self.pos_cross_norms_q = None
            self.pos_cross_norms_kv = None
            self.pos_cross_attns = None

        # === 输出投影 ===
        # 不再 zero-init: 之前的 zero-init 与 LoRA-B zero-init / branch alpha 0.1
        # 叠加, 在空 prompt 训练下形成多重 zero-init 冷启动死锁,
        # 导致 motion 通道长期梯度极弱、模型学不出 motion transfer.
        # 改用 PyTorch 默认 kaiming_uniform 初始化 (with bias zero), 让 motion 通道
        # 从第一步就有非零输出, 提供有效梯度信号给下游 LoRA 学习.
        self.output_norm = nn.LayerNorm(internal_dim)
        self.output_proj = nn.Linear(internal_dim, output_dim)
        # PyTorch nn.Linear 默认就是 kaiming_uniform_ (a=sqrt(5)) for weight 和
        # uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for bias. 这里显式重置 bias 为 0
        # 防止初始 bias 引入无意义偏置, 但保留 weight 的标准 kaiming 初始化.
        nn.init.zeros_(self.output_proj.bias)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_motion(
        self,
        motion_input: Union[torch.Tensor, Sequence],
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """编码 reference RGB video -> mask 注入前的运动表示.

        Args:
            motion_input: 可为以下任一形式 (单个 ref video):
                * list[PIL.Image]
                * numpy uint8 [T, H, W, 3]
                * torch.Tensor [3, T, H, W] 或 [T, 3, H, W] (float in [0, 1] 或 uint8)
              也可以是上述任意形式的 list/tuple, 视作 batch (每条样本独立跑 MLLM).
            instruction: 可选, 覆盖默认的 user instruction.
        Returns:
            motion_queries: [B, K, internal_dim], dtype = dismo_dtype, device = encoder device.
        """
        # Detect batch vs single sample.
        if self._is_single_ref_video(motion_input):
            return self.mllm_encoder(motion_input, instruction=instruction)
        # batched: list of samples
        return self.mllm_encoder.forward_batch(list(motion_input), instruction=instruction)

    @staticmethod
    def _is_single_ref_video(x) -> bool:
        """Decide whether x represents a single ref video or a batch of them.

        Single ref video patterns:
            * list[PIL.Image]
            * numpy [T, H, W, 3]
            * torch.Tensor [3, T, H, W] or [T, 3, H, W]
        Batch pattern: list/tuple whose elements are themselves single ref videos.
        """
        from PIL import Image
        import numpy as np

        if isinstance(x, torch.Tensor):
            return True
        if isinstance(x, np.ndarray):
            return True
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return True  # empty -> treat as single (will likely error downstream)
            first = x[0]
            if isinstance(first, Image.Image):
                # list[PIL] -> single video
                return True
            # otherwise assume batch of samples
            return False
        # Unknown -> let downstream raise.
        return True

    def inject_target_mask(
        self,
        motion_queries: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_mask_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """将 target mask 的空间位置注入到运动表示中.

        Args:
            motion_queries: [B, K, internal_dim] mask 注入前的运动表示
            target_mask: [B, 1, H, W] target image 上的物体 mask (可选, 默认全 1)
            target_mask_size: (H, W), 仅在 target_mask=None 时用于构造默认全 1 mask
                (新版 motion_input 是 RGB 视频, 不再隐含分辨率信息).
        Returns:
            conditioned_queries: [B, K, internal_dim]
        """
        # use_mask=False: 直接 passthrough, 不做任何 mask 注入
        if not self.use_mask:
            return motion_queries

        B = motion_queries.shape[0]
        device = motion_queries.device
        dtype = motion_queries.dtype

        if target_mask is None:
            if target_mask_size is None:
                # 默认 latent 分辨率 64x64; 不重要, 因为最终会被 conv pool 到 mask_pool_size.
                H, W = 64, 64
            else:
                H, W = target_mask_size
            target_mask = torch.ones(B, 1, H, W, device=device, dtype=dtype)
        else:
            target_mask = target_mask.to(device=device, dtype=dtype)
        mask_feats = self.mask_encoder(target_mask)

        # 2 层 cross-attn 反复对照 mask, 残差连接
        x = motion_queries
        for norm_q, norm_kv, attn in zip(
            self.pos_cross_norms_q, self.pos_cross_norms_kv, self.pos_cross_attns,
        ):
            q = norm_q(x)
            kv = norm_kv(mask_feats)
            x = x + attn(q, kv, kv, need_weights=False)[0]
        return x

    def project_motion_queries(self, motion_queries: torch.Tensor) -> torch.Tensor:
        """把 internal_dim 的运动表示投影成 DiT hidden_dim motion tokens."""
        return self.output_proj(self.output_norm(motion_queries))

    def forward(
        self,
        motion_input: Union[torch.Tensor, Sequence],
        target_mask: Optional[torch.Tensor] = None,
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode reference video + (optional) target mask -> motion tokens.

        Returns:
            motion_tokens: [B, K, output_dim]
        """
        motion_queries = self.encode_motion(motion_input, instruction=instruction)
        conditioned = self.inject_target_mask(motion_queries, target_mask)
        return self.project_motion_queries(conditioned)


class MotionTokenProcessor(nn.Module):
    """Motion Token 处理器 (DisMo motion encoder).

    本类是 ``MotionTokenEncoder`` 的薄包装, 提供 ``get_motion_tokens`` /
    ``get_motion_queries_before_mask`` 等给训练 / 推理脚本使用的稳定接口,
    以及 ckpt 序列化所需的 metadata.

    旧版 (Qwen3-VL MLLM 编码器) 已彻底移除.
    """

    def __init__(
        self,
        hidden_dim: int,                          # = DiT hidden_dim
        use_mask: bool = True,
        adapter_internal_dim: int = 768,
        adapter_num_heads: int = 12,
        # ---- DisMo encoder 参数 ----
        num_video_queries: int = 512,
        dismo_ckpt_path: Optional[str] = None,
        dismo_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        dismo_max_frames: Optional[int] = None,
        dismo_dtype: torch.dtype = torch.bfloat16,
        dismo_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_mask = use_mask
        self.adapter_internal_dim = adapter_internal_dim
        self.adapter_num_heads = adapter_num_heads
        self.num_video_queries = num_video_queries
        self.dismo_ckpt_path = dismo_ckpt_path
        self.dismo_repo_path = dismo_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.dismo_max_frames = dismo_max_frames
        self.dismo_dtype = dismo_dtype
        self.dismo_gradient_checkpointing = dismo_gradient_checkpointing

        self.encoder = MotionTokenEncoder(
            output_dim=hidden_dim,
            internal_dim=adapter_internal_dim,
            adapter_num_heads=adapter_num_heads,
            use_mask=use_mask,
            num_video_queries=num_video_queries,
            dismo_ckpt_path=dismo_ckpt_path,
            dismo_repo_path=dismo_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            dismo_max_frames=dismo_max_frames,
            dismo_dtype=dismo_dtype,
            dismo_gradient_checkpointing=dismo_gradient_checkpointing,
        )

    # ------------------------------------------------------------------
    # LM LoRA helpers (no-op for DisMo; kept for API parity with the old
    # Qwen3-VL path so trainer/inference scripts can call through unchanged).
    # ------------------------------------------------------------------

    def enable_lm_lora(
        self,
        rank: int,
        alpha: Optional[float] = None,
        target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> int:
        """No-op: DisMo has no language model. Always returns 0."""
        return self.encoder.mllm_encoder.enable_lm_lora(
            rank=rank, alpha=alpha, target_modules=target_modules,
        )

    @property
    def lm_lora_enabled(self) -> bool:
        return self.encoder.mllm_encoder.lm_lora_enabled

    @property
    def lm_lora_rank(self) -> int:
        return self.encoder.mllm_encoder.lm_lora_rank

    @property
    def lm_lora_alpha(self) -> float:
        return self.encoder.mllm_encoder.lm_lora_alpha

    @property
    def lm_lora_target_modules(self) -> Tuple[str, ...]:
        return self.encoder.mllm_encoder.lm_lora_target_modules

    def lm_lora_state_dict(self) -> dict:
        return self.encoder.mllm_encoder.lm_lora_state_dict()

    def load_lm_lora_state_dict(self, state: dict, strict: bool = False):
        return self.encoder.mllm_encoder.load_lm_lora_state_dict(state, strict=strict)

    def lm_lora_named_parameters(self):
        """Yield nothing — DisMo path has no LM LoRA params."""
        return iter(())

    def get_motion_queries_before_mask(
        self,
        batch_size: int,
        motion_input: Optional[Union[torch.Tensor, Sequence]] = None,
        video_latent: Optional[Union[torch.Tensor, Sequence]] = None,
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """获取 mask 注入前的运动表示, 用于 token-level T_ref 不变性约束.

        Args:
            batch_size: 仅作向后兼容, 未使用.
            motion_input: reference RGB video (PIL list / numpy / torch RGB).
            video_latent: 旧参数名兼容; 作为 motion_input fallback.
            instruction: 仅 API 兼容, DisMo 路径忽略.
        Returns:
            motion_queries: [B, K, internal_dim]
        """
        del batch_size
        if motion_input is None:
            motion_input = video_latent
        assert motion_input is not None, "motion_input (ref RGB video) is required"
        return self.encoder.encode_motion(motion_input, instruction=instruction)

    def get_motion_tokens(
        self,
        batch_size: int,
        motion_input: Optional[Union[torch.Tensor, Sequence]] = None,
        video_latent: Optional[Union[torch.Tensor, Sequence]] = None,
        target_mask: Optional[torch.Tensor] = None,
        motion_queries_before_mask: Optional[torch.Tensor] = None,
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """统一接口获取 motion tokens.

        Args:
            batch_size: 仅作向后兼容, 未使用.
            motion_input: reference RGB video (PIL list / numpy / torch RGB).
            video_latent: 旧参数名兼容; 作为 motion_input fallback.
            target_mask: [B, 1, H, W] target mask (可选, 默认全 1).
            motion_queries_before_mask: 预计算的 mask 注入前运动表示;
                传入后复用它生成最终 tokens, 避免重复跑 DisMo encoder.
            instruction: 仅 API 兼容, DisMo 路径忽略.
        Returns:
            motion_tokens: [B, K, hidden_dim]
        """
        del batch_size
        if motion_input is None:
            motion_input = video_latent
        if motion_queries_before_mask is None:
            assert motion_input is not None, "motion_input (ref RGB video) is required"
            motion_queries_before_mask = self.encoder.encode_motion(
                motion_input, instruction=instruction
            )
        conditioned = self.encoder.inject_target_mask(
            motion_queries_before_mask, target_mask,
        )
        return self.encoder.project_motion_queries(conditioned)

    # === Checkpoint 方法 ===

    def load_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        print(f"Loaded motion tokens from {checkpoint_path}")

    def save_checkpoint(self, save_path: str, global_step: Optional[int] = None, **kwargs):
        # Persist *only* the trainable adapter weights — strip the frozen DisMo
        # backbone (DINOv2 + transformer + heads). Trainable keys retained:
        #   encoder.mllm_encoder.{queries, input_proj, resampler.*, out_norm}
        #   encoder.{mask_encoder, pos_cross_*, output_norm, output_proj}
        full_sd = self.state_dict()
        adapter_sd = {
            k: v for k, v in full_sd.items()
            if "encoder.mllm_encoder.dismo." not in k
        }
        checkpoint = {
            'state_dict': adapter_sd,
            'hidden_dim': self.hidden_dim,
            'adapter_internal_dim': self.adapter_internal_dim,
            'adapter_num_heads': self.adapter_num_heads,
            'num_video_queries': self.num_video_queries,
            'dismo_ckpt_path': self.dismo_ckpt_path,
            'dino_ckpt_path': self.dino_ckpt_path,
            'dismo_max_frames': self.dismo_max_frames,
            'dismo_dtype': self.dismo_dtype,
        }
        if global_step is not None:
            checkpoint['global_step'] = global_step
        checkpoint.update(kwargs)
        torch.save(checkpoint, save_path)
        print(f"Saved motion tokens checkpoint to {save_path}")
