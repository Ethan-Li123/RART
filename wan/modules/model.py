# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


@torch.no_grad()
def _compute_motion_attn_mass(q, k_cat, motion_len, text_len, sample=256):
    """Diagnostic: fraction of attention mass query allocates to motion tokens.

    Used in 'text' motion_attn_mode where motion / text K/V share one softmax.
    We sub-sample ``sample`` query positions to keep the [L_q, L_kv] matmul
    cheap (full matmul on L_q≈4096 × L_kv≈1500 × heads=24 is ~150M ops/layer
    in fp32 — sampling 256 cuts that to ~10M).

    Args:
        q       : [b, L_q, n, d]  -- post-norm queries
        k_cat   : [b, L_kv, n, d] -- concat([k_text, k_motion]) post-view
        motion_len, text_len: token counts in the concat (motion comes second).
        sample  : how many query positions to sample.
    Returns:
        Python float in [0, 1]: average fraction of attention probability that
        falls on the motion segment, averaged over sampled queries, heads, batch.
    """
    b, L_q, n, d = q.shape
    L_kv = k_cat.shape[1]
    assert L_kv == text_len + motion_len, \
        f"k_cat seq mismatch: {L_kv} vs text({text_len})+motion({motion_len})"

    s = min(sample, L_q)
    if s < L_q:
        idx = torch.randperm(L_q, device=q.device)[:s]
        q_s = q.index_select(1, idx)
    else:
        q_s = q

    # [b, n, s, d] @ [b, n, d, L_kv] -> [b, n, s, L_kv]
    qf = q_s.permute(0, 2, 1, 3).float()
    kf = k_cat.permute(0, 2, 3, 1).float()
    scores = torch.matmul(qf, kf) / (d ** 0.5)
    probs = scores.softmax(dim=-1)
    # Motion segment is the LAST motion_len entries (text first, motion second).
    motion_mass = probs[..., text_len:].sum(dim=-1).mean()
    return float(motion_mass.item())


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        # 保存 init args 供 enable_motion_lora 使用
        self._qk_norm_flag = qk_norm
        self._eps = eps

        # Motion token processor 引用 (由 enable_motion_tokens 设置)
        self.motion_processor = None
        # Text token 长度 (用于在 forward 里解构 [motion, text] context). 默认为 0,
        # 只有启用 motion 时才设成 self.text_len (= 512).
        self.text_context_length = 0

        # === Motion LoRA path (encoder-mode) ===
        # 设计: motion tokens 与 video latent 同源, 用 sibling self-attention 的
        # to_k / to_v (Wan 预训练) 作为 motion 的 K, V 投影起点, 再加 LoRA delta
        # 学习 motion-specific 的小调整.
        # 字段先占位, enable_motion_lora() 才真正装配.
        self.motion_lora_enabled = False
        self.motion_lora_rank = 0
        # 'text' = motion 融入 text branch 共享 softmax;
        # 'branch' = 独立 motion attention branch + 加性融合 (alpha_motion_branch zero-init gate)
        self.motion_attn_mode = 'text'
        self.lora_scale = 1.0  # 等价 alpha/r=1 的标准 LoRA scale

        # 用 object.__setattr__ 防止子模块自动注册导致 from_pretrained meta tensor 残留
        # (这是 FFMMT_wan 修过的关键 bug, 必须保留 - sibling self_attn 引用 + 占位字段)
        object.__setattr__(self, 'lora_k_motion_A', None)
        object.__setattr__(self, 'lora_k_motion_B', None)
        object.__setattr__(self, 'lora_v_motion_A', None)
        object.__setattr__(self, 'lora_v_motion_B', None)
        object.__setattr__(self, 'alpha_motion', None)         # buffer (固定 1.0)
        object.__setattr__(self, 'alpha_motion_branch', None)  # branch mode gate (Param)
        object.__setattr__(self, 'norm_k_motion', None)
        # sibling self-attn 引用: 用 object.__setattr__ 避免 nn.Module 把它注册成子模块
        object.__setattr__(self, '_sibling_self_attn', None)

        # === Diagnostics probe (off by default; enabled per-step by trainer) ===
        # 'text' mode 下, 把 motion / text K/V 拼起来共享 softmax 后, 我们想知道
        # query 实际把多少注意力质量分给了 motion segment.
        # 训练侧设置 self._probe_enabled = True (主进程, 每 logging_steps 打开一次),
        # forward 末尾 detach 计算 motion_mass = softmax(qk^T)[:, motion_seg].sum(-1).mean()
        # 写入 self._probe_motion_mass (Python float). 不开销默认路径.
        object.__setattr__(self, '_probe_enabled', False)
        object.__setattr__(self, '_probe_motion_mass', None)
        # 抽样 query 数量, 控制 probe 开销 (L_q 通常 4000+).
        object.__setattr__(self, '_probe_query_sample', 256)

    def enable_motion_lora(self, rank: int = 16, mode: str = 'text'):
        """启用 motion LoRA path: 在 sibling self-attn 的 K/V 投影上加 LoRA delta.

        Args:
            rank: LoRA rank (e.g., 16)
            mode: 'text' = motion 融入 text branch (与 text 共享 softmax);
                  'branch' = 独立 motion attention branch + 加性融合 (zero-init gate).
        """
        if self._sibling_self_attn is None:
            raise RuntimeError(
                "enable_motion_lora: _sibling_self_attn 未 link. "
                "请在 WanAttentionBlock 构造后 / WanModel.enable_motion_tokens 里 link."
            )
        if mode not in ('text', 'branch'):
            raise ValueError(f"motion_attn_mode 必须为 'text' 或 'branch', 收到 {mode!r}")

        dim = self.dim
        qk_norm = getattr(self, '_qk_norm_flag', True)
        eps = getattr(self, '_eps', 1e-6)
        self.motion_lora_rank = int(rank)
        self.motion_attn_mode = mode

        # 在这里**才**实际创建 nn.Module / nn.Parameter, 让它们走 PyTorch 自动注册成
        # cross_attn 的子模块. 注意此时 from_pretrained 已经完成 (在外部), 不再有 meta
        # 残留问题.
        self.lora_k_motion_A = nn.Linear(dim, rank, bias=False)
        self.lora_k_motion_B = nn.Linear(rank, dim, bias=False)
        self.lora_v_motion_A = nn.Linear(dim, rank, bias=False)
        self.lora_v_motion_B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_k_motion_A.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_v_motion_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_k_motion_B.weight)
        nn.init.zeros_(self.lora_v_motion_B.weight)

        # alpha_motion 是固定 buffer (=1.0). 用 nn.Parameter(zeros) 会形成 LoRA B + alpha
        # 双零死锁: alpha=0 → delta=0 → B 无梯度; B=0 → delta=0 → alpha 无梯度.
        # LoRA 标准做法: scale 是固定超参, B zero-init 保护初始安全, B 自己学 delta 大小.
        if 'alpha_motion' in self.__dict__:
            del self.__dict__['alpha_motion']
        self.register_buffer('alpha_motion', torch.ones(1), persistent=True)

        # branch mode gate (可学习). 初始化 0.1 避免 zero-init 冷启动延迟,
        # 同时幅度够小不会严重扰动预训练分布.
        self.alpha_motion_branch = nn.Parameter(torch.full((1,), 0.1))

        # motion K 用一个独立的 RMSNorm 防 norm 跑飞
        self.norm_k_motion = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.motion_lora_enabled = True

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C] - [motion(可选), text]
            context_lens(Tensor): Shape [B]
        """
        # 解构 context
        # 没启用 motion: context 全是 text
        # 启用 motion: context = [motion(M), text(text_context_length=512)]
        has_motion = (
            hasattr(self, 'motion_processor') and self.motion_processor is not None
        )
        if has_motion and self.text_context_length > 0:
            context_motion = context[:, :-self.text_context_length]
            context_text = context[:, -self.text_context_length:]
        else:
            context_motion = None
            context_text = context

        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Q
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        # text K/V (走原有投影器, 不动)
        k_text = self.norm_k(self.k(context_text))
        v_text = self.v(context_text)

        # === motion K/V 计算 (两个 mode 共用这一段) ===
        motion_present = (
            has_motion and context_motion is not None and context_motion.shape[1] > 0
        )
        k_motion = v_motion = None
        if motion_present:
            if self.motion_lora_enabled:
                sa = self._sibling_self_attn
                k_motion_base = sa.k(context_motion)
                v_motion_base = sa.v(context_motion)
                k_motion_delta = self.lora_k_motion_B(self.lora_k_motion_A(context_motion))
                v_motion_delta = self.lora_v_motion_B(self.lora_v_motion_A(context_motion))
                k_motion = self.norm_k_motion(k_motion_base + self.lora_scale * k_motion_delta)
                v_motion = v_motion_base + self.lora_scale * v_motion_delta
            else:
                # 旧路径 fallback (motion 共用 text path 投影器)
                k_motion = self.norm_k(self.k(context_motion))
                v_motion = self.v(context_motion)

        # === 根据 motion_attn_mode 决定 motion 怎么参与 attention ===
        # 5B 没有 image branch, 简单很多 (相比 2.1 i2v-14B 还有 image attention 加性融合)
        if motion_present and self.motion_lora_enabled and self.motion_attn_mode == 'branch':
            # 独立 motion attention branch + 加性融合 (alpha_motion_branch zero-init gate)
            k_t = k_text.view(b, -1, n, d)
            v_t = v_text.view(b, -1, n, d)
            k_m = k_motion.view(b, -1, n, d)
            v_m = v_motion.view(b, -1, n, d)
            text_ctx_lens = (
                torch.full((b,), self.text_context_length, dtype=torch.long, device=q.device)
                if context_lens is not None else None
            )
            x_text = flash_attention(q, k_t, v_t, k_lens=text_ctx_lens)
            x_motion = flash_attention(q, k_m, v_m, k_lens=None)
            x_motion = x_motion * self.alpha_motion_branch
            # --- diag probe (branch mode): output-side injection ratio ---
            # 与 'text' mode 的 motion_mass (softmax 概率质量) 不同, branch mode
            # 在 attn 输出做加性融合, 所以从输出端测 ||alpha·x_motion|| / ||x_total||
            # 更直接, 反映 motion 段对 cross-attn 输出的实际贡献占比.
            if self._probe_enabled:
                with torch.no_grad():
                    n_motion = x_motion.detach().float().norm()
                    n_total = (x_text + x_motion).detach().float().norm().clamp(min=1e-8)
                    self._probe_motion_mass = float((n_motion / n_total).item())
            x = x_text + x_motion
        elif motion_present:
            # mode 'text' (默认): motion 拼到 text K/V 共享 softmax
            k_cat = torch.cat([k_text, k_motion], dim=1).view(b, -1, n, d)
            v_cat = torch.cat([v_text, v_motion], dim=1).view(b, -1, n, d)
            # --- diag probe: motion attention mass (no_grad, sub-sampled) ---
            if self._probe_enabled:
                self._probe_motion_mass = _compute_motion_attn_mass(
                    q, k_cat,
                    motion_len=k_motion.shape[1],
                    text_len=k_text.shape[1],
                    sample=self._probe_query_sample,
                )
            x = flash_attention(q, k_cat, v_cat, k_lens=context_lens)
        else:
            # 没有 motion: 只有 text
            k = k_text.view(b, -1, n, d)
            v = v_text.view(b, -1, n, d)
            x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        # Link sibling self_attn 到 cross_attn (motion LoRA path 用其 to_k/to_v 作 base).
        # 用 object.__setattr__ 跳过 nn.Module 的子模块自动注册 (避免 self_attn 被
        # 重复挂到 cross_attn 名下, 导致 from_pretrained 时新增 meta 参数).
        if hasattr(self.cross_attn, '_sibling_self_attn'):
            object.__setattr__(self.cross_attn, '_sibling_self_attn', self.self_attn)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        return_pre_ffn=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            attn_delta = self.cross_attn(self.norm3(x), context, context_lens)
            x = x + attn_delta
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            if return_pre_ffn:
                return x, attn_delta
            return x

        return cross_attn_ffn(x, context, context_lens, e)


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Motion token processor (encoder-mode), 默认不启用; 由 enable_motion_tokens() 装配
        self.motion_processor = None

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        video_latent=None,
        motion_input=None,
        target_mask=None,
        motion_context=None,
        return_features=False,
        feature_layer_indices=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, seq_len] (per-pixel)
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            video_latent (Tensor, *optional*):
                Legacy name for reference motion input. Kept as fallback for old callers.
            motion_input (Tensor, *optional*):
                Optical flow / dense motion field for motion encoder, shape [B, 2, T, H, W].
                Only used when motion_processor is enabled (encoder mode).
            target_mask (Tensor, *optional*):
                First-frame target mask for motion encoder, shape [B, 1, H, W].
                Only used when motion_processor is enabled.
            motion_context (Tensor, *optional*):
                Precomputed motion tokens, shape [B, M, C]. If provided, skip
                motion_processor.get_motion_tokens and reuse these tokens.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context (text)
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # 注入 motion tokens (如果启用 encoder 模式).
        # 拼接顺序: [motion(M), text(text_len)]. WanCrossAttention 用
        # text_context_length=text_len 反推 motion 段在前.
        if self.motion_processor is not None:
            if motion_context is None:
                motion_context = self.motion_processor.get_motion_tokens(
                    batch_size=context.size(0),
                    motion_input=motion_input,
                    video_latent=video_latent,
                    target_mask=target_mask,
                )
            context = torch.cat([motion_context, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        feature_layers = set(feature_layer_indices or [])
        collect_features = bool(return_features)
        hidden_features = []
        for block_idx, block in enumerate(self.blocks):
            take_feature = collect_features and (
                block_idx in feature_layers
                or (not feature_layers and block_idx == len(self.blocks) - 1)
            )
            if take_feature:
                x, pre_ffn_x = block(x, return_pre_ffn=True, **kwargs)
                hidden_features.append(
                    pre_ffn_x[:, :seq_lens.max().item()].detach()
                )
            else:
                x = block(x, **kwargs)

        # head
        returned_features = None
        if collect_features:
            if hidden_features:
                returned_features = torch.stack(hidden_features, dim=0).mean(dim=0)
            else:
                returned_features = x[:, :seq_lens.max().item()].detach()

        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        out = [u.float() for u in x]
        if collect_features:
            return out, returned_features
        return out

    def enable_motion_tokens(self, use_mask=True,
                             motion_lora_rank=0,
                             motion_attn_mode='text',
                             adapter_internal_dim=768, adapter_num_heads=12,
                             # ---- DisMo encoder 参数 ----
                             num_video_queries=512,
                             dismo_ckpt_path=None,
                             dismo_repo_path=None,
                             dino_ckpt_path=None,
                             dismo_max_frames=None,
                             dismo_dtype=torch.bfloat16,
                             dismo_gradient_checkpointing=False):
        """
        启用 motion token 机制 (DisMo motion encoder + 可选 motion LoRA path).

        架构:
            ref RGB video [3, T, H, W] in [0, 1]
              -> DisMoMotionEncoder (frozen DisMo backbone, bf16, learnable
                 resampler queries; 最后一层 zero-init)            -> [B, K, internal_dim]
              -> position cross-attn (target mask)                  -> internal_dim
              -> zero-init output_proj                              -> motion tokens [B, K, dim]

        其中 K = ``num_video_queries``, motion token 数量恒定不依赖输入分辨率.

        Args:
            use_mask: 是否使用 target mask 注入, 默认 True.
            motion_lora_rank: > 0 时启用 motion LoRA path. 默认 0 = 关闭走旧路径
                (motion 共用 text path 的 self.k/self.v 投影).
            motion_attn_mode: 'text' (默认) / 'branch'. 仅当 motion_lora_rank > 0 时生效.
            adapter_internal_dim / adapter_num_heads: motion adapter 内部维度与
                position cross-attn / resampler head 数 (resampler 复用同样的 head 数).
            num_video_queries: 学习 query 数量 K (= 输出 motion token 数).
            dismo_ckpt_path: 本地 DisMo motion_extractor_large 权重文件路径; None 则
                通过 torch.hub.load_state_dict_from_url 从 HuggingFace 下载.
            dismo_repo_path: 可选, 本地 CompVis/DisMo 仓库路径. 提供则注入 sys.path,
                便于 ``from dismo.model import MotionExtractor_Large``.
            dino_ckpt_path: 可选, 本地 DINOv2 (dinov2_vitl14_reg) 权重 .pt 路径.
                若提供, 跳过 facebookresearch/dinov2 的 torch.hub 下载, 适合离线集群.
            dismo_max_frames: 可选, ref video 帧数上限 (uniform-stride subsample).
            dismo_dtype: DisMo backbone + resampler 计算 dtype, 默认 bf16.
            dismo_gradient_checkpointing: 是否对 resampler 启用 torch.utils.checkpoint.
        """
        if self.model_type not in ['ti2v', 'i2v', 'flf2v']:
            raise ValueError(
                f"Motion tokens only supported for ti2v/i2v/flf2v models, got {self.model_type}"
            )

        from .motiontoken_processor import MotionTokenProcessor

        # 创建 motion processor
        self.motion_processor = MotionTokenProcessor(
            hidden_dim=self.dim,
            use_mask=use_mask,
            adapter_internal_dim=adapter_internal_dim,
            adapter_num_heads=adapter_num_heads,
            num_video_queries=num_video_queries,
            dismo_ckpt_path=dismo_ckpt_path,
            dismo_repo_path=dismo_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            dismo_max_frames=dismo_max_frames,
            dismo_dtype=dismo_dtype,
            dismo_gradient_checkpointing=dismo_gradient_checkpointing,
        )

        # 在所有 cross_attn 层中设置 motion_processor 引用 + text_context_length
        for block in self.blocks:
            if hasattr(block.cross_attn, 'motion_processor'):
                block.cross_attn.motion_processor = self.motion_processor
                block.cross_attn.text_context_length = self.text_len  # = 512
                if motion_lora_rank and motion_lora_rank > 0 \
                        and hasattr(block.cross_attn, 'enable_motion_lora'):
                    block.cross_attn.enable_motion_lora(
                        rank=motion_lora_rank, mode=motion_attn_mode)

        dismo_tag = (f"dismo=(K={num_video_queries}, "
                     f"max_frames={dismo_max_frames}, "
                     f"ckpt={dismo_ckpt_path or 'hub:CompVis/DisMo'})")
        adapter_tag = f"adapter=(dim={adapter_internal_dim},heads={adapter_num_heads})"
        lora_str = ""
        if motion_lora_rank and motion_lora_rank > 0:
            lora_str = (f", DiT-LoRA rank={motion_lora_rank} "
                        f"mode={motion_attn_mode}")
        print(f"Motion tokens enabled: mask={use_mask}{lora_str}, "
              f"{adapter_tag}, {dismo_tag}")

    def disable_motion_tokens(self):
        """禁用 motion token 机制."""
        self.motion_processor = None
        for block in self.blocks:
            if hasattr(block.cross_attn, 'motion_processor'):
                block.cross_attn.motion_processor = None
                block.cross_attn.text_context_length = 0

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
