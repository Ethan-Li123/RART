from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from einops import rearrange


def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(cond_features, features, bias=False)
        nn.init.zeros_(self.linear.weight)

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        # removed one additional expansion here
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm=None, dropout=0.0, d_out=None):
        super().__init__()
        d_out = d_out or d_model
        self.norm = RMSNorm(d_model) if d_cond_norm is None else AdaRMSNorm(d_model, d_cond_norm)
        self.up_proj = nn.Linear(d_model, d_ff * 2, bias=False)
        self.down_proj = nn.Linear(d_ff, d_out, bias=False)
        self.skip_proj = nn.Identity() if d_out == d_model else nn.Linear(d_model, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x, cond_norm=None, **kwargs):
        skip = self.skip_proj(x)
        x = self.norm(x) if cond_norm is None else self.norm(x, cond_norm)
        x, gate = self.up_proj(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_cross=None,
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.d_cross = d_cross
        self.n_heads = d_model // d_head

        self.self_norm = RMSNorm(d_model) if d_cond_norm is None else AdaRMSNorm(d_model, d_cond_norm)
        self.self_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.self_scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.self_out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.self_out.weight)

        if d_cross is not None:
            self.cross_norm_q = RMSNorm(d_model) if d_cond_norm is None else AdaRMSNorm(d_model, d_cond_norm)
            self.cross_norm_kv = RMSNorm(d_cross)
            self.cross_q = nn.Linear(d_model, d_model, bias=False)
            self.cross_kv = nn.Linear(d_cross, d_model * 2, bias=False)
            self.cross_scale = nn.Parameter(torch.full([self.n_heads], 10.0))
            self.cross_out = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.cross_out.weight)

        d_ff = d_model * ff_expand
        # self.ffn_norm = RMSNorm(d_model) if d_cond_norm is None else AdaRMSNorm(d_model, d_cond_norm)
        # self.ffn_up = nn.Linear(d_model, d_ff * 2, bias=False)
        # self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def fwd_self(
        self, 
        x: Float[torch.Tensor, "b l d"], 
        theta: Float[torch.Tensor, "b n_h l d_head"], 
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ):
        skip = x
        if cond_norm is not None:
            x = self.self_norm(x, cond_norm)
        else:
            x = self.self_norm(x)
        qkv = self.self_qkv(x)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.self_scale[:, None, None], torch.tensor(1e-6, device=x.device))

        # theta = self.self_pos_emb(pos.to(qkv.dtype)).movedim(-2, -3)
        # q = self.self_pos_emb.apply_emb(q, theta)
        # k = self.self_pos_emb.apply_emb(k, theta)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")
        x = self.dropout(x)
        x = self.self_out(x)
        return x + skip
    
    def fwd_cross(
        self,
        x: Float[torch.Tensor, "b l d"],
        theta: Float[torch.Tensor, "b n_h l d_head"],
        x_cross: Float[torch.Tensor, "b l_cross d_cross"], 
        theta_cross: Float[torch.Tensor, "b n_h l_cross d_head"],
        cond_norm: Float[torch.Tensor, "b d"],
    ) -> Float[torch.Tensor, "b l d"]:
        skip = x
        if cond_norm is not None:
            x = self.cross_norm_q(x, cond_norm)
        else:
            x = self.cross_norm_q(x)

        x_cross = self.cross_norm_kv(x_cross)
        q = self.cross_q(x)
        kv = self.cross_kv(x_cross)

        q = rearrange(q, "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.cross_scale[:, None, None], torch.tensor(1e-6, device=x.device))

        # pos = pos.to(q.dtype)
        # pos_cross = pos_cross.to(q.dtype)
        # theta = self.cross_pos_emb(pos)
        # theta_cross = self.cross_pos_emb(pos_cross)
        # theta = theta.movedim(-2, -3)
        # theta_cross = theta_cross.movedim(-2, -3)
        # q = self.cross_pos_emb.apply_emb(q, theta)
        # k = self.cross_pos_emb.apply_emb(k, theta_cross)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta_cross)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.cross_out(x)
        return x + skip
    
    def fwd_ffn(
        self,
        x: Float[torch.Tensor, "b l d"],
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ) -> Float[torch.Tensor, "b l d"]:
        return self.ff(x, cond_norm=cond_norm)

    def forward(
        self, 
        x: Float[torch.Tensor, "b l d"], 
        theta: Float[torch.Tensor, "b n_h l d_head"],
        x_cross: Float[torch.Tensor, "b l_cross d_cross"] | None = None, 
        theta_cross: Float[torch.Tensor, "b n_h l_cross d_head"] | None = None,
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ):
        x = self.fwd_self(x, theta=theta, cond_norm=cond_norm)
        if x_cross is not None:
            x = self.fwd_cross(x, theta=theta, x_cross=x_cross, theta_cross=theta_cross, cond_norm=cond_norm)
        x = self.fwd_ffn(x, cond_norm=cond_norm)
        return x


class Transformer(nn.Module):
    def __init__(
        self, 
        width: int,
        depth: int,
        d_cross: int | None = None,
        pos_emb_cls = None,
        **layer_params,
    ):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model=width, d_cross=d_cross, **layer_params) for _ in range(depth)])
        d_head = layer_params.get("d_head", 64)
        n_heads = width // d_head
        self.pos_emb = pos_emb_cls(d_head=d_head, n_heads=n_heads)

    def forward(
        self,
        x: Float[torch.Tensor, "b l d"],
        pos: Float[torch.Tensor, "b l 2"],
        x_cross: Float[torch.Tensor, "b l_cross d_cross"] | None = None,
        pos_cross: Float[torch.Tensor, "b l_cross 2"] | None = None,
        **kwargs,
    ) -> Float[torch.Tensor, "b l d_out"]:
        B, *DIMS, D = x.shape
        x = x.reshape(B, -1, D)
        pos = pos.reshape(B, -1, pos.shape[-1])
        theta = self.pos_emb(pos).movedim(-2, -3)
        theta_cross = None if pos_cross is None else self.pos_emb(pos_cross).movedim(-2, -3)
        for layer in self.layers:
            x = layer(x, x_cross=x_cross, theta=theta, theta_cross=theta_cross, **kwargs)
        x = x.reshape(B, *DIMS, D)
        return x
