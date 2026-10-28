import math
from functools import partial
from typing import Union, List, Optional, Any
import numpy as np
import torch
from torch import nn
import diffusers
from jaxtyping import Float
from einops import rearrange
from dismo.transformer import Transformer, FeedForwardBlock, RMSNorm
from dismo.dinov2 import DinoFeatureExtractor



# ================================================================================================
# RoPE Positional Embedding Utilities
# ================================================================================================

def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_sinusoidal_pos_1d(x, width, temperature=10_000.0, dtype=np.float32):
    assert width % 2 == 0, 'Width must be mult of 2 for sincos posemb'
    omega = torch.arange(width // 2, device=x.device) / (width // 2 - 1)
    omega = 1.0 / (temperature**omega)
    x = torch.einsum('m,d->md', x.flatten(), omega)
    pe = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    return pe


def make_axial_pos_2d(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2

    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)

    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def make_axial_pos_3d(
    t,
    h,
    w,
    pixel_aspect_ratio=1.0,
    align_corners=False,
    dtype=None,
    device=None,
    relative_pos=True,
):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2
    
    if align_corners:
        t_pos = torch.arange(t, dtype=dtype, device=device)
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        t_pos = torch.arange(t, dtype=dtype, device=device)
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)

    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).unsqueeze(0)
    grid = torch.cat([
        torch.ones_like(grid[:, :, :, :1]) * t_pos.view(-1, 1, 1, 1),
        grid.repeat(t_pos.size(0), 1, 1, 1),
    ], dim=-1)
    t, h, w, d = grid.shape
    return grid.view(t * h * w, d)


class AxialRoPE2D(nn.Module):
    def __init__(self, d_head: int, n_heads: int):
        super().__init__()
        # we only apply RoPE to half of the token
        assert d_head % 2 == 0, "Number of head dimensions must be even"
        d_head //= 2
        min_freq = math.pi
        max_freq = 10.0 * math.pi
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        # 2 * 2 for sin and cos, height and width
        freqs = torch.stack([torch.linspace(log_min, log_max, n_heads * d_head // (2 * 2) + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freqs.view(2, d_head // (2 * 2), n_heads).mT.contiguous(), requires_grad=False)

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class AxialRoPE3D(nn.Module):
    def __init__(self, d_head, n_heads, temporal_theta=100):
        super().__init__()
        n_freqs = n_heads * d_head // 8
        min_freq = math.pi
        max_freq = 10.0 * math.pi
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        spatial_freqs = torch.linspace(log_min, log_max, n_freqs + 1)[:-1].exp()
        spatial_freqs = torch.stack([spatial_freqs] * 2)
        temporal_freqs = 1.0 / (temporal_theta ** (torch.arange(0, n_freqs).float() / (n_freqs)))
        self.spatial_freqs = nn.Parameter(spatial_freqs.view(2, d_head // 8, n_heads).mT.contiguous(), requires_grad=False)
        self.temporal_freqs = nn.Parameter(temporal_freqs.view(d_head // 8, n_heads).T.contiguous(), requires_grad=False)

    def forward(self, pos):
        theta_t = pos[..., None, 0:1] * self.temporal_freqs
        theta_h = pos[..., None, 1:2] * self.spatial_freqs[0]
        theta_w = pos[..., None, 2:3] * self.spatial_freqs[1]
        result = torch.cat((theta_t, theta_h, theta_w), dim=-1)
        return result



# ================================================================================================
# Frame Generator Modules
# ================================================================================================

class TinyAutoencoderKL(nn.Module):
    def __init__(self, repo="madebyollin/taesd"):
        super().__init__()
        self.ae = diffusers.AutoencoderTiny.from_pretrained(repo)
        self.ae.eval()
        self.ae.compile()
        self.ae.requires_grad_(False)

    def forward(self, img):
        return self.encode(img)

    @torch.no_grad()
    def encode(self, img):
        img = img.movedim(-1, 1)  # Move channel dimension to the front
        latent = self.ae.encode(img, return_dict=False)[0]
        latent = latent.movedim(1, -1)  # Move channel dimension back to the end
        return latent

    @torch.no_grad()
    def decode(self, latent):
        latent = latent.movedim(-1, 1)  # Move channel dimension to the front
        rec = self.ae.decode(latent, return_dict=False)[0]
        rec = rec.movedim(1, -1)
        return rec


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([FeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x
    

class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Linear(in_features * self.h * self.w, out_features, bias=False)

    def forward(self, x, pos):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=self.h, nw=self.w).mean(dim=-2)
        return self.proj(x), pos
    

class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.norm = RMSNorm(in_features)
        self.proj = nn.Linear(in_features, out_features * self.h * self.w, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return x
    

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FrameGenerator(nn.Module):
    def __init__(
        self, 
        width: int,
        depth: int,
        d_head: int,
        d_motion: int = 128,
        source_encoder_params: dict[str, Any] = {},
        patch_size: tuple[int, int] = (2, 2),
        mapping_depth: int = 2,
        mapping_width: int = 1024,
        mapping_dropout: float = 0,
    ):
        super().__init__()

        self.source_encoder = DinoFeatureExtractor(**source_encoder_params)

        self.transformer = Transformer(
            width=width,
            depth=depth,
            d_cross=self.source_encoder.embed_dim,
            d_cond_norm=mapping_width,
            d_head=d_head,
            pos_emb_cls=AxialRoPE2D,
            dropout=0,
            ff_expand=3,
        )

        self.token_merge = TokenMerge(4, width, patch_size=patch_size)
        self.token_split = TokenSplit(width, 4, patch_size=patch_size)

        self.ae = TinyAutoencoderKL()

        self.mapping = MappingNetwork(mapping_depth, mapping_width, mapping_width * 3, dropout=mapping_dropout)

        self.time_emb = FourierFeatures(1, mapping_width)
        self.time_in_proj = nn.Linear(mapping_width, mapping_width, bias=False)
        
        self.motion_proj = nn.Linear(d_motion + mapping_width, mapping_width, bias=False)
        self.patch_size = patch_size

        nn.init.zeros_(self.motion_proj.weight)

    def get_pos(self, x: Float[torch.Tensor, "B *DIMS C"]) -> Float[torch.Tensor, "B *DIMS c"]:
        B, *DIMS, _ = x.shape
        pos = make_axial_pos_2d(*DIMS, device=x.device).view(1, *DIMS, -1).expand(B, -1, -1, -1)
        return pos

    def forward(self, x: Float[torch.Tensor, "b h w c"], **data_kwargs) -> Float[torch.Tensor, "b"]:
        B = x.shape[0]

        # encode with pre-trained vae
        x = self.ae.encode(x)

        # logit sigmoid timestep sampling
        t = torch.sigmoid(torch.randn((B,), device=x.device))
        texp = t.view([B, *([1] * len(x.shape[1:]))])

        # construct noisy input
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        dtype = x.dtype
        zt, t = zt.to(dtype), t.to(dtype)

        # get conditionings
        cond_dict = self.get_conditioning(t, **data_kwargs)
        pos = self.get_pos(zt)

        # forward pass
        zt, pos = self.token_merge(zt, pos)
        vtheta = self.transformer(zt, pos=pos, **cond_dict)
        vtheta = self.token_split(vtheta)

        # compute loss and return
        return ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))

    @torch.no_grad()
    def sample(
        self, z: Float[torch.Tensor, "b ... c"], sample_steps=50, **data_kwargs
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        B = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * B, device=z.device, dtype=z.dtype).view([B, *([1] * len(z.shape[1:]))])
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * B, device=z.device, dtype=z.dtype)
            pos = self.get_pos(z)
            cond_dict = self.get_conditioning(t, **data_kwargs)
            z, pos = self.token_merge(z, pos)
            vc_cond = self.transformer(z, pos=pos, **cond_dict)
            vc_cond = self.token_split(vc_cond)
            z = z - dt * vc_cond
        x = self.ae.decode(z)
        return x

    def get_conditioning(
        self,
        t: Float[torch.Tensor, "b"],
        source_frames: Float[torch.Tensor, "b h w c"],
        motion_embeddings: Float[torch.Tensor, "b d"],
        delta_times: Optional[Float[torch.Tensor, "b"]] = None,
    ) -> dict[str, torch.Tensor]:
        
        # cond_time
        time_emb = self.time_in_proj(self.time_emb(t[..., None]))

        # cond_motion
        dt_embs = make_sinusoidal_pos_1d(delta_times * 100, width=time_emb.shape[-1])
        motion_dt_emb = torch.cat([motion_embeddings, dt_embs], dim=-1)
        motion_dt_emb = self.motion_proj(motion_dt_emb)
        cond_motion_and_time = self.mapping(motion_dt_emb + time_emb)

        # cond_source
        cond_source = self.source_encoder(source_frames)
        Hs, Ws = cond_source.shape[1:3]
        source_pos = make_axial_pos_2d(Hs, Ws, device=cond_source.device).expand(cond_source.shape[0], Hs * Ws, 2)
        cond_source = cond_source.flatten(1, 2)

        # cond_dict
        cond_dict = {}
        cond_dict['cond_norm'] = cond_motion_and_time
        cond_dict['x_cross'] = cond_source
        cond_dict['pos_cross'] = source_pos
        return cond_dict



# ================================================================================================
# Motion Extractor Module
# ================================================================================================

class MotionExtractor(Transformer):
    def __init__(
        self, 
        width: int,
        depth: int,
        d_head: int,
        d_motion: int, 
        frame_encoder_params: dict[str, Any] = {},
        max_delta_time: int = 1,
        train_resolution: list = [8, 256, 256],
    ):
        super().__init__(width=width, depth=depth, d_head=d_head, pos_emb_cls=AxialRoPE3D)
        self.d_motion = d_motion
        self.frame_encoder = DinoFeatureExtractor(**frame_encoder_params)
        self.proj = nn.Linear(self.frame_encoder.embed_dim, width, bias=False)
        self.head = nn.Sequential(
            RMSNorm(width),
            FeedForwardBlock(width, 3 * width),
            FeedForwardBlock(width, 3 * width, d_out=d_motion),
            RMSNorm(d_motion),
        )
        self.query_tokens = nn.Parameter(torch.empty(1, 1, width))
        self.max_delta_time = max_delta_time
        self.train_resolution = train_resolution
        nn.init.normal_(self.query_tokens, std=0.02)
    
    def forward(self, x: Float[torch.Tensor, "b t h w c"]) -> Float[torch.Tensor, "b t d"]:
        # Compute frame tokens
        B, T, H, W, _ = x.shape
        if self.frame_encoder is not None:
            x = rearrange(x, "b t h w c -> (b t) h w c")
            x = self.frame_encoder(x)
            x = rearrange(x, "(b t) h w c -> b t h w c", b=B)
            B, T, H, W, _ = x.shape

        x = self.proj(x)
        x = x.flatten(1, 3)
        pos = make_axial_pos_3d(T, H, W, device=x.device).expand(B, -1, 3)

        # Compute query tokens
        query_tokens = self.query_tokens.repeat(B, T - self.max_delta_time, 1)
        query_pos = make_axial_pos_3d(T - self.max_delta_time, 1, 1, device=x.device).expand(B, -1, 3)

        # Forward pass
        motion_embeddings = super().forward(
            x=torch.cat([query_tokens, x], dim=1),
            pos=torch.cat([query_pos, pos], dim=1),
        )
        motion_embeddings = motion_embeddings[:, :query_tokens.shape[1]]
        motion_embeddings = self.head(motion_embeddings)
        return motion_embeddings

    def forward_sliding(
        self, 
        x: Float[torch.Tensor, "b t h w c"], 
        lookahead: int = None
    ) -> Float[torch.Tensor, "b t d"]:
        if lookahead is None:
            lookahead = self.max_delta_time
        
        B, T, H, W, _ = x.shape
        assert T >= lookahead, f"Input video length {T} must be at least lookahead={lookahead}."

        T_max = min(self.train_resolution[0], T)

        if self.frame_encoder is not None:
            x = rearrange(x, "b t h w c -> (b t) h w c")
            x = self.frame_encoder(x)
            x = rearrange(x, "(b t) h w c -> b t h w c", b=B)
            B, T, H, W, _ = x.shape

        x = self.proj(x)
        pos = make_axial_pos_3d(T_max, H, W, device=x.device).expand(B, -1, 3)

        # Compute query tokens
        query_tokens = self.query_tokens.repeat(B, T_max - lookahead, 1)
        query_pos = make_axial_pos_3d(T_max - lookahead, 1, 1, device=x.device).expand(B, -1, 3)
        
        window_offsets = list(range(max(T - self.train_resolution[0], 0) + 1))
        x_b = torch.cat([
            torch.cat([query_tokens, x[:, t:(t+T_max)].flatten(1, 3)], dim=1)
            for t in window_offsets
        ])
        pos_b = torch.cat([query_pos, pos], dim=1).repeat(len(window_offsets), 1, 1)
        embs_b = super().forward(x=x_b, pos=pos_b)
        embs_b = embs_b[:, :query_tokens.shape[1]]
        embs_b = self.head(embs_b)
        embs_b = rearrange(embs_b, "(a b) t d -> a b t d", a=len(window_offsets))

        motion_embeddings = torch.cat([embs_b[0]] + list(embs_b[1:, :, -1:].unbind(0)), dim=1)
        return motion_embeddings



# ================================================================================================
# DisMo Model
# ================================================================================================

class DisMo(nn.Module):
    def __init__(
        self, 
        motion_extractor_params: dict[str, Any], 
        frame_generator_params: dict[str, Any], 
        compile: bool = True,
    ):
        super().__init__()
        self.motion_extractor = MotionExtractor(**motion_extractor_params)
        self.frame_generator = FrameGenerator(**frame_generator_params)
        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True, mode='reduce-overhead', dynamic=False)

    def forward(
        self, 
        motion_frames: Float[torch.Tensor, "b t h w c"], 
        source_frames: Float[torch.Tensor, "b p h w c"], 
        target_frames: Float[torch.Tensor, "b p h w c"], 
        delta_times: Float[torch.Tensor, "b p"], 
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        motion_embeddings: torch.Tensor = self.motion_extractor(motion_frames) # b p d
        loss = self.frame_generator(
            x=target_frames.flatten(0, 1),
            motion_embeddings=motion_embeddings.flatten(0, 1),
            source_frames=source_frames.flatten(0, 1),
            delta_times=delta_times.flatten(0, 1),
        )
        metrics = {
            'motion_std_batch': motion_embeddings.std(dim=(0, 2)).mean(),
            'motion_std_video': motion_embeddings.std(dim=2).mean(),
        }
        return loss.mean(), metrics


DisMo_Large = partial(
    DisMo, 
    motion_extractor_params=dict(
        width=1024,
        depth=20,
        d_head=64,
        d_motion=128,
        frame_encoder_params=dict(
            model_version="dinov2_vitl14_reg",
            gradient_last_blocks=2,
        ),
        max_delta_time=4,
        train_resolution=[8, 256, 256],
    ),
    frame_generator_params=dict(
        width=1152,
        depth=28,
        d_head=72,
        source_encoder_params=dict(
            model_version="dinov2_vitl14_reg",
            gradient_last_blocks=2,
        ),
        patch_size=(2, 2),
        mapping_depth=2,
        mapping_width=1024,
        mapping_dropout=0.0,
        d_motion=128,
    ),
)

MotionExtractor_Large = partial(
    MotionExtractor,
    width=1024,
    depth=20,
    d_head=64,
    d_motion=128,
    frame_encoder_params=dict(
        model_version="dinov2_vitl14_reg",
        gradient_last_blocks=2,
    ),
    max_delta_time=4,
    train_resolution=[8, 256, 256],
)
