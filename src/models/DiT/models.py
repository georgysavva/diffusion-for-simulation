import math

import einops
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self, hidden_size, num_heads, mlp_ratio, enable_conditioning, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.enable_conditioning = enable_conditioning
        if enable_conditioning:
            self.cross_obs_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn_obs = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )

            self.cross_act_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn_act = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )

    def forward(self, x, t, prev_obs, prev_act):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=1)
        )
        if self.enable_conditioning:
            # Cross-Attention on prev_obs
            obs_conditioned, _ = self.cross_attn_obs(
                query=self.cross_obs_norm(x),  # (N, T, D)
                key=prev_obs,  # (N, steps * T, D)
                value=prev_obs,  # (N, steps * T, D)
                need_weights=False,
            )
            x = x + obs_conditioned  # (N, T, D)
            # Cross-Attention on prev_act
            act_conditioned, _ = self.cross_attn_act(
                query=self.cross_act_norm(x),  # (N, T, D)
                key=prev_act,  # (N, steps, D)
                value=prev_act,  # (N, steps, D)
                need_weights=False,
            )
            x = x + act_conditioned  # (N, T, D)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, prev_obs, prev_act):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ActionEmbedder(nn.Module):
    """
    Embeds actions into vector representations.
    """

    def __init__(self, num_actions, num_conditioning_steps, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_actions, hidden_size)
        time_embed = get_1d_sincos_pos_embed_from_grid(
            hidden_size, np.arange(num_conditioning_steps, dtype=np.float32)
        )
        self.time_embed = nn.Parameter(
            torch.from_numpy(time_embed).float().unsqueeze(0), requires_grad=False
        )

    def forward(self, actions):
        embeddings = self.embedding_table(actions)
        embeddings = embeddings + self.time_embed
        return embeddings


class PreviousObservationEmbedder(nn.Module):
    """
    Embeds previous observations into vector representations.
    """

    def __init__(
        self, num_conditioning_steps, image_size, patch_size, in_channels, hidden_size
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.num_patches = self.patch_embed.num_patches
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            hidden_size,
            np.arange(self.num_patches, dtype=np.float32),
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False
        )
        time_embed = get_1d_sincos_pos_embed_from_grid(
            hidden_size,
            np.arange(num_conditioning_steps, dtype=np.float32),
        )
        self.time_embed = nn.Parameter(
            torch.from_numpy(time_embed).float().unsqueeze(0), requires_grad=False
        )

    def forward(self, prev_obs):
        N, steps = prev_obs.shape[:2]
        prev_obs = einops.rearrange(prev_obs, "N steps C H W -> (N steps) C H W")
        prev_obs = self.patch_embed(prev_obs)  # (N * steps, T, embed_dim)
        prev_obs = prev_obs + self.pos_embed
        prev_obs = einops.rearrange(
            prev_obs, "(N steps) T embed_dim -> (N T) steps embed_dim", N=N, steps=steps
        )
        prev_obs = prev_obs + self.time_embed
        prev_obs = einops.rearrange(
            prev_obs,
            "(N T) steps embed_dim -> N (steps T) embed_dim",
            N=N,
            T=self.num_patches,
        )
        return prev_obs


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        num_actions,
        num_conditioning_steps,
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        time_frequency_embedding_size,
        enable_conditioning=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.noised_obs_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size, time_frequency_embedding_size)
        self.enable_conditioning = enable_conditioning
        if enable_conditioning:
            self.previous_obs_embedder = PreviousObservationEmbedder(
                num_conditioning_steps, input_size, patch_size, in_channels, hidden_size
            )
            self.act_embedder = ActionEmbedder(
                num_actions, num_conditioning_steps, hidden_size
            )
        num_patches = self.noised_obs_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.noised_obs_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_conditioning=enable_conditioning,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def load_pretrained_weights(self, weights):
        missing_keys, unexpected_keys = self.load_state_dict(
            weights, strict=False
        )  # This doesn't work yet. Fix the layers.
        print(
            f"Loaded pretrained weights. Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}"
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        noised_obs_pos_embed = get_2d_sincos_pos_embed(
            self.noised_obs_pos_embed.shape[-1],
            int(self.noised_obs_embedder.num_patches**0.5),
        )
        self.noised_obs_pos_embed.data.copy_(
            torch.from_numpy(noised_obs_pos_embed).float().unsqueeze(0)
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.noised_obs_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.noised_obs_embedder.proj.bias, 0)
        if self.enable_conditioning:
            w = self.previous_obs_embedder.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.previous_obs_embedder.patch_embed.proj.bias, 0)

            # Initialize action embedding table:
            nn.init.normal_(self.act_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.noised_obs_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, noised_obs, t, prev_obs, prev_act):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        noised_obs = (
            self.noised_obs_embedder(noised_obs) + self.noised_obs_pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        if self.enable_conditioning:
            prev_obs = self.previous_obs_embedder(prev_obs)  # (N, steps * T, D)
            prev_act = self.act_embedder(prev_act)  # (N, steps, D)
        else:
            prev_obs = prev_act = None

        for block in self.blocks:
            noised_obs = block(noised_obs, t, prev_obs, prev_act)  # (N, T, D)
        noised_obs = self.final_layer(
            noised_obs, t, prev_obs, prev_act
        )  # (N, T, patch_size ** 2 * out_channels)
        noised_obs = self.unpatchify(noised_obs)  # (N, out_channels, H, W)
        return noised_obs

    @property
    def device(self):
        return next(self.parameters()).device


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)
