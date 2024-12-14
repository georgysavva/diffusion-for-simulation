import math

import einops
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale, T):

    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = einops.rearrange(x, "(b t) n m-> b (t n) m", b=B, t=T, n=N, m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = einops.rearrange(x, "b (t n) m-> (b t) n m", b=B, t=T, n=N, m=M)
    return x


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
        self,
        hidden_size,
        num_heads,
        mlp_ratio,
        condition_on_actions,
        **block_kwargs,
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
        self.condition_on_actions = condition_on_actions
        if self.condition_on_actions:
            self.cross_act_norm = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.cross_attn_act = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, batch_first=True
            )
        self.temporal_norm1 = nn.LayerNorm(hidden_size)
        self.temporal_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.temporal_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, t, prev_obs, prev_act):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t).chunk(6, dim=1)
        )
        B = t.shape[0]
        BT, N, M = x.shape
        T = BT // B
        x = einops.rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T, n=N, m=M)
        res_temporal = self.temporal_attn(self.temporal_norm1(x))
        res_temporal = self.temporal_fc(res_temporal)
        res_temporal = einops.rearrange(
            res_temporal, "(b n) t m -> (b t) n m", b=B, t=T, n=N, m=M
        )
        x = einops.rearrange(x, "(b n) t m -> (b t) n m", b=B, t=T, n=N, m=M)
        x = x + res_temporal
        # Cross-Attention on prev_act
        if self.condition_on_actions:
            # this doesn't work yet
            act_conditioned, _ = self.cross_attn_act(
                query=self.cross_act_norm(x),  # (N, T, D)
                key=prev_act,  # (N, steps, D)
                value=prev_act,  # (N, steps, D)
                need_weights=False,
            )
            x = x + act_conditioned  # (N, T, D)

        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, T))
        attn = einops.rearrange(attn, "(b t) n m-> b (t n) m", b=B, t=T, n=N, m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = einops.rearrange(attn, "b (t n) m-> (b t) n m", b=B, t=T, n=N, m=M)
        x = x + attn

        mlp = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, T))
        mlp = einops.rearrange(mlp, "(b t) n m-> b (t n) m", b=B, t=T, n=N, m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = einops.rearrange(mlp, "b (t n) m-> (b t) n m", b=B, t=T, n=N, m=M)
        x = x + mlp
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
        BT = x.shape[0]
        B = t.shape[0]
        T = BT // B
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, T)
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


class ObservationEmbedder(nn.Module):
    """
    Embeds previous observations into vector representations.
    """

    def __init__(
        self, input_size, patch_size, in_channels, num_conditioning_steps, hidden_size
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
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

    def forward(self, obs):
        N, steps = obs.shape[:2]
        obs = einops.rearrange(obs, "N steps C H W -> (N steps) C H W")
        obs = self.patch_embed(obs)  # (N * steps, T, embed_dim)
        obs = obs + self.pos_embed
        obs = einops.rearrange(
            obs, "(N steps) T embed_dim -> (N T) steps embed_dim", N=N, steps=steps
        )
        obs = obs + self.time_embed
        obs = einops.rearrange(
            obs,
            "(N T) steps embed_dim -> (N steps) T embed_dim",
            N=N,
            T=self.num_patches,
        )
        return obs


class VDT(nn.Module):
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
        condition_on_actions,
        learn_sigma,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.t_embedder = TimestepEmbedder(hidden_size, time_frequency_embedding_size)
        self.condition_on_actions = condition_on_actions
        self.seq_len = num_conditioning_steps + 1
        self.obs_embedder = ObservationEmbedder(
            input_size, patch_size, in_channels, self.seq_len, hidden_size
        )
        if condition_on_actions:
            self.act_embedder = ActionEmbedder(
                num_actions, num_conditioning_steps, hidden_size
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    condition_on_actions=condition_on_actions,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def load_pretrained_weights(self, weights):
        key_mapping = {
            "pos_embed": "obs_embedder.pos_embed",
            "x_embedder.proj.weight": "obs_embedder.patch_embed.proj.weight",
            "x_embedder.proj.bias": "obs_embedder.patch_embed.proj.bias",
        }
        for map_from, map_to in key_mapping.items():
            if map_from in weights:
                weights[map_to] = weights.pop(map_from)
        missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
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

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.obs_embedder.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.obs_embedder.patch_embed.proj.bias, 0)

        # Initialize action embedding table:
        if self.condition_on_actions:
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
        p = self.obs_embedder.patch_embed.patch_size[0]
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

        t = self.t_embedder(t)  # (N, D)
        obs = torch.concat(
            [prev_obs, noised_obs.unsqueeze((1))], dim=1
        )  # (N, steps + 1, C, H, W)
        obs = self.obs_embedder(obs)  # (N * (steps + 1), T, D)
        if self.condition_on_actions:
            prev_act = self.act_embedder(prev_act)  # (N, steps, D)
        else:
            prev_act = None

        for block in self.blocks:
            obs = block(obs, t, prev_obs, prev_act)  # (N * (steps + 1), T, D)
        obs = einops.rearrange(
            obs, "(N steps) T M -> N steps T M", N=t.shape[0], steps=self.seq_len
        )
        noised_obs = obs[:, -1]  # (N, patch_size ** 2 * out_channels)
        noised_obs = self.final_layer(
            noised_obs, t, prev_obs, prev_act
        )  # (N * (steps + 1), T, patch_size ** 2 * out_channels)
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
