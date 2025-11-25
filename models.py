import math
import random
import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from util.rope import RotaryPositionEmbedding2D, PositionGetter
from util.generate_mask import generate_connected_masks
from util.misc import broadcast


# ---------------------------
# Utilities: PatchEmbed
# ---------------------------
class PatchEmbed(nn.Module):
    """Image to patch tokens"""
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/ps, W/ps)
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, E)
        return x, (H, W)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------
# Alternative Attention
# ---------------------------
class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.qk_norm = qk_norm

    def forward(self, view, view_pos):
        B, L, C = view.shape

        qkv = self.qkv(view).reshape(B, L, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = qkv.unbind(2)
        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        q = self.rope(q, view_pos)
        k = self.rope(k, view_pos)
        view = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        view = view.transpose(1, 2).reshape(B, L, C)
        view = self.proj(view)
        view = self.proj_drop(view)
        return view

class AABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.frame_attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.frame_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.global_atten = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm4 = norm_layer(dim)
        self.global_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, view, view_pos):
        B, V, L, C = view.shape
        view = view.view(B*V, L, C)
        view_pos = view_pos.view(B*V, L, 2)
        view = view + self.drop_path(self.frame_attn(self.norm1(view), view_pos))
        view = view + self.drop_path(self.frame_mlp(self.norm2(view)))

        view = view.view(B, V*L, C)
        view_pos = view_pos.view(B, V*L, 2)
        view = view + self.drop_path(self.global_atten(self.norm3(view), view_pos))
        view = view + self.drop_path(self.global_mlp(self.norm4(view)))

        view = view.view(B, V, L, C)
        return view


# ---------------------------
# Full Multi-view Croco
# ---------------------------
class MultiViewCroco(nn.Module):
    def __init__(
        self,
        block_type=AABlock,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        rope_freq=100,
        enable_checkpoint=False,
        num_ref=-1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.depth = depth
        self.num_register_tokens = num_register_tokens
        self.enable_checkpoint = enable_checkpoint
        self.num_ref = num_ref

        # register tokens per view (learnable)
        self.register_tokens = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))

        self.mask_placeholder = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()

        # create transformer blocks
        self.blocks = nn.ModuleList([
            block_type(embed_dim, num_heads, mlp_ratio, qkv_bias=True, rope=self.rope)
            for _ in range(depth)
        ])

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans * 2, bias=True) # decoder to patch

    def random_masking(self, x, Hp, Wp, mask_mode, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, V, L, C = x.shape  # batch, length, dim

        i = random.randint(0, len(mask_mode)-1)
        mode, ratio = mask_mode[i], mask_ratio[i]

        if self.num_ref >= 0:
            num_ref = self.num_ref
        else:
            num_ref = random.randint(V//4,V//2)

        mask = generate_connected_masks(B, V-num_ref, Hp, Wp, ratio=ratio,
                                        device=x.device, dtype=torch.float32,
                                        mode=mode)
        mask = mask.flatten(2,3)
        mask = torch.cat((torch.zeros(B,num_ref,L, device=x.device), mask), dim=1)

        x = torch.where(mask.unsqueeze(-1)>0.5, self.mask_placeholder, x)

        return x, mask

    def compute_confloss(self, imgs, pred, conf, mask):
        """
        imgs: [B, V, 3, H, W]
        pred: [B, V, L, p*p*3]
        conf: [B, V, L, p*p*3]
        mask: [B, V, L], 0=keep, 1=remove
        """
        target = F.pixel_unshuffle(imgs, downscale_factor=self.patch_size)
        target = target.permute(0,1,3,4,2).flatten(2, 3)

        mse = (pred - target) ** 2
        loss = ((conf+0.1) * mse).mean(dim=-1)  # [N, L], mean loss per patch

        per_instance_loss = (loss * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2))

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        conf_reg = -torch.log(conf + 1e-6).mean(dim=-1)
        conf_reg = (conf_reg * mask).mean()

        return {
            'mse': (mse.detach().mean(dim=-1) * mask).sum() / mask.sum(),
            'conf_reg': conf_reg,
            'loss': loss + 0.1 * conf_reg,
        }, per_instance_loss.detach()

    def forward(self, images: torch.Tensor, random_aspect_ratio=False, random_num_views=False,
                mask_mode=('random',), mask_ratio=(0.9,)):
        """
        images: (B, V, C, H, W)
        returns: per-view token lists
        """
        if random_aspect_ratio and images.shape[-1] == 512:
            # randomly select an aspect ratio from 1:1, 4:3, 16:9.
            H, W = random.choice([(512, 288), (512, 384), (512, 512), (288, 512), (384, 512), (512, 512)])

            # synchronize ratio across GPUs to avoid waiting.
            H, W = broadcast((H, W))
            margin_H, margin_W = (512-H) // 2, (512-W) // 2
            images = images[..., margin_H:512-margin_H, margin_W:512-margin_W]
        if random_num_views:
            V = random.choice([1,2,4,8])
            V, = broadcast((V,))
            total_views = images.shape[0] * images.shape[1]
            images = images.view(total_views//V, V, *images.shape[2:])
        B, V, _, H, W = images.shape

        x, (Hp, Wp) = self.patch_embed(images.view(B*V,3,H,W))

        x = x.view(B, V, Hp*Wp, -1) # (B, N_patches, E)
        x, mask = self.random_masking(x, Hp, Wp, mask_mode=mask_mode, mask_ratio=mask_ratio)
        x = torch.cat((self.register_tokens.expand(B, V, -1, -1), x), dim=2)

        pos = self.position_getter(B * V, Hp, Wp, device=images.device).view(B, V, Hp*Wp, -1)
        pos = pos + 1
        pos = torch.cat((torch.zeros(B, V, self.num_register_tokens, 2).to(pos), pos), dim=2)

        # run blocks
        for i, blk in enumerate(self.blocks):
            if self.enable_checkpoint:
                x = checkpoint.checkpoint(blk, x, pos)
            else:
                x = blk(x, pos)

        # final norm on concatenated views
        x = x[:,:,self.num_register_tokens:]
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        pred, conf = torch.chunk(x, 2, dim=-1)
        conf = torch.sigmoid(conf)

        # loss_dict, per_instance_loss = self.compute_confloss(images, pred, conf, mask)
        loss_dict, per_instance_loss = self.compute_loss(images, pred, conf, mask)

        pred = F.pixel_shuffle(pred.view(B,V,Hp,Wp,-1).permute(0,1,4,2,3), upscale_factor=self.patch_size)
        conf = F.pixel_shuffle(conf.view(B,V,Hp,Wp,-1).permute(0,1,4,2,3), upscale_factor=self.patch_size)

        return loss_dict, per_instance_loss, pred, conf, mask

    def extract_feature(self, images: torch.Tensor):
        B, V, _, H, W = images.shape
        x, (Hp, Wp) = self.patch_embed(images.view(B*V,3,H,W))

        x = x.view(B, V, Hp*Wp, -1) # (B, N_patches, E)
        x = torch.cat((self.register_tokens.expand(B, V, -1, -1), x), dim=2)

        pos = self.position_getter(B * V, Hp, Wp, device=images.device).view(B, V, Hp*Wp, -1)
        pos = pos + 1
        pos = torch.cat((torch.zeros(B, V, self.num_register_tokens, 2).to(pos), pos), dim=2)

        # run blocks
        saved_xs = []
        for i, blk in enumerate(self.blocks):
            if self.enable_checkpoint:
                x = checkpoint.checkpoint(blk, x, pos)
            else:
                x = blk(x, pos)
            saved_xs.append(x[:,:,self.num_register_tokens:].view(B, V, Hp, Wp, -1))

        return saved_xs


def small(**kwargs):
    return MultiViewCroco(embed_dim=384, depth=6, num_heads=6, **kwargs)

def base(**kwargs):
    return MultiViewCroco(embed_dim=768, depth=6, num_heads=12, **kwargs)

def large(**kwargs):
    return MultiViewCroco(embed_dim=1024, depth=12, num_heads=16, **kwargs)

def huge(**kwargs):
    return MultiViewCroco(embed_dim=1280, depth=16, num_heads=16, **kwargs)
