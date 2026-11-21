# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (licenses/LICENSE-EfficientSAM).
# This file comes from EfficientSAM (https://github.com/yformer/EfficientSAM) and was
# modified for FoSAM; see NOTICE.

# FoSAM change: the ViT blocks run on the gaze-selected tokens only.

import math
from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_selector import TokenSelector


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_mask=None):
        """key_mask: [B, N] bool, False for padding tokens that must not be attended to."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            attn = attn.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, key_mask=None):
        x = x + self.attn(self.norm1(x), key_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def get_abs_pos(abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int,
            patch_size: int,
            in_chans: int,
            patch_embed_dim: int,
            normalization_type: str,
            depth: int,
            num_heads: int,
            mlp_ratio: float,
            neck_dims: List[int],
            act_layer: Type[nn.Module],
            cut_ratio: float,
            min_tokens: int,
            num_classes: int,
            selector_downsample: int = 4,
            selector_depth: int = 2,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
            cut_ratio (float): Token Importance Score threshold, relative to the peak (tau in Eq. 1).
            min_tokens (int): Minimum number of tokens kept per image (epsilon in the paper).
            num_classes (int): Number of categories of the auxiliary classification loss.
            selector_downsample (int): Downsampling ratio v of the image given to the TIEncoder.
            selector_depth (int): Number of TIEncoder layers.
        """
        super().__init__()
        self.cut_ratio = cut_ratio
        self.min_tokens = min_tokens
        self.img_size = img_size
        self.selector_downsample = selector_downsample
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.token_selector = TokenSelector(dim=patch_embed_dim, num_classes=num_classes,
                                            depth=selector_depth)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)

        num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True)
            self.blocks.append(vit_block)
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding + positional embedding, [B, 3, H, W] -> [B, C, H/16, W/16]."""
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]])
        return x.permute(0, 3, 1, 2)

    def forward(
            self,
            x: torch.Tensor,
            gaze: torch.Tensor,
            gt_mask: Optional[torch.Tensor] = None,
            gt_class: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: [B, 3, img_size, img_size] normalised image.
            gaze: [B, 2] gaze point (x, y), normalised to [0, 1].
            gt_mask, gt_class: ground truth of the gazed object, for the selector losses.
        Returns:
            image embedding [B, C_out, img_size/16, img_size/16], classification loss, NLL loss
        """
        assert (
                x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        tokens = self.tokenize(x)                                 # [B, C, ts, ts]
        B, C, ts, _ = tokens.shape

        # Low-resolution copy for the TIEncoder, tokenised with the same patch embedding.
        ds_size = self.img_size // self.selector_downsample
        x_ds = F.avg_pool2d(x, kernel_size=self.img_size // ds_size)
        x_ds = F.interpolate(x_ds, size=(ds_size, ds_size), mode="bilinear", align_corners=False)
        tokens_ds = self.tokenize(x_ds)

        selected, indices, loss_cls, loss_nll = self.token_selector(
            tokens, tokens_ds, gaze,
            cut_ratio=self.cut_ratio, min_tokens=self.min_tokens,
            gt_mask=gt_mask, gt_class=gt_class,
        )

        # The ViT sees only the selected tokens.  Images of a batch keep different numbers
        # of tokens; the padding of the shorter ones is masked out of the attention, so the
        # result of an image does not depend on what it is batched with.
        valid = indices >= 0                                      # [B, K]
        key_mask = None if bool(valid.all()) else valid
        x = selected.permute(0, 2, 1)                             # [B, K, C]
        for blk in self.blocks:
            x = blk(x, key_mask)
        x = x.permute(0, 2, 1)                                    # [B, C, K]

        # Put the tokens back at their positions; everything not selected stays zero.
        full = x.new_zeros(B, C, ts * ts)
        for b in range(B):
            full[b].scatter_(1, indices[b, valid[b]].unsqueeze(0).expand(C, -1), x[b][:, valid[b]])
        x = self.neck(full.view(B, C, ts, ts))

        return x, loss_cls, loss_nll
