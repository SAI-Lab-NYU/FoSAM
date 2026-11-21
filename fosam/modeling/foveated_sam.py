# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (licenses/LICENSE-EfficientSAM).
# This file comes from EfficientSAM (https://github.com/yformer/EfficientSAM) and was
# modified for FoSAM; see NOTICE.

# FoSAM change: the model takes a gaze point, encodes only the tokens selected around it
# and decodes one mask.

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, PromptEncoder
from .two_way_transformer import TwoWayTransformer

# name -> (patch embedding dimension, attention heads) of the EfficientSAM encoder
VARIANTS = {"small": (384, 6), "tiny": (192, 3)}

# EfficientSAM predicts four masks per prompt.  FoSAM keeps this one.
_ESAM_MASK_INDEX = 1


class FoveatedSam(nn.Module):
    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False)

    @property
    def token_selector(self):
        return self.image_encoder.token_selector

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Resize to the model input size and normalize pixel values."""
        size = self.image_encoder.img_size
        if x.shape[2] != size or x.shape[3] != size:
            x = F.interpolate(x, (size, size), mode="bilinear")
        return (x - self.pixel_mean) / self.pixel_std

    def forward(
            self,
            images: torch.Tensor,
            gaze: torch.Tensor,
            gt_mask: Optional[torch.Tensor] = None,
            gt_class: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            images: [B, 3, H, W] RGB in [0, 1].
            gaze: [B, 2] gaze point in pixels of the input image, as (x, y).
            gt_mask: [B, 1, H, W] binary mask of the gazed object. Training only: with it the
                token selector losses are computed.
            gt_class: [B] category index of the gazed object. Training only.
        Returns:
            masks: [B, 1, H, W] mask logits (the mask is `masks > 0`)
            iou_pred: [B, 1] predicted mask quality
            loss_cls, loss_nll: token selector losses, None without ground truth
        """
        B, _, H, W = images.shape
        scale = torch.tensor([W, H], dtype=torch.float32, device=images.device)
        gaze = gaze.to(device=images.device, dtype=torch.float32)

        image_embeddings, loss_cls, loss_nll = self.image_encoder(
            self.preprocess(images), gaze / scale, gt_mask, gt_class)

        # The gaze point is the (single, positive) point prompt of the SAM decoder.
        size = self.image_encoder.img_size
        points = (gaze * size / scale).view(B, 1, 2)
        labels = torch.ones(B, 1, device=images.device)
        sparse_embeddings = self.prompt_encoder(points, labels)
        sparse_embeddings = sparse_embeddings.view(B, 1, sparse_embeddings.shape[1], sparse_embeddings.shape[2])

        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=False,
        )
        masks = F.interpolate(low_res_masks, (H, W), mode="bicubic")
        return {"masks": masks, "iou_pred": iou_pred, "loss_cls": loss_cls, "loss_nll": loss_nll}


def build_foveated_sam(
        variant: str = "small",
        img_size: int = 640,
        num_classes: int = 51,
        cut_ratio: float = 0.25,
        min_tokens: int = 50,
        selector_downsample: int = 4,
        selector_depth: int = 2,
) -> FoveatedSam:
    """
    Args:
        variant: "small" (FoSAM-S, on EfficientSAM-S) or "tiny" (FoSAM-T, on EfficientSAM-Ti).
        img_size: input resolution; the token grid is img_size / 16.
        num_classes: categories of the auxiliary classification loss (training only).
        cut_ratio: Token Importance Score threshold relative to the peak (tau in Eq. 1).
        min_tokens: minimum number of tokens kept (epsilon).
        selector_downsample: downsampling ratio of the image given to the TIEncoder.
        selector_depth: number of TIEncoder layers.
    """
    patch_embed_dim, num_heads = VARIANTS[variant]
    activation = nn.GELU
    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        patch_embed_dim=patch_embed_dim,
        normalization_type="layer_norm",
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        neck_dims=[256, 256],
        act_layer=activation,
        cut_ratio=cut_ratio,
        min_tokens=min_tokens,
        num_classes=num_classes,
        selector_downsample=selector_downsample,
        selector_depth=selector_depth,
    )
    embed_size = image_encoder.image_embedding_size
    dim = image_encoder.transformer_output_dim
    return FoveatedSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=dim,
            image_embedding_size=(embed_size, embed_size),
            input_image_size=(img_size, img_size),
        ),
        mask_decoder=MaskDecoder(
            transformer_dim=dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=dim,
                num_heads=8,
                mlp_dim=2048,
                activation=activation,
                normalize_before_activation=False,
            ),
            num_multimask_outputs=1,
            activation=activation,
            normalization_type="layer_norm",
            normalize_before_activation=False,
            iou_head_depth=2,
            iou_head_hidden_dim=256,
            upscaling_layer_dims=[64, 32],
        ),
    )


def load_efficient_sam_weights(model: FoveatedSam, checkpoint: str) -> None:
    """Initialise FoSAM from an official EfficientSAM checkpoint (efficient_sam_vits.pt / vitt.pt).

    The image encoder, prompt encoder and mask decoder are taken over, with the decoder
    reduced to a single mask output.  The token selector has no pretrained weights.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    k = _ESAM_MASK_INDEX
    out = {}
    for name, value in state.items():
        if name.startswith("mask_decoder.output_hypernetworks_mlps."):
            head, rest = name[len("mask_decoder.output_hypernetworks_mlps."):].split(".", 1)
            if int(head) != k:
                continue
            name = f"mask_decoder.output_hypernetworks_mlps.0.{rest}"
        elif name in ("mask_decoder.mask_tokens.weight",
                      "mask_decoder.iou_prediction_head.fc.weight",
                      "mask_decoder.iou_prediction_head.fc.bias"):
            value = value[k:k + 1]
        out[name] = value
    missing, unexpected = model.load_state_dict(out, strict=False)
    not_pretrained = ("image_encoder.token_selector.", "mask_decoder.cls_token.")
    missing = [m for m in missing if not m.startswith(not_pretrained)]
    if missing or unexpected:
        raise RuntimeError(f"{checkpoint} does not match the model: "
                           f"missing {missing}, unexpected {list(unexpected)}")


def load_checkpoint(model: FoveatedSam, checkpoint: str) -> dict:
    """Load a checkpoint written by scripts/train_fosam.py. Returns the checkpoint dict."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
