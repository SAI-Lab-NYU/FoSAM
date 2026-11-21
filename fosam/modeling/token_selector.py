"""Gaze-guided token selection (Sections 3.2.1 - 3.2.3 of the paper).

The selector looks at a 4x-downsampled copy of the frame, encodes it with a small
ViT (the Token Importance Encoder, TIEncoder), reads the token under the gaze point
and predicts from it the shape of a 2D Gaussian centred on that token.  The Gaussian
is the Token Importance Score map over the full-resolution token grid; the tokens
whose score is at least `cut_ratio` of the peak are kept (Eq. 1), with a floor of
`min_tokens`.

Training adds two losses on the TIEncoder outputs (Eq. 2 and 3): every token inside
the ground-truth mask must predict a Gaussian that covers the mask (negative
log-likelihood), and must be classified into the category of the object.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Numerical guards for the Gaussian parameters.  The likelihood contains
# log(sigma_x * sigma_y * sqrt(1 - rho^2)): rho -> +-1 makes 1 - rho^2 cancel to 0 in
# fp32 and sigma -> 0 sends the log to -inf, and both are reachable in a single
# optimizer step.  SIGMA_MIN is well below one token (1/40 of the image at 640 px),
# so the floor does not change the token count.
SIGMA_MIN = 0.01
RHO_MAX = 0.90


class TIEncoder(nn.Module):
    """Token Importance Encoder: a shallow ViT over the low-resolution tokens."""

    def __init__(self, dim: int, depth: int = 2, heads: int = 3, mlp_dim: Optional[int] = None):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim or dim * 4,
            batch_first=True,
            activation="gelu",
        )
        # enable_nested_tensor only matters with a padding mask, which is never passed here
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, h, w] -> [B, h, w, C]"""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(self.transformer(x))
        return x.reshape(B, H, W, C)


def select_tokens(
    tokens: torch.Tensor, scores: torch.Tensor, cut_ratio: float, min_tokens: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive token sampling (Eq. 1).

    A token is kept when its score exceeds `cut_ratio` times the largest score of the
    image.  If fewer than `min_tokens` pass, the `min_tokens` highest-scoring tokens are
    kept instead.

    Args:
        tokens: [B, C, H, W] token map.
        scores: [B, H, W] Token Importance Scores.
    Returns:
        selected: [B, C, K] kept tokens, K being the largest count in the batch; images
            that keep fewer tokens are zero-padded.
        indices: [B, K] flat position (row * W + col) of every kept token, -1 for padding.
        counts: [B] number of tokens kept per image.
    """
    B, C, H, W = tokens.shape
    scores = scores.reshape(B, -1)
    keep = scores > scores.max(dim=1, keepdim=True).values * cut_ratio
    counts = keep.sum(dim=1).int()

    for b in range(B):
        if counts[b] < min_tokens:
            top = torch.sort(scores[b], descending=True).indices[:min_tokens]
            keep[b] = False
            keep[b, top] = True
            counts[b] = min_tokens

    K = int(counts.max())
    tokens = tokens.reshape(B, C, -1)
    selected = tokens.new_zeros(B, C, K)
    indices = torch.full((B, K), -1, dtype=torch.long, device=tokens.device)
    for b in range(B):
        idx = torch.nonzero(keep[b]).squeeze(1)
        selected[b, :, : idx.numel()] = tokens[b, :, idx]
        indices[b, : idx.numel()] = idx
    return selected, indices, counts


class TokenSelector(nn.Module):
    """TIEncoder + Gaussian predictor + adaptive sampling, with the training losses."""

    def __init__(self, dim: int, num_classes: int, depth: int = 2, heads: int = 3,
                 temperature: float = 0.5, mask_cover: float = 0.2):
        """
        Args:
            dim: token dimension (the patch embedding dimension of the image encoder).
            num_classes: number of object categories for the auxiliary classification loss.
            depth, heads: size of the TIEncoder.
            temperature: softmax temperature of the classification loss.
            mask_cover: a low-resolution token counts as inside the object when the mask
                covers more than this fraction of it.
        """
        super().__init__()
        self.tiencoder = TIEncoder(dim=dim, depth=depth, heads=heads)
        # Outputs (d_mu_x, d_mu_y, sigma_x, sigma_y, rho).  The two mean offsets are not
        # used: the Gaussian is centred on the token it is predicted from.
        self.gaussian_head = nn.Linear(dim, 5)
        self.cls_head = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.temperature = temperature
        self.mask_cover = mask_cover

        # Filled by every forward pass, for logging and analysis.
        self.scores = None        # [B, H, W] Token Importance Score map
        self.token_counts = None  # [B] tokens kept
        self.sigma_x = None       # [B] Gaussian parameters read at the gaze token
        self.sigma_y = None
        self.rho = None

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_lowres: torch.Tensor,
        gaze: torch.Tensor,
        cut_ratio: float,
        min_tokens: int,
        gt_mask: Optional[torch.Tensor] = None,
        gt_class: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            tokens: [B, C, H, W] patch tokens of the full-resolution image.
            tokens_lowres: [B, C, h, w] patch tokens of the downsampled image.
            gaze: [B, 2] gaze point as (x, y), normalised to [0, 1].
            gt_mask: [B, 1, H_img, W_img] binary mask of the gazed object (training).
            gt_class: [B] category index of the gazed object (training).
        Returns:
            selected [B, C, K], indices [B, K], and the two losses (None without ground truth).
        """
        B, _, H, W = tokens.shape
        emb = self.tiencoder(tokens_lowres)                      # [B, h, w, C]
        h, w = emb.shape[1], emb.shape[2]
        batch = torch.arange(B, device=emb.device)

        gaze_col = torch.clamp((gaze[:, 0] * w).long(), 0, w - 1)
        gaze_row = torch.clamp((gaze[:, 1] * h).long(), 0, h - 1)

        params = self.gaussian_head(self.dropout(emb.reshape(-1, emb.shape[-1]))).view(B, h, w, 5)
        sigma_x = F.softplus(params[..., 2] + 1e-6) + SIGMA_MIN   # [B, h, w], one Gaussian per token
        sigma_y = F.softplus(params[..., 3] + 1e-6) + SIGMA_MIN
        rho = torch.tanh(params[..., 4]) * RHO_MAX
        ys = torch.linspace(0, 1, h, device=emb.device)           # token centres, normalised
        xs = torch.linspace(0, 1, w, device=emb.device)

        # Score map from the Gaussian of the gaze token, evaluated on the low-resolution
        # grid and upsampled to the full token grid.
        sx = sigma_x[batch, gaze_row, gaze_col].view(B, 1, 1)
        sy = sigma_y[batch, gaze_row, gaze_col].view(B, 1, 1)
        r = rho[batch, gaze_row, gaze_col].view(B, 1, 1)
        zx = (xs.view(1, 1, w) - xs[gaze_col].view(B, 1, 1)) / sx
        zy = (ys.view(1, h, 1) - ys[gaze_row].view(B, 1, 1)) / sy
        norm = 1 / (2 * torch.pi * sx * sy * torch.sqrt(1 - r ** 2))
        scores = norm * torch.exp(-(zx ** 2 - 2 * r * zx * zy + zy ** 2) / (2 * (1 - r ** 2)))
        scores = F.interpolate(scores.unsqueeze(1), size=(H, W), mode="bilinear",
                               align_corners=False).squeeze(1)

        selected, indices, counts = select_tokens(tokens, scores.detach(), cut_ratio, min_tokens)

        self.scores = scores.detach()
        self.token_counts = counts.detach()
        self.sigma_x, self.sigma_y, self.rho = sx.detach().view(B), sy.detach().view(B), r.detach().view(B)

        if gt_mask is None:
            return selected, indices, None, None

        # Fraction of every low-resolution token covered by the object.
        cover = F.avg_pool2d(gt_mask.float(), kernel_size=(gt_mask.shape[2] // h, gt_mask.shape[3] // w))
        cover = cover[:, 0]                                       # [B, h, w]
        loss_nll, loss_cls = [], []
        for b in range(B):
            inside = torch.nonzero(cover[b] > self.mask_cover)    # tokens that predict (reference tokens)
            if inside.numel() == 0:                               # tiny object: fall back to the gaze token
                inside = torch.stack([gaze_row[b], gaze_col[b]]).view(1, 2)
            rows, cols = inside[:, 0], inside[:, 1]

            logits = self.cls_head(emb[b, rows, cols])            # [N, num_classes]
            loss_cls.append(F.cross_entropy(logits / self.temperature, gt_class[b].expand(logits.shape[0])))

            # Likelihood of every token touched by the object under each reference Gaussian.
            target = torch.nonzero(cover[b])
            ty = (target[:, 0].float() / h).view(1, -1)
            tx = (target[:, 1].float() / w).view(1, -1)
            mx, my = xs[cols].view(-1, 1), ys[rows].view(-1, 1)
            sgx, sgy = sigma_x[b, rows, cols].view(-1, 1), sigma_y[b, rows, cols].view(-1, 1)
            rr = rho[b, rows, cols].view(-1, 1)
            log_norm = torch.log(2 * math.pi * sgx * sgy * torch.sqrt(1 - rr ** 2))
            z = ((tx - mx) ** 2 / sgx ** 2
                 - 2 * rr * (tx - mx) * (ty - my) / (sgx * sgy)
                 + (ty - my) ** 2 / sgy ** 2)
            loss_nll.append((log_norm + z / (2 * (1 - rr ** 2))).mean())

        return selected, indices, torch.stack(loss_cls).mean(), torch.stack(loss_nll).mean()
