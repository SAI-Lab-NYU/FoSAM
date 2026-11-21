import torch


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Dice loss between mask logits and a binary mask, [B, 1, H, W] each (Eq. 4)."""
    prob = torch.sigmoid(logits.float()).flatten(1)
    target = target.float().flatten(1)
    inter = (prob * target).sum(1)
    return (1 - (2 * inter + eps) / (prob.sum(1) + target.sum(1) + eps)).mean()


def mask_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU per sample of two boolean masks [B, ...]."""
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred & target).sum(1).float()
    union = (pred | target).sum(1).float().clamp(min=1)
    return inter / union
