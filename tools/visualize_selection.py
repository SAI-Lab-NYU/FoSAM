"""Run FoSAM on one image and one gaze point, and draw what it did.

The output shows the image with the tokens that were not selected dimmed, the predicted
mask and its outline, and the gaze point.

    python tools/visualize_selection.py --ckpt checkpoints/my_run/fosam_latest.pth \
        --image photo.jpg --gaze 320 240 --out selection.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fosam import build_foveated_sam


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="FoSAM checkpoint from scripts/train_fosam.py")
    p.add_argument("--image", required=True)
    p.add_argument("--gaze", type=float, nargs=2, required=True, metavar=("X", "Y"),
                   help="gaze point in pixels of the image")
    p.add_argument("--out", default="selection.png")
    p.add_argument("--cut_ratio", type=float, help="score threshold tau; default: the training value")
    p.add_argument("--min_tokens", type=int, help="minimum number of tokens; default: the training value")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    size = cfg["img_size"]
    model = build_foveated_sam(
        cfg["variant"], size, cfg["num_classes"],
        cfg["cut_ratio"] if args.cut_ratio is None else args.cut_ratio,
        cfg["min_tokens"] if args.min_tokens is None else args.min_tokens)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # Longest side to the model input size, then zero-pad to a square (as in training).
    img = Image.open(args.image).convert("RGB")
    scale = size / max(img.size)
    w, h = round(img.width * scale), round(img.height * scale)
    left, top = (size - w) // 2, (size - h) // 2
    x = torch.from_numpy(np.array(img.resize((w, h), Image.BILINEAR))).permute(2, 0, 1).float() / 255
    x = F.pad(x, (left, size - w - left, top, size - h - top))[None].to(device)
    gaze = torch.tensor([[args.gaze[0] * scale + left, args.gaze[1] * scale + top]], device=device)

    with torch.no_grad():
        out = model(x, gaze)
    selector = model.token_selector
    grid = size // 16
    mask = (out["masks"][0, 0] > 0).cpu().numpy()
    keep = selector.scores[0] > selector.scores[0].max() * model.image_encoder.cut_ratio
    if int(keep.sum()) < model.image_encoder.min_tokens:
        keep = torch.zeros_like(keep).flatten()
        keep[selector.scores[0].flatten().topk(model.image_encoder.min_tokens).indices] = True
        keep = keep.view(grid, grid)
    keep = keep.cpu().numpy().repeat(16, 0).repeat(16, 1)
    print(f"{int(selector.token_counts[0])} of {grid * grid} tokens selected, "
          f"mask covers {mask.mean() * 100:.1f} % of the canvas")

    canvas = x[0].permute(1, 2, 0).cpu().numpy()
    canvas = np.where(keep[..., None], canvas, canvas * 0.3)             # dim what the encoder never saw
    color = np.array([0.15, 0.55, 1.0])
    canvas[mask] = canvas[mask] * 0.5 + color * 0.5
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    canvas[edge] = color
    canvas = Image.fromarray((canvas * 255).astype(np.uint8)).crop((left, top, left + w, top + h))
    gx, gy = float(gaze[0, 0]) - left, float(gaze[0, 1]) - top
    draw = ImageDraw.Draw(canvas)
    draw.ellipse([gx - 7, gy - 7, gx + 7, gy + 7], outline=(255, 255, 255), width=2)
    draw.ellipse([gx - 2, gy - 2, gx + 2, gy + 2], fill=(255, 60, 60))
    canvas.save(args.out)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
