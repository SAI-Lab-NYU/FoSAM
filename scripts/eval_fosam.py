"""Evaluate gaze-prompted segmentation on gaze-conditioned LVIS samples.

Reports the IoU with the ground-truth mask of the gazed object and the number of image
tokens that go through the encoder, overall and by object size.  Run from the repository
root:

    # FoSAM
    python scripts/eval_fosam.py --ckpt checkpoints/my_run/fosam_latest.pth --val_dir <lvis>/test --coco_root <coco2017>
    # SAM ViT-B, prompted with the same gaze point (needs `pip install segment-anything`)
    python scripts/eval_fosam.py --model sam --sam_ckpt sam_vit_b_01ec64.pth --val_dir <lvis>/test --coco_root <coco2017>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fosam import build_foveated_sam
from fosam.data import GazeLVIS
from fosam.metrics import mask_iou

SIZE_BINS = [0, 32, 64, 96, 128, 192, 256, 384, 10 ** 9]   # sqrt(object area) in pixels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="fosam", choices=["fosam", "sam"])
    p.add_argument("--ckpt", help="FoSAM checkpoint from scripts/train_fosam.py")
    p.add_argument("--sam_ckpt", help="SAM ViT-B checkpoint (sam_vit_b_01ec64.pth), for --model sam")
    p.add_argument("--val_dir", required=True, help="evaluation samples from tools/preprocess_lvis.py")
    p.add_argument("--coco_root", required=True, help="folder with train2017/ and val2017/")
    p.add_argument("--cut_ratio", type=float, help="score threshold tau; default: the training value")
    p.add_argument("--min_tokens", type=int, help="minimum number of tokens; default: the training value")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0, help="evaluate the first N samples only")
    p.add_argument("--latency", type=int, default=0,
                   help="also time N single-image forward passes on this GPU (FoSAM only)")
    p.add_argument("--out", default="", help="result file; default results/<model>_<samples>.json")
    return p.parse_args()


def load_fosam(args, device):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    cut_ratio = cfg["cut_ratio"] if args.cut_ratio is None else args.cut_ratio
    min_tokens = cfg["min_tokens"] if args.min_tokens is None else args.min_tokens
    model = build_foveated_sam(cfg["variant"], cfg["img_size"], cfg["num_classes"], cut_ratio, min_tokens)
    model.load_state_dict(ckpt["model"])
    print(f"FoSAM-{cfg['variant']} from {args.ckpt} (step {ckpt['step']}), "
          f"cut_ratio {cut_ratio}, min_tokens {min_tokens}", flush=True)
    return model.to(device).eval(), cfg["img_size"]


@torch.no_grad()
def run_fosam(model, loader, device):
    ious, tokens, areas = [], [], []
    for image, gaze, mask, _, _ in tqdm(loader):
        image, mask = image.to(device), mask.to(device)
        out = model(image, gaze.to(device))
        ious.append(mask_iou(out["masks"] > 0, mask > 0.5).cpu())
        tokens.append(model.token_selector.token_counts.cpu())
        areas.append(mask.flatten(1).sum(1).cpu())
    return torch.cat(ious).numpy(), torch.cat(tokens).numpy().astype(float), torch.cat(areas).numpy()


@torch.no_grad()
def run_sam(args, loader, device):
    """SAM ViT-B with the gaze point as a single positive point prompt.

    SAM returns three masks per prompt; the one its own IoU head ranks first is scored.
    """
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt).to(device).eval()
    size = sam.image_encoder.img_size
    resize = ResizeLongestSide(size)
    ious, areas = [], []
    for image, gaze, mask, _, _ in tqdm(loader):
        B, _, H, W = image.shape
        mask = mask.to(device)
        emb = sam.image_encoder(sam.preprocess(resize.apply_image_torch(image.to(device) * 255.0)))
        points = resize.apply_coords_torch(gaze.to(device).view(B, 1, 2), (H, W))
        sparse, dense = sam.prompt_encoder(points=(points, torch.ones(B, 1, device=device)),
                                           boxes=None, masks=None)
        low_res, scores = [], []
        for i in range(B):                      # SAM's decoder takes one image at a time
            m, s = sam.mask_decoder(image_embeddings=emb[i:i + 1],
                                    image_pe=sam.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse[i:i + 1],
                                    dense_prompt_embeddings=dense[i:i + 1], multimask_output=True)
            low_res.append(m)
            scores.append(s)
        low_res, scores = torch.cat(low_res), torch.cat(scores)
        pred = sam.postprocess_masks(low_res, (size, size), (H, W)) > sam.mask_threshold
        pred = pred[torch.arange(B), scores.argmax(1)]
        ious.append(mask_iou(pred, mask[:, 0] > 0.5).cpu())
        areas.append(mask.flatten(1).sum(1).cpu())
    ious = torch.cat(ious).numpy()
    tokens = np.full(len(ious), float((size // 16) ** 2))
    return ious, tokens, torch.cat(areas).numpy()


@torch.no_grad()
def measure_latency(model, dataset, n, device):
    times = []
    for i in range(min(n, len(dataset)) + 10):
        image, gaze, _, _, _ = dataset[i % len(dataset)]
        image, gaze = image[None].to(device), gaze[None].to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(image, gaze)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times[10:]))            # the first passes warm up the GPU


def main():
    args = parse_args()
    device = torch.device("cuda")
    if args.model == "fosam":
        if not args.ckpt:
            sys.exit("--ckpt is required for --model fosam")
        model, canvas = load_fosam(args, device)
    else:
        if not args.sam_ckpt:
            sys.exit("--sam_ckpt is required for --model sam")
        canvas = 640

    dataset = GazeLVIS(args.val_dir, args.coco_root, canvas=canvas)
    if args.max_samples:
        dataset.fnames = dataset.fnames[:args.max_samples]
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    print(f"{len(dataset)} samples from {args.val_dir}", flush=True)

    if args.model == "fosam":
        ious, tokens, areas = run_fosam(model, loader, device)
    else:
        ious, tokens, areas = run_sam(args, loader, device)

    result = {"model": args.model, "ckpt": args.ckpt or args.sam_ckpt, "val_dir": args.val_dir,
              "samples": len(ious), "iou": float(ious.mean()), "tokens": float(tokens.mean()), "by_size": []}
    print(f"\nIoU {ious.mean():.4f} | tokens: mean {tokens.mean():.1f}, median {np.median(tokens):.0f}, "
          f"min {tokens.min():.0f}, max {tokens.max():.0f}")
    print(f"{'sqrt(area) px':>14} {'samples':>8} {'IoU':>7} {'tokens':>7}")
    size = np.sqrt(areas)
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        sel = (size >= lo) & (size < hi)
        if sel.any():
            label = f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}"
            print(f"{label:>14} {sel.sum():8d} {ious[sel].mean():7.4f} {tokens[sel].mean():7.1f}")
            result["by_size"].append({"range": label, "samples": int(sel.sum()),
                                      "iou": float(ious[sel].mean()), "tokens": float(tokens[sel].mean())})

    if args.latency and args.model == "fosam":
        result["latency_ms"] = measure_latency(model, dataset, args.latency, device)
        print(f"latency: {result['latency_ms']:.1f} ms per image "
              f"(batch 1, fp32, {torch.cuda.get_device_name(0)})")

    result["per_sample"] = [{"file": f, "iou": float(i), "tokens": float(t)}
                            for f, i, t in zip(dataset.fnames, ious, tokens)]
    out = args.out or os.path.join("results", f"{args.model}_{len(ious)}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
