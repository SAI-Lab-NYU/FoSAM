"""Count the FLOPs of one FoSAM forward pass at a fixed number of selected tokens.

    python tools/count_flops.py --variant small --tokens 100 200 1600

1600 tokens is the whole 40 x 40 grid of a 640 x 640 image, i.e. no token selection.
"""
import argparse
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fosam import build_foveated_sam


def gflops(fn) -> float:
    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="small", choices=["small", "tiny"])
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--tokens", type=int, nargs="+", default=[100, 200])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.rand(1, 3, args.img_size, args.img_size, device=device)
    gaze = torch.full((1, 2), args.img_size / 2, device=device)
    print(f"FoSAM-{args.variant}, {args.img_size} x {args.img_size} input")
    print(f"{'tokens':>7} {'selector':>9} {'encoder':>9} {'decoder':>9} {'total GFLOPs':>13}")
    for k in args.tokens:
        # A threshold at the peak keeps nothing, so exactly min_tokens = k tokens are selected.
        model = build_foveated_sam(args.variant, args.img_size, cut_ratio=1.0, min_tokens=k).to(device).eval()
        enc = model.image_encoder
        with torch.no_grad():
            x = model.preprocess(image)
            total = gflops(lambda: model(image, gaze))
            encoder = gflops(lambda: enc(x, gaze / args.img_size))
            tokens = enc.tokenize(x)
            tokens_ds = enc.tokenize(torch.nn.functional.avg_pool2d(x, enc.selector_downsample))
            selector = gflops(lambda: enc.tokenize(torch.nn.functional.avg_pool2d(x, enc.selector_downsample)))
            selector += gflops(lambda: enc.token_selector(tokens, tokens_ds, gaze / args.img_size, 1.0, k))
        assert int(enc.token_selector.token_counts[0]) == k
        print(f"{k:7d} {selector:9.2f} {encoder - selector:9.2f} {total - encoder:9.2f} {total:13.2f}")
    print("selector: low-resolution tokenisation, TIEncoder and Gaussian predictor; "
          "encoder: patch embedding, ViT blocks on the selected tokens and neck.")


if __name__ == "__main__":
    main()
