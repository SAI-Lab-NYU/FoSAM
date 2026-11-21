"""Build gaze-conditioned samples from LVIS.

LVIS annotates every object of an image; FoSAM segments only the one the user looks at.
A sample is therefore one object instance with a gaze point drawn uniformly inside its
mask.  Samples are drawn class-balanced over 50 LVIS categories, and each one is written
as a PNG mask whose name carries the gaze point (see fosam/data.py).  Images are not
copied: they are read from the COCO 2017 folders at training time.

    python tools/preprocess_lvis.py --lvis_json <lvis_v1_train.json> --out <lvis>/train --num_samples 120000
    python tools/preprocess_lvis.py --lvis_json <lvis_v1_val.json> --out <lvis>/valid --num_samples 24000 --seed 1
    python tools/preprocess_lvis.py --lvis_json <lvis_v1_val.json> --out <lvis>/test --num_samples 8000 --seed 2
"""
import argparse
import json
import os

import numpy as np
import skimage.draw
from PIL import Image
from tqdm import trange

# LVIS v1 category ids of the 50 object classes; the position in this list is the class
# index used by the classification loss.
CATEGORY_IDS = [77, 421, 816, 1115, 225, 173, 1052, 982, 1021, 1050, 76, 110, 361, 1097, 631, 422,
                378, 687, 1077, 912, 183, 445, 1019, 818, 496, 817, 3, 698, 703, 766, 1123, 19, 1202,
                1071, 1042, 694, 296, 569, 1117, 1064, 961, 350, 461, 169, 1043, 392, 36, 90, 181, 139]


def wrap_name(name: str) -> str:
    """Category name without the characters that separate the fields of a file name."""
    return "-".join(part for part in name.translate(str.maketrans("_ ().", "-----")).split("-") if part)


def rasterize(polygons, height: int, width: int) -> np.ndarray:
    """LVIS polygons [x0, y0, x1, y1, ...] -> boolean mask [H, W]."""
    mask = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        polygon = np.round(np.array(polygon)).astype(np.int32)
        cols = np.clip(polygon[0::2], 0, width - 1)
        rows = np.clip(polygon[1::2], 0, height - 1)
        rr, cc = skimage.draw.polygon(rows, cols)
        mask[rr, cc] = True
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lvis_json", required=True, help="lvis_v1_train.json or lvis_v1_val.json")
    p.add_argument("--out", required=True, help="output folder for the samples")
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--canvas", type=int, default=640, help="images are zero-padded to canvas x canvas")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"reading {args.lvis_json}", flush=True)
    with open(args.lvis_json) as f:
        lvis = json.load(f)
    images = {img["id"]: img for img in lvis["images"]}
    names = {cat["id"]: cat["name"] for cat in lvis["categories"]}
    class_index = {cid: k for k, cid in enumerate(CATEGORY_IDS)}
    by_class = {cid: [] for cid in CATEGORY_IDS}
    for a, ann in enumerate(lvis["annotations"]):
        if ann["category_id"] in by_class:
            by_class[ann["category_id"]].append(a)
    categories = [cid for cid in CATEGORY_IDS if by_class[cid]]
    print(f"{sum(len(v) for v in by_class.values())} instances of {len(categories)} categories", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    written = 0
    for n in trange(args.num_samples):
        cid = categories[n % len(categories)]
        a = int(rng.choice(by_class[cid]))
        ann = lvis["annotations"][a]
        img = images[ann["image_id"]]
        H, W = img["height"], img["width"]
        if H > args.canvas or W > args.canvas:
            continue
        mask = rasterize(ann["segmentation"], H, W)
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            continue
        g = int(rng.integers(len(rows)))                       # gaze point, uniform over the mask

        pad_left = (args.canvas - W) // 2
        pad_right = args.canvas - W - pad_left
        pad_top = (args.canvas - H) // 2
        pad_bottom = args.canvas - H - pad_top
        image_id = os.path.splitext(os.path.basename(img["coco_url"]))[0]
        fname = (f"{wrap_name(names[cid])}_c{cid}_k{class_index[cid]}_a{a}_{image_id}_"
                 f"{rows[g] + pad_top}x{cols[g] + pad_left}_"
                 f"{pad_left}x{pad_right}x{pad_top}x{pad_bottom}_{H}x{W}.png")
        Image.fromarray(mask).save(os.path.join(args.out, fname))
        written += 1
    print(f"{written} samples written to {args.out}")


if __name__ == "__main__":
    main()
