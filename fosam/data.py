"""Gaze-conditioned LVIS samples written by tools/preprocess_lvis.py.

One sample is one object instance with a gaze point drawn inside it.  The mask is stored
as a PNG of the size of the image; the image itself is read from the COCO 2017 folders.
Everything else about the sample is in the file name:

    <category>_c<LVIS category id>_k<class index>_a<annotation index>_<COCO image id>_
    <gaze row>x<gaze col>_<pad left>x<right>x<top>x<bottom>_<H>x<W>.png

The paddings centre the image on the square canvas the model works on, and the gaze point
is given on that canvas.
"""
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

COCO_SPLITS = ("train2017", "val2017", "test2017")


class GazeLVIS(torch.utils.data.Dataset):
    """
    Returns
        image  [3, S, S]  RGB in [0, 1], zero-padded to the S x S canvas
        gaze   [2]        gaze point in canvas pixels, (x, y)
        mask   [1, S, S]  binary mask of the gazed object
        label  []         class index of the object
        index  []         position of the sample in the dataset
    """

    def __init__(self, sample_dir: str, coco_root: str, canvas: int = 640):
        self.sample_dir = sample_dir
        self.coco_root = coco_root
        self.canvas = canvas
        self.fnames = sorted(f for f in os.listdir(sample_dir) if f.endswith(".png"))
        if not self.fnames:
            raise FileNotFoundError(f"no samples (*.png) in {sample_dir}")
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.fnames)

    def _image_path(self, image_id: str) -> str:
        for split in COCO_SPLITS:
            path = os.path.join(self.coco_root, split, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{image_id}.jpg not found under {self.coco_root}")

    def __getitem__(self, i):
        fname = self.fnames[i]
        try:
            _, _, kid, _, image_id, gaze, paddings, _ = fname.split(".")[0].split("_")
            pad = [int(v) for v in paddings.split("x")]           # left, right, top, bottom
            row, col = [int(v) for v in gaze.split("x")]

            mask = self.to_tensor(Image.open(os.path.join(self.sample_dir, fname)).convert("L"))
            image = self.to_tensor(Image.open(self._image_path(image_id)).convert("RGB"))
            mask = F.pad((mask > 0.5).to(torch.float32), pad)
            image = F.pad(image, pad)
            if mask.shape[-2:] != (self.canvas, self.canvas) or image.shape[-2:] != mask.shape[-2:]:
                raise ValueError(f"unexpected shape {tuple(image.shape)} {tuple(mask.shape)}")
        except Exception as e:                                    # skip the odd unreadable sample
            print(f"[dataset] skipping {fname}: {e}", flush=True)
            return self[(i + 1) % len(self)]
        return (image, torch.tensor([col, row], dtype=torch.float32), mask,
                torch.tensor(int(kid[1:]), dtype=torch.int64), i)
