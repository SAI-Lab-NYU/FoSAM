# FoSAM: Foveated Segment Anything

<p align="center" style="font-size: 24px;">
  IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027
</p>

Code for the WACV 2027 paper "Foveated Segment Anything". FoSAM is a gaze-guided segmentation model for
AR/VR devices built on EfficientSAM: instead of encoding the whole front-view frame, it selects the
image tokens around the object the user is looking at and runs the encoder only on those.

## ⭐ Highlights

- **Segment what the user looks at:** the gaze point from the device's eye tracker picks the instance
  of interest. FoSAM encodes only the tokens that cover it, where SAM encodes the full frame and then
  decodes one object from it.
- **Gaussian Token Importance Scores:** a two-layer Token Importance Encoder (TIEncoder) reads a
  4x-downsampled copy of the frame, and a Gaussian predictor turns the token under the gaze into a 2D
  Gaussian over the token grid. One Gaussian keeps the selection spatially coherent, which SAM's mask
  decoder needs.
- **Adaptive token sampling:** tokens above a fraction of the peak score are kept, so the token count
  follows the size of the object: small objects cost fewer tokens than large ones.
- **Frozen encoder:** the EfficientSAM image encoder is not trained. Only the token selector (3.6 M
  parameters for FoSAM-S) and the mask decoder are; a run takes about half an hour on one GPU.
- **Train it yourself:** `scripts/train_fosam.py` trains FoSAM-S or FoSAM-T from the official
  EfficientSAM weights. No FoSAM weights are shipped.
- **Results:** FoSAM cuts the computation of SAM by more than 50x at comparable
  accuracy, runs in about 20 ms on the Snapdragon XR2 Gen 2, and is preferred over SAM in 96.5 % of
  the trials of a user study on the Meta Quest Pro.

## 📦 Contents

```
fosam/
  modeling/
    token_selector.py            TIEncoder, Gaussian predictor, adaptive token sampling and their losses
    image_encoder.py             EfficientSAM ViT encoder running on the selected tokens
    foveated_sam.py              the FoSAM model, model builder, EfficientSAM weight loading
    mask_decoder.py              prompt encoder and mask decoder (EfficientSAM)
    two_way_transformer.py, mlp.py
  data.py                        gaze-conditioned LVIS dataset
  metrics.py                     Dice loss, IoU
scripts/
  train_fosam.py                 training
  eval_fosam.py                  evaluation (FoSAM and the SAM ViT-B baseline)
tools/
  preprocess_lvis.py             LVIS -> gaze-conditioned samples (object mask + gaze point)
  count_flops.py                 FLOPs at a given token budget
  visualize_selection.py         selected tokens and predicted mask for one image and gaze point
```

## 🔧 Environment

* Python 3.10 or newer and a CUDA GPU; tested with Python 3.12, PyTorch 2.9.0 / CUDA 12.8 on an
  RTX A6000. Training FoSAM-S with batch size 8 needs about 3 GB of GPU memory.
* `pip install -r requirements.txt`
* The SAM baseline of `scripts/eval_fosam.py` additionally needs `pip install segment-anything`.

## 📥 Data and models

* **EfficientSAM weights**, the initialisation of FoSAM: `efficient_sam_vits.pt` (FoSAM-S, shipped as
  `efficient_sam_vits.pt.zip`) and `efficient_sam_vitt.pt` (FoSAM-T) from the `weights/` folder of
  https://github.com/yformer/EfficientSAM.
* **LVIS v1 annotations**: `lvis_v1_train.json` and `lvis_v1_val.json` from
  https://www.lvisdataset.org/dataset.
* **COCO 2017 images**, which LVIS annotates: `train2017/` and `val2017/` from
  https://cocodataset.org/#download, side by side in one folder `<coco2017>`. Both are needed, the
  LVIS validation set uses images of both.
* **SAM ViT-B** (baseline only): `sam_vit_b_01ec64.pth` from
  https://github.com/facebookresearch/segment-anything.

## 🛠 Preprocessing

All commands run from the repository root.

LVIS annotates every object of an image, FoSAM segments the one under the gaze. A sample is one
object instance with a gaze point drawn at random inside its mask, class-balanced over 50 LVIS
categories:

```bash
python tools/preprocess_lvis.py --lvis_json <lvis_v1_train.json> --out <lvis>/train --num_samples 120000
python tools/preprocess_lvis.py --lvis_json <lvis_v1_val.json>   --out <lvis>/valid --num_samples 24000 --seed 1
python tools/preprocess_lvis.py --lvis_json <lvis_v1_val.json>   --out <lvis>/test  --num_samples 8000  --seed 2
```

Each sample is a PNG mask whose name holds the gaze point (format in `fosam/data.py`); the images
stay in `<coco2017>`. The training set takes about 45 minutes and 0.5 GB.

## 🎯 Training

```bash
python scripts/train_fosam.py --train_dir <lvis>/train --val_dir <lvis>/valid --coco_root <coco2017> \
    --esam_ckpt efficient_sam_vits.pt --save_dir checkpoints/my_run
```

For FoSAM-T add `--variant tiny` and pass `efficient_sam_vitt.pt`. The run writes
`fosam_latest.pth` (the model of the last step; restarting the same command resumes from it),
`fosam_best.pth` (best validation IoU) and TensorBoard logs to `--save_dir`; the default 30,000 steps
take about half an hour on one RTX A6000.

`--cut_ratio` (default 0.25) is the score threshold relative to the peak and `--min_tokens`
(default 50) the minimum number of tokens kept; together they set the token budget. The log prints
the number of tokens selected per image next to the losses.

## 👁 Evaluation

```bash
# FoSAM
python scripts/eval_fosam.py --ckpt checkpoints/my_run/fosam_latest.pth --val_dir <lvis>/test --coco_root <coco2017>
# SAM ViT-B, prompted with the same gaze point
python scripts/eval_fosam.py --model sam --sam_ckpt sam_vit_b_01ec64.pth --val_dir <lvis>/test --coco_root <coco2017>
```

The script reports the IoU with the mask of the gazed object and the number of tokens that went
through the encoder, overall and by object size. `--cut_ratio` and `--min_tokens` override the
training values, which moves the model along the accuracy / token budget curve without retraining.
`--latency N` also times N single-image forward passes on the GPU, and `--max_samples N` evaluates
the first N samples. Per-sample results are written as JSON to `results/`.

## 🤝 Acknowledgements and license

See `LICENSE` and `NOTICE`. The image encoder, prompt encoder and mask decoder are derived from
EfficientSAM, which builds on Segment Anything.

## 📚 Citation

```bibtex
@inproceedings{fosam2027,
  title     = {Foveated Segment Anything},
  author    = {Wang, Ziqi and Xia, Tianhua and Wang, Haiyu and Dong, Zhenyuan and Tang, Qiance and
               Lin, Jieyu and Li, Ziyun and De Salvo, Barbara and Zhang, Sai Qian},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```
