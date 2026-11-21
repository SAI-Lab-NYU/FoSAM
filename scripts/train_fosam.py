"""Train FoSAM on gaze-conditioned LVIS samples.

The EfficientSAM image encoder stays frozen.  The token selector (TIEncoder, Gaussian
predictor, classification head) and the mask decoder are trained with

    L = lambda_nll * L_nll + lambda_cls * L_cls + lambda_dice * L_dice

(Eq. 2-4 of the paper).  Run from the repository root:

    python scripts/train_fosam.py --train_dir <lvis>/train --val_dir <lvis>/valid \
        --coco_root <coco2017> --esam_ckpt efficient_sam_vits.pt --save_dir checkpoints/my_run
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fosam import build_foveated_sam, load_efficient_sam_weights
from fosam.data import GazeLVIS
from fosam.metrics import dice_loss, mask_iou


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_dir", required=True, help="training samples from tools/preprocess_lvis.py")
    p.add_argument("--val_dir", required=True, help="validation samples from tools/preprocess_lvis.py")
    p.add_argument("--coco_root", required=True, help="folder with train2017/ and val2017/")
    p.add_argument("--esam_ckpt", required=True, help="official EfficientSAM checkpoint matching --variant")
    p.add_argument("--save_dir", default="checkpoints/fosam")
    p.add_argument("--variant", default="small", choices=["small", "tiny"],
                   help="FoSAM-S (EfficientSAM-S) or FoSAM-T (EfficientSAM-Ti)")
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--num_classes", type=int, default=51,
                   help="size of the classification head; must exceed the largest class index of the samples")
    p.add_argument("--cut_ratio", type=float, default=0.25, help="score threshold tau, relative to the peak")
    p.add_argument("--min_tokens", type=int, default=50, help="minimum number of tokens kept per image")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--lr_decoder", type=float, default=1e-4)
    p.add_argument("--lr_selector", type=float, default=1e-4)
    p.add_argument("--lambda_nll", type=float, default=1.0)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_dice", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=500, help="linear warm-up steps")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--val_batches", type=int, default=250, help="validation batches per evaluation")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(json.dumps(vars(args), indent=2), flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    # bfloat16 autocast, entered per step.  The weight-cast cache is disabled so that the
    # forward pass always sees the weights the optimizer has just updated.
    amp = dict(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ----------------------------------------------------------------- data
    ds_train = GazeLVIS(args.train_dir, args.coco_root, canvas=args.img_size)
    ds_val = GazeLVIS(args.val_dir, args.coco_root, canvas=args.img_size)
    print(f"train samples {len(ds_train)}, validation samples {len(ds_val)}", flush=True)

    def loader(ds, train):
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=train, drop_last=train,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    dl_train, dl_val = loader(ds_train, True), loader(ds_val, False)

    # ----------------------------------------------------------------- model
    model = build_foveated_sam(args.variant, args.img_size, args.num_classes,
                               args.cut_ratio, args.min_tokens).to(device)
    load_efficient_sam_weights(model, args.esam_ckpt)
    selector = model.token_selector
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.mask_decoder, selector):
        for p in module.parameters():
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_all / 1e6:.1f} M, trained {n_train / 1e6:.1f} M "
          f"(token selector {sum(p.numel() for p in selector.parameters()) / 1e6:.1f} M)", flush=True)

    base_lrs = [args.lr_decoder, args.lr_selector]
    opt = torch.optim.Adam([
        {"params": model.mask_decoder.parameters(), "lr": args.lr_decoder},
        {"params": selector.parameters(), "lr": args.lr_selector},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=30 * len(ds_train) // args.batch_size, T_mult=2, eta_min=5e-5)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))

    step, best_iou = 0, -1.0
    latest = os.path.join(args.save_dir, "fosam_latest.pth")
    if os.path.exists(latest):
        st = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best_iou = st["step"], st["best_iou"]
        print(f"resumed from {latest} at step {step}", flush=True)

    def save(name, with_optimizer):
        state = {"model": model.state_dict(), "args": vars(args), "step": step, "best_iou": best_iou}
        if with_optimizer:
            state.update(opt=opt.state_dict(), sched=sched.state_dict())
        tmp = os.path.join(args.save_dir, f".{name}.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(args.save_dir, name))

    stop = {"now": False}

    def on_signal(signum, frame):
        print(f"signal {signum}: saving and exiting", flush=True)
        stop["now"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def forward(batch):
        image, gaze, mask, label, _ = batch
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(**amp):
            out = model(image, gaze.to(device), mask, label.to(device))
            l_dice = dice_loss(out["masks"], mask)
            loss = (args.lambda_dice * l_dice + args.lambda_cls * out["loss_cls"]
                    + args.lambda_nll * out["loss_nll"])
        iou = mask_iou(out["masks"] > 0, mask > 0.5)
        return loss, l_dice, out["loss_cls"], out["loss_nll"], iou

    @torch.no_grad()
    def validate():
        model.eval()
        ious, tokens = [], []
        for i, batch in enumerate(dl_val):
            if i >= args.val_batches:
                break
            ious.append(forward(batch)[4].cpu())
            tokens.append(selector.token_counts.cpu())
        model.train()
        return torch.cat(ious).mean().item(), torch.cat(tokens).float().mean().item()

    # ----------------------------------------------------------------- training
    model.train()
    t0, step0 = time.time(), step
    running, tokens = [], []
    while step < args.max_steps and not stop["now"]:
        for batch in dl_train:
            if step >= args.max_steps or stop["now"]:
                break
            loss, l_dice, l_cls, l_nll, iou = forward(batch)
            step += 1
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                # degenerate sample, e.g. an empty mask leaves the likelihood undefined
                print(f"step {step}: non-finite loss, batch skipped", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"] if p.grad is not None], args.grad_clip)
            opt.step()
            sched.step()
            if step < args.warmup:
                for group, lr in zip(opt.param_groups, base_lrs):
                    group["lr"] = lr * (step + 1) / args.warmup
            running.append([loss.item(), l_dice.item(), l_cls.item(), l_nll.item(), iou.mean().item()])
            tokens.append(selector.token_counts.cpu())

            if step % args.log_every == 0:
                m = np.mean(running, 0)
                tk = torch.cat(tokens).float()
                speed = (step - step0) * args.batch_size / (time.time() - t0)
                print(f"step {step:6d} | loss {m[0]:.4f} dice {m[1]:.4f} cls {m[2]:.4f} nll {m[3]:+.4f} "
                      f"IoU {m[4]:.4f} | tokens mean {tk.mean():6.1f} min {int(tk.min()):4d} max {int(tk.max()):4d} "
                      f"| sigma {selector.sigma_x.float().mean():.3f} {selector.sigma_y.float().mean():.3f} "
                      f"| {speed:.1f} img/s", flush=True)
                for name, value in [("loss/total", m[0]), ("loss/dice", m[1]), ("loss/cls", m[2]),
                                    ("loss/nll", m[3]), ("train/iou", m[4]), ("train/tokens", tk.mean().item()),
                                    ("lr/selector", opt.param_groups[1]["lr"])]:
                    writer.add_scalar(name, value, step)
                running, tokens = [], []

            if step % args.val_every == 0:
                iou_val, tokens_val = validate()
                print(f"  [val] step {step}: IoU {iou_val:.4f}, tokens {tokens_val:.1f}", flush=True)
                writer.add_scalar("val/iou", iou_val, step)
                writer.add_scalar("val/tokens", tokens_val, step)
                if iou_val > best_iou:
                    best_iou = iou_val
                    save("fosam_best.pth", with_optimizer=False)

            if step % args.ckpt_every == 0 or step % args.val_every == 0:
                save("fosam_latest.pth", with_optimizer=True)

    save("fosam_latest.pth", with_optimizer=True)
    print(f"finished at step {step} after {(time.time() - t0) / 3600:.2f} h, "
          f"best validation IoU {best_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
