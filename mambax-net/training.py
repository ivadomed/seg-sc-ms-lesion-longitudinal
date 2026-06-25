"""
Training loop for longitudinal MS lesion segmentation (consecutive-pair task).

Input:
    --unet   : path to the pretrained UNet model
    --data   : path to the MSD-style dataset JSON (with image1/image2/label1/label2)
    --output : path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import os
import json
import time
from loguru import logger
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import nibabel as nib
import wandb
from monai.utils import set_determinism

from mambaxnet import MambaXNet
from mambaxnet_v2 import MambaXNetV2
from load_dataset import get_dataloaders
from wandb_logging import log_validation_images
from metrics import compute_all_metrics, compute_dice
from loss import CombinedLoss, DeepSupervisionLoss, DiceLoss
from debuging import save_batch_patches
from utils import PolyLRScheduler

# ──────────────────────────────────────────────────────────────────────────────
# Forward pass
# ──────────────────────────────────────────────────────────────────────────────

def forward_pair(model: nn.Module, batch: dict, device: torch.device,
                 zero_prev_mask: bool = False):
    image1 = batch["image1"].to(device)   # (B, 1, *spatial) — t-1
    label1 = batch["label1"].to(device)   # (B, 1, *spatial) — t-1 mask
    image2 = batch["image2"].to(device)   # (B, 1, *spatial) — t
    label2 = batch["label2"].to(device)   # (B, 1, *spatial) — t target

    if zero_prev_mask:
        # Ablation: hide the previous-timepoint mask to test how much the model
        # relies on it (vs. the images). Trains/evaluates a no-prior baseline.
        label1 = torch.zeros_like(label1)

    targets = label2.squeeze(1)            # (B, *spatial)
    preds   = model(image2, image1, label1)
    return preds, targets


# ──────────────────────────────────────────────────────────────────────────────
# Train / validate
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, n_classes, epoch, global_step, scaler,
                    zero_prev_mask=False, deep_supervision=True):
    model.train()
    # Multi-scale outputs when deep supervision is enabled; otherwise the model
    # returns only the full-resolution output. (validate() always sets False.)
    model.deep_supervision = deep_supervision
    total_loss = 0.0
    total_dice = 0.0

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type):
            preds, targets = forward_pair(model, batch, device, zero_prev_mask=zero_prev_mask)
            loss = criterion(preds, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # Deep supervision returns a list (highest-res first); score the finest.
            finest = preds[0] if isinstance(preds, (list, tuple)) else preds
            # TODO: change next function to function on binary not logits (similar as in val)
            dice = compute_dice(finest.detach(), targets, n_classes)

        total_loss += loss.item()
        total_dice += dice
        global_step += 1

        wandb.log({"train/batch_loss": loss.item(), "train/batch_dice": dice}, step=global_step)
        logger.info(
            f"  Epoch {epoch} | batch {batch_idx+1}/{len(loader)} "
            f"| loss {loss.item():.4f} | dice {dice:.4f}"
        )

    return total_loss / len(loader), total_dice / len(loader), global_step


@torch.no_grad()
def validate(model, loader, criterion, device, overlap_ratio: float = 0.1,
             zero_prev_mask: bool = False):
    """Patch-level validation: forward the (centre-cropped) validation patches
    directly and average the loss + overlap/lesion metrics. No sliding window.

    Returns a dict of mean metrics over the validation set:
        loss, dice, lesion_f1, lesion_ppv, lesion_sensitivity
    plus dice_nonempty (Dice averaged only over patches whose GT has lesions —
    this is the honest overlap number, since empty/empty cases score Dice 1.0
    and otherwise inflate the mean).
    """
    model.eval()
    model.deep_supervision = False   # single full-res output (no deep supervision)
    agg = {k: [] for k in
           ("loss", "dice", "lesion_f1", "lesion_ppv", "lesion_sensitivity")}
    dice_nonempty = []

    for batch in loader:
        logits, targets = forward_pair(model, batch, device,
                                       zero_prev_mask=zero_prev_mask)  # (B, C, *spatial)
        agg["loss"].append(criterion(logits, targets).item())

        pred_labels = logits.argmax(dim=1)                     # (B, *spatial)
        for b in range(pred_labels.shape[0]):
            p = (pred_labels[b] == 1).cpu().numpy()
            g = (targets[b]     == 1).cpu().numpy()
            m = compute_all_metrics(p, g, overlap_ratio=overlap_ratio)
            for k in ("dice", "lesion_f1", "lesion_ppv", "lesion_sensitivity"):
                agg[k].append(m[k])
            if g.any():
                dice_nonempty.append(m["dice"])

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    out["dice_nonempty"] = float(np.mean(dice_nonempty)) if dice_nonempty else 0.0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet",           type=str, required=True)
    parser.add_argument("--data",           type=str, required=True)
    parser.add_argument("--output",         type=str, required=True)
    parser.add_argument("--epochs",         type=int,   default=1000,
                        help="Number of epochs (nnU-Net default: 1000).")
    parser.add_argument("--lr",             type=float, default=1e-2)
    parser.add_argument("--n_classes",      type=int,   default=2)
    parser.add_argument("--wandb_project",  type=str,   default="mambaxnet-longitudinal")
    parser.add_argument("--wandb_run",      type=str,   default=None)
    parser.add_argument("--wandb_offline",    action="store_true")
    parser.add_argument("--freeze-encoder",   action="store_true",
                        help="Freeze encoder weights (stem + all stages).")
    parser.add_argument("--debug-save-patches", action="store_true",
                        help="Save the first training batch's patches as NIfTI, then stop.")
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v1",
                        help="v1: M-CAM at 3 finest levels. v2: + bottleneck fusion.")
    parser.add_argument("--deep-supervision", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Apply the loss at every decoder resolution (nnU-Net "
                             "deep supervision). Use --no-deep-supervision to train "
                             "on the full-resolution output only.")
    parser.add_argument("--zero-prev-mask", action="store_true",
                        help="Ablation: zero the previous-timepoint mask everywhere "
                             "(train + eval) to measure reliance on the prior.")
    parser.add_argument("--roi-size", type=int, nargs=3, default=[64, 64, 160],
                        help="Patch size for training and validation (RPI order).")
    parser.add_argument("--overlap-ratio", type=float, default=0.1,
                        help="Lesion-wise detection overlap threshold (fraction).")
    parser.add_argument("--val-interval", type=int, default=1,
                        help="Run patch-level validation every N epochs.")
    parser.add_argument("--best-metric", type=str, default="lesion_f1",
                        choices=["lesion_f1", "dice", "dice_nonempty",
                                 "lesion_sensitivity", "lesion_ppv"],
                        help="Validation metric used to select the best checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (RNGs, foreground "
                             "sampler, MONAI augmentations, and random init of "
                             "the SEM/M-CAM modules).")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    output_path = os.path.join(args.output, f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(output_path, exist_ok=True)

    log_path = os.path.join(output_path, "training.log")
    logger.add(log_path, rotation="10 MB")
    logger.info(f"Dataset : {args.data}")
    logger.info(f"Output  : {output_path}")

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args), dir=output_path)
    logger.info(f"W&B run: {wandb.run.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Seed everything before constructing the loaders (sampler + augmentations)
    # and the model (random init of SEM/M-CAM). set_determinism also sets
    # cudnn.deterministic; it does NOT force torch.use_deterministic_algorithms,
    # so the mamba-ssm CUDA kernels won't error (full GPU bitwise determinism is
    # therefore not guaranteed, but the run is reproducible RNG-wise).
    set_determinism(seed=args.seed)
    logger.info(f"Random seed: {args.seed} (monai.set_determinism)")

    roi_size = tuple(args.roi_size)
    logger.info("Loading dataset …")
    # Train and validate on patches (validation = centre crop to roi_size).
    # One shuffled pass over the full dataset per epoch.
    train_loader, val_loader, _ = get_dataloaders(
        json_path=args.data, batch_size=2, target_shape=roi_size, eval_full_volume=False)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    if args.zero_prev_mask:
        logger.info("ABLATION: previous-timepoint mask zeroed (train + eval).")

    logger.info(f"Initialising MambaXNet ({args.model_version}) …")
    plans_json = os.path.join(args.unet, "plans.json")
    model = MambaXNet(plans_json=plans_json, n_channels=1, n_classes=args.n_classes)
    model.load_pretrained_resenc(args.unet)
    logger.info("Loaded pretrained nnU-Net weights into encoder/decoder.")
    model.to(device)

    if args.freeze_encoder:
        encoder_modules = [
            model.enc_stem,
            model.enc_stage0, model.enc_stage1, model.enc_stage2,
            model.enc_stage3, model.enc_stage4, model.enc_stage5,
        ]
        for m in encoder_modules:
            for p in m.parameters():
                p.requires_grad = False
        logger.info("Encoder frozen.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    base_criterion = CombinedLoss(n_classes=args.n_classes, batch_dice=False)

    if args.deep_supervision:
        # Deep-supervision weights: 1, 1/2, 1/4, … one per decoder output, with
        # the coarsest level dropped (weight 0) then normalised — like nnU-Net.
        n_ds = len(model.seg_layers)
        ds_weights = np.array([1 / (2 ** i) for i in range(n_ds)], dtype=float)
        ds_weights[-1] = 0.0
        ds_weights = ds_weights / ds_weights.sum()
        criterion = DeepSupervisionLoss(base_criterion, ds_weights)
        logger.info(f"Loss: Dice(batch_dice={False})+CE | deep-supervision "
                    f"weights {np.round(ds_weights, 3).tolist()}")
    else:
        criterion = base_criterion
        logger.info(f"Loss: Dice(batch_dice={False})+CE | deep supervision OFF")

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr,
        momentum     = 0.99,
        weight_decay = 3e-5,
        nesterov     = True,
    )
    scheduler = PolyLRScheduler(optimizer, max_epochs=args.epochs)
    scaler    = torch.cuda.amp.GradScaler()
    logger.info(f"SGD | lr={args.lr} | momentum=0.99 | weight_decay=3e-5 | nesterov=True")
    logger.info(f"PolyLR | exponent=0.9 | max_epochs={args.epochs}")

    if args.debug_save_patches:
        debug_dir = os.path.join(output_path, "debug_patches")
        logger.info(f"Debug mode: saving first training batch patches → {debug_dir}")
        first_batch = next(iter(train_loader))
        save_batch_patches(first_batch, debug_dir)
        logger.info("Debug patches saved. Stopping.")
        wandb.finish()
        return

    best_metric_val = 0.0
    global_step     = 0
    history = {"train_loss": [], "train_dice": [], "val": []}

    logger.info(f"Starting training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_dice, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.n_classes,
            epoch, global_step, scaler, zero_prev_mask=args.zero_prev_mask,
            deep_supervision=args.deep_supervision
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        wandb.log({
            "epoch":            epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_dice": train_dice,
            "lr":               current_lr,
        }, step=global_step)

        # Patch-level validation — run every --val-interval epochs.
        run_val = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        if run_val:
            val = validate(model, val_loader, criterion, device,
                           overlap_ratio=args.overlap_ratio,
                           zero_prev_mask=args.zero_prev_mask)
            log_validation_images(model, val_loader, device, global_step)

            elapsed = time.perf_counter() - t0
            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
                f"val loss {val['loss']:.4f} dice {val['dice_nonempty']:.4f} "
                f"lesionF1 {val['lesion_f1']:.4f} sens {val['lesion_sensitivity']:.4f} "
                f"ppv {val['lesion_ppv']:.4f} | lr {current_lr:.2e} | {elapsed:.1f}s"
            )
            wandb.log({
                "val/loss":               val["loss"],
                "val/dice":               val["dice"],
                "val/dice_nonempty":      val["dice_nonempty"],
                "val/lesion_f1":          val["lesion_f1"],
                "val/lesion_ppv":         val["lesion_ppv"],
                "val/lesion_sensitivity": val["lesion_sensitivity"],
                "epoch_time_s":           elapsed,
            }, step=global_step)
            history["val"].append({"epoch": epoch, **val})

            metric_val = val[args.best_metric]
            if metric_val > best_metric_val:
                best_metric_val = metric_val
                torch.save({
                    "state_dict": model.state_dict(),
                    "plans_json": plans_json,
                    "model_version": args.model_version,
                }, os.path.join(output_path, "best_model.pth"))
                logger.info(f"  New best val {args.best_metric}: {best_metric_val:.4f}")
                wandb.log({f"val/best_{args.best_metric}": best_metric_val}, step=global_step)
        else:
            logger.info(
                f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} "
                f"dice {train_dice:.4f} | lr {current_lr:.2e} "
                f"| {time.perf_counter()-t0:.1f}s (val skipped)"
            )

    history_path = os.path.join(output_path, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Done. Best val {args.best_metric}: {best_metric_val:.4f}")
    wandb.summary[f"best_val_{args.best_metric}"] = best_metric_val
    wandb.finish()


if __name__ == "__main__":
    main()