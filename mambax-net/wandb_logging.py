"""
W&B image logging for validation: plots image2 / GT label2 / prediction
for up to 3 patches that contain at least one foreground voxel in label2.

Orientation choice (axial vs sagittal) is made per-sample by comparing the
pixel count of the two planes:
  - axial   : slice along I-S (dim 2 in RPI), plane is H × W
  - sagittal: slice along R-L (dim 0 in RPI), plane is W × D
The plane with more pixels is used.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


# ──────────────────────────────────────────────────────────────────────────────

def _best_slice_index(mask: np.ndarray, axis: int) -> int:
    """Return the slice index along `axis` with the most foreground voxels."""
    counts = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis))
    return int(np.argmax(counts))


def _get_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2-D slice from a 3-D array along `axis` at position `idx`."""
    return np.take(volume, idx, axis=axis)


def _norm(img: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] for display."""
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        return (img - vmin) / (vmax - vmin)
    return np.zeros_like(img)


def _make_panel(image2: np.ndarray, gt2: np.ndarray, pred: np.ndarray,
                axis: int, slice_idx: int, plane_name: str,
                image2_name: str) -> plt.Figure:
    """
    Build a 1×3 matplotlib figure: image2 | GT label2 | prediction.

    All arrays are (H, W, D) spatial volumes (no channel dim).
    """
    img_sl  = _norm(_get_slice(image2, axis, slice_idx))
    gt_sl   = _get_slice(gt2,    axis, slice_idx)
    pred_sl = _get_slice(pred,   axis, slice_idx)

    # Size each panel to the true slice aspect ratio so the spinal cord is not
    # squished into a fixed-size box. Arrays are shown transposed (arr.T), so
    # the displayed image has shape (cols, rows) = arr.shape; rows along the
    # vertical axis. With a 64x64x160 RPI patch the sagittal plane is 64 x 160,
    # i.e. a tall panel that must keep its 1:3 ratio.
    disp_rows, disp_cols = img_sl.shape[1], img_sl.shape[0]
    panel_h = 5.0
    panel_w = max(1.5, panel_h * (disp_cols / disp_rows))

    fig, axes = plt.subplots(1, 3, figsize=(3 * panel_w, panel_h))
    titles = [f"image2 ({plane_name} #{slice_idx})",
              "GT label2",
              "Prediction"]
    data  = [img_sl, gt_sl, pred_sl]
    cmaps = ["gray", "hot", "hot"]

    for ax, title, arr, cmap in zip(axes, titles, data, cmaps):
        # aspect="equal" guarantees square voxels (no compression) regardless
        # of the figure/axes box; the figsize above keeps whitespace minimal.
        ax.imshow(arr.T, origin="lower", cmap=cmap, aspect="equal",
                  vmin=0, vmax=1 if cmap == "gray" else arr.max() or 1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(image2_name, fontsize=8, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def log_validation_images(model: torch.nn.Module,
                           val_loader,
                           device: torch.device,
                           global_step: int,
                           n_images: int = 3) -> None:
    """
    Sample up to `n_images` validation patches that contain foreground in
    label2, run inference, and log image2/GT/pred panels to W&B.

    Args:
        model       : trained model (in eval mode)
        val_loader  : validation DataLoader
        device      : torch device
        global_step : current training step (for W&B x-axis)
        n_images    : how many panels to log (default 3)
    """
    model.eval()
    collected     = []   # list of wandb.Image
    seen_subjects = set()

    for batch in val_loader:
        if len(collected) >= n_images:
            break

        image1 = batch["image1"].to(device)   # (B, 1, H, W, D)
        label1 = batch["label1"].to(device)
        image2 = batch["image2"].to(device)
        label2 = batch["label2"].to(device)

        preds = model(image2, image1, label1)          # (B, n_classes, H, W, D)
        pred_labels = preds.argmax(dim=1)              # (B, H, W, D)

        B = image2.shape[0]
        for b in range(B):
            if len(collected) >= n_images:
                break

            subject = batch["subject"][b]
            if subject in seen_subjects:
                continue

            gt_vol = label2[b, 0].cpu().numpy()       # (H, W, D)
            if gt_vol.max() == 0:
                continue

            seen_subjects.add(subject)

            img2_vol  = image2[b, 0].cpu().numpy()    # (H, W, D)
            pred_vol  = pred_labels[b].cpu().numpy().astype(np.float32)

            H, W, D = img2_vol.shape

            # Choose orientation: axial (axis=2, plane H×W) vs sagittal (axis=0, plane W×D)
            if H * W >= W * D:
                axis, plane_name = 2, "axial"
            else:
                axis, plane_name = 0, "sagittal"

            slice_idx   = _best_slice_index(gt_vol, axis)
            image2_name = os.path.basename(batch["image2_path"][b])

            fig = _make_panel(img2_vol, gt_vol, pred_vol,
                              axis, slice_idx, plane_name, image2_name)

            session2 = batch["session2"][b]
            caption  = f"{subject} | {session2} | {plane_name} slice {slice_idx}"
            collected.append(wandb.Image(fig, caption=caption))
            plt.close(fig)

    for i, img in enumerate(collected):
        wandb.log({f"val/image_{i+1}": img}, step=global_step)
