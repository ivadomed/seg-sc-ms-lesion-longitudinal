"""
Segmentation metrics for MS lesion evaluation.

Two families:
  - Voxel-wise Dice (overlap; dominated by large lesions, sensitive to boundary
    noise — useful but not the whole story for small MS lesions).
  - Lesion-wise detection (F1 / sensitivity / PPV) via 3D connected components.
    Far more robust to inter-rater boundary disagreement, which is the regime
    that caps voxel Dice for MS lesions.

Lesion-wise functions adapted from:
  https://github.com/npnl/atlas2_grand_challenge/blob/main/isles/scoring.py

Author: Pierre-Louis Benveniste
"""

from scipy import ndimage
import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Voxel-wise
# ──────────────────────────────────────────────────────────────────────────────

def dice_score(prediction, groundtruth, smooth: float = 1e-5) -> float:
    """Soft/hard Dice with additive smoothing.

    With `smooth=1`, two empty masks score 1.0 (model correctly predicts
    nothing), which is the convention used throughout this module.
    """
    prediction = np.asarray(prediction).astype(np.float32)
    groundtruth = np.asarray(groundtruth).astype(np.float32)
    numer = (prediction * groundtruth).sum()
    denom = (prediction + groundtruth).sum()
    return float((2 * numer + smooth) / (denom + smooth))


# Compute dice score for the entire batch
def compute_dice(preds: torch.Tensor, targets: torch.Tensor,
                 n_classes: int, smooth: float = 1e-5) -> float:
    pred_labels = preds.argmax(dim=1)
    dice_scores = []
    for cls in range(1, n_classes):
        pred_c = (pred_labels == cls).float().view(-1)
        tgt_c  = (targets == cls).float().view(-1)
        inter  = (pred_c * tgt_c).sum()
        denom  = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            # empty GT and empty prediction for this class → perfect
            dice_scores.append(1.0)
            continue
        dice_scores.append(((2.0 * inter + smooth) / (denom + smooth)).item())
    return float(np.mean(dice_scores)) if dice_scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Lesion-wise detection
# ──────────────────────────────────────────────────────────────────────────────

def lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio: float = 0.1):
    """Lesion-wise TP / FP / FN via 3D connected-component analysis.

    A ground-truth lesion counts as a true positive if at least `overlap_ratio`
    of its voxels overlap the prediction; otherwise it is a false negative.
    A predicted lesion with no overlapping ground-truth voxel is a false
    positive.

    Parameters
    ----------
    truth, prediction : array-like
        3D arrays (cast to bool internally).
    overlap_ratio : float
        Minimum fraction of a GT lesion's voxels that must be predicted for it
        to count as detected (default 0.1 = 10%).

    Returns
    -------
    (tp, fp, fn) : tuple[int, int, int]
    """
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    tp, fp, fn = 0, 0, 0

    # TP / FN: iterate over ground-truth connected components.
    labeled_truth, n_truth = ndimage.label(truth)
    for idx in range(1, n_truth + 1):
        lesion = labeled_truth == idx
        n_lesion_voxels = int(lesion.sum())
        overlapping = int((lesion & prediction).sum())
        if n_lesion_voxels > 0 and overlapping / n_lesion_voxels >= overlap_ratio:
            tp += 1
        else:
            fn += 1

    # FP: iterate over predicted connected components with no GT overlap.
    labeled_pred, n_pred = ndimage.label(prediction)
    for idx in range(1, n_pred + 1):
        lesion = labeled_pred == idx
        if not np.any(lesion & truth):
            fp += 1

    return tp, fp, fn


def lesion_f1_score(truth, prediction, overlap_ratio: float = 0.1) -> float:
    """Lesion-wise F1 = TP / (TP + (FP + FN)/2). Empty/empty → 1.0."""
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    if not truth.any() and not prediction.any():
        return 1.0
    if truth.any() != prediction.any():        # exactly one is empty
        return 0.0

    tp, fp, fn = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
    denom = tp + (fp + fn) / 2
    return float(tp / denom) if denom != 0 else 1.0


def lesion_ppv(truth, prediction, overlap_ratio: float = 0.1) -> float:
    """Lesion-wise positive predictive value = TP / (TP + FP). Empty/empty → 1.0."""
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    if not truth.any() and not prediction.any():
        return 1.0
    if truth.any() != prediction.any():
        return 0.0

    tp, fp, _ = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
    denom = tp + fp
    return float(tp / denom) if denom != 0 else 1.0


def lesion_sensitivity(truth, prediction, overlap_ratio: float = 0.1) -> float:
    """Lesion-wise sensitivity (detection rate) = TP / (TP + FN). Empty/empty → 1.0."""
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    if not truth.any() and not prediction.any():
        return 1.0
    if truth.any() != prediction.any():
        return 0.0

    tp, _, fn = lesion_wise_tp_fp_fn(truth, prediction, overlap_ratio)
    denom = tp + fn
    return float(tp / denom) if denom != 0 else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(prediction, truth, overlap_ratio: float = 0.1) -> dict:
    """Return all metrics for one binary prediction/GT pair as a dict.

    `prediction` and `truth` are 3D binary arrays in the same space.
    """
    return {
        "dice":                dice_score(prediction, truth),
        "lesion_f1":           lesion_f1_score(truth, prediction, overlap_ratio),
        "lesion_ppv":          lesion_ppv(truth, prediction, overlap_ratio),
        "lesion_sensitivity":  lesion_sensitivity(truth, prediction, overlap_ratio),
    }
