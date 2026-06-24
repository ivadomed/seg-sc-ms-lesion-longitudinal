"""
Affine registration of a previous-timepoint spinal-cord scan onto the current
timepoint, for the *registered-data* comparison strategy.

The transform is estimated from **disc-level point labels**, which are exact
anatomical correspondences between the two sessions (e.g. the C2/C3 disc has
the same integer label in both `--disc1` and `--disc2`).

IMPORTANT — disc centres are (near-)collinear along the cord, so a full 12-DOF
affine is rank-deficient and would fit the points perfectly while shearing
everything off the cord line into nonsense. We therefore default to an Umeyama
**similarity** transform (rotation + uniform scale + translation), which is
well-posed for collinear landmarks. `--transform affine` is offered but
auto-downgrades to similarity whenever the landmarks fail to span 3D (i.e.
essentially always for discs). One DOF is genuinely unobservable from
collinear points alone — the *roll about the cord axis* — and is left at
identity; recovering it (and any non-rigid cord bending) requires an
intensity- or cord-shape-driven method such as SCT `sct_register_multimodal`.

The spinal-cord segmentations are used for **quality control** only: after
registration we report the Dice between the warped previous cord and the
current cord, plus the landmark residual. They are not part of the estimate
(disc points already pin the cord along its length), but warping them lets you
eyeball the result.

Everything is done in world coordinates, so the two images may have different
shapes, orientations, and voxel sizes.

Outputs (into --output):
    <stem>_reg.nii.gz        warped previous image      (linear)
    <stem>_sccord_reg.nii.gz warped previous cord mask  (nearest)
    <stem>_disc_reg.nii.gz   warped previous disc labels(nearest, QC)
    <stem>_lesion_reg.nii.gz warped previous lesion mask(nearest)  [if --label1]
    affine_world.txt         the 4x4 world->world transform (prev -> current)
    registration_qc.json     landmark RMSE + cord Dice

For deformable / cord-following registration (which can hide real lesion
change and is therefore deliberately avoided here), the Spinal Cord Toolbox
`sct_register_multimodal` is the standard alternative.

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc


# ──────────────────────────────────────────────────────────────────────────────
# Landmark extraction
# ──────────────────────────────────────────────────────────────────────────────

def _label_centroids_world(disc_path: str) -> dict:
    """Return {label_value: world_coord (3,)} for every non-zero disc label."""
    img = nib.load(disc_path)
    data = np.asarray(img.get_fdata())
    affine = img.affine

    centroids = {}
    for value in np.unique(data):
        if value == 0:
            continue
        vox = np.argwhere(data == value).mean(axis=0)            # voxel centroid
        world = nib.affines.apply_affine(affine, vox)            # -> mm
        centroids[int(round(float(value)))] = world
    return centroids


def matched_landmarks(disc1_path: str, disc2_path: str):
    """Return (P1, P2) arrays (N, 3) of matched disc world coords (prev, current)."""
    c1 = _label_centroids_world(disc1_path)
    c2 = _label_centroids_world(disc2_path)
    shared = sorted(set(c1) & set(c2))
    if not shared:
        raise RuntimeError(
            "No disc labels are shared between the two images. "
            f"prev labels={sorted(c1)} current labels={sorted(c2)}"
        )
    P1 = np.stack([c1[k] for k in shared])
    P2 = np.stack([c2[k] for k in shared])
    return P1, P2, shared


# ──────────────────────────────────────────────────────────────────────────────
# Transform estimation (world -> world, mapping prev onto current)
# ──────────────────────────────────────────────────────────────────────────────

def _affine_lstsq(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Least-squares 12-DOF affine T s.t. P2 ≈ T(P1). Returns 4x4.

    Only valid when the landmarks span 3D (rank-3 configuration). For collinear
    points (e.g. spinal disc centres along the cord) this is rank-deficient and
    must NOT be used — see `estimate_transform`.
    """
    n = P1.shape[0]
    P1_h = np.hstack([P1, np.ones((n, 1))])          # (N, 4)
    A, *_ = np.linalg.lstsq(P1_h, P2, rcond=None)    # (4, 3): P1_h @ A ≈ P2
    T = np.eye(4)
    T[:3, :] = A.T                                   # (3, 4)
    return T


def _umeyama(P1: np.ndarray, P2: np.ndarray, with_scale: bool) -> np.ndarray:
    """Umeyama least-squares similarity (with_scale) or rigid fit. Returns 4x4.

    Well-posed for collinear point sets: it recovers the minimal rotation that
    aligns the two lines (no spurious roll about the line) and returns identity
    when P1 == P2. Roll about the line of collinear points is genuinely
    unobservable and is left at identity — see module docstring.
    """
    mu1, mu2 = P1.mean(0), P2.mean(0)
    X, Y = P1 - mu1, P2 - mu2
    C = (Y.T @ X) / P1.shape[0]
    U, Dvals, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:     # reflection guard
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var1 = (X ** 2).sum() / P1.shape[0]
        scale = float((Dvals * np.diag(S)).sum() / var1) if var1 > 0 else 1.0
    else:
        scale = 1.0
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = mu2 - scale * R @ mu1
    return T


def _spans_3d(P: np.ndarray, tol: float = 1e-3) -> bool:
    """True if the centred points span 3D (needed for a non-degenerate affine)."""
    s = np.linalg.svd(P - P.mean(0), compute_uv=False)
    return s[0] > 0 and (s[2] / s[0]) > tol


def estimate_transform(P1, P2, mode: str = "affine") -> np.ndarray:
    """Estimate a world->world transform mapping P1 (prev) onto P2 (current).

    mode:
      - "affine"     : 12-DOF, but auto-downgrades to similarity when the
                       landmarks are collinear/coplanar (always the case for
                       disc centres along the cord) since affine is then degenerate.
      - "similarity" : rotation + uniform scale + translation (7 DOF).
      - "rigid"      : rotation + translation (6 DOF).
    """
    n = P1.shape[0]
    if n < 3:
        raise RuntimeError(f"Need >=3 matched landmarks, got {n}.")

    if mode == "affine":
        if n >= 4 and _spans_3d(P1) and _spans_3d(P2):
            return _affine_lstsq(P1, P2)
        print("  Landmarks do not span 3D (collinear/coplanar) or n<4 — "
              "affine is degenerate; using a similarity transform instead.")
        return _umeyama(P1, P2, with_scale=True)
    if mode == "similarity":
        return _umeyama(P1, P2, with_scale=True)
    if mode == "rigid":
        return _umeyama(P1, P2, with_scale=False)
    raise RuntimeError(f"Unknown transform mode: {mode}")


def _landmark_rmse(T, P1, P2) -> float:
    P1_pred = nib.affines.apply_affine(T, P1)
    return float(np.sqrt(((P1_pred - P2) ** 2).sum(axis=1).mean()))


# ──────────────────────────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────────────────────────

def warp_to_reference(moving_path: str, T: np.ndarray, reference: nib.Nifti1Image,
                      order: int) -> nib.Nifti1Image:
    """Warp `moving` into the reference grid using world transform T (prev->current).

    Trick: pre-multiplying the moving image's affine by T moves its world
    coordinates into the reference's world space, after which a plain regrid
    onto the reference voxel map performs the resampling.
    """
    moving = nib.load(moving_path)
    moved = nib.Nifti1Image(np.asarray(moving.get_fdata()), T @ moving.affine, moving.header)
    return nibproc.resample_from_to(moved, (reference.shape[:3], reference.affine), order=order)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image1",  required=True, help="Previous-timepoint image (moving)")
    p.add_argument("--image2",  required=True, help="Current-timepoint image (reference)")
    p.add_argument("--sc-seg1", required=True, help="Previous cord segmentation (QC)")
    p.add_argument("--sc-seg2", required=True, help="Current cord segmentation (QC)")
    p.add_argument("--disc1",   required=True, help="Previous disc-level point labels")
    p.add_argument("--disc2",   required=True, help="Current disc-level point labels")
    p.add_argument("--label1",  default=None,  help="Previous lesion mask to warp too (optional)")
    p.add_argument("--output",  required=True, help="Output directory")
    p.add_argument("--transform", choices=["affine", "similarity", "rigid"],
                   default="similarity",
                   help="affine auto-downgrades to similarity for collinear discs; "
                        "similarity (default) = rotation+uniform scale+translation.")
    p.add_argument("--stem", default=None, help="Output filename stem (default: image1 basename)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    stem = args.stem or os.path.basename(args.image1).replace(".nii.gz", "").replace(".nii", "")

    # 1. Landmarks + transform
    P1, P2, shared = matched_landmarks(args.disc1, args.disc2)
    print(f"Matched {len(shared)} disc landmarks: {shared}")
    T = estimate_transform(P1, P2, mode=args.transform)
    rmse = _landmark_rmse(T, P1, P2)
    print(f"Landmark RMSE after registration: {rmse:.3f} mm")

    reference = nib.load(args.image2)

    # 2. Warp moving volumes into the current-timepoint grid
    out_img = warp_to_reference(args.image1, T, reference, order=1)
    nib.save(out_img, os.path.join(args.output, f"{stem}_reg.nii.gz"))

    out_cord = warp_to_reference(args.sc_seg1, T, reference, order=0)
    nib.save(out_cord, os.path.join(args.output, f"{stem}_sccord_reg.nii.gz"))

    out_disc = warp_to_reference(args.disc1, T, reference, order=0)
    nib.save(out_disc, os.path.join(args.output, f"{stem}_disc_reg.nii.gz"))

    if args.label1:
        out_lesion = warp_to_reference(args.label1, T, reference, order=0)
        nib.save(out_lesion, os.path.join(args.output, f"{stem}_lesion_reg.nii.gz"))

    # 3. QC: warped previous cord vs current cord
    cord_dice = _dice(np.asarray(out_cord.get_fdata()),
                      np.asarray(nib.load(args.sc_seg2).get_fdata()))
    print(f"Cord Dice (warped prev vs current): {cord_dice:.3f}")

    np.savetxt(os.path.join(args.output, "affine_world.txt"), T)
    with open(os.path.join(args.output, "registration_qc.json"), "w") as f:
        json.dump({
            "transform": args.transform,
            "n_landmarks": len(shared),
            "shared_labels": shared,
            "landmark_rmse_mm": rmse,
            "cord_dice": cord_dice,
        }, f, indent=2)
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
