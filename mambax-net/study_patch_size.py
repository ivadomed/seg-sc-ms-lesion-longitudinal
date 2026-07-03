"""
Scan all NIfTI images under a BIDS dataset and
plot the distribution of each anatomical axis size (R-L, A-P, I-S) and voxel
spacing to help decide on a training patch size.

Inputs:
    --bids:   Path to the BIDS dataset root
    --output: Directory to save plots (default: ./patch_study)

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path
import os

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Maps any orientation code letter to its canonical axis name
_CODE_TO_AXIS = {
    "R": "R-L", "L": "R-L",
    "A": "A-P", "P": "A-P",
    "S": "I-S", "I": "I-S",
}

ANAT_AXES = ["R-L", "A-P", "I-S"]


def _axis_name(code: str) -> str:
    return _CODE_TO_AXIS[code.upper()]


# ──────────────────────────────────────────────────────────────────────────────

def collect_stats(bids_root: Path) -> dict:
    """
    Walk bids_root and accumulate voxel counts and spacings keyed by anatomical
    axis name (R-L / A-P / I-S) rather than array dimension index.
    """
    images = sorted(bids_root.rglob("*.nii.gz"))
    images = [str(img) for img in images]
    images = [img for img in images if "derivatives" not in img]  # exclude labels
    images = [img for img in images if "SHA256" not in img]         # only anatomical images

    sizes    = {"R-L": [], "A-P": [], "I-S": []}
    spacings = {"R-L": [], "A-P": [], "I-S": []}

    for img_path in images:
        nii   = nib.load(img_path)
        shape = nii.shape[:3]
        zooms = nii.header.get_zooms()[:3]

        # aff2axcodes returns e.g. ('R', 'A', 'S') — one letter per dim
        codes = nib.aff2axcodes(nii.affine)

        for dim, code in enumerate(codes):
            aname = _axis_name(code)
            sizes[aname].append(shape[dim] * zooms[dim])   # physical size in mm
            spacings[aname].append(zooms[dim])

    logger.info(f"Scanned {len(images)} images.")
    return {
        "n":       len(images),
        "sizes":   {k: np.array(v) for k, v in sizes.items()},
        "spacings": {k: np.array(v) for k, v in spacings.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────

def print_summary(stats: dict) -> None:
    col_w = 10
    header = f"{'':>6}" + "".join(
        f"  {f'{a} (mm)':>{col_w}}  {f'{a} sp(mm)':>{col_w}}"
        for a in ANAT_AXES
    )
    logger.info(header)
    logger.info("-" * len(header))

    fns = {
        "min": np.min,
        "p25": lambda x: np.percentile(x, 25),
        "med": np.median,
        "p75": lambda x: np.percentile(x, 75),
        "max": np.max,
    }
    for label, fn in fns.items():
        row = f"{label:>6}"
        for a in ANAT_AXES:
            row += f"  {fn(stats['sizes'][a]):>{col_w}.1f}  {fn(stats['spacings'][a]):>{col_w}.3f}"
        logger.info(row)


# ──────────────────────────────────────────────────────────────────────────────

def plot_stats(stats: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    n      = stats["n"]

    # ── Figure 1: axis size distributions ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, aname, color in zip(axes, ANAT_AXES, colors):
        data = stats["sizes"][aname]
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(data), color="black", linestyle="--", linewidth=1.2,
                   label=f"median={np.median(data):.0f}")
        ax.axvline(np.percentile(data, 25), color="gray", linestyle=":", linewidth=1,
                   label=f"p25={np.percentile(data, 25):.0f}")
        ax.axvline(np.percentile(data, 75), color="gray", linestyle=":", linewidth=1,
                   label=f"p75={np.percentile(data, 75):.0f}")
        ax.set_title(f"{aname} axis", fontsize=11)
        ax.set_xlabel("Physical size (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"Spatial axis sizes  (n={n} images)", fontsize=13)
    fig.tight_layout()
    out = output_dir / "axis_sizes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")

    # ── Figure 2: voxel spacing distributions ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, aname, color in zip(axes, ANAT_AXES, colors):
        data = stats["spacings"][aname]
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(data), color="black", linestyle="--", linewidth=1.2,
                   label=f"median={np.median(data):.3f}")
        ax.set_title(f"{aname} spacing", fontsize=11)
        ax.set_xlabel("Voxel size (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"Voxel spacing  (n={n} images)", fontsize=13)
    fig.tight_layout()
    out = output_dir / "voxel_spacing.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")

    # ── Figure 3: 2-D scatter — R-L vs A-P, coloured by I-S ─────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(stats["sizes"]["R-L"], stats["sizes"]["A-P"],
                    c=stats["sizes"]["I-S"],
                    cmap="viridis", alpha=0.6, edgecolors="none", s=20)
    fig.colorbar(sc, ax=ax, label="I-S size (mm)")
    ax.set_xlabel("R-L size (mm)")
    ax.set_ylabel("A-P size (mm)")
    ax.set_title(f"R-L vs A-P  (colour = I-S, n={n})", fontsize=11)
    fig.tight_layout()
    out = output_dir / "axis_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids",   type=str, required=True,
                        help="Path to the BIDS dataset root")
    parser.add_argument("--output", type=str, default="./patch_study",
                        help="Directory to save plots (default: ./patch_study)")
    return parser.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the logger
    log_path = os.path.join(output_dir, "study_patch_size.log")
    logger.add(log_path, rotation="10 MB")

    stats = collect_stats(Path(args.bids))
    print_summary(stats)
    plot_stats(stats, output_dir)


if __name__ == "__main__":
    main()