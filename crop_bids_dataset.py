"""
Takes a BIDS dataset and produces a duplicate with SC-cropped images and lesion labels.

Arguments:
    -i: Path to the source BIDS dataset
    -o: Path for the cropped output dataset (created if absent)
    
Author: Pierre-Louis Benveniste
"""

import argparse
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import tqdm
from sc_crop import crop, detect


# ---------------------------------------------------------------------------
# BIDS helpers
# ---------------------------------------------------------------------------

def find_cases(bids_root: Path) -> dict:
    """Return {image_path: label_path | None} for every image under sub-*/ses-*/anat/."""
    cases = {}
    for img in sorted(bids_root.glob("sub-*/ses-*/anat/*.nii.gz")):
        stem  = img.name.replace(".nii.gz", "")
        lbl   = (bids_root / "derivatives" / "labels"
                 / img.relative_to(bids_root).parent
                 / f"{stem}_label-lesion_seg.nii.gz")
        cases[img] = lbl if lbl.exists() else None
    return cases


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_case(image_path, label_path, src_root, dst_root, pad,tmp_dir):
    """Detect SC, crop image and label, run QC. Returns (qc | None, error | None)."""
    bbox = detect(image_path, **pad)

    dst_image = dst_root / image_path.relative_to(src_root)
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    nib.save(crop(nib.load(image_path), bbox), dst_image)

    # Same for the label
    label_nii = nib.load(label_path)
    dst_label = dst_root / label_path.relative_to(src_root)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    nib.save(crop(label_nii, bbox), dst_label)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Crop a BIDS dataset to the spinal cord.")
    p.add_argument("-i", "--input",    required=True, help="Source BIDS dataset directory")
    p.add_argument("-o", "--output",   required=True, help="Output directory for cropped dataset")
    p.add_argument("--pad-sup",  type=float, default=40,  help="Superior padding mm  (default: 40)")
    p.add_argument("--pad-inf",  type=float, default=100, help="Inferior padding mm  (default: 100)")
    p.add_argument("--pad-rl",   type=float, default=20,  help="Right-Left padding mm (default: 20)")
    p.add_argument("--pad-ap",   type=float, default=20,  help="A-P padding mm        (default: 20)")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    pad = dict(
        pad_superior=args.pad_sup,
        pad_inferior=args.pad_inf,
        pad_rl=args.pad_rl,
        pad_ap=args.pad_ap,
    )

    cases = find_cases(src)
    print(f"Found {len(cases)} images")

    with tempfile.TemporaryDirectory(prefix="sc_crop_") as tmp:
        tmp_dir = Path(tmp)
        for image_path, label_path in tqdm.tqdm(cases.items()):
            process_case(image_path, label_path, src, dst, pad, tmp_dir)


if __name__ == "__main__":
    main()