"""
Takes a BIDS dataset and produces a duplicate with SC-cropped images and lesion labels.

Arguments:
    -i: Path to the source BIDS dataset
    -o: Path for the cropped output dataset (created if absent)
    -pad-sup: Superior padding in mm (default: 40)
    -pad-inf: Inferior padding in mm (default: 100)
    -pad-rl: Right-Left padding in mm (default: 20)
    -canproco: If we crop the canproco dataset, then we only deal with PSIR and STIR data.
    --exclude-canproco: Path to the yml file containing the list of subjects to exclude from the canproco dataset
    
Author: Pierre-Louis Benveniste
"""

import argparse
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
import tqdm
from sc_crop import crop, detect
import os
import yaml


# ---------------------------------------------------------------------------
# BIDS helpers
# ---------------------------------------------------------------------------

def find_cases(bids_root: Path, canproco: bool, exclude_file: Path = None) -> dict:
    """Return {image_path: label_path | None} for every image under sub-*/ses-*/anat/."""
    cases = {}
    list_images = sorted(bids_root.glob("sub-*/ses-*/anat/*.nii.gz"))
    if canproco:
        list_images = [img for img in list_images if "PSIR" in img.name or "STIR" in img.name]
        # We load the exclude file in the canproco dataset
        subjects_to_remove = ["sub-cal123"]
        with open(exclude_file, 'r') as file:
            exclude_list = yaml.load(file, Loader=yaml.FullLoader)
            exclude_list = exclude_list["PSIR"] + exclude_list["STIR"]
        subjects_to_remove.extend(exclude_list)
        # Remove the session from the subjects to exclude
        subjects_to_remove = [sub.split("_")[0] for sub in subjects_to_remove]
        list_images = [img for img in list_images if img.parts[-4] not in subjects_to_remove]
    
    for img in list_images:
        stem  = img.name.replace(".nii.gz", "")
        lbl   = (bids_root / "derivatives" / "labels"
                 / img.relative_to(bids_root).parent
                 / f"{stem}_label-lesion_seg.nii.gz")
        if canproco:
            lbl = (bids_root / "derivatives" / "labels"
                   / img.relative_to(bids_root).parent
                   / f"{stem}_lesion-manual.nii.gz")
        if canproco and not lbl.exists():
            # Then in this case, we segment the lesions on the original image
            pred_lesion_seg = (bids_root / "derivatives" / "labels-pred"
                               / img.relative_to(bids_root).parent
                               / f"{stem}_lesion-manual.nii.gz")
            lbl = pred_lesion_seg
            if not lbl.exists():
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {img} -o {pred_lesion_seg} -v 0") == 0
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
    # Predicted segmentations live under derivatives/labels-pred in the source;
    # store their cropped version under derivatives/labels in the output dataset.
    rel_parts = tuple("labels" if part == "labels-pred" else part
                      for part in label_path.relative_to(src_root).parts)
    dst_label = dst_root.joinpath(*rel_parts)
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
    p.add_argument("--canproco", action="store_true", help="If we crop the canproco dataset, then we only deal with PSIR and STIR data.")
    p.add_argument("--exclude-canproco", type=str, default=None, help="Path to the yml file containing the list of subjects to exclude from the canproco dataset")
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

    cases = find_cases(src, args.canproco, args.exclude_canproco)
    print(f"Found {len(cases)} images")

    with tempfile.TemporaryDirectory(prefix="sc_crop_") as tmp:
        tmp_dir = Path(tmp)
        for image_path, label_path in tqdm.tqdm(cases.items()):
            process_case(image_path, label_path, src, dst, pad, tmp_dir)

if __name__ == "__main__":
    main()