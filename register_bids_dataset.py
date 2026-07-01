"""
Takes a BIDS dataset and produces an affine-registered version where follow-up
sessions are registered to the baseline session.

Registration is done WITHIN groups of same subject, same contrast and same chunk
(mirroring the pairing used to build the MSD dataset in mambax-net/create_msd.py),
so a follow-up is only ever registered onto a baseline that shares its contrast
and chunk. The first (earliest) session of each group is the baseline.

Steps per group:
  1. Segment spinal cord (sct_deepseg spinalcord)
  2. Detect disc labels (sct_deepseg spine)
  3. Register each follow-up to baseline (sct_register_multimodal)
  4. Apply the warping field to lesion segmentations (sct_apply_transfo)

Arguments:
    -i: Path to the source BIDS dataset
    -o: Path for the registered output dataset (created if absent)

Author: Pierre-Louis Benveniste
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import tqdm
import os


def run(cmd: str):
    """Run a shell command, raising on failure."""
    print(f"  >> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def get_chunk(image_path: Path) -> str:
    """
    Extracts the chunk identifier (e.g. 'chunk-2') from a BIDS filename, or '' if
    none is present. Used to keep different chunks of the same subject/contrast
    from being registered together (matches mambax-net/create_msd.py).
    """
    for token in image_path.name.split("_"):
        if token.startswith("chunk-"):
            return token
    return ""


def find_groups(bids_root: Path) -> dict:
    """
    Return {(subject, contrast, chunk): sorted list of (session_dir, image_path, label_path)}.

    Images are grouped by subject, contrast and chunk so that a follow-up is only
    ever paired with a baseline that shares its contrast and chunk (same grouping
    as mambax-net/create_msd.py). Sessions within a group are sorted chronologically
    by session folder name (ses-YYYYMMDD), so the first entry is the baseline.
    """
    groups = {}
    for img in sorted(bids_root.rglob("*.nii.gz")):
        if "derivatives" in img.parts or "SHA256" in img.parts:
            continue
        sub = img.parts[-4]
        # the session folder (ses-YYYYMMDD); img.parent is the anat/ folder
        session_dir = img.parents[1]
        stem = img.name.replace(".nii.gz", "")
        # the label is the BIDS folder but folder derivatives/labels/ and then the relative path to the image, with the suffix _label-lesion_seg.nii.gz
        label = bids_root / "derivatives" / "labels" / img.relative_to(bids_root).parent / f"{stem}_label-lesion_seg.nii.gz"
        if not label.exists():
            label = bids_root / "derivatives" / "labels" / img.relative_to(bids_root).parent / f"{stem}_lesion-manual.nii.gz"
        # This is a temporary fix since the canproco segs need to be manually corrected
        if not label.exists():
            label = bids_root / "derivatives" / "labels-pred" / img.relative_to(bids_root).parent / f"{stem}_lesion-manual.nii.gz"
        if not label.exists():
            # This is image has no label, we don't include the image in the list of cases
            continue
        contrast = stem.split("_")[-1]
        key = (sub, contrast, get_chunk(img))
        groups.setdefault(key, []).append((session_dir, img, label))
    for key in groups:
        groups[key].sort(key=lambda x: x[0].name)
    return groups


def segment_sc(image: Path, output: Path):
    """Segment spinal cord"""
    run(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {image} -o {output}")


def segment_discs(image: Path, tmp_output: Path, output: Path):
    """
    Detect disc labels.
    Multiple files are produced by sct_deepseg spine, but we only keep
    the _totalspineseg_discs.nii.gz file and move it to the output path.
    """
    run(f"SCT_USE_GPU=1 sct_deepseg spine -i {image} -o {tmp_output}")
    # Now we move the _totalspineseg_discs.nii.gz file to the output path
    disc_file = tmp_output.parent / f"{tmp_output.stem.replace('.nii','')}_totalspineseg_discs.nii.gz"
    shutil.move(disc_file, output)


def log_disc_seg_failure(image: Path, out_root: Path):
    """Append the path of an image whose disc segmentation failed to a tracking file."""
    failures_file = out_root / "disc_seg_failures.txt"
    with open(failures_file, "a") as f:
        f.write(f"{image}\n")


def keep_common_levels_only(levels_1, levels_2, out_1, out_2):
    """
    This function keeps only the common disc levels between two level segmentations.
    The filtered results are written to new files (out_1, out_2) without overwriting the inputs.
    """
    run(f"sct_label_utils -i {levels_1} -remove-sym {levels_2} -o {out_1} {out_2}")
    return out_1, out_2


def register(moving_img, fixed_img, moving_seg, fixed_seg, moving_disc, fixed_disc, output, qc_folder):
    """Affine registration of moving to fixed using SC seg and disc labels."""
    run(
        f"sct_register_multimodal"
        f" -i {moving_img}"
        f" -d {fixed_img}"
        f" -iseg {moving_seg}"
        f" -dseg {fixed_seg}"
        f" -ilabel {moving_disc}"
        f" -dlabel {fixed_disc}"
        f" -o {output}"
        f" -param step=0,type=label,dof=Tx_Ty_Tz:step=1,type=seg,algo=centermassrot"
        f" -qc {qc_folder}"
    )



def apply_transfo(input_file, dest_file, warp_field, output_file):
    """Apply warping field to a label image."""
    run(
        f"sct_apply_transfo"
        f" -i {input_file}"
        f" -d {dest_file}"
        f" -w {warp_field}"
        f" -o {output_file}"
        f" -x nn"
    )


def main():
    parser = argparse.ArgumentParser(description="Affine-register a BIDS dataset to baseline sessions.")
    parser.add_argument("-i", required=True, help="Path to source BIDS dataset")
    parser.add_argument("-o", required=True, help="Path to output registered dataset")
    parser.add_argument("-p", required=True, help="Path to cache SC/disc predictions. Predictions are saved there and reused on subsequent runs.")
    args = parser.parse_args()

    bids_root = Path(args.i)
    out_root = Path(args.o)
    out_root.mkdir(parents=True, exist_ok=True)

    # QC folder for the registration, inside the output registered dataset
    qc_dir = Path(os.path.join(out_root, "QC"))
    qc_dir.mkdir(parents=True, exist_ok=True)

    groups = find_groups(bids_root)
    print(f"Found {len(groups)} (subject, contrast, chunk) groups")

    for (sub, contrast, chunk), sessions in tqdm.tqdm(groups.items(), desc="Groups"):
        group_label = f"{sub} [{contrast}{(' ' + chunk) if chunk else ''}]"
        if len(sessions) < 2:
            print(f"  Skipping {group_label}: only {len(sessions)} session(s)")
            continue

        baseline_dir, baseline_img, baseline_label = sessions[0]
        baseline_ses = baseline_dir.name

        # --- Check if all output files already exist ---
        out_baseline_anat = out_root / sub / baseline_ses / "anat"
        out_baseline_label = out_root / "derivatives" / "labels" / sub / baseline_ses / "anat"
        all_exist = (out_baseline_anat / baseline_img.name).exists() and (out_baseline_label / baseline_label.name).exists()
        for ses_dir, fu_img, fu_label in sessions[1:]:
            fu_ses = ses_dir.name
            out_fu_anat = out_root / sub / fu_ses / "anat"
            out_fu_label = out_root / "derivatives" / "labels" / sub / fu_ses / "anat"
            if not (out_fu_anat / fu_img.name).exists() or not (out_fu_label / fu_label.name).exists():
                all_exist = False
                break
        if all_exist:
            print(f"  Skipping {group_label}: all registered outputs already exist")
            continue

        # Pred directory for SC/disc predictions
        pred_dir = Path(args.p) / sub

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # --- Segment baseline ---
            baseline_stem = baseline_img.stem.replace('.nii', '')
            baseline_sc_seg = pred_dir / baseline_ses / f"{baseline_stem}_sc-seg.nii.gz"
            if not baseline_sc_seg.exists():
                segment_sc(baseline_img, baseline_sc_seg)

            baseline_temp_disc = tmpdir / f"{baseline_stem}_disc-labels.nii.gz"
            baseline_disc = pred_dir / baseline_ses / f"{baseline_stem}_disc-labels.nii.gz"
            if not baseline_disc.exists():
                try:
                    segment_discs(baseline_img, baseline_temp_disc, baseline_disc)
                except subprocess.CalledProcessError:
                    print(f"  Skipping {group_label}: baseline disc segmentation failed for {baseline_img.name}")
                    log_disc_seg_failure(baseline_img, out_root)
                    continue

            # --- Copy baseline to output as-is ---
            out_baseline_anat.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_img, out_baseline_anat / baseline_img.name)
            # Copy the baseline label into derivatives/labels
            out_baseline_label.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_label, out_baseline_label / baseline_label.name)

            # --- Process each follow-up ---
            for ses_dir, fu_img, fu_label in sessions[1:]:
                # Logging
                print("----------------------------------------------------------------")
                print(f"  Registering {fu_img.name} to baseline {baseline_img.name}")
                fu_ses = ses_dir.name

                # Segment follow-up
                fu_stem = fu_img.stem.replace('.nii', '')
                fu_sc_seg = pred_dir / fu_ses / f"{fu_stem}_sc-seg.nii.gz"
                if not fu_sc_seg.exists():
                    segment_sc(fu_img, fu_sc_seg)

                fu_temp_disc = tmpdir / f"{fu_stem}_disc-labels.nii.gz"
                fu_disc = pred_dir / fu_ses / f"{fu_stem}_disc-labels.nii.gz"
                if not fu_disc.exists():
                    try:
                        segment_discs(fu_img, fu_temp_disc, fu_disc)
                    except subprocess.CalledProcessError:
                        print(f"  Skipping follow-up {fu_img.name} in {group_label}: disc segmentation failed")
                        log_disc_seg_failure(fu_img, out_root)
                        continue

                # Both baseline and follow-up disc files need to have the same discs present, so we only keep the common ones.
                # Write to new files so the original disc segmentations (including the shared baseline) are not overwritten.
                fu_disc_common = pred_dir / fu_ses / f"{fu_stem}_disc-labels_common.nii.gz"
                baseline_disc_common = pred_dir / baseline_ses / f"{baseline_stem}_disc-labels_common_{fu_ses}.nii.gz"
                keep_common_levels_only(fu_disc, baseline_disc, fu_disc_common, baseline_disc_common)

                # Output directories
                out_fu_anat = out_root / sub / fu_ses / "anat"
                out_fu_anat.mkdir(parents=True, exist_ok=True)
                out_fu_label = out_root / "derivatives" / "labels" / sub / fu_ses / "anat"
                out_fu_label.mkdir(parents=True, exist_ok=True)

                # Register follow-up to baseline
                reg_temp_output = tmpdir / f"{fu_stem}_reg.nii.gz"
                print("adadad")
                print(reg_temp_output)
                reg_output = out_root / sub / fu_ses / "anat" / fu_img.name
                if not reg_output.exists():
                    register(fu_img, baseline_img, fu_sc_seg, baseline_sc_seg, fu_disc_common, baseline_disc_common, reg_temp_output, qc_dir)
                    # Copy registered image
                    shutil.copy2(reg_temp_output, reg_output)

                    # Find the warping field produced by sct_register_multimodal
                    warp_field = tmpdir / f"warp_{fu_img.name.replace('.nii.gz', '')}2{baseline_img.name.replace('.nii.gz', '')}.nii.gz"
                    # Apply warping field to lesion label, save into derivatives/labels
                    reg_label = tmpdir / f"{fu_label.stem.replace('.nii', '')}_reg.nii.gz"
                    apply_transfo(fu_label, baseline_img, warp_field, reg_label)
                    shutil.copy2(reg_label, out_fu_label / fu_label.name)

    print("Done.")


if __name__ == "__main__":
    main()
