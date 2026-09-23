"""
This script correlates the longitudinal change in spinal cord lesion volume
(M0 -> M12) with the longitudinal change in EDSS sub-scores, for the canproco
dataset.

Lesion volumes are computed from either the MANUAL lesion labels or from
lesion segmentations PREDICTED by sct_deepseg (task lesion_ms, with
test-time augmentation), selected via --seg-source {manual,predicted}
(default: manual).

For each subject with a manual lesion label at both ses-M0 and ses-M12 under
<data>/derivatives/labels-ms-spinal-cord-only:
  - with --seg-source manual: the lesion volume is computed directly from
    the manual label mask.
  - with --seg-source predicted: the manual label path is used only to (a)
    confirm the session exists and (b) derive the path to the corresponding
    raw image (by dropping the "derivatives/labels-..." prefix and the
    "_label-lesion_seg" suffix). sct_deepseg is then run on that raw image:

        SCT_USE_GPU=1 sct_deepseg lesion_ms -i image.nii.gz -o output_path.nii.gz -test-time-aug

    Predictions are cached under <output>/predictions/sub-X/ses-Y/anat/ and
    are not recomputed on subsequent runs unless --overwrite-predictions is
    passed. The lesion volume (mm3) at each timepoint is then computed from
    the resulting predicted mask instead of the manual one.

The subject is matched to its EDSS records via the SC_ID column of the EDSS
csv (SC_ID is expected to hold the subject id without the "sub-" prefix, e.g.
"van222" for "sub-van222"), and the EDSS sub-scores are read at the row where
"Data Collection Point" equals "M0" / "M12".

Subjects (or individual sub-X_ses-Y sessions) listed in the exclude yml file
are skipped entirely.

For each EDSS column of interest (EDSSVisual, EDSSBrainstem, EDSSPyramidal,
EDSSCerebellar, EDSSSensory, EDSSBladderBowel, EDSSMental, EDSSAmbulation,
EDSSTotal), the script correlates the delta EDSS score (M12 - M0) against
both the absolute lesion volume change (mm3) and the percent lesion volume
change, using Pearson and Spearman correlation.

Outputs (written to --output), tagged with the --seg-source used:
    predictions/sub-X/ses-Y/anat/*_pred_seg.nii.gz     predicted lesion masks (--seg-source predicted only)
    lesion_edss_data_<seg-source>.csv                  per-subject lesion volumes, EDSS scores and deltas
    lesion_edss_correlations_<seg-source>.csv          correlation coefficients and p-values per EDSS column
    plots_<seg-source>/delta_<column>.png              scatter plot of delta lesion volume vs delta EDSS

Arguments:
    --data:       Path to the canproco BIDS dataset root (contains derivatives/labels-ms-spinal-cord-only)
    --edss:       Path to the canproco EDSS.csv file
    --exclude:    Path to a yml file listing sub-X_ses-Y entries to exclude
    --output:     Path to the output directory where results and predictions are saved
    --seg-source: 'manual' (default) to use the manual labels, or 'predicted' to run sct_deepseg
    --no-plots:   Skip generating scatter plots
    --no-gpu:     Run sct_deepseg on CPU (SCT_USE_GPU=0) instead of GPU (--seg-source predicted only)
    --overwrite-predictions: Re-run sct_deepseg even if a prediction already exists (--seg-source predicted only)

Pierre-Louis Benveniste
"""

import argparse
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import pearsonr, spearmanr

SESSIONS = ["ses-M0", "ses-M12"]

EDSS_COLUMNS = [
    "EDSSVisual",
    "EDSSBrainstem",
    "EDSSPyramidal",
    "EDSSCerebellar",
    "EDSSSensory",
    "EDSSBladderBowel",
    "EDSSMental",
    "EDSSAmbulation",
    "EDSSTotal",
]

VOLUME_METRICS = ["delta_lesion_volume_mm3", "pct_change_lesion_volume"]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Correlate longitudinal lesion volume change with longitudinal EDSS score change for canproco"
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the canproco BIDS dataset root")
    parser.add_argument("--edss", type=str, required=True, help="Path to the canproco EDSS.csv file")
    parser.add_argument("--exclude", type=str, required=True, help="Path to a yml file listing sub-X_ses-Y entries to exclude")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory where results and predictions are saved")
    parser.add_argument("--seg-source", type=str, choices=["manual", "predicted"], default="manual",
                        help="Use the manual lesion labels or run sct_deepseg to predict them (default: manual)")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating scatter plots")
    parser.add_argument("--no-gpu", action="store_true", help="Run sct_deepseg on CPU (SCT_USE_GPU=0) instead of GPU (--seg-source predicted only)")
    parser.add_argument("--overwrite-predictions", action="store_true",
                        help="Re-run sct_deepseg even if a prediction already exists (--seg-source predicted only)")
    return parser


def load_exclude_set(exclude_path: str) -> set:
    """Loads a flat set of 'sub-X_ses-Y' strings from a YAML exclude file."""
    with open(exclude_path, "r") as f:
        exclude_list = yaml.load(f, Loader=yaml.FullLoader)
    return set(exclude_list["to_exclude"])


def find_session_label_file(subject_dir: Path, ses: str) -> Path:
    """Returns the lesion label file for a subject/session, or None if missing."""
    anat_dir = subject_dir / ses / "anat"
    if not anat_dir.exists():
        return None
    candidates = sorted(anat_dir.glob("*_label-lesion_seg.nii.gz"))
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(f"Multiple lesion label files found in {anat_dir}, using {candidates[0].name}")
    return candidates[0]


def compute_lesion_volume_mm3(label_path: Path) -> float:
    """Computes the lesion volume (mm3) of a binary/probabilistic segmentation mask."""
    img = nib.load(str(label_path))
    data = img.get_fdata()
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))
    n_lesion_voxels = int(np.count_nonzero(data > 0))
    return n_lesion_voxels * voxel_volume


def derive_image_path(label_file: Path, labels_root: Path, dataset_path: Path) -> Path:
    """Maps a derivative label file to its matching raw image path under the dataset root, by
    dropping the derivatives/labels-... prefix and the _label-lesion_seg suffix."""
    relative = label_file.relative_to(labels_root)
    image_name = relative.name.replace("_label-lesion_seg.nii.gz", ".nii.gz")
    return dataset_path / relative.parent / image_name


def build_prediction_path(label_file: Path, labels_root: Path, predictions_root: Path) -> Path:
    """Returns the path where the sct_deepseg prediction for a given manual label file is stored."""
    relative = label_file.relative_to(labels_root)
    pred_name = relative.name.replace("_label-lesion_seg.nii.gz", "_pred_seg.nii.gz")
    return predictions_root / relative.parent / pred_name


def run_sct_deepseg_prediction(image_path: Path, output_path: Path, use_gpu: bool, overwrite: bool) -> bool:
    """Runs sct_deepseg lesion_ms (with test-time augmentation) on image_path, writing to output_path.
    Skips running if output_path already exists and overwrite is False. Returns True on success."""
    if output_path.exists() and not overwrite:
        logger.info(f"Prediction already exists, skipping: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SCT_USE_GPU"] = "1" if use_gpu else "0"
    cmd = ["sct_deepseg", "lesion_ms", "-i", str(image_path), "-o", str(output_path), "-test-time-aug"]

    logger.info(f"Running (SCT_USE_GPU={env['SCT_USE_GPU']}): {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sct_deepseg failed for {image_path}:\n{result.stderr}")
        return False
    if not output_path.exists():
        logger.error(f"sct_deepseg reported success but no output was found at {output_path}")
        return False
    return True


def get_segmentation_files(
    subject_id: str, label_files: dict, labels_root: Path, data_path: Path,
    seg_source: str, predictions_root: Path, use_gpu: bool, overwrite: bool,
) -> dict:
    """Returns {ses: Path} to the mask to compute lesion volume from, for each session, according
    to seg_source. For 'manual' this is just the label files. For 'predicted', the raw image is
    located from the label file and sct_deepseg is run (or its cached output reused). Returns None
    if any session's mask could not be obtained (missing image, or a failed prediction)."""
    if seg_source == "manual":
        return dict(label_files)

    seg_files = {}
    for ses in SESSIONS:
        image_path = derive_image_path(label_files[ses], labels_root, data_path)
        if not image_path.exists():
            logger.warning(f"{subject_id}: raw image not found at {image_path}, skipping")
            return None

        pred_path = build_prediction_path(label_files[ses], labels_root, predictions_root)
        if not run_sct_deepseg_prediction(image_path, pred_path, use_gpu, overwrite):
            logger.warning(f"{subject_id}: prediction failed for {ses}, skipping")
            return None

        seg_files[ses] = pred_path

    return seg_files


def collect_lesion_volumes(
    data_path: Path, exclude_set: set, seg_source: str,
    predictions_root: Path = None, use_gpu: bool = True, overwrite: bool = False,
) -> pd.DataFrame:
    """Computes M0/M12 lesion volumes for every subject with both sessions available and not
    excluded, from either the manual labels or sct_deepseg predictions (see seg_source)."""
    labels_root = data_path / "derivatives" / "labels-ms-spinal-cord-only"
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_root}")

    subject_dirs = sorted(d for d in labels_root.iterdir() if d.is_dir() and d.name.startswith("sub-"))

    rows = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        if any(f"{subject_id}_{ses}" in exclude_set for ses in SESSIONS):
            logger.info(f"{subject_id}: excluded via exclude file, skipping")
            continue

        label_files = {ses: find_session_label_file(subject_dir, ses) for ses in SESSIONS}
        if any(label_files[ses] is None for ses in SESSIONS):
            missing = [ses for ses in SESSIONS if label_files[ses] is None]
            logger.warning(f"{subject_id}: missing lesion label for {missing}, skipping")
            continue

        seg_files = get_segmentation_files(
            subject_id, label_files, labels_root, data_path, seg_source, predictions_root, use_gpu, overwrite
        )
        if seg_files is None:
            continue

        volume_m0 = compute_lesion_volume_mm3(seg_files["ses-M0"])
        volume_m12 = compute_lesion_volume_mm3(seg_files["ses-M12"])
        delta_volume = volume_m12 - volume_m0
        pct_change = (delta_volume / volume_m0 * 100) if volume_m0 > 0 else np.nan

        rows.append({
            "subject": subject_id,
            "sc_id": subject_id.replace("sub-", ""),
            "lesion_volume_mm3_M0": volume_m0,
            "lesion_volume_mm3_M12": volume_m12,
            "delta_lesion_volume_mm3": delta_volume,
            "pct_change_lesion_volume": pct_change,
        })

    logger.info(f"Collected lesion volumes for {len(rows)} subjects (M0 and M12 both present, not excluded)")
    return pd.DataFrame(rows)


def load_edss(edss_path: str) -> pd.DataFrame:
    """Loads the EDSS csv and normalizes the SC_ID / Data Collection Point columns for matching."""
    edss_df = pd.read_csv(edss_path, dtype=str)
    edss_df["SC_ID_norm"] = edss_df["SC_ID"].astype(str).str.strip().str.lower().str.replace("^sub-", "", regex=True)
    edss_df["DCP_norm"] = edss_df["Data Collection Point"].astype(str).str.strip().str.upper()
    return edss_df


def get_edss_row(edss_df: pd.DataFrame, sc_id: str, dcp_label: str) -> pd.Series:
    """Returns the EDSS row matching a subject's SC_ID and a Data Collection Point (e.g. 'M0'), or None."""
    matches = edss_df[(edss_df["SC_ID_norm"] == sc_id.lower()) & (edss_df["DCP_norm"] == dcp_label.upper())]
    if matches.empty:
        return None
    if len(matches) > 1:
        logger.warning(f"Multiple EDSS rows found for SC_ID={sc_id}, {dcp_label}, using the first one")
    return matches.iloc[0]


def collect_edss_deltas(lesion_df: pd.DataFrame, edss_df: pd.DataFrame) -> pd.DataFrame:
    """Adds EDSS M0/M12/delta columns to the lesion volume dataframe, for each EDSS column of interest."""
    rows = []
    for _, lesion_row in lesion_df.iterrows():
        sc_id = lesion_row["sc_id"]
        row_m0 = get_edss_row(edss_df, sc_id, "M0")
        row_m12 = get_edss_row(edss_df, sc_id, "M12")

        record = lesion_row.to_dict()
        if row_m0 is None or row_m12 is None:
            missing = "M0" if row_m0 is None else "M12"
            logger.warning(f"{lesion_row['subject']}: no EDSS record found for {missing} (SC_ID={sc_id}), scores will be NaN")

        for col in EDSS_COLUMNS:
            val_m0 = pd.to_numeric(row_m0[col], errors="coerce") if row_m0 is not None else np.nan
            val_m12 = pd.to_numeric(row_m12[col], errors="coerce") if row_m12 is not None else np.nan
            record[f"edss_{col}_M0"] = val_m0
            record[f"edss_{col}_M12"] = val_m12
            record[f"delta_edss_{col}"] = val_m12 - val_m0

        rows.append(record)

    return pd.DataFrame(rows)


def compute_correlations(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Pearson and Spearman correlations between each lesion volume change metric
    and each delta EDSS column, over subjects with non-missing values for both."""
    results = []
    for col in EDSS_COLUMNS:
        delta_col = f"delta_edss_{col}"
        for volume_metric in VOLUME_METRICS:
            paired = merged_df[[volume_metric, delta_col]].dropna()
            n = len(paired)

            if n < 3:
                logger.warning(f"{col} vs {volume_metric}: only {n} paired observations, skipping correlation")
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r, pearson_p = pearsonr(paired[volume_metric], paired[delta_col])
                spearman_r, spearman_p = spearmanr(paired[volume_metric], paired[delta_col])

            results.append({
                "edss_column": col,
                "volume_metric": volume_metric,
                "n": n,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })

    return pd.DataFrame(results)


def plot_correlations(merged_df: pd.DataFrame, plots_dir: Path):
    """Saves a scatter plot of delta lesion volume (mm3) vs delta EDSS for each EDSS column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    volume_metric = "delta_lesion_volume_mm3"
    for col in EDSS_COLUMNS:
        delta_col = f"delta_edss_{col}"
        paired = merged_df[[volume_metric, delta_col]].dropna()
        if paired.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(paired[volume_metric], paired[delta_col], alpha=0.7)
        if len(paired) >= 2:
            slope, intercept = np.polyfit(paired[volume_metric], paired[delta_col], 1)
            x_line = np.linspace(paired[volume_metric].min(), paired[volume_metric].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1)
        ax.set_xlabel("Delta lesion volume (mm3), M12 - M0")
        ax.set_ylabel(f"Delta {col}, M12 - M0")
        ax.set_title(f"Lesion volume change vs {col} change (n={len(paired)})")
        fig.tight_layout()
        fig.savefig(plots_dir / f"delta_{col}.png", dpi=150)
        plt.close(fig)

    logger.info(f"Saved scatter plots to {plots_dir}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    exclude_set = load_exclude_set(args.exclude)
    logger.info(f"Loaded {len(exclude_set)} exclude entries from {args.exclude}")

    logger.info(f"Using --seg-source={args.seg_source} lesion segmentations")
    predictions_root = output_path / "predictions"
    lesion_df = collect_lesion_volumes(
        data_path, exclude_set, args.seg_source,
        predictions_root=predictions_root, use_gpu=not args.no_gpu, overwrite=args.overwrite_predictions,
    )
    if lesion_df.empty:
        logger.error("No subjects with lesion volumes at both M0 and M12 were found, exiting")
        return

    edss_df = load_edss(args.edss)
    merged_df = collect_edss_deltas(lesion_df, edss_df)

    data_csv = output_path / f"lesion_edss_data_{args.seg_source}.csv"
    merged_df.to_csv(data_csv, index=False)
    logger.info(f"Saved per-subject lesion/EDSS data to {data_csv}")

    correlations_df = compute_correlations(merged_df)
    correlations_csv = output_path / f"lesion_edss_correlations_{args.seg_source}.csv"
    correlations_df.to_csv(correlations_csv, index=False)
    logger.info(f"Saved correlation results to {correlations_csv}")
    logger.info("Correlation summary:\n" + correlations_df.to_string(index=False))

    if not args.no_plots:
        plot_correlations(merged_df, output_path / f"plots_{args.seg_source}")


if __name__ == "__main__":
    main()
