"""
This script correlates the longitudinal change in spinal cord lesion volume
(M0 -> M12) with the longitudinal change in EDSS sub-scores, for the canproco
dataset.

For each subject with a lesion segmentation at both ses-M0 and ses-M12 under
<data>/derivatives/labels-ms-spinal-cord-only, the lesion volume (mm3) is
computed at each timepoint from the binary segmentation mask. The subject is
matched to its EDSS records via the SC_ID column of the EDSS csv (SC_ID is
expected to hold the subject id without the "sub-" prefix, e.g. "van222" for
"sub-van222"), and the EDSS sub-scores are read at the row where
"Data Collection Point" equals "M0" / "M12".

Subjects (or individual sub-X_ses-Y sessions) listed in the exclude yml file
are skipped entirely.

For each EDSS column of interest (EDSSVisual, EDSSBrainstem, EDSSPyramidal,
EDSSCerebellar, EDSSSensory, EDSSBladderBowel, EDSSMental, EDSSAmbulation,
EDSSTotal), the script correlates the delta EDSS score (M12 - M0) against
both the absolute lesion volume change (mm3) and the percent lesion volume
change, using Pearson and Spearman correlation.

Outputs (written to --output):
    lesion_edss_data.csv          per-subject lesion volumes, EDSS scores and deltas
    lesion_edss_correlations.csv  correlation coefficients and p-values per EDSS column
    plots/delta_<column>.png      scatter plot of delta lesion volume vs delta EDSS

Arguments:
    --data:    Path to the canproco BIDS dataset root (contains derivatives/labels-ms-spinal-cord-only)
    --edss:    Path to the canproco EDSS.csv file
    --exclude: Path to a yml file listing sub-X_ses-Y entries to exclude
    --output:  Path to the output directory where results are saved
    --no-plots: Skip generating scatter plots

Pierre-Louis Benveniste
"""

import argparse
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
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory where results are saved")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating scatter plots")
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


def collect_lesion_volumes(data_path: Path, exclude_set: set) -> pd.DataFrame:
    """Computes M0/M12 lesion volumes for every subject with both sessions available and not excluded."""
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

        volume_m0 = compute_lesion_volume_mm3(label_files["ses-M0"])
        volume_m12 = compute_lesion_volume_mm3(label_files["ses-M12"])
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


def plot_correlations(merged_df: pd.DataFrame, output_dir: Path):
    """Saves a scatter plot of delta lesion volume (mm3) vs delta EDSS for each EDSS column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
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

    lesion_df = collect_lesion_volumes(data_path, exclude_set)
    if lesion_df.empty:
        logger.error("No subjects with lesion volumes at both M0 and M12 were found, exiting")
        return

    edss_df = load_edss(args.edss)
    merged_df = collect_edss_deltas(lesion_df, edss_df)

    data_csv = output_path / "lesion_edss_data.csv"
    merged_df.to_csv(data_csv, index=False)
    logger.info(f"Saved per-subject lesion/EDSS data to {data_csv}")

    correlations_df = compute_correlations(merged_df)
    correlations_csv = output_path / "lesion_edss_correlations.csv"
    correlations_df.to_csv(correlations_csv, index=False)
    logger.info(f"Saved correlation results to {correlations_csv}")
    logger.info("Correlation summary:\n" + correlations_df.to_string(index=False))

    if not args.no_plots:
        plot_correlations(merged_df, output_path)


if __name__ == "__main__":
    main()
