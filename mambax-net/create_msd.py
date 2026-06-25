"""
This file creates the MSD-style JSON datalist to train a longitudinal model.
Creates pairs of consecutive labeled images of the same contrast for each subject.

Multiple BIDS datasets can be pooled into a single MSD dataset: longitudinal
pairs are built INDEPENDENTLY within each dataset (a pair never spans two
datasets), every pair is tagged with its source `site`, and the splits are
made at the (site, subject) level so the same subject ID appearing in two
cohorts is never merged or leaked across train/val/test.

Arguments:
    --data:   One or more BIDS dataset roots (space separated)
    --sites:  Optional source label per dataset (defaults to each dataset
              folder name); must match the number of --data paths if given
    --output: Path to the output directory where dataset json is saved
    --seed:   Random seed for reproducibility

Pierre-Louis Benveniste
"""

import os
import json
from tqdm import tqdm
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from datetime import date
from pathlib import Path
from collections import defaultdict


def get_parser():
    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for longitudinal lesion segmentation')
    parser.add_argument('--data', type=str, required=True, nargs='+',
                        help='One or more BIDS dataset roots (space separated). Pairs are built within each dataset.')
    parser.add_argument('--sites', type=str, nargs='+', default=None,
                        help='Optional source label per dataset (defaults to the dataset folder name). '
                             'If given, must match the number of --data paths.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory where dataset json is saved')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser


def get_session_date(derivative_path: Path) -> str:
    """
    Extracts the session date string (e.g. '20231121') from a BIDS path containing 'ses-YYYYMMDD'.
    Returns the raw string so it can be sorted lexicographically (ISO date format sorts correctly).
    """
    for part in derivative_path.parts:
        if part.startswith('ses-'):
            return part.replace('ses-', '')
    return ''


def get_contrast(derivative_path: Path) -> str:
    """Extracts the contrast identifier from the filename (last underscore-separated token before .nii.gz)."""
    return derivative_path.name.replace('_label-lesion_seg.nii.gz', '.nii.gz').replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')


def get_subject(derivative_path: Path) -> str:
    """Extracts the subject ID from the filename."""
    return derivative_path.name.split('_')[0]


def build_longitudinal_pairs(derivatives: list, site: str) -> list:
    """
    Groups derivatives by (subject, contrast), sorts sessions chronologically,
    and builds consecutive pairs (session N, session N+1).

    Each pair is a dict with:
        image1, label1  -> earlier timepoint
        image2, label2  -> later timepoint
        subject, contrast, session1, session2, site

    Only pairs where all four files exist on disk are included.

    Input:
        derivatives : list[Path] : all label files found under one dataset
        site        : str        : source label written into each pair

    Returns:
        pairs : list[dict]
    """
    # Group by (subject, contrast)
    groups = defaultdict(list)
    for deriv in derivatives:
        subject  = get_subject(deriv)
        contrast = get_contrast(deriv)
        session  = get_session_date(deriv)
        groups[(subject, contrast)].append((session, deriv))

    pairs = []
    for (subject, contrast), entries in groups.items():
        # Sort by session date (lexicographic sort works for YYYYMMDD)
        entries_sorted = sorted(entries, key=lambda x: x[0])

        for i in range(len(entries_sorted) - 1):
            ses1, label1_path = entries_sorted[i]
            ses2, label2_path = entries_sorted[i + 1]

            image1_path = str(label1_path).replace('_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
            image2_path = str(label2_path).replace('_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels/', '')

            if site=="canproco":
                image1_path = str(label1_path).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
                image2_path = str(label2_path).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels/', '')

            # Only keep pairs where all four files exist
            if not all(os.path.exists(p) for p in [str(label1_path), str(label2_path), image1_path, image2_path]):
                missing = [p for p in [str(label1_path), str(label2_path), image1_path, image2_path] if not os.path.exists(p)]
                logger.warning(f"Skipping pair ({site}, {subject}, {contrast}, {ses1}->{ses2}): missing files: {missing}")
                continue

            pairs.append({
                "image1":    image1_path,
                "label1":    str(label1_path),
                "image2":    image2_path,
                "label2":    str(label2_path),
                "subject":   subject,
                "contrast":  contrast,
                "session1":  ses1,
                "session2":  ses2,
                "site":      site,
            })

    return pairs


def _split_key(pair: dict) -> tuple:
    """Composite identity used for splitting: a subject is unique within its site."""
    return (pair["site"], pair["subject"])


def split_pairs_by_subject(pairs: list, test_size: float = 0.1, random_state: int = 42):
    """
    Splits pairs into train / val / test by (site, subject) — no subject from a
    given cohort appears in two splits, and identical subject IDs from different
    cohorts are treated as distinct.

    Input:
        pairs        : list[dict] : pooled output of build_longitudinal_pairs()
        test_size    : float      : fraction of subjects held out for test (and for val)
        random_state : int

    Returns:
        train, val, test : list[dict]
    """
    subjects = list({_split_key(p) for p in pairs})

    subj_train, subj_test = train_test_split(subjects, test_size=test_size, random_state=random_state)
    subj_train, subj_val  = train_test_split(subj_train, test_size=test_size / (1 - test_size), random_state=random_state)

    subj_train = set(subj_train)
    subj_val   = set(subj_val)
    subj_test  = set(subj_test)

    train = [p for p in pairs if _split_key(p) in subj_train]
    val   = [p for p in pairs if _split_key(p) in subj_val]
    test  = [p for p in pairs if _split_key(p) in subj_test]

    return train, val, test


def print_pairs_distribution(pairs: list, split_name: str):
    """Logs site + contrast distribution and subject count for a given split."""
    contrasts = [p["contrast"] for p in pairs]
    sites     = [p["site"] for p in pairs]
    subjects  = {_split_key(p) for p in pairs}
    logger.info(f"[{split_name}] {len(pairs)} pairs | {len(subjects)} subjects")
    for s in sorted(set(sites)):
        n_site = sites.count(s)
        n_subj = len({p["subject"] for p in pairs if p["site"] == s})
        logger.info(f"  site {s}: {n_site} pairs | {n_subj} subjects")
    for c in sorted(set(contrasts)):
        logger.info(f"  contrast {c}: {contrasts.count(c)} pairs")


def main():
    parser = get_parser()
    args = parser.parse_args()
    data_paths  = args.data
    output_path = args.output
    test_size   = 0.1

    # Resolve a source label for each dataset (folder name by default).
    if args.sites is not None:
        if len(args.sites) != len(data_paths):
            parser.error(f"--sites ({len(args.sites)}) must match the number of --data paths ({len(data_paths)})")
        sites = args.sites
    else:
        sites = [Path(p.rstrip('/')).name for p in data_paths]
    if len(set(sites)) != len(sites):
        parser.error(f"Duplicate site labels {sites}; pass distinct --sites so cohorts stay separable.")

    # ------------------------------------------------------------------ #
    # 1-2. Discover labels and build consecutive pairs, per dataset
    # ------------------------------------------------------------------ #
    all_pairs = []
    for data_path, site in zip(data_paths, sites):
        derivatives = list(Path(data_path).rglob('*_label-lesion_seg.nii.gz'))
        if site=="canproco":
            derivatives = list(Path(data_path).rglob('*_lesion-manual.nii.gz'))
        logger.info(f"[{site}] Found {len(derivatives)} label files under {data_path}")
        site_pairs = build_longitudinal_pairs(derivatives, site=site)
        logger.info(f"[{site}] Built {len(site_pairs)} valid consecutive pairs")
        all_pairs.extend(site_pairs)

    logger.info(f"Pooled {len(all_pairs)} pairs from {len(data_paths)} dataset(s): {sites}")

    # ------------------------------------------------------------------ #
    # 3. Train / val / test split (per-cohort subject level)
    # ------------------------------------------------------------------ #
    train_pairs, val_pairs, test_pairs = split_pairs_by_subject(
        all_pairs, test_size=test_size, random_state=args.seed
    )

    for split_name, split_pairs in [("train", train_pairs), ("validation", val_pairs), ("test", test_pairs)]:
        print_pairs_distribution(split_pairs, split_name)

    # ------------------------------------------------------------------ #
    # 4. Assemble the JSON
    # ------------------------------------------------------------------ #
    params = {
        "description":        "ms-lesion-longitudinal",
        "labels":             {"0": "background", "1": "ms-lesion-seg"},
        "license":            "plb",
        "modality":           {"0": "MRI"},
        "name":               "ms-lesion-longitudinal",
        "seed":               args.seed,
        "reference":          "NeuroPoly",
        "tensorImageSize":    "3D",
        "task":               "consecutive-pair segmentation",
        "sites":              sites,
        "train":              train_pairs,
        "validation":         val_pairs,
        "test":               test_pairs,
        "numTraining":        len(train_pairs),
        "numValidation":      len(val_pairs),
        "numTest":            len(test_pairs),
        "numSubjects":        len({_split_key(p) for p in all_pairs}),
    }

    total = params["numTraining"] + params["numValidation"] + params["numTest"]
    logger.info(f"Total pairs in dataset: {total}")
    logger.info(f"Total unique subjects:  {params['numSubjects']}")

    # ------------------------------------------------------------------ #
    # 5. Write outputs
    # ------------------------------------------------------------------ #
    os.makedirs(output_path, exist_ok=True)
    today = str(date.today())

    json_path = os.path.join(output_path, f"dataset_{today}.json")
    with open(json_path, "w") as f:
        f.write(json.dumps(params, indent=4, sort_keys=True))
    logger.info(f"Dataset JSON saved to {json_path}")


if __name__ == "__main__":
    main()
