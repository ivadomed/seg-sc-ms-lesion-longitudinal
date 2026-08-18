"""
This file creates the MSD-style JSON datalist to train a longitudinal model.
Creates pairs of consecutive labeled images of the same contrast for each subject.

Multiple BIDS datasets can be pooled into a single MSD dataset: longitudinal
pairs are built INDEPENDENTLY within each dataset (a pair never spans two
datasets), every pair is tagged with its source `site`, and the splits are
made at the (site, subject) level so the same subject ID appearing in two
cohorts is never merged or leaked across train/val/test.

Each pair records: the path to the 2 images, the path to the 2 labels, the
2 session dates, the subject ID and the site name.

canproco pools multiple acquisition sites under one dataset root; subjects
are distinguished by ID prefix (sub-tor001 -> Toronto, sub-mon002 -> Montreal,
sub-cal003 -> Calgary, sub-van004 -> Vancouver, sub-edm005 -> Edmonton), so
canproco pairs are tagged with site "canproco-<city>" instead of a single
"canproco" site.

Derivative labels live under either <dataset>/derivatives/labels-ms-spinal-cord-only
or, if that folder does not exist, <dataset>/derivatives/labels. The matching raw
image lives at the same relative sub-X/ses-Y/anat path directly under <dataset>.

Sessions follow either "ses-M<number>" (e.g. canproco) or "ses-<YYYYMMDD>" (e.g.
ms-ucsf-2025); they are sorted chronologically accordingly, falling back to
lexicographic order otherwise. Consecutive sessions (after exclusions) are paired,
and within a pair, images/labels are matched by their shared BIDS entities (e.g.
acq-ax_chunk-1_T2w) so only images that exist at both timepoints are paired.

The dataset named in TEST_SET is held out entirely as an external test set; every
other dataset's pairs are split into train/validation/test at the (site, subject)
level.

Arguments:
    --data:    Path to a parent folder containing one or more BIDS datasets folder.
    --output:  Path to the output directory where dataset json is saved

Pierre-Louis Benveniste
"""

import os
import re
import json
import yaml
from tqdm import tqdm
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict


TEST_SET = "ms-ucsf-2025"

# canproco pools these sites under one dataset root, distinguished by subject ID prefix.
CANPROCO_SITE_PREFIXES = {
    "tor": "toronto",
    "mon": "montreal",
    "cal": "calgary",
    "van": "vancouver",
    "edm": "edmonton",
}

# Exclude file path
EXCLUDE_FILE = Path(__file__).parent / "exclude.yml"

# Let's set the seed for reproducibility
seed = 42


def get_parser():
    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for longitudinal lesion segmentation')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to a parent folder containing one or more BIDS datasets folder')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory where dataset json is saved')
    return parser


def load_exclude_list(exclude_path: str) -> list:
    """Loads a flat list of strings from a YAML exclude file. Returns [] if no path is given."""
    with open(exclude_path, 'r') as f:
        exclude_list = yaml.load(f, Loader=yaml.FullLoader)
    exclude_list = exclude_list["to_exclude"]
    return exclude_list


def find_labels_root(dataset_path: Path) -> Path:
    """Returns the derivatives folder holding lesion labels for a dataset."""
    cord_only_root = dataset_path / "derivatives" / "labels-ms-spinal-cord-only"
    if cord_only_root.exists():
        return cord_only_root
    return dataset_path / "derivatives" / "labels"


def derive_image_path(label_file: Path, labels_root: Path, dataset_path: Path) -> Path:
    """Maps a derivative label file to its matching raw image path under the dataset root."""
    relative = label_file.relative_to(labels_root)
    image_name = relative.name.replace("_label-lesion_seg.nii.gz", ".nii.gz")
    return dataset_path / relative.parent / image_name


def get_entity_key(filename: str, subject_id: str, ses_full: str) -> str:
    """Returns the BIDS entities (contrast/acq/chunk...) of a label filename, stripped of
    the subject, session and label suffix, so that images can be matched across sessions."""
    key = filename.replace("_label-lesion_seg.nii.gz", "")
    prefix = f"{subject_id}_{ses_full}_"
    if key.startswith(prefix):
        key = key[len(prefix):]
    return key


def session_sort_key(ses_full: str):
    """Returns a sortable chronological key for a BIDS session label.
    Handles ses-M<number> (month offset) and ses-<YYYYMMDD> (calendar date),
    falling back to lexicographic order for anything else."""
    ses_label = ses_full.replace("ses-", "")
    month_match = re.fullmatch(r"M(\d+)", ses_label)
    if month_match:
        return (0, int(month_match.group(1)))
    date_match = re.fullmatch(r"\d{8}", ses_label)
    if date_match:
        return (1, datetime.strptime(ses_label, "%Y%m%d"))
    return (2, ses_label)


def get_site(dataset_name: str, subject_id: str) -> str:
    """Returns the site name for a subject, splitting canproco into its pooled sites."""
    if dataset_name == "canproco":
        prefix = subject_id.replace("sub-", "")[:3]
        city = CANPROCO_SITE_PREFIXES.get(prefix)
        if city is None:
            logger.warning(f"Unknown canproco site prefix for subject {subject_id}, using dataset name as site")
            return dataset_name
        return f"canproco-{city}"
    return dataset_name


def build_dataset_pairs(dataset_name: str, dataset_path: Path, exclude_list: list) -> list:
    """Builds the longitudinal pairs for a single BIDS dataset, independently of any other dataset."""
    labels_root = find_labels_root(dataset_path)
    if not labels_root.exists():
        logger.warning(f"No derivatives labels folder found for {dataset_name}, skipping")
        return []

    label_files = sorted(labels_root.rglob("*_label-lesion_seg.nii.gz"))

    # sessions_by_subject[subject_id][ses_full] -> list of {"image", "label", "entity_key"}
    sessions_by_subject = defaultdict(lambda: defaultdict(list))

    for label_file in label_files:
        relative = label_file.relative_to(labels_root)
        if len(relative.parts) < 3 or not relative.parts[1].startswith("ses-"):
            logger.warning(f"Skipping {label_file}: expected a sub-X/ses-Y/anat structure")
            continue

        subject_id, ses_full = relative.parts[0], relative.parts[1]
        if f"{subject_id}_{ses_full}" in exclude_list:
            continue

        image_path = derive_image_path(label_file, labels_root, dataset_path)
        if not image_path.exists():
            logger.warning(f"Image not found for label {label_file}")
            continue

        entity_key = get_entity_key(relative.name, subject_id, ses_full)
        sessions_by_subject[subject_id][ses_full].append({
            "label": str(label_file),
            "image": str(image_path),
            "entity_key": entity_key,
        })

    dataset_pairs = []
    for subject_id, sessions in sessions_by_subject.items():
        site = get_site(dataset_name, subject_id)
        ordered_sessions = sorted(sessions.keys(), key=session_sort_key)

        for ses_a, ses_b in zip(ordered_sessions, ordered_sessions[1:]):
            entities_a = {item["entity_key"]: item for item in sessions[ses_a]}
            entities_b = {item["entity_key"]: item for item in sessions[ses_b]}

            for key in sorted(entities_a.keys() & entities_b.keys()):
                dataset_pairs.append({
                    "images": [entities_a[key]["image"], entities_b[key]["image"]],
                    "labels": [entities_a[key]["label"], entities_b[key]["label"]],
                    "sessions": [ses_a, ses_b],
                    "subject": subject_id,
                    "site": site,
                })

    logger.info(f"{dataset_name}: built {len(dataset_pairs)} longitudinal pairs")
    return dataset_pairs


def split_subjects(subjects: list, test_size: float, random_state: int):
    """Splits a list of subjects into train/validation/test. Too few subjects to split
    meaningfully (< 3) all go to train."""
    subjects = sorted(subjects)
    if len(subjects) < 3:
        return subjects, [], []
    train, test = train_test_split(subjects, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train, test_size=test_size / (1 - test_size), random_state=random_state)
    return sorted(train), sorted(val), sorted(test)


def main():
    parser = get_parser()
    args = parser.parse_args()
    data_path  = Path(args.data)
    output_path = args.output
    test_size   = 0.1

    # Load exclude list
    exclude_list = load_exclude_list(EXCLUDE_FILE)
    if exclude_list:
        logger.info(f"Loaded {len(exclude_list)} exclude patterns from {EXCLUDE_FILE}")

    # List all BIDS datasets (list of all folder in the data input path)
    datasets = sorted(d for d in os.listdir(data_path) if (data_path / d).is_dir())
    logger.info(f"Found {len(datasets)} datasets: {datasets}")

    # Build longitudinal pairs independently for each dataset, keeping the held-out
    # external test set separate from the pool that gets split into train/val/test.
    external_test_pairs = []
    pool_pairs = []

    for dataset_name in tqdm(datasets, desc="Building pairs per dataset"):
        dataset_pairs = build_dataset_pairs(dataset_name, data_path / dataset_name, exclude_list)
        if dataset_name == TEST_SET:
            external_test_pairs.extend(dataset_pairs)
        else:
            pool_pairs.extend(dataset_pairs)

    # Split the pooled pairs into train/validation/test at the (site, subject) level,
    # so a subject's pairs never end up split across two sets.
    pairs_by_site = defaultdict(list)
    for pair in pool_pairs:
        pairs_by_site[pair["site"]].append(pair)

    train_pairs, val_pairs, test_pairs = [], [], []
    for site, site_pairs in pairs_by_site.items():
        subjects = {p["subject"] for p in site_pairs}
        train_subjects, val_subjects, test_subjects = split_subjects(subjects, test_size, seed)
        for pair in site_pairs:
            if pair["subject"] in train_subjects:
                train_pairs.append(pair)
            elif pair["subject"] in val_subjects:
                val_pairs.append(pair)
            else:
                test_pairs.append(pair)
        logger.info(f"Site {site}: {len(train_subjects)} train / {len(val_subjects)} val / {len(test_subjects)} test subjects")

    train_pairs = sorted(train_pairs, key=lambda p: (p["site"], p["subject"], p["sessions"]))
    val_pairs = sorted(val_pairs, key=lambda p: (p["site"], p["subject"], p["sessions"]))
    test_pairs = sorted(test_pairs, key=lambda p: (p["site"], p["subject"], p["sessions"]))
    external_test_pairs = sorted(external_test_pairs, key=lambda p: (p["site"], p["subject"], p["sessions"]))

    params = {}
    params["description"] = "ms-lesion-longitudinal"
    params["labels"] = {
        "0": "background",
        "1": "ms-lesion-seg",
    }
    params["license"] = "plb"
    params["modality"] = {
        "0": "MRI",
    }
    params["name"] = "ms-lesion-longitudinal"
    params["seed"] = seed
    params["reference"] = "NeuroPoly"
    params["tensorImageSize"] = "3D"
    params["train"] = train_pairs
    params["validation"] = val_pairs
    params["test"] = test_pairs
    params["externalTest"] = external_test_pairs
    params["numTrain"] = len(train_pairs)
    params["numValidation"] = len(val_pairs)
    params["numTest"] = len(test_pairs)
    params["numExternalTest"] = len(external_test_pairs)

    logger.info(f"Number of pairs -- train: {params['numTrain']}, validation: {params['numValidation']}, "
                f"test: {params['numTest']}, external test ({TEST_SET}): {params['numExternalTest']}")

    os.makedirs(output_path, exist_ok=True)
    output_file = Path(output_path) / f"dataset_{date.today()}_seed{seed}.json"
    with open(output_file, "w") as f:
        json.dump(params, f, indent=4, sort_keys=True)
    logger.info(f"Saved dataset json to {output_file}")


if __name__ == "__main__":
    main()
