import json
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Sampler
from monai import transforms as T
from monai.data import MetaTensor


# ------------------------------------------------------------------ #
# 1. Dataset class
# ------------------------------------------------------------------ #

class LongitudinalLesionDataset(Dataset):
    """
    Loads consecutive image pairs for longitudinal MS lesion segmentation.
    Each sample is a dict with keys: image1, label1, image2, label2.
    All volumes are returned as float32 tensors with a channel dim: (1, H, W, D).
    """

    def __init__(self, json_path: str, split: str = "train", transform=None):
        """
        Args:
            json_path  : path to the dataset JSON produced by create_datalist.py
            split      : "train", "validation", or "test"
            transform  : optional MONAI/callable transform applied to each sample dict
        """
        with open(json_path) as f:
            dataset = json.load(f)

        assert split in ("train", "validation", "test"), \
            f"split must be train/validation/test, got {split}"

        self.samples   = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _load_nifti(self, path: str) -> MetaTensor:
        """Load a NIfTI file and return a (1, H, W, D) float32 MetaTensor.

        The affine is preserved so that downstream MONAI transforms
        (Orientationd, Spacingd, …) operate in real anatomical space.
        Without it, reorientation and resampling silently become no-ops.
        """
        nii = nib.load(path)
        vol = nii.get_fdata(dtype=np.float32)
        tensor = torch.from_numpy(vol).unsqueeze(0)             # add channel dim
        affine = torch.as_tensor(nii.affine, dtype=torch.float32)
        return MetaTensor(tensor, affine=affine)

    def __getitem__(self, idx: int) -> dict:
        entry = self.samples[idx]

        sample = {
            "image1":       self._load_nifti(entry["image1"]),
            "label1":       self._load_nifti(entry["label1"]),
            "image2":       self._load_nifti(entry["image2"]),
            "label2":       self._load_nifti(entry["label2"]),
            # metadata — not tensors, kept out of the collate stack
            "subject":      entry["subject"],
            "contrast":     entry["contrast"],
            "session1":     entry["session1"],
            "session2":     entry["session2"],
            "image2_path":  entry["image2"],
        }

        if self.transform:
            sample = self.transform(sample)
            # RandCropByPosNegLabeld (training) returns a list of `num_samples`
            # crops; we use num_samples=1, so unwrap back to a single dict.
            if isinstance(sample, list):
                sample = sample[0]

        return sample


# ------------------------------------------------------------------ #
# 2. MONAI transforms
# ------------------------------------------------------------------ #

def get_initial_patch_size(patch_size, rotation: float = 0.52, scale_min: float = 0.7):
    """Enlarged patch to crop BEFORE spatial augmentation (nnU-Net's
    "no black borders" trick).

    Rotation/zoom are applied to a patch larger than the final one, which is then
    centre-cropped to `patch_size` — so the augmentations never pull zero-padding
    into the final patch. The enlarged size is the bounding box of `patch_size`
    under the max single-axis rotation, divided by the minimum zoom factor (this
    mirrors nnU-Net's `get_patch_size`).
    """
    p = np.asarray(patch_size, dtype=float)

    def bbox(a: int, b: int, ang: float):
        out = p.copy()
        ca, sa = abs(np.cos(ang)), abs(np.sin(ang))
        out[a] = p[a] * ca + p[b] * sa
        out[b] = p[a] * sa + p[b] * ca
        return out

    candidates = np.vstack([p, bbox(1, 2, rotation), bbox(0, 2, rotation), bbox(0, 1, rotation)])
    initial = candidates.max(axis=0) / scale_min
    return tuple(int(np.ceil(s)) for s in initial)


def get_transforms(split: str, target_shape=(64, 64, 160), crop: bool = True,
                   oversample_rate: float = 0.33):
    """
    Returns a MONAI Compose for training or inference.
    Spatial augmentations are applied IDENTICALLY to both timepoints and labels.

    `target_shape` is given in RPI axis order (R-L, P-A, I-S). The default
    (64, 64, 160) is long along I-S (dim 2) so every patch contains a large
    extent of the spinal cord.

    Training (split="train") uses nnU-Net-style patch sampling:
      * current timepoint (image2/label2): a foreground-oversampled RANDOM crop
        — `oversample_rate` of patches are centred on a lesion voxel, the rest on
        background (nnU-Net's oversample_foreground_percent).
      * previous timepoint (image1/label1): a centre pad/crop, fed as fixed
        global context (matches the sliding-window validation predictor).
      * both are cropped to an ENLARGED patch first; spatial augmentation runs on
        it and a final centre crop trims to `target_shape`, so rotation/zoom
        never leave black borders.

    `crop` only affects val/test: when True, centre pad/crop to `target_shape`
    (patch-level eval); when False (full-volume eval), keep native size so
    sliding-window inference can tile the whole image — set DataLoader
    batch_size to 1 because volumes then have different shapes.
    """
    image_keys = ["image1", "image2"]
    label_keys = ["label1", "label2"]
    all_keys   = image_keys + label_keys

    # --- shared geometry: orient to RPI + resample to 1mm iso (every split) ---
    pre = [
        T.Orientationd(keys=all_keys, axcodes="RPI", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        T.Spacingd(keys=image_keys + label_keys, pixdim=(1.0, 1.0, 1.0),
                   mode=["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)),
    ]

    if split == "train":
        # Crop to an enlarged patch so rotation/zoom never introduce black borders.
        initial = get_initial_patch_size(target_shape)
        crop_tf = [
            # Current timepoint: foreground-oversampled random crop.
            # pos/(pos+neg) = oversample_rate → that fraction of patches are
            # centred on a lesion voxel (nnU-Net oversample_foreground_percent).
            T.SpatialPadd(keys=["image2", "label2"], spatial_size=initial),
            T.RandCropByPosNegLabeld(
                keys=["image2", "label2"], label_key="label2",
                spatial_size=initial,
                pos=oversample_rate, neg=1.0 - oversample_rate,
                num_samples=1, allow_smaller=False,
            ),
            # Previous timepoint: centre pad/crop, used as fixed global context.
            T.ResizeWithPadOrCropd(keys=["image1", "label1"], spatial_size=initial),
        ]
        normalise = [
            # Intensity normalise images only
            T.NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            T.ToTensord(keys=all_keys),
        ]
        augment = [
            # ── Spatial (applied identically to both timepoints and labels) ───
            #
            # Mirror transform: flip each axis independently with p=0.5
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
            # Rotation ±30° (0.52 rad) on all axes, p=0.2
            # Images: bilinear interpolation; labels: nearest-neighbour
            T.RandRotated(
                keys=all_keys,
                range_x=0.52, range_y=0.52, range_z=0.52,
                prob=0.2,
                mode=["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys),
                padding_mode="zeros",
            ),
            # Scaling 0.7–1.4, p=0.2; keep_size crops/pads back to original shape
            T.RandZoomd(
                keys=all_keys,
                min_zoom=0.7, max_zoom=1.4,
                prob=0.2,
                mode=["trilinear"] * len(image_keys) + ["nearest"] * len(label_keys),
                keep_size=True,
            ),

            # ── Intensity (images only — never applied to labels) ─────────────
            #
            # Gaussian noise: nnUNet uses variance ~ U(0, 0.1) → std up to 0.316
            T.RandGaussianNoised(keys=image_keys, mean=0.0, std=0.1, prob=0.1),
            # Gaussian blur: sigma U(0.5, 1.0), p=0.2
            T.RandGaussianSmoothd(
                keys=image_keys,
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
                prob=0.2,
            ),
            # Multiplicative brightness: multiplier U(0.75, 1.25), p=0.15
            # RandScaleIntensity multiplies by (1 + factor), so factors=0.25 → [0.75, 1.25]
            T.RandScaleIntensityd(keys=image_keys, factors=0.25, prob=0.15),
            # Contrast adjustment: gamma U(0.75, 1.25), p=0.15
            T.RandAdjustContrastd(keys=image_keys, gamma=(0.75, 1.25), prob=0.15),
            # Low-resolution simulation: downsample then upsample, p=0.25
            T.RandSimulateLowResolutiond(
                keys=image_keys,
                zoom_range=(0.5, 1.0),
                prob=0.25,
            ),
            # Gamma correction: gamma U(0.7, 1.5), p=0.3
            T.RandAdjustContrastd(keys=image_keys, gamma=(0.7, 1.5), prob=0.3),
        ]
        # Trim the augmented enlarged patch down to the final patch size.
        final_crop = [T.CenterSpatialCropd(keys=all_keys, roi_size=target_shape)]
        return T.Compose(pre + crop_tf + normalise + augment + final_crop)

    # --- validation / test ---
    base = list(pre)
    if crop:
        # Patch-level eval: centre pad/crop to a uniform size.
        base.append(T.ResizeWithPadOrCropd(keys=all_keys, spatial_size=target_shape))
    base += [
        T.NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        T.ToTensord(keys=all_keys),
    ]
    return T.Compose(base)


# ------------------------------------------------------------------ #
# 3. Fixed-length sampler (nnU-Net-style epoch)
# ------------------------------------------------------------------ #

class FixedLengthRandomSampler(Sampler):
    """
    Yields a fixed number of random indices (with replacement) per epoch, so an
    "epoch" is a fixed number of iterations decoupled from the dataset size —
    matching nnU-Net (num_iterations_per_epoch × batch_size samples per epoch,
    1000 epochs).

    Foreground oversampling is no longer the sampler's job: it is handled at the
    patch level by RandCropByPosNegLabeld in get_transforms, which centres a
    fraction of crops on a lesion voxel (closer to nnU-Net than picking whole
    foreground-containing volumes, and it removes the startup label scan).
    """

    def __init__(self, dataset_len: int, num_samples: int):
        self.dataset_len = dataset_len
        self.num_samples = num_samples

    def __iter__(self):
        return iter(np.random.randint(0, self.dataset_len, size=self.num_samples).tolist())

    def __len__(self) -> int:
        return self.num_samples


# ------------------------------------------------------------------ #
# 4. Custom collate — handles the string metadata fields
# ------------------------------------------------------------------ #

def longitudinal_collate(batch: list) -> dict:
    """
    Default torch collate breaks on string fields.
    This stacks tensors and collects strings into lists.
    """
    tensor_keys = {"image1", "label1", "image2", "label2"}
    string_keys = {"subject", "contrast", "session1", "session2", "image2_path"}

    out = {}
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch])
    for key in string_keys:
        out[key] = [b[key] for b in batch]
    return out


# ------------------------------------------------------------------ #
# 5. DataLoader factory
# ------------------------------------------------------------------ #

def get_dataloaders(json_path: str,
                    target_shape=(64, 64, 160),
                    batch_size: int = 2,
                    num_workers: int = 4,
                    oversample_rate: float = 0.33,
                    eval_full_volume: bool = True,
                    num_iterations_per_epoch: int = 250):
    """
    Returns (train, val, test) DataLoaders.

    Training mirrors nnU-Net: each epoch is a fixed `num_iterations_per_epoch`
    batches (FixedLengthRandomSampler), and `oversample_rate` of the patches are
    centred on a lesion voxel via RandCropByPosNegLabeld in get_transforms.
    Training always operates on `target_shape` patches.

    `eval_full_volume`: when True (default), the val/test loaders return whole
    volumes (no crop) at batch_size 1, so evaluation can use sliding-window
    inference over the full image rather than scoring arbitrary crops. When
    False, val/test also crop to `target_shape` (the old patch-level behaviour).
    """
    loaders = {}
    for split in ("train", "validation", "test"):
        is_train = split == "train"
        crop = True if is_train else (not eval_full_volume)
        ds = LongitudinalLesionDataset(
            json_path  = json_path,
            split      = split,
            transform  = get_transforms(split, target_shape, crop=crop,
                                        oversample_rate=oversample_rate),
        )
        if is_train:
            # Fixed-length epoch (decoupled from dataset size); foreground
            # oversampling is done at the patch level in get_transforms.
            sampler = FixedLengthRandomSampler(
                len(ds), num_samples=num_iterations_per_epoch * batch_size
            )
            loaders[split] = DataLoader(
                ds,
                batch_size  = batch_size,
                sampler     = sampler,        # replaces shuffle=True
                num_workers = num_workers,
                pin_memory  = True,
                collate_fn  = longitudinal_collate,
            )
        else:
            # Full-volume eval cannot stack variable-sized volumes → batch_size 1.
            eval_bs = 1 if eval_full_volume else batch_size
            loaders[split] = DataLoader(
                ds,
                batch_size  = eval_bs,
                shuffle     = False,
                num_workers = num_workers,
                pin_memory  = True,
                collate_fn  = longitudinal_collate,
            )

    return loaders["train"], loaders["validation"], loaders["test"]