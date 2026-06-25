import os
import numpy as np
import nibabel as nib
from loguru import logger

#  ──────────────────────────────────────────────────────────────────────────────
# Debug — save the patches seen by the model
# ──────────────────────────────────────────────────────────────────────────────
def save_batch_patches(batch: dict, out_dir: str):
    """
    Dump every volume in a batch (image1/label1/image2/label2) to NIfTI so the
    exact patches fed to the model can be inspected. Affine is taken from the
    MetaTensor when available, otherwise identity.
    """
    os.makedirs(out_dir, exist_ok=True)
    keys = ["image1", "label1", "image2", "label2"]
    bsz  = batch["image1"].shape[0]

    for b in range(bsz):
        subject  = batch.get("subject",  ["unknown"] * bsz)[b]
        session1 = batch.get("session1", ["s1"] * bsz)[b]
        session2 = batch.get("session2", ["s2"] * bsz)[b]
        prefix   = f"b{b}_{subject}_{session1}_{session2}"

        for key in keys:
            vol = batch[key][b]                       # (1, H, W, D)
            arr = vol.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Recover the affine from the MONAI MetaTensor if present
            affine = getattr(vol, "affine", None)
            if affine is not None:
                affine = affine.detach().cpu().numpy()
            else:
                affine = np.eye(4)

            path = os.path.join(out_dir, f"{prefix}_{key}.nii.gz")
            nib.save(nib.Nifti1Image(arr, affine), path)

        logger.info(f"Saved debug patches for sample {b} ({subject}) → {out_dir}")