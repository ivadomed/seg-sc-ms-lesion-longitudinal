"""
Standalone GradCAM analysis for MambaXNet.

Produces per-input-branch heatmaps showing which spatial regions drive the
model's lesion prediction.  Four target layers are supported:

  1. enc_stage0 on i_t      — where in the current image the model looks
  2. enc_stage0 on i_prev   — where in the previous image the model looks
  3. sem (output)            — which prior-lesion regions matter
  4. m_cam1 (output)         — where temporal cross-attention changes the decision

Usage:
    python gradcam.py \
        --weights /path/to/best_model.pth \
        --data   /path/to/dataset.json \
        --output /path/to/gradcam_output \
        [--n_samples 5] [--target_class 1]

Author: Pierre-Louis Benveniste
"""

import argparse
import os
import json
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, axcodes2ornt, apply_orientation
from scipy.ndimage import zoom as scipy_zoom

from mambaxnet import MambaXNet
from load_dataset import LongitudinalLesionDataset, get_transforms


# ──────────────────────────────────────────────────────────────────────────────
# GradCAM core
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D models.

    Hooks into a target layer, captures its activations on the forward pass
    and the gradients on the backward pass, then produces a spatial heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, i_t, i_prev, m_prev, target_class=1):
        """
        Returns a (B, D, H, W) heatmap in [0, 1] at the spatial resolution
        of the model output.
        """
        self.model.zero_grad()
        output = self.model(i_t, i_prev, m_prev)

        # Score for the target class, summed over spatial dims
        score = output[:, target_class].sum()
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture activations/gradients. "
                               "Check that the target layer is on the forward path.")

        # Global-average-pool gradients over spatial dims -> channel weights
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, D', H', W')
        cam = F.relu(cam)

        # Upsample to output spatial size
        spatial_size = output.shape[2:]
        cam = F.interpolate(cam, size=spatial_size, mode='trilinear', align_corners=False)
        cam = cam.squeeze(1)  # (B, D, H, W)

        # Normalize per sample
        for b in range(cam.shape[0]):
            c = cam[b]
            cmin, cmax = c.min(), c.max()
            if cmax - cmin > 0:
                cam[b] = (c - cmin) / (cmax - cmin)
            else:
                cam[b] = torch.zeros_like(c)

        return cam, output


# ──────────────────────────────────────────────────────────────────────────────
# Layer accessors — handle the fact that enc_stage0 is shared between i_t and
# i_prev, so we need wrapper modules to capture each branch separately.
# ──────────────────────────────────────────────────────────────────────────────

class BranchCapture(torch.nn.Module):
    """Identity wrapper inserted around a layer to capture a specific branch."""
    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Identity()

    def forward(self, x):
        return self.identity(x)


def patch_model_for_gradcam(model):
    """
    Patches MambaXNet.forward to insert branch-specific capture points.
    Returns a dict of {name: capture_module} that can be used as GradCAM targets.
    """
    captures = {
        # Current time-point encoder — all 6 stages
        "enc_stage0_it":    BranchCapture(),
        "enc_stage1_it":    BranchCapture(),
        "enc_stage2_it":    BranchCapture(),
        "enc_stage3_it":    BranchCapture(),
        "enc_stage4_it":    BranchCapture(),
        "enc_stage5_it":    BranchCapture(),
        # Previous time-point encoder — only first 3 stages are computed
        "enc_stage0_iprev": BranchCapture(),
        "enc_stage1_iprev": BranchCapture(),
        "enc_stage2_iprev": BranchCapture(),
        # Other branches
        "sem_output":       BranchCapture(),
        "mcam1_output":     BranchCapture(),
    }
    for name, cap in captures.items():
        cap.to(next(model.parameters()).device)

    original_forward = model.forward

    def patched_forward(i_t, i_prev, m_prev):
        # Encoder for current time-point
        e1 = model.enc_stage0(model.enc_stem(i_t))
        e1 = captures["enc_stage0_it"](e1)
        e2 = model.enc_stage1(e1)
        e2 = captures["enc_stage1_it"](e2)
        e3 = model.enc_stage2(e2)
        e3 = captures["enc_stage2_it"](e3)
        e4 = model.enc_stage3(e3)
        e4 = captures["enc_stage3_it"](e4)
        e5 = model.enc_stage4(e4)
        e5 = captures["enc_stage4_it"](e5)
        e6 = model.enc_stage5(e5)
        e6 = captures["enc_stage5_it"](e6)

        # Encoder for previous time-point
        e1_prev = model.enc_stage0(model.enc_stem(i_prev))
        e1_prev = captures["enc_stage0_iprev"](e1_prev)
        e2_prev = model.enc_stage1(e1_prev)
        e2_prev = captures["enc_stage1_iprev"](e2_prev)
        e3_prev = model.enc_stage2(e2_prev)
        e3_prev = captures["enc_stage2_iprev"](e3_prev)

        # SEM
        m_prev_shape = model.sem(m_prev)
        m_prev_shape = captures["sem_output"](m_prev_shape)

        # M-CAM
        e1_mcam = model.m_cam1(e1, e1_prev, m_prev_shape)
        e1_mcam = captures["mcam1_output"](e1_mcam)
        e2_mcam = model.m_cam2(e2, e2_prev, m_prev_shape)
        e3_mcam = model.m_cam3(e3, e3_prev, m_prev_shape)

        # Decoder
        d = model.transpconvs[0](e6)
        d = model.dec_stages[0](torch.cat([d, e5], dim=1))
        d = model.transpconvs[1](d)
        d = model.dec_stages[1](torch.cat([d, e4], dim=1))
        d = model.transpconvs[2](d)
        d = model.dec_stages[2](torch.cat([d, e3_mcam], dim=1))
        d = model.transpconvs[3](d)
        d = model.dec_stages[3](torch.cat([d, e2_mcam], dim=1))
        d = model.transpconvs[4](d)
        d = model.dec_stages[4](torch.cat([d, e1_mcam], dim=1))

        out = model.seg_layers[4](d)
        return out

    model.forward = patched_forward
    return captures


# ──────────────────────────────────────────────────────────────────────────────
# Save utilities
# ──────────────────────────────────────────────────────────────────────────────

def _rpi_affine():
    """Affine for RPI orientation at 1mm isotropic (RAS+ convention).

    In RPI voxel order, increasing index goes R / P / I, i.e. +x / -y / -z
    in RAS+ space → diag(+1, -1, -1). Used only as a fallback when the
    transformed sample's own affine is unavailable.
    """
    aff = np.zeros((4, 4), dtype=np.float64)
    aff[0, 0] =  1.0  # +index → R
    aff[1, 1] = -1.0  # +index → P
    aff[2, 2] = -1.0  # +index → I
    aff[3, 3] =  1.0
    return aff


def save_nifti(array: np.ndarray, path: str, affine=None):
    if affine is None:
        affine = _rpi_affine()
    img = nib.Nifti1Image(array, affine)
    nib.save(img, path)


def save_gradcam_results(output_dir, sample_idx, cam_dict, inputs, pred, target, affine=None):
    """Save all heatmaps and inputs for one sample as NIfTI files.

    Every array here lives in the transformed patch space — RPI orientation,
    1mm isotropic, shape (64, 64, 160). They all share the SAME affine so they
    overlay exactly. Prefer passing the transformed sample's own affine (which
    correctly describes that patch grid); the canonical RPI affine is only a
    fallback. (The original full-volume affine must NOT be used: it describes
    the un-resampled, un-cropped volume and would mis-place the patch.)
    """
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    if affine is None:
        affine = _rpi_affine()

    # Save inputs
    save_nifti(inputs["i_t"][0, 0].cpu().numpy(), os.path.join(sample_dir, "image_current.nii.gz"), affine)
    save_nifti(inputs["i_prev"][0, 0].cpu().numpy(), os.path.join(sample_dir, "image_previous.nii.gz"), affine)
    save_nifti(inputs["m_prev"][0, 0].cpu().numpy(), os.path.join(sample_dir, "mask_previous.nii.gz"), affine)

    # Save prediction and target
    pred_labels = pred[0].argmax(dim=0).cpu().numpy().astype(np.float32)
    save_nifti(pred_labels, os.path.join(sample_dir, "prediction.nii.gz"), affine)
    save_nifti(target[0].cpu().numpy().astype(np.float32), os.path.join(sample_dir, "target.nii.gz"), affine)

    # Save heatmaps
    for name, cam in cam_dict.items():
        save_nifti(cam[0].cpu().numpy(), os.path.join(sample_dir, f"gradcam_{name}.nii.gz"), affine)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GradCAM analysis for MambaXNet")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to trained MambaXNet checkpoint (best_model.pth)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to dataset JSON")
    p.add_argument("--output", type=str, required=True,
                   help="Output directory for heatmaps")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of samples to process")
    p.add_argument("--target_class", type=int, default=1,
                   help="Class index to compute GradCAM for (1=lesion)")
    p.add_argument("--target_shape", type=int, nargs=3, default=[64, 64, 160],
                   help="Patch size in RPI axis order (R-L, P-A, I-S). Default is "
                        "long along I-S so each patch contains a large extent of the SC.")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(args.output, f"gradcam_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    plans_json = checkpoint["plans_json"]
    model = MambaXNet(plans_json=plans_json, n_channels=1, n_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Patch forward to insert branch capture points
    captures = patch_model_for_gradcam(model)

    # Load dataset (no augmentation)
    target_shape = tuple(args.target_shape)
    ds = LongitudinalLesionDataset(
        json_path=args.data, split=args.split,
        transform=get_transforms("validation", target_shape),
    )
    print(f"Dataset: {len(ds)} samples in '{args.split}' split")

    n = min(args.n_samples, len(ds))
    target_names = list(captures.keys())

    for idx in range(n):
        print(f"\nProcessing sample {idx+1}/{n}...")
        sample = ds[idx]
        i_t    = sample["image2"].unsqueeze(0).to(device)
        i_prev = sample["image1"].unsqueeze(0).to(device)
        m_prev = sample["label1"].unsqueeze(0).to(device)
        target = sample["label2"]             # (1, D, H, W) — keep leading dim so target[0] in save is the full volume

        cam_dict = {}
        for tname in target_names:
            gc = GradCAM3D(model, captures[tname].identity)
            i_t.requires_grad_(True)
            i_prev.requires_grad_(True)
            m_prev.requires_grad_(True)
            cam, pred = gc(i_t, i_prev, m_prev, target_class=args.target_class)
            cam_dict[tname] = cam
            gc.remove_hooks()
            # Detach for next iteration
            i_t = i_t.detach().requires_grad_(False)
            i_prev = i_prev.detach().requires_grad_(False)
            m_prev = m_prev.detach().requires_grad_(False)

        inputs = {"i_t": i_t, "i_prev": i_prev, "m_prev": m_prev}
        # Use the transformed sample's own affine (correct RPI 1mm patch grid).
        sample_affine = getattr(sample["image2"], "affine", None)
        if sample_affine is not None:
            sample_affine = sample_affine.detach().cpu().numpy()
        save_gradcam_results(output_dir, idx, cam_dict, inputs, pred.detach(), target,
                             affine=sample_affine)
        print(f"  Saved to {output_dir}/sample_{idx:03d}/")

    print(f"\nDone. {n} samples processed → {output_dir}")


if __name__ == "__main__":
    main()
