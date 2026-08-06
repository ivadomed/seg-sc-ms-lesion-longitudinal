"""
Longitudinal attention segmentation model.

The follow-up scan is the primary input; the baseline scan is treated as a
contextual prompt (in the spirit of the modality-prompt / modular-fusion
design in doc/Modular-Cross-Attention-Fusion) and is fused into the
follow-up encoder features via cross-attention at every resolution level.
By default a single pretrained nnU-Net ResidualEncoderUNet encoder is
shared (weight-tied) between the two time points, the same pattern used
for the image encoder in the mambax-net-sc-lesion model. Set
`share_encoder=False` to give each time point its own encoder instead
(see LongitudinalAttentionUNet).

Two fusion mechanisms are selectable at inference via `fusion_type`:
    "cross_attention"  – spatial multi-head cross-attention over patch
                          tokens (query=follow-up, key/value=baseline).
    "se_gate_fusion"   – channel-wise squeeze-and-excitation-style gating,
                          matching `AttentionFusion` in
                          doc/Modular-Cross-Attention-Fusion's
                          modular_fusion_wrapper.py.

Two inputs:
    image_followup – (B, 1, *spatial)  current / follow-up scan  (primary)
    image_baseline  – (B, 1, *spatial)  prior / baseline scan     (contextual prompt)

One output:
    logits          – (B, n_classes, *spatial)  lesion segmentation at follow-up

Author: Pierre-Louis Benveniste
"""

import copy
import json
import os
import pydoc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from monai.networks.blocks import CrossAttentionBlock


# ──────────────────────────────────────────────────────────────────────────────
# Weight loading
# ──────────────────────────────────────────────────────────────────────────────

def load_nnunet_weights(model_folder: str, fold: int = 0, checkpoint_name: str = "checkpoint_best.pth"):
    """Load a pretrained nnU-Net ResidualEncoderUNet from a fold checkpoint.

    Returns the model together with the `features_per_stage` list from the
    plans, so callers can size the fusion modules to match.
    """
    checkpoint_path = os.path.join(model_folder, f"fold_{fold}", checkpoint_name)
    plans_path = os.path.join(model_folder, "plans.json")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("network_weights", checkpoint.get("state_dict", checkpoint))

    with open(plans_path) as f:
        plans = json.load(f)
    arch_kwargs = dict(plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])

    init_args = dict(arch_kwargs)
    for key in ("conv_op", "norm_op", "nonlin"):
        if isinstance(init_args.get(key), str):
            init_args[key] = pydoc.locate(init_args[key])

    model = ResidualEncoderUNet(input_channels=1, num_classes=2, **init_args)
    model.load_state_dict(state_dict)

    return model, arch_kwargs["features_per_stage"]


# ──────────────────────────────────────────────────────────────────────────────
# Cross-attention fusion (follow-up queries, baseline is the contextual prompt)
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """Conv3d patch embedding: (B, C, D, H, W) -> (B, N, E) token sequence."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        d, h, w = x.shape[2:]
        p = self.patch_size
        pad_d, pad_h, pad_w = (-d) % p, (-h) % p, (-w) % p
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        x = self.proj(x)                                    # (B, E, D', H', W')
        patch_shape = x.shape[2:]
        tokens = self.norm(x.flatten(2).transpose(1, 2))     # (B, N, E)
        return tokens, patch_shape


class PatchUnembed3D(nn.Module):
    """Inverse of PatchEmbed3D: (B, N, E) -> (B, C, D, H, W), resized to target."""

    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, tokens: torch.Tensor, patch_shape, target_shape):
        b = tokens.shape[0]
        x = self.proj(tokens).transpose(1, 2).reshape(b, -1, *patch_shape)  # (B, C, D', H', W')
        if x.shape[2:] != tuple(target_shape):
            x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Fuses follow-up features (query) with baseline features (contextual
    prompt, key/value) at a single encoder resolution:

        f_fused = ReLU(CrossAttn(Q=followup, K=V=baseline) + followup)
    """

    def __init__(self, in_channels: int, embed_dim: int = 32, num_heads: int = 4, patch_size: int = 2):
        super().__init__()
        self.embed_followup = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.embed_baseline = PatchEmbed3D(in_channels, embed_dim, patch_size)
        # batch_first=True so tokens are fed directly as (B, N, E)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.unembed = PatchUnembed3D(embed_dim, in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, followup_feat: torch.Tensor, baseline_feat: torch.Tensor) -> torch.Tensor:
        q_tokens, patch_shape = self.embed_followup(followup_feat)
        kv_tokens, _ = self.embed_baseline(baseline_feat)

        # need_weights=False routes to the fused SDPA kernel instead of
        # materialising the full N x N attention matrix.
        attn_tokens, _ = self.cross_attn(query=q_tokens, key=kv_tokens, value=kv_tokens, need_weights=False)
        attn_spatial = self.unembed(attn_tokens, patch_shape, followup_feat.shape[2:])

        return self.relu(attn_spatial + followup_feat)


# ──────────────────────────────────────────────────────────────────────────────
# Channel-attention fusion (follow-up queries, baseline is the contextual prompt)
# ──────────────────────────────────────────────────────────────────────────────

def _group_norm_groups(num_channels: int, max_groups: int = 8) -> int:
    """Largest divisor of `num_channels` that is <= max_groups (>= 1)."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class SqueezeExciteGateFusion(nn.Module):
    """
    Squeeze-and-excitation style gating fusion, matching
    what is done in Modular-Cross-Attention-Fusion paper.
    Global-average-pools concat(followup, baseline) down to
    one value per channel, uses it to compute a per-channel sigmoid gate,
    applies that gate to the baseline (prompt) features, then fuses:

        gate  = sigmoid(Conv(ReLU(Conv(AvgPool(concat(followup, baseline))))))
        fused = ReLU(GroupNorm(Conv(concat(followup, gate * baseline))))
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid = max(in_channels // 4, 1)
        groups = _group_norm_groups(in_channels)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv3d(in_channels * 2, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, in_channels, 1),
            nn.Sigmoid(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3, padding=1),
            nn.GroupNorm(groups, in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, followup_feat: torch.Tensor, baseline_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([followup_feat, baseline_feat], dim=1)
        gate = self.channel_attention(self.global_pool(combined))
        gated_baseline = baseline_feat * gate
        return self.fusion_conv(torch.cat([followup_feat, gated_baseline], dim=1))


FUSION_TYPES = ("cross_attention", "se_gate_fusion")


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class LongitudinalAttentionUNet(nn.Module):
    """
    Segments the follow-up scan using the baseline scan as a contextual
    prompt. Its decoder (from `resenc_model`) is reused as-is. At every
    encoder resolution (including the bottleneck), baseline features are
    fused into follow-up features before being passed to the decoder.

    The fusion mechanism is controlled by `fusion_type` (see FUSION_TYPES):
        "cross_attention"    (default) — CrossAttentionFusion, spatial
                              multi-head cross-attention over patch tokens.
        "se_gate_fusion"   — SqueezeExciteGateFusion, squeeze-and-excitation
                              style per-channel gating (matches what is done 
                              in Modular-Cross-Attention-Fusion paper).

    Encoder sharing is controlled by `share_encoder`:
        True  (default) — a single pretrained encoder is weight-tied and
                           called on both time points (fewer parameters,
                           forces both time points into the same feature
                           space).
        False            — each time point gets its own encoder. By default
                           the baseline encoder and the follow-up encoder are
                           deep copies of the same pretrained encoder (same
                           pretrained initialisation, independent weights
                           thereafter).
    """

    def __init__(
        self,
        resenc_model: ResidualEncoderUNet,
        features_per_stage,
        n_classes: int = 2,
        fusion_type: str = "cross_attention",
        fusion_embed_dim: int = 32,
        fusion_num_heads: int = 4,
        fusion_patch_size: int = 2,
        share_encoder: bool = True,
    ):
        super().__init__()

        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got {fusion_type!r}")

        self.fusion_type = fusion_type

        self.share_encoder = share_encoder

        if share_encoder:
            self.encoder_baseline = resenc_model.encoder
            self.encoder_followup = self.encoder_followup
        else:
            self.encoder_baseline = resenc_model.encoder
            self.encoder_followup = copy.deepcopy(resenc_model.encoder)

        # Decoder is reused unchanged; its skip inputs receive fused features.
        decoder = resenc_model.decoder
        self.transpconvs = decoder.transpconvs
        self.dec_stages = decoder.stages
        self.seg_layers = decoder.seg_layers

        if fusion_type == "cross_attention":
            self.fusion_blocks = nn.ModuleList([
                CrossAttentionFusion(
                    in_channels=c,
                    embed_dim=fusion_embed_dim,
                    num_heads=fusion_num_heads,
                    patch_size=fusion_patch_size,
                )
                for c in features_per_stage
            ])
        else:
            self.fusion_blocks = nn.ModuleList([
                SqueezeExciteGateFusion(in_channels=c) for c in features_per_stage
            ])

    @staticmethod
    def _encode(x: torch.Tensor, encoder):
        """Run `encoder`, returning per-stage features finest -> bottleneck."""
        x = encoder.stem(x)
        feats = []
        for stage in encoder.stages:
            x = stage(x)
            feats.append(x)
        return feats

    def forward(self, image_followup: torch.Tensor, image_baseline: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_classes, *spatial) logits for the lesion mask at follow-up."""
        followup_feats = self._encode(image_followup, self.encoder_followup)
        baseline_feats = self._encode(image_baseline, self.encoder_baseline)

        fused = [
            fusion(f_t, f_prompt)
            for fusion, f_t, f_prompt in zip(self.fusion_blocks, followup_feats, baseline_feats)
        ]

        *skips, bottleneck = fused
        d = bottleneck
        for i in range(len(self.dec_stages)):
            d = self.transpconvs[i](d)
            skip = skips[-(i + 1)]
            d = self.dec_stages[i](torch.cat([d, skip], dim=1))

        return self.seg_layers[-1](d)
