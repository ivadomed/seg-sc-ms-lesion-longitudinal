"""
MambaXNet v2 — adds temporal fusion at the bottleneck.

v1 fuses the two time-points only at the three finest encoder resolutions
(M-CAM at e1/e2/e3); the bottleneck (e6) is purely current-timepoint, so the
deepest, most semantic level never sees temporal context. v2 adds a bottleneck
M-CAM:

    - the previous-timepoint encoder is run through ALL six stages (v1 stops
      at stage 2),
    - an M-CAM block fuses (e6, e6_prev, SEM) before the decoder upsamples it.

Because the bottleneck spatial extent is tiny (≈ /32 of the patch), the
bottleneck M-CAM uses patch_size=1 so the strided patch-embedding does not
collapse the feature map.

Everything else (encoder/decoder weights, the three fine-level M-CAMs, SEM) is
inherited from MambaXNet, so `load_pretrained_resenc` still initialises the
nnU-Net weights; only the new bottleneck block is trained from scratch.

Author: Pierre-Louis Benveniste
"""

import json
import torch

from mambaxnet import MambaXNet
from mcam import MCAM


def _features_per_stage(plans_json: str):
    """Read the per-stage channel counts from a nnU-Net plans.json."""
    with open(plans_json) as f:
        cfg = json.load(f)
    return cfg["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]["features_per_stage"]


class MambaXNetV2(MambaXNet):
    def __init__(self, plans_json: str, n_channels: int = 1, n_classes: int = 2,
                 bottleneck_embed_dim: int = 256, bottleneck_heads: int = 8):
        super().__init__(plans_json=plans_json, n_channels=n_channels, n_classes=n_classes)

        feats = _features_per_stage(plans_json)
        c_bottleneck = int(feats[-1])           # channels of e6

        # Bottleneck temporal fusion. patch_size=1: at /32 resolution the
        # default patch_size=4 strided embed would zero out the sequence.
        self.m_cam_bottleneck = MCAM(
            in_channels=c_bottleneck,
            embed_dim=bottleneck_embed_dim,
            num_heads=bottleneck_heads,
            sem_channels=32,
            patch_size=1,
        )

    def forward(self, i_t: torch.Tensor,
                i_prev: torch.Tensor,
                m_prev: torch.Tensor) -> torch.Tensor:
        # Encoder — current time-point (all six stages)
        e1 = self.enc_stage0(self.enc_stem(i_t))
        e2 = self.enc_stage1(e1)
        e3 = self.enc_stage2(e2)
        e4 = self.enc_stage3(e3)
        e5 = self.enc_stage4(e4)
        e6 = self.enc_stage5(e5)

        # Encoder — previous time-point (all six stages, vs three in v1)
        e1_prev = self.enc_stage0(self.enc_stem(i_prev))
        e2_prev = self.enc_stage1(e1_prev)
        e3_prev = self.enc_stage2(e2_prev)
        e4_prev = self.enc_stage3(e3_prev)
        e5_prev = self.enc_stage4(e4_prev)
        e6_prev = self.enc_stage5(e5_prev)

        # Shape features from previous mask
        m_prev_shape = self.sem(m_prev)

        # M-CAM cross-attention: three finest resolutions + bottleneck
        e1_mcam = self.m_cam1(e1, e1_prev, m_prev_shape)
        e2_mcam = self.m_cam2(e2, e2_prev, m_prev_shape)
        e3_mcam = self.m_cam3(e3, e3_prev, m_prev_shape)
        e6_mcam = self.m_cam_bottleneck(e6, e6_prev, m_prev_shape)

        # Decoder — bottleneck now carries temporal context (e6_mcam). A seg head
        # is applied at every resolution so deep supervision can use them.
        seg_outputs = []
        d = self.transpconvs[0](e6_mcam)
        d = self.dec_stages[0](torch.cat([d, e5], dim=1))
        seg_outputs.append(self.seg_layers[0](d))
        d = self.transpconvs[1](d)
        d = self.dec_stages[1](torch.cat([d, e4], dim=1))
        seg_outputs.append(self.seg_layers[1](d))
        d = self.transpconvs[2](d)
        d = self.dec_stages[2](torch.cat([d, e3_mcam], dim=1))
        seg_outputs.append(self.seg_layers[2](d))
        d = self.transpconvs[3](d)
        d = self.dec_stages[3](torch.cat([d, e2_mcam], dim=1))
        seg_outputs.append(self.seg_layers[3](d))
        d = self.transpconvs[4](d)
        d = self.dec_stages[4](torch.cat([d, e1_mcam], dim=1))
        seg_outputs.append(self.seg_layers[4](d))

        # Highest-resolution output first (nnU-Net convention).
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]
