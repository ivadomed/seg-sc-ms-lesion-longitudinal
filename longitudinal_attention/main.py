"""
Smoke test for LongitudinalAttentionUNet on artificially created tensors.

Builds a small, randomly-initialised ResidualEncoderUNet (no pretrained
weights required) purely to check that the longitudinal wiring — encoder(s),
fusion of the baseline prompt into the follow-up features, and decoder —
produces the expected output shape, for every combination of `fusion_type`
("cross_attention" / "channel_attention") and `share_encoder` (True / False).

To use real pretrained weights instead, replace `build_dummy_resenc_unet(...)`
with `load_nnunet_weights(model_folder)` from model.py.

Author: Pierre-Louis Benveniste
"""

import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from model import FUSION_TYPES, LongitudinalAttentionUNet


def build_dummy_resenc_unet(features_per_stage=(16, 32, 64, 128), n_classes: int = 2) -> ResidualEncoderUNet:
    """Small randomly-initialised ResidualEncoderUNet for a quick CPU smoke test."""
    n_stages = len(features_per_stage)
    return ResidualEncoderUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=list(features_per_stage),
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=[1] + [2] * (n_stages - 1),
        n_blocks_per_stage=[1] * n_stages,
        num_classes=n_classes,
        n_conv_per_stage_decoder=[1] * (n_stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def run_case(
    fusion_type: str,
    share_encoder: bool,
    device: torch.device,
    features_per_stage=(16, 32, 64, 128),
    n_classes: int = 2,
) -> int:
    label = f"fusion_type={fusion_type}, share_encoder={share_encoder}"
    print(f"\n--- {label} ---")

    resenc_model = build_dummy_resenc_unet(features_per_stage, n_classes)
    model = LongitudinalAttentionUNet(
        resenc_model=resenc_model,
        features_per_stage=features_per_stage,
        n_classes=n_classes,
        fusion_type=fusion_type,
        share_encoder=share_encoder,
    )
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Total parameters: {n_params:,}")

    shape = (2, 1, 32, 64, 64)
    image_followup = torch.randn(*shape, device=device)
    image_baseline = torch.randn(*shape, device=device)

    with torch.no_grad():
        logits = model(image_followup, image_baseline)

    expected_shape = (shape[0], n_classes, *shape[2:])
    print(f"Input shape (followup / baseline): {tuple(shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    assert tuple(logits.shape) == expected_shape, (
        f"Expected output shape {expected_shape}, got {tuple(logits.shape)}"
    )
    print(f"OK: output shape matches expectation ({label}).")

    return n_params


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_params = {}
    for fusion_type in FUSION_TYPES:
        for share_encoder in (True, False):
            n_params[(fusion_type, share_encoder)] = run_case(fusion_type, share_encoder, device)

    for fusion_type in FUSION_TYPES:
        shared = n_params[(fusion_type, True)]
        separate = n_params[(fusion_type, False)]
        assert separate > shared, (
            f"[{fusion_type}] separate-encoder model should have more parameters than shared-encoder model."
        )
        print(f"\nOK [{fusion_type}]: separate encoders ({separate:,} params) "
              f"> shared encoder ({shared:,} params), as expected.")


if __name__ == "__main__":
    main()
