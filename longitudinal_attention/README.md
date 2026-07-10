# Longitudinal Attention UNet

Segmentation model for **longitudinal MS lesion segmentation**: given a pair
of scans from the same subject, it segments the lesion on the **follow-up**
(current) scan while using the **baseline** (prior) scan purely as a
*contextual prompt* — extra information that helps the model, but is never
segmented itself.

The design combines two ideas from the reference repos:

- the shared-encoder pattern used for the two time points in the
  `mambax-net-sc-lesion` model on `plb/mambax-net`,
- the "primary input + prompt, fused by attention" pattern used for
  modality prompts in [`doc/Modular-Cross-Attention-Fusion`](../doc/Modular-Cross-Attention-Fusion)
  (`ModularFusionWrapper` / `AttentionFusion`), applied here across
  **time points** instead of across **MRI modalities**.

## Files

| File | Purpose |
|---|---|
| [`model.py`](model.py) | `LongitudinalAttentionUNet` model, `CrossAttentionFusion` fusion block, `load_nnunet_weights` checkpoint loader. |
| [`main.py`](main.py) | Smoke test: builds a small randomly-initialised encoder/decoder and runs a forward pass on artificial tensors, for both the shared- and separate-encoder configurations. |

## Problem framing

Two co-registered volumes per subject:

- `image_followup` — the scan to segment (**primary input**)
- `image_baseline` — an earlier scan of the same subject (**contextual prompt**)

```
logits = model(image_followup, image_baseline)   # (B, n_classes, D, H, W)
```

The baseline never appears in the loss directly — it exists only to give the
model temporal context (e.g. "this lesion was already present three months
ago" vs. "this is new"), the same role the non-reference MRI contrasts play
as *prompts* in the Modular-Cross-Attention-Fusion model relative to its T2
primary branch.

## Architecture

```
image_followup ──► encoder_followup ──► f_followup[0..L]  (finest → bottleneck)
                                                │
image_baseline  ──► encoder_baseline ──► f_baseline[0..L] │
                                                │          │
                                                ▼          ▼
                                    fusion_blocks[0..L]         (fusion_type selects the class:
                                    (query/primary = f_followup,  CrossAttentionFusion or
                                     context/prompt = f_baseline)  ChannelAttentionFusion)
                                                │
                                                ▼
                                        f_fused[0..L]
                                                │
                                     shared pretrained decoder
                                     (transposed convs + skip
                                      connections from f_fused)
                                                │
                                                ▼
                                   logits (B, n_classes, D, H, W)
```

### 1. Backbone: pretrained nnU-Net `ResidualEncoderUNet`

The encoder and decoder both come from a pretrained nnU-Net
`ResidualEncoderUNet` (`dynamic_network_architectures.architectures.unet`),
loaded via `load_nnunet_weights(model_folder)`. This mirrors exactly how
`mambaxnet_sc_lesion.py` reuses a pretrained ResEncUNet on `plb/mambax-net`:
`load_nnunet_weights` reads `checkpoint_best.pth` and `plans.json` from an
nnU-Net results folder and instantiates the network with the architecture
kwargs stored in the plans.

The **encoder** (`stem` + a `ModuleList` of residual stages, finest → deepest
/ bottleneck) is run independently on each time point (see "Encoder sharing"
below). The **decoder** (`transpconvs`, `stages`, `seg_layers`) is reused
completely unmodified — the only thing that changes relative to a standard
single-timepoint UNet is *what* gets fed into its skip connections.

### 2. Encoder sharing: `share_encoder`

`LongitudinalAttentionUNet` supports two modes, selected by the
`share_encoder` constructor flag:

- **`share_encoder=True` (default).** A single encoder (from `resenc_model`)
  is weight-tied and called twice, once on `image_followup` and once on
  `image_baseline`. Both time points are therefore projected into exactly
  the same feature space, and there is only one encoder's worth of
  parameters. This is the pattern used in `mambaxnet_sc_lesion.py`, where the
  same `enc_stage0..5` weights process both `image_M12` and `image_M0`.

- **`share_encoder=False`.** Each time point gets its **own** encoder
  (`encoder_followup` and `encoder_baseline`), so the two branches can learn
  different feature extractors — useful if baseline and follow-up scans
  differ systematically (e.g. different scanner/protocol generation, or if
  you simply want more model capacity). By default `encoder_baseline` is a
  **deep copy** of `resenc_model.encoder`, so both start from the same
  pretrained weights but diverge independently during training. You can
  instead seed the baseline encoder from a *different* pretrained checkpoint
  by passing `resenc_model_baseline` (its `.encoder` is used, its `.decoder`
  is discarded — only one decoder is ever used, the one from `resenc_model`).

Everything downstream (fusion blocks, decoder) is identical in both modes;
only which encoder computes `f_baseline` changes.

### 3. Fusion mechanisms: baseline as a contextual prompt

At **every** encoder resolution — from the finest skip connection down to
the bottleneck — a fusion block injects baseline information into the
follow-up feature map. Two interchangeable fusion mechanisms are
implemented; which one is used is selected once, model-wide, via the
`fusion_type` constructor argument (`model.FUSION_TYPES = ("cross_attention",
"channel_attention")`). One `fusion_blocks[i]` is instantiated per encoder
stage regardless of which type is chosen (`fusion_blocks`, a `ModuleList`
sized from `features_per_stage`), each scoped to that stage's channel count.

#### 3a. `fusion_type="cross_attention"` (default) — `CrossAttentionFusion`

Spatial, token-level, multi-head cross-attention:

```
f_fused = ReLU( CrossAttention(Q = f_followup, K = V = f_baseline) + f_followup )
```

Concretely, for each resolution level:

1. **Patch-embed** both `f_followup` and `f_baseline` from
   `(B, C, D, H, W)` into token sequences `(B, N, E)` with a strided `Conv3d`
   (`PatchEmbed3D`). Spatial dims are zero-padded up to a multiple of
   `patch_size` first, so this works for any input shape without crashing on
   odd/small volumes (relevant for the bottleneck, which can be very small).
2. **Multi-head cross-attention** (MONAI's `CrossAttentionBlock`):
   follow-up tokens are the *query* (`x`), baseline tokens are *key* and
   *value* (`context`). This lets every follow-up spatial location attend to
   the most relevant baseline locations — not just the co-located voxel —
   which matters when there is residual misalignment between time points.
   `use_flash_attention=True` routes it to PyTorch's fused
   scaled-dot-product-attention kernel instead of materialising the full
   `N × N` attention matrix.
3. **Unpack** the attended tokens back to a spatial map (`PatchUnembed3D`:
   linear projection + reshape + `trilinear` interpolation back to the
   follow-up feature map's exact spatial shape).
4. **Residual connection + ReLU** with the original follow-up features, so
   the block can learn to add temporal context without being forced to
   overwrite the purely-spatial features if the baseline isn't informative
   at that location.

This is deliberately a **lighter version** of the `MCAM` module used in
`mambaxnet_sc_lesion.py` on `plb/mambax-net`: `MCAM` additionally runs a
Mamba state-space block on each token sequence before the cross-attention,
which requires `mamba_ssm` (a CUDA-only dependency). `CrossAttentionFusion`
drops the Mamba step and keeps only the patch-embed → cross-attention →
unpack → residual pipeline, so the whole model runs on CPU (needed for the
`main.py` smoke test) and has one fewer dependency. Swapping in a Mamba block
before the cross-attention would recover the `MCAM`-equivalent design if
desired later.

#### 3b. `fusion_type="channel_attention"` — `ChannelAttentionFusion`

Channel-wise, squeeze-and-excitation-style gating — a direct port of
`AttentionFusion` from
[`doc/Modular-Cross-Attention-Fusion/nnunetv2/training/network_architecture/modular_fusion_wrapper.py`](../doc/Modular-Cross-Attention-Fusion/nnunetv2/training/network_architecture/modular_fusion_wrapper.py),
renamed here from `t2/prompt` to `followup/baseline`:

```
gate   = sigmoid( Conv(ReLU(Conv( AvgPool(concat(f_followup, f_baseline)) ))) )     # (B, C, 1, 1, 1)
f_fused = ReLU( GroupNorm( Conv( concat(f_followup, gate * f_baseline) ) ) )
```

1. **Global-average-pool** `concat(f_followup, f_baseline)` over all spatial
   dimensions down to one vector per channel.
2. **Two 1×1×1 convolutions + sigmoid** turn that vector into a per-channel
   gate in `[0, 1]` (a squeeze-and-excitation bottleneck).
3. The gate multiplies `f_baseline` — channels of the baseline features the
   network finds relevant (given the current follow-up/baseline pair) are
   kept, others are suppressed.
4. `concat(f_followup, gated f_baseline)` is fused back down to
   `in_channels` by a `3×3×3` `Conv3d` + `GroupNorm` + `ReLU`.

The reference `AttentionFusion` constructor also accepts `fusion_channels`
and `num_heads` arguments, but never uses either of them in its `forward` —
`ChannelAttentionFusion` omits both as dead parameters. `GroupNorm`'s group
count is chosen automatically as the largest divisor of `in_channels` that
is `<= 8`, so this works for arbitrary channel counts, not just multiples
of 8 as in the original.

The key difference from `CrossAttentionFusion`: the gate is a **single
scalar per channel per sample**, identical at every spatial location. It
cannot express "this voxel should borrow from a *different* voxel in the
baseline" — only "this channel of the baseline is/isn't useful right now,
uniformly across the volume". It is correspondingly cheaper and has far
fewer parameters (no attention matrix, no `E`-dimensional token
projections).

#### Comparison

| | `cross_attention` | `channel_attention` |
|---|---|---|
| Reference | `MCAM` (`plb/mambax-net`), minus the Mamba block | `AttentionFusion` (`doc/Modular-Cross-Attention-Fusion`) |
| Granularity | per spatial location (token) | per channel, uniform over space |
| Can align spatially-shifted lesions | yes — query/key/value attention across locations | no — only reweights channels |
| Relative cost | higher (attention over `N` tokens) | lower (global pooling + 1×1×1 / 3×3×3 convs) |
| Extra dependencies | none (built on `nn.MultiheadAttention`) | none |

### 4. Decoder

The decoder is the pretrained UNet decoder, used exactly as in a standard
single-timepoint UNet, except every skip connection now receives a *fused*
feature map instead of a plain follow-up feature map:

```python
*skips, bottleneck = fused            # fused[i] = fusion_blocks[i] applied at stage i
d = bottleneck
for i in range(len(dec_stages)):
    d = transpconvs[i](d)
    d = dec_stages[i](torch.cat([d, skips[-(i + 1)]], dim=1))
logits = seg_layers[-1](d)
```

Only the finest segmentation head (`seg_layers[-1]`) is used — deep
supervision heads at intermediate decoder resolutions are not wired up here.

## Parameter reference

`LongitudinalAttentionUNet(resenc_model, features_per_stage, n_classes=2, fusion_type="cross_attention", fusion_embed_dim=32, fusion_num_heads=4, fusion_patch_size=2, share_encoder=True, resenc_model_baseline=None)`

| Argument | Meaning |
|---|---|
| `resenc_model` | Pretrained `ResidualEncoderUNet` (e.g. from `load_nnunet_weights`). Supplies the follow-up encoder and, always, the decoder. |
| `features_per_stage` | Output channel count of each encoder stage, finest → bottleneck (from the nnU-Net plans; also returned by `load_nnunet_weights`). Used to size each fusion block. |
| `n_classes` | Number of segmentation classes (background included). |
| `fusion_type` | `"cross_attention"` (default) or `"channel_attention"` — selects `CrossAttentionFusion` or `ChannelAttentionFusion` for every stage (see [Fusion mechanisms](#3-fusion-mechanisms-baseline-as-a-contextual-prompt)). See `model.FUSION_TYPES`. |
| `fusion_embed_dim` | Token embedding dimension `E` inside each `CrossAttentionFusion` block. Ignored when `fusion_type="channel_attention"`. |
| `fusion_num_heads` | Number of attention heads in each `CrossAttentionFusion` block. Ignored when `fusion_type="channel_attention"`. |
| `fusion_patch_size` | Patch size used to tokenize feature maps before attention (larger = fewer, coarser tokens = cheaper attention). Ignored when `fusion_type="channel_attention"`. |
| `share_encoder` | `True` → one weight-tied encoder for both time points. `False` → independent `encoder_followup` / `encoder_baseline`. |
| `resenc_model_baseline` | Only used when `share_encoder=False`. If given, its `.encoder` seeds `encoder_baseline`; otherwise `encoder_baseline` is a deep copy of `resenc_model.encoder`. Must be `None` when `share_encoder=True`. |

## Running the smoke test

```bash
cd longitudinal_attention
python main.py
```

`main.py` does not require any pretrained checkpoint: it builds a small
randomly-initialised `ResidualEncoderUNet` (4 stages, CPU-friendly), then
runs `LongitudinalAttentionUNet` on random `(2, 1, 32, 64, 64)` follow-up /
baseline tensors for **all four** combinations of `fusion_type`
(`"cross_attention"` / `"channel_attention"`) × `share_encoder`
(`True` / `False`), asserting the output shape is `(2, n_classes, 32, 64,
64)` in every case, and that for each `fusion_type` the separate-encoder
model has more parameters than the shared-encoder one.

To use real pretrained weights instead of the random dummy encoder, and to
pick a fusion mechanism explicitly:

```python
from model import load_nnunet_weights, LongitudinalAttentionUNet

resenc_model, features_per_stage = load_nnunet_weights("/path/to/nnUNet_results/DatasetXXX/.../fold_0/..")
model = LongitudinalAttentionUNet(
    resenc_model, features_per_stage,
    fusion_type="channel_attention",  # or "cross_attention"
    share_encoder=True,
)
logits = model(image_followup, image_baseline)
```

## Dependencies

- `torch`
- [`dynamic_network_architectures`](https://github.com/MIC-DKFZ/dynamic-network-architectures) (for `ResidualEncoderUNet`)

Neither the model nor the smoke test requires `mamba_ssm`, unlike the
`MCAM`-based models on `plb/mambax-net`.
