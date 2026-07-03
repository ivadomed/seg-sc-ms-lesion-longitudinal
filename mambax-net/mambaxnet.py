import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import json
import pydoc # Useful for importing classes from strings

import os
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from mcam import MCAM


def _parse_arch_kwargs(json_config_path):
    """Parse and resolve architecture kwargs from a nnU-Net plans.json."""
    with open(json_config_path, 'r') as f:
        init_args = json.load(f)
    init_args = init_args['configurations']['3d_fullres']['architecture']['arch_kwargs']
    # Convert strings to actual objects/classes
    # These specific keys are strings in the JSON but must be objects for the class init
    if isinstance(init_args.get('conv_op'), str):
        init_args['conv_op'] = pydoc.locate(init_args['conv_op'])
    if isinstance(init_args.get('norm_op'), str):
        init_args['norm_op'] = pydoc.locate(init_args['norm_op'])
    if isinstance(init_args.get('nonlin'), str):
        init_args['nonlin'] = pydoc.locate(init_args['nonlin'])
    return init_args


def build_resenc_unet(json_config_path):
    """
    Build a ResidualEncoderUNet from a plans.json config (random weights).
    """
    init_args = _parse_arch_kwargs(json_config_path)
    return ResidualEncoderUNet(input_channels=1, num_classes=2, **init_args)


def load_nnunet_weights(model_folder):
    """
    Build a ResidualEncoderUNet and load pretrained nnU-Net weights into it.
    Used at training time to initialise MambaXNet with pretrained encoder/decoder.
    """
    checkpoint_path = f'{model_folder}/fold_0/checkpoint_best.pth'
    json_config_path = f'{model_folder}/plans.json'

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'network_weights' in checkpoint:
        state_dict = checkpoint['network_weights']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)

    res_enc_unet = build_resenc_unet(json_config_path)
    res_enc_unet.load_state_dict(state_dict)

    return res_enc_unet


class ShapeExtractorModule(nn.Module):
    """
    SEM: three sequential (Conv3d + ReLU) blocks.

    Args:
        in_channels:  number of mask channels (e.g. 1 for lesion masks)
        out_channels: latent channel dim (set to match encoder feature channels)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        mid = max(out_channels // 2, in_channels)
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, mid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask : [B, C_mask, D, H, W]
        return: [B, out_channels, D, H, W]
        """
        return self.blocks(mask)


class MambaXNet(nn.Module):
    def __init__(self, plans_json: str, n_channels=1, n_classes=2):
        super(MambaXNet, self).__init__()
        resenc_model = build_resenc_unet(plans_json)
        # Reconstruction of the encoder layers
        enc = resenc_model.encoder
        self.enc_stem = enc.stem
        self.enc_stage0 = enc.stages[0]
        self.enc_stage1 = enc.stages[1]
        self.enc_stage2 = enc.stages[2]
        self.enc_stage3 = enc.stages[3]
        self.enc_stage4 = enc.stages[4]
        self.enc_stage5 = enc.stages[5]

        # Decoder layers — pulled from the actual decoder, not decoder.encoder
        dec = resenc_model.decoder
        self.transpconvs = dec.transpconvs   # ModuleList of 5 ConvTranspose3d
        self.dec_stages   = dec.stages        # ModuleList of 5 StackedConvBlocks
        self.seg_layers   = dec.seg_layers    # ModuleList of 5 Conv3d heads

        # Build the SEM module to extract shape features from the previous time point
        self.sem = ShapeExtractorModule(in_channels=1, out_channels=32)
        
        # M-CAM blocks integrated at the last three upsampling levels
        self.m_cam1 = MCAM(in_channels=32, embed_dim=128, num_heads=8, sem_channels=32)
        self.m_cam2 = MCAM(in_channels=64, embed_dim=64, num_heads=8, sem_channels=32)
        self.m_cam3 = MCAM(in_channels=128, embed_dim=32, num_heads=8, sem_channels=32)

    def load_pretrained_resenc(self, model_folder: str):
        """
        Load pretrained nnU-Net weights into the encoder and decoder.
        Call this at training time for weight initialisation.
        """
        resenc_model = load_nnunet_weights(model_folder)
        enc = resenc_model.encoder
        self.enc_stem.load_state_dict(enc.stem.state_dict())
        for i, stage in enumerate(enc.stages):
            getattr(self, f'enc_stage{i}').load_state_dict(stage.state_dict())
        dec = resenc_model.decoder
        self.transpconvs.load_state_dict(dec.transpconvs.state_dict())
        self.dec_stages.load_state_dict(dec.stages.state_dict())
        self.seg_layers.load_state_dict(dec.seg_layers.state_dict())

    def forward(self, i_t: torch.Tensor,
                i_prev: torch.Tensor,
                m_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            i_t    : (B, 1, *spatial)  image at the current time-point
            i_prev : (B, 1, *spatial)  image at the previous time-point
            m_prev : (B, 1, *spatial)  segmentation mask at the previous time-point

        Returns:
            out    : (B, n_classes, *spatial)  logits for the current time-point
        """
        # Encoder features for current time-point
        e1 = self.enc_stage0(self.enc_stem(i_t))
        e2 = self.enc_stage1(e1)
        e3 = self.enc_stage2(e2)
        e4 = self.enc_stage3(e3)
        e5 = self.enc_stage4(e4)
        e6 = self.enc_stage5(e5)

        # Encoder features for previous time-point (first three levels only,
        # used as context in M-CAM)
        e1_prev = self.enc_stage0(self.enc_stem(i_prev))
        e2_prev = self.enc_stage1(e1_prev)
        e3_prev = self.enc_stage2(e2_prev)

        # Shape features from previous mask
        m_prev_shape = self.sem(m_prev)

        # M-CAM cross-attention at the three finest encoder resolutions
        e1_mcam = self.m_cam1(e1, e1_prev, m_prev_shape)
        e2_mcam = self.m_cam2(e2, e2_prev, m_prev_shape)
        e3_mcam = self.m_cam3(e3, e3_prev, m_prev_shape)

        # Decoder (transpconv → cat with skip → stage)
        # stage 0: bottleneck e6 → upsample → cat(e5) → 640→320
        d = self.transpconvs[0](e6)
        d = self.dec_stages[0](torch.cat([d, e5], dim=1))
        # stage 1: → cat(e4) → 512→256
        d = self.transpconvs[1](d)
        d = self.dec_stages[1](torch.cat([d, e4], dim=1))
        # stage 2: → cat(e3_mcam) → 256→128
        d = self.transpconvs[2](d)
        d = self.dec_stages[2](torch.cat([d, e3_mcam], dim=1))
        # stage 3: → cat(e2_mcam) → 128→64
        d = self.transpconvs[3](d)
        d = self.dec_stages[3](torch.cat([d, e2_mcam], dim=1))
        # stage 4: → cat(e1_mcam) → 64→32
        d = self.transpconvs[4](d)
        d = self.dec_stages[4](torch.cat([d, e1_mcam], dim=1))

        # Final segmentation head
        out = self.seg_layers[4](d)   # (B, n_classes, *spatial)

        return out


def main():
    # Use CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check it torch had access to cuda
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation and GPU configuration.")
        return
    else:
        print("CUDA is available. Proceeding with GPU computations.")

    # Try loading the nnU-Net weights into MambaXNet
    model_folder = '/home/plbenveniste/net/longitudinal_mamba/trained_resencUnet/nnUNetTrainerDiceCELoss_noSmooth_4000epochs_fromScratch__nnUNetResEncUNetL1x1x1_Model2_Plans__3d_fullres'
    plans_json = f'{model_folder}/plans.json'

    # Initialize MambaXNet from architecture config
    model = MambaXNet(plans_json=plans_json, n_channels=1, n_classes=2)
    # Load pretrained nnU-Net weights into encoder/decoder
    model.load_pretrained_resenc(model_folder)
    print("MambaXNet initialised with pretrained nnU-Net weights.")
    model.to(device)
    model.eval()
    print("MambaXNet initialized with pretrained nnU-Net weights.")

    # Generate a random input tensor to test the forward pass
    i_t    = torch.randn(1, 1, 32, 128, 128).to(device)
    i_prev = torch.randn(1, 1, 32, 128, 128).to(device)
    m_prev = torch.randn(1, 1, 32, 128, 128).to(device)
    output = model(i_t, i_prev, m_prev)
    print("MambaXNet forward pass successful. Output shape:", output.shape)


if __name__ == "__main__":
   main()