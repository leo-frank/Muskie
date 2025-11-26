import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from models import AABlock, RotaryPositionEmbedding2D, PositionGetter
from models import small, base, large, huge


def prepare_model(chkpt_dir, model_type, enable_checkpoint):
    if model_type == 'muskie_small':
        model = small(enable_checkpoint=enable_checkpoint)
    elif model_type == 'muskie_base':
        model = base(enable_checkpoint=enable_checkpoint)
    elif model_type == 'muskie_large':
        model = large(enable_checkpoint=enable_checkpoint)
    elif model_type == 'muskie_huge':
        model = huge(enable_checkpoint=enable_checkpoint)
    else:
        raise ValueError(f"Invalid encoder type {model_type}")

    if chkpt_dir:
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded encoder from {chkpt_dir}. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
    else:
        print("No checkpoint provided. Initializing encoder with random weights.")
    
    return model


class Encoder(nn.Module):
    def __init__(self, cfg):
        encoder_type = cfg.model.encoder_type
        super(Encoder, self).__init__()
        args = cfg.model.encoder_variants[encoder_type]

        self.encoder_type = encoder_type

        self.embed_dim = args.embed_dim
        self.patch_size = args.patch_size
        self.decoder_depth = args.decoder_depth
        self.num_heads = args.num_heads
        self.enable_checkpoint = cfg.enable_checkpoint

        mlp_ratio = 4.0
        rope_freq = 100
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()

        self.encoder = prepare_model(args.pretrained_ckpt, encoder_type, self.enable_checkpoint)
        for name, value in (("_img_mean", [0.5, 0.5, 0.5]), ("_img_std", [0.5, 0.5, 0.5])):
            self.register_buffer(name, torch.FloatTensor(
                value).view(1, 1, 3, 1, 1), persistent=False)

        # Additional AA blocks besides main encoder (each AA block equals to 2 layers)
        if self.decoder_depth > 0:
            self.additional_blocks = nn.ModuleList([
                AABlock(self.embed_dim, self.num_heads,
                        mlp_ratio, qkv_bias=True, rope=self.rope)
                for _ in range(self.decoder_depth)
            ])

    def forward(self, samples):

        images = samples['images']
        B, V, _, H, W = images.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size

        # Normalize images, vggt do not normalize at datasets
        images = (images - self._img_mean) / self._img_std

        # Get latent embeddings
        enc_tokens = self.encoder.get_latent_embeddings(images)

        if hasattr(self, 'additional_blocks'):
            pos = self.position_getter(
                B * V, Hp, Wp, device=images.device).view(B, V, Hp*Wp, -1)
            x = enc_tokens[-1].view(B, V, Hp*Wp, -1)
            for blk in self.additional_blocks:
                if self.enable_checkpoint:
                    x = checkpoint.checkpoint(blk, x, pos, use_reentrant=False)
                else:
                    x = blk(x, pos)
                enc_tokens.append(x.view(B*V, Hp*Wp, -1))

        return enc_tokens
