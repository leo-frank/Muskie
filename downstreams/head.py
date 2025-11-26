import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_block import DPTOutputAdapter
from .camera import CameraHead
from .loss import homogenize_points


class Head(nn.Module):
    def __init__(self, cfg):
        super(Head, self).__init__()
        encoder_type = cfg.model.encoder_type
        self.enable_depth = cfg.model.head['enable_depth']
        self.enable_point = cfg.model.head['enable_point']
        self.enable_camera = cfg.model.head['enable_camera']
        args = cfg.model.head[encoder_type]
        self.patch_size = args.patch_size
        self.embed_dim = args.embed_dim
        self.hooks = [i + args.hooks_offset for i in args.hooks_start]

        if self.enable_depth:
            self.depth_head = DPTOutputAdapter(
                num_channels=2,
                patch_size=self.patch_size,
                hooks=self.hooks,
                feature_dim=self.embed_dim,
                layer_dims=[self.embed_dim//4, self.embed_dim//2, self.embed_dim, self.embed_dim],
                dim_tokens_enc=self.embed_dim,
                head_type='regression',
                activation="exp",
                conf_activation="expp1"
            )
        
        if self.enable_point:
            # like pi3, predict local pointmap
            self.point_head = DPTOutputAdapter(
                num_channels=4,
                patch_size=self.patch_size,
                hooks=self.hooks,
                feature_dim=self.embed_dim,
                layer_dims=[self.embed_dim//4, self.embed_dim//2, self.embed_dim, self.embed_dim],
                dim_tokens_enc=self.embed_dim,
                head_type='regression',
                activation="pi3",
                conf_activation="expp1"
            )
        
        if self.enable_camera:
            self.camera_head = CameraHead(
                dim = self.embed_dim
            )


    def forward(self, enc_tokens, H, W, B, V):
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        predictions = {}

        if self.enable_depth:
            depth, depth_conf = self.depth_head(
                enc_tokens, (H, W)
            )
            predictions["depth"] = depth.view(B, V, H, W, 1)
            predictions["depth_conf"] = depth_conf.view(B, V, H, W)

        if self.enable_point:
            pts3d, pts3d_conf = self.point_head(
                enc_tokens, (H, W)
            )
            predictions["local_points"] = pts3d.view(B, V, H, W, 3)
            predictions["local_points_conf"] = pts3d_conf.view(B, V, H, W)

        if self.enable_camera:
            camera_poses = self.camera_head(enc_tokens[-1], patch_h, patch_w).reshape(B, V, 4, 4)
            predictions["camera_poses"] = camera_poses # c2w
            predictions["points"] = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(predictions["local_points"]))[..., :3]

        return predictions
