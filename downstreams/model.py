import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .head import Head

from hydra.utils import instantiate

class MVEstimator(nn.Module):

    def __init__(self, cfg):
        super(MVEstimator, self).__init__()
        self.encoder = Encoder(cfg)
        self.head = Head(cfg)
        self.loss = instantiate(cfg.model.loss)

    def forward(self, samples, compute_loss=True):
        images = samples['images']
        B, V, _, H, W = images.shape
        enc_tokens = self.encoder(samples)
        predictions = self.head(enc_tokens, H, W, B, V)

        if compute_loss:
            loss = self.compute_loss(predictions, samples)
            return loss, _, _, _
        else:
            return predictions
    
    def compute_loss(self, predictions, batch):
        loss_dict = self.loss(predictions, batch)
        return loss_dict