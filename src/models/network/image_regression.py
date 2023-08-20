import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from .layers import GNMLPLayerSimple
from .embedding import sinusoidalembedding
from .loss import ConditionShapeMLPLoss

class ImageMLPNet(Module):
    def __init__(self, config):
        super().__init__()
        self.context_dim = config.context_dim
        self.flame_dim = config.flame_dim

        if self.arch == 'arch1':
            self.layers = ModuleList([
                    GNMLPLayerSimple(self.context_dim, self.context_dim), # 512 - 256
                    GNMLPLayerSimple(self.context_dim, self.context_dim//2), # 512 - 256
                    GNMLPLayerSimple(self.context_dim//2, self.context_dim//4), # 256 - 128
                    GNMLPLayerSimple(self.context_dim//4, self.context_dim//2), # 128 - 256
                    GNMLPLayerSimple((2*self.context_dim//2), self.context_dim), # 256 - 512
                    GNMLPLayerSimple((2*self.context_dim), self.flame_dim, islast=True) # 256 - 512
            ])
            self.outmodule = GNMLPLayerSimple(self.context_dim, self.flame_dim, islast=True)
            self.skip_layers = [3,4]
        if self.arch == 'arch3':
            self.layers = ModuleList([
                    GNMLPLayerSimple(self.context_dim, self.context_dim), # 512 - 256
                    GNMLPLayerSimple(self.context_dim, self.context_dim//2), # 512 - 256
                    GNMLPLayerSimple(self.context_dim//2, self.context_dim//4), # 256 - 128
                    GNMLPLayerSimple(self.context_dim//4, self.context_dim//2), # 128 - 256
                    GNMLPLayerSimple((2*self.context_dim//2), self.context_dim), # 256 - 512
                    GNMLPLayerSimple((2*self.context_dim), self.flame_dim, islast=True) # 256 - 512
            ])
            self.outmodule = GNMLPLayerSimple(self.context_dim, self.flame_dim, islast=True)
            self.skip_layers = [3,4]

    def forward(self, imgemb):
        """
        Args:
            x:  Image embedding input, (B, d).
        """
        batch_size = imgemb.size(0)
        imgemb = imgemb.view(batch_size, -1)
        unet_out = []
        out = imgemb.clone()
        k = 1
        for i, layer in enumerate(self.layers):
            out = layer(x=out)

            if i < len(self.layers) - 1:
                unet_out.append(out.clone())
                if i in self.skip_layers:
                    out = torch.cat([out, unet_out[i-2*k]],dim=1)
                    k += 1
        out = out + self.outmodule(x=imgemb)
        return out.view(batch_size, -1)

