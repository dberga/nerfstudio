import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

from nerfstudio.field_components.encodings import Encoding
from nerfstudio.field_components.embedding import Embedding

# this module encodes appearance, it depends on number of embeddings (parameter N_a, default is 48)
class E_attr(Encoding): # default heritance was from class nn.Module
  def __init__(self, input_dim_a=3, output_nc=8):
    super(E_attr, self).__init__()
    dim = 64
    self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),  ## size
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),  ## size/2
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/4
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/8
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),  ## size/16
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))  ## 1*1
    return

  def forward(self, x):
    x = self.model(x)
    output = x.view(x.size(0), -1)
    return output

# this module embeds occlusion through mask mapping
class implicit_mask(Embedding): # default heritance was from class nn.Module
    def __init__(self, latent=128, W=256, in_channels_dir=42):
        super().__init__()
        self.mask_mapping = nn.Sequential(
                            nn.Linear(latent + in_channels_dir, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, W), nn.ReLU(True),
                            nn.Linear(W, 1), nn.Sigmoid())

    def forward(self, x):
        mask = self.mask_mapping(x)
        return mask

"""
Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
Used for defining embedding_uv, embedding_xyz and embedding_dir
"""
class PosEmbedding(Embedding): # default heritance was from class nn.Module
    def __init__(self, max_logscale, N_freqs, logscale=True):
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)