import torch
import math
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


def sinusoidalembeddingwithbeta(t, betas, dim):
    t = torch.tensor(t).to('cuda')
    half_dim = dim / 4
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device='cuda')* -emb).to('cuda')
    timeemb = t[:, None] * emb[None, :]
    betaemb = betas[:, None] * emb[None, :]
    emb = torch.cat((timeemb.sin(), timeemb.cos(), betaemb.sin(), betaemb.cos()), dim=-1)
    return emb

def sinusoidalembedding(t, dim):
    t1 = t.clone().detach().to('cuda')
    half_dim = dim / 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device='cuda')* -emb).to('cuda')
    timeemb = t1[:, None] * emb[None, :]
    emb = torch.cat((timeemb.sin(), timeemb.cos()), dim=-1)
    #print(emb.shape)
    return emb


def timeembedding(t, dim):
    t = torch.tensor(t).to('cuda')
    half_dim = dim / 2
    emb = math.log(4000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device='cuda')* -emb).to('cuda')
    timeemb = t[:, None] * emb[None, :]
    emb = torch.cat((timeemb.sin(), timeemb.cos()), dim=-1)
    return emb

def binarytimeembedding(t, dim):
    stremb = '{0:0'+str(dim)+'b}'
    emb = np.array(['{0:0512b}'.format(num) for num in t])
    emb1 = []
    for e in emb:
        emb1.append([int(c) for c in list(e)])
    #print(stremb)
    emb = torch.tensor(np.array(emb1)).to('cuda').float()
    #print(emb.shape)
    return emb

