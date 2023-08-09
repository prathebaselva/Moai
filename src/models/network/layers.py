import torch
import math
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class MLPLayerSimple(Module):
    def __init__(self, dim_in, dim_out, islast=False):
        super(MLPLayerSimple, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(dim_out)
        self.islast = islast

    def forward(self, x, islast=False):
        if self.islast:
            return self.mlp(x)
        ret = self.relu(self.bn(self.mlp(x)))
        return ret

class MLPLayerSigmoid(Module):
    def __init__(self, dim_in, dim_out, islast=False):
        super(MLPLayerSigmoid, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.bn = torch.nn.BatchNorm1d(dim_out)

    def forward(self, x, islast=False):
        out = self.bn(self.mlp(x))
        #print(out[0:10])
        #print(out.shape)
        #out -= out.min(0, keepdim=True)[0]
        #out /= out.max(0, keepdim=True)[0]

        return torch.sigmoid(out)


class ScaleShiftMLPLayer(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, dim_time):
        super(ScaleShiftMLPLayer, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        dim_context = dim_time
        if dim_ctx > 0:
            dim_context += dim_ctx
        self.shift1 = Linear(dim_context, dim_out, bias=False)
        self.shift2 = Linear(dim_out, dim_out, bias=False)
        self.scale1 = Linear(dim_context, dim_out)
        self.scale2 = Linear(dim_out, dim_out)
        self.silu = torch.nn.SiLU()
        self.gelu = torch.nn.GELU()

    def forward(self, ctx, time, x):
        context = time
        if ctx is not None:
            context = torch.cat([time, ctx], dim=-1)
        scale = self.scale2(self.gelu(self.scale1(context)))
        shift = self.shift2(self.gelu(self.shift1(context)))
        ret = self.mlp(x) * (scale ) + shift
        ret = self.silu(ret)
        return ret


class ScaleShiftMLPLayerSimple(Module):
    def __init__(self, dim_in, dim_out, dim_context, dim_time, islast=False):
        super(ScaleShiftMLPLayerSimple, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        dim_ctx = dim_time
        if dim_context is not None:
            dim_ctx = dim_context + dim_time
        self.scale = Linear(dim_ctx, dim_out)
        self.shift = Linear(dim_ctx, dim_out, bias=False)
        self.relu = torch.nn.ReLU()
        self.islast = islast

    def forward(self, ctx, time, x):
        context = time
        if ctx is not None:
            context = torch.cat([time, ctx], dim=-1).to('cuda')
        s = self.scale(context).to('cuda')
        scale = torch.sigmoid(s)
        shift = self.shift(context)
        if not self.islast:
            ret = self.relu(self.mlp(x)) * scale + shift
        else:
            ret = self.mlp(x) * scale + shift
        return ret


class GNMLPLayerSimple(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False):
        super(GNMLPLayerSimple, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.silu = torch.nn.SiLU()
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            return self.mlp(x)
        ret = self.gn(self.silu(self.mlp(x)))
        return ret

class GNMLPLayerSimpleRelu(Module):
    def __init__(self, dim_in, dim_out, groups=8, islast=False):
        super(GNMLPLayerSimpleRelu, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.relu = torch.nn.ReLU()
        
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        self.islast = islast

    def forward(self, x):
        if self.islast:
            return self.mlp(x)
        ret = self.gn(self.relu(self.mlp(x)))
        return ret


class ScaleShiftGNMLPLayerSimpleTanh(Module):
    def __init__(self, dim_in, dim_out, dim_context, dim_time, groups=8, islast=False):
        super(ScaleShiftGNMLPLayerSimpleTanh, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.scale = None
        self.shift = None
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        dim_ctx =  dim_time
        if dim_context is not None:
            dim_ctx = dim_context + dim_time
            self.scale = Linear(dim_ctx, dim_out)
            self.shift = Linear(dim_ctx, dim_out, bias=False)
        self.islast = islast
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)

    def forward(self, ctx, time, x):
        if (time is not None) and (self.scale is not None):
            context = time
            if ctx is not None:
                context = torch.cat([time, ctx], dim=-1).to('cuda')
            if self.scale is not None:
                s = self.scale(context).to('cuda')
                scale = torch.sigmoid(s)
                shift = self.shift(context)
                if self.islast:
                    return self.tanh(self.mlp(x) * scale + shift)
                if x.shape[0] > 1:
                    ret = self.gn(self.relu(self.mlp(x))) * scale + shift
                else:
                    ret = (self.relu(self.mlp(x))) * scale + shift
        else:
            if self.islast:
                return self.tanh(self.mlp(x))
            if x.shape[0] > 1:
                ret = self.gn(self.relu(self.mlp(x)))
            else:
                ret = (self.relu(self.mlp(x)))
        return ret

class ScaleShiftGNMLPLayerSimpleRelu(Module):
    def __init__(self, dim_in, dim_out, dim_context, dim_time, groups=8, islast=False):
        super(ScaleShiftGNMLPLayerSimpleRelu, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.scale = None
        self.shift = None
        self.relu = torch.nn.ReLU()

        dim_ctx =  dim_time
        if dim_context is not None:
            dim_ctx = dim_context + dim_time
            self.scale = Linear(dim_ctx, dim_out)
            self.shift = Linear(dim_ctx, dim_out, bias=False)
        self.islast = islast
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)

    def forward(self, ctx, time, x):
        if (time is not None) and (self.scale is not None):
            context = time
            if ctx is not None:
                context = torch.cat([time, ctx], dim=-1).to('cuda')
            if self.scale is not None:
                s = self.scale(context).to('cuda')
                scale = torch.sigmoid(s)
                shift = self.shift(context)
                if self.islast:
                    return self.mlp(x) * scale + shift
                if x.shape[0] > 1:
                    ret = self.gn(self.relu(self.mlp(x))) * scale + shift
                else:
                    ret = (self.relu(self.mlp(x))) * scale + shift
        else:
            if self.islast:
                return self.mlp(x)
            if x.shape[0] > 1:
                ret = self.gn(self.relu(self.mlp(x)))
            else:
                ret = (self.relu(self.mlp(x)))
        return ret

class ScaleShiftGNMLPLayerSimple(Module):
    def __init__(self, dim_in, dim_out, dim_context, dim_time, groups=8, islast=False):
        super(ScaleShiftGNMLPLayerSimple, self).__init__()
        self.mlp = Linear(dim_in, dim_out)
        self.scale = None
        self.shift = None
        dim_ctx =  dim_time
        if dim_context is not None:
            dim_ctx = dim_context + dim_time
            self.scale = Linear(dim_ctx, dim_out)
            self.shift = Linear(dim_ctx, dim_out, bias=False)
        self.silu = torch.nn.SiLU()
        self.islast = islast
        if not islast:
            self.gn = torch.nn.GroupNorm(groups, dim_out)
        

    def forward(self, ctx, time, x):
        if (time is not None) and (self.scale is not None):
            context = time
            if ctx is not None:
                context = torch.cat([time, ctx], dim=-1).to('cuda')
            if self.scale is not None:
                s = self.scale(context).to('cuda')
                scale = torch.sigmoid(s)
                shift = self.shift(context)
                if self.islast:
                    return self.mlp(x) * scale + shift
                if x.shape[0] > 1:
                    ret = self.gn(self.silu(self.mlp(x))) * scale + shift
                else:
                    ret = (self.silu(self.mlp(x))) * scale + shift
        else:
            if self.islast:
                return self.mlp(x)
            if x.shape[0] > 1:
                ret = self.gn(self.silu(self.mlp(x)))
            else:
                ret = (self.silu(self.mlp(x)))
        return ret

def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr


def getProjection(x_3d, focal_length):
    #print(x_3d.shape)
    x_2d = (x_3d[:,:,:2]/x_3d[:,:,2:3])[:,:,:3]
    #print(x_2d.shape)
    #print(focal_length.shape)
    f = (320 * (focal_length / 2))
    x_2d *=  f.view(-1,1,1)
    x_2d += torch.tensor([[320/2, 320/2]]).to('cuda')
    x_2d[:,:,0] = 320 - x_2d[:,:,0]
    return x_2d


def getoneProjection(x_3d, focal_length):
    #print(x_3d.shape)
    x_2d = (x_3d[:,:2]/x_3d[:,2:3])[:,:3]
    #print(x_2d)
    #print(x_2d.shape)
    #print(focal_length.shape)
    f = (320 * (focal_length / 2))
    x_2d *=  f
    x_2d += torch.tensor([[320/2, 320/2]]).to('cuda')
    x_2d[:,0] = 320 - x_2d[:,0]
    return x_2d


def getoneProjectionnp(x_3d, focal_length):
    #print(x_3d.shape)
    x_2d = (x_3d[:,:2]/x_3d[:,2:3])[:,:3]
    #print(x_2d.shape)
    #print(focal_length.shape)
    f = (1 * (focal_length / 2))
    x_2d *=  f
    x_2d += ([[1/2, 1/2]])
    x_2d[:,0] = 1 - x_2d[:,0]
    return x_2d


def getAllCenterAndRadius(mesh):
    maxdim = torch.max(mesh[:,:,:3], dim=1)[0]
    mindim = torch.min(mesh[:,:,:3], dim=1)[0]
    m = (maxdim - mindim)
    center = (maxdim+mindim)/2
    radius = torch.max(m, dim = 1)[0]
    #print(center.shape)
    #print(radius.shape)
    return center,radius

def getCenterAndRadius(mesh):
    maxdim = torch.max(mesh[:,:3], dim=0)[0]
    mindim = torch.min(mesh[:,:3], dim=0)[0]

    m = maxdim - mindim
    center = (maxdim+mindim)/2
    radius = torch.max(m)
    return center,radius
