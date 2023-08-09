import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from .layers import ScaleShiftGNMLPLayerSimple, GNMLPLayerSimple, ScaleShiftGNMLPLayerSimpleRelu
from .embedding import sinusoidalembedding,binarytimeembedding,timeembedding

from .loss import ConditionShapeMLPLoss
import math
from utils.writeply import *
from utils.utils import *


class VarianceScheduleTestSampling(Module):
    def __init__(self, num_steps, beta_1, beta_T, eta=0,mode='linear'):
        super().__init__()
        assert mode in ('linear', 'cosine' )
        self.mode = mode
        #print(self.mode)
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.num_steps = num_steps


        if self.mode == 'linear':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=(self.num_steps))
            self.num_steps = len(betas)

        elif self.mode == 'cosine':
            s = 0.008
            warmupfrac = 1
            frac_steps = int(self.num_steps * warmupfrac)
            rem_steps = self.num_steps - frac_steps
            ft = [math.cos(((t/self.num_steps + s)/(1+s))*(math.pi/2))**2 for t in range(num_steps+1)]
            #ft = [math.cos(((t/frac_steps + s)/(1+s))*(math.pi/2))**2 for t in range(frac_steps+1)]
            alphabar = [(ft[t]/ft[0]) for t in range(frac_steps+1)]
            betas = np.zeros(self.num_steps)
            for i in range(1,frac_steps+1):
                betas[i-1] = min(1-(alphabar[i]/alphabar[i-1]), 0.999)
            #betas[frac_steps:] = [beta_T]*rem_steps
            self.num_steps = len(betas)

        self.num_steps = len(betas)
        print("num steps samp = ", self.num_steps)

        betas = np.array(betas, dtype=np.float32)
        assert((betas > 0).all() and (betas <=1).all())

        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)
        alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])
        sigma = eta*np.sqrt((1-(alpha_cumprod/alpha_cumprod_prev))*(1-alpha_cumprod_prev)/(1-alpha_cumprod))
        sigma = torch.tensor(sigma)
        alphas_cumprod_prev = torch.tensor(alpha_cumprod_prev)
        sqrt_alpha_cumprod = torch.tensor(np.sqrt(alpha_cumprod))
        sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - alpha_cumprod))
        log_one_minus_alpha_cumprod = torch.tensor(np.log(1.0 - alpha_cumprod))
        sqrt_recip_alpha_cumprod = torch.tensor(np.sqrt(1.0/alpha_cumprod))
        sqrt_recip_minus_one_alpha_cumprod = torch.tensor(np.sqrt((1.0/alpha_cumprod) -1))
        sqrt_recip_one_minus_alpha_cumprod = np.sqrt(1.0/(1 - alpha_cumprod))


        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        posterior_mean_coeff1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        posterior_mean_coeff2 = torch.tensor((1.0 - alpha_cumprod_prev)*np.sqrt(alphas) / (1 - alpha_cumprod))
        posterior_mean_coeff3 = torch.tensor((betas * sqrt_recip_one_minus_alpha_cumprod))
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        alphas_cumprod = torch.tensor(alpha_cumprod)
        posterior_variance = torch.tensor(posterior_variance)

        self.register_buffer('test_betas', betas)
        self.register_buffer('test_alphas', alphas)
        self.register_buffer('test_sigma', sigma)
        self.register_buffer('test_alphas_cumprod', alphas_cumprod)
        self.register_buffer('test_alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('test_sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('test_sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)
        self.register_buffer('test_log_one_minus_alpha_cumprod', log_one_minus_alpha_cumprod)
        self.register_buffer('test_sqrt_recip_alpha_cumprod', sqrt_recip_alpha_cumprod)
        self.register_buffer('test_sqrt_recip_minus_one_alpha_cumprod', sqrt_recip_minus_one_alpha_cumprod)

        self.register_buffer('test_posterior_variance', posterior_variance)
        self.register_buffer('test_posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('test_posterior_mean_coeff1', posterior_mean_coeff1)
        self.register_buffer('test_posterior_mean_coeff2', posterior_mean_coeff2)
        self.register_buffer('test_posterior_mean_coeff3', posterior_mean_coeff3)

class VarianceScheduleMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.num_steps = config.num_steps
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T
        assert self.mode in ('linear', 'cosine' )

        if self.mode == 'linear':
            betas = torch.linspace(self.beta_1, self.beta_T, steps=(self.num_steps))
            self.num_steps = len(betas)

        elif self.mode == 'cosine':
            s = 0.008
            warmupfrac = 1
            frac_steps = int(self.num_steps * warmupfrac)
            rem_steps = self.num_steps - frac_steps
            ft = [math.cos(((t/self.num_steps + s)/(1+s))*(math.pi/2))**2 for t in range(num_steps+1)]
            #ft = [math.cos(((t/frac_steps + s)/(1+s))*(math.pi/2))**2 for t in range(frac_steps+1)]
            alphabar = [(ft[t]/ft[0]) for t in range(frac_steps+1)]
            betas = np.zeros(self.num_steps)
            for i in range(1,frac_steps+1):
                betas[i-1] = min(1-(alphabar[i]/alphabar[i-1]), 0.999)
            #betas[frac_steps:] = [beta_T]*rem_steps
            self.num_steps = len(betas)

        betas = np.array(betas, dtype=np.float32)
        assert((betas > 0).all() and (betas <=1).all())

        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas)
        alpha_cumprod_prev = np.append(1., alpha_cumprod[:-1])

        alphas_cumprod_prev = torch.tensor(alpha_cumprod_prev)
        sqrt_alpha_cumprod = torch.tensor(np.sqrt(alpha_cumprod))
        sqrt_recip_alpha = torch.tensor(np.sqrt(1.0/alphas))
        sqrt_one_minus_alpha_cumprod = torch.tensor(np.sqrt(1.0 - alpha_cumprod))
        log_one_minus_alpha_cumprod = torch.tensor(np.log(1.0 - alpha_cumprod))
        sqrt_recip_alpha_cumprod = torch.tensor(np.sqrt(1.0/alpha_cumprod))
        sqrt_recip_minus_one_alpha_cumprod = torch.tensor(np.sqrt((1.0/alpha_cumprod) -1))
        sqrt_recip_one_minus_alpha_cumprod = np.sqrt(1.0/(1 - alpha_cumprod))



        posterior_variance = (betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
        posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)))
        posterior_mean_coeff1 = torch.tensor(betas * np.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod))
        posterior_mean_coeff2 = torch.tensor((1.0 - alpha_cumprod_prev)*np.sqrt(alphas) / (1 - alpha_cumprod))
        posterior_mean_coeff3 = torch.tensor((betas * sqrt_recip_one_minus_alpha_cumprod))
        betas = torch.tensor(betas)
        alphas = torch.tensor(alphas)
        sqrt_alpha = torch.tensor(np.sqrt(alphas))
        alphas_cumprod = torch.tensor(alpha_cumprod)
        one_minus_alpha_cumprod = torch.tensor(1.0 - alpha_cumprod)
        mean_coeff = (one_minus_alpha_cumprod / betas)
        posterior_variance = torch.tensor(posterior_variance)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alpha_cumprod', sqrt_alpha_cumprod)
        self.register_buffer('sqrt_alpha', sqrt_alpha)
        self.register_buffer('sqrt_recip_alpha', sqrt_recip_alpha)
        self.register_buffer('sqrt_one_minus_alpha_cumprod', sqrt_one_minus_alpha_cumprod)
        self.register_buffer('log_one_minus_alpha_cumprod', log_one_minus_alpha_cumprod)
        self.register_buffer('sqrt_recip_alpha_cumprod', sqrt_recip_alpha_cumprod)
        self.register_buffer('sqrt_recip_minus_one_alpha_cumprod', sqrt_recip_minus_one_alpha_cumprod)

        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coeff1', posterior_mean_coeff1)
        self.register_buffer('posterior_mean_coeff2', posterior_mean_coeff2)
        self.register_buffer('posterior_mean_coeff3', posterior_mean_coeff3)
        self.register_buffer('mean_coeff', mean_coeff)


    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(self.num_steps), batch_size)
#        ts[0] = 5
        ts[0] = 5
#        ts[2] = 250
#        ts[3] = 500
#        ts[4] = 700
#        ts[5] = 800
        ts[1] = 999
        return ts.tolist()

class MoaiMLPNet(Module):
    def __init__(self, config):
        super().__init__()
        context_dim = config.context_dim
        time_dim = config.time_dim
        self.moai_dim = config.moai_dim
        self.arch = config.arch
        self.embedlayers = None
        self.tanh = torch.nn.Tanh()


        if self.arch == 'archjaw4':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 4, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(4, 8, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(8, 16, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(16, 32, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(32, 16, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(32, 8, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(16, 4, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(8, self.moai_dim, context_dim,time_dim,1, islast=True),
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [4,5,6]

        if self.arch == 'archjaw3':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 4, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(4, 8, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(8, 16, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(16, 32, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(32, 64, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(64, 128, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(128, 64, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(128, 32, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(64, 16, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(32, 8, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(16, 4, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(8, self.moai_dim, context_dim,time_dim,1, islast=True),
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [6,7,8,9,10]

        if self.arch == 'archjaw2':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1, islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = []

        if self.arch == 'archjaw0':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1, islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [3,4]


        if self.arch == 'archexp11':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 256, context_dim,time_dim,16),
                    ScaleShiftGNMLPLayerSimple(256, 512, context_dim,time_dim,32),
                    ScaleShiftGNMLPLayerSimple(512, 1024, context_dim,time_dim,64),
                    ScaleShiftGNMLPLayerSimple(1024, 2048, context_dim,time_dim,128),
                    ScaleShiftGNMLPLayerSimple(2048, 1024, context_dim,time_dim,64),
                    ScaleShiftGNMLPLayerSimple((2*1024),512 , context_dim,time_dim,128),
                    ScaleShiftGNMLPLayerSimple((2*512),256, context_dim,time_dim,32),
                    ScaleShiftGNMLPLayerSimple((2*256), self.moai_dim, context_dim,time_dim,1, islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [4,5,6]

        if self.arch == 'archexp10':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1, islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [5,6,7,8]



        if self.arch == 'archexp0':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*self.moai_dim), self.moai_dim, context_dim,time_dim,1, islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [3,4]

        if self.arch == 'archexp1':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim,1, islast=True)
            ])
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = []

        if self.arch == 'archexp2':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 64, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(64, 32, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(32, 16, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(16, 32, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple((2*32), 64, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple((2*64), self.moai_dim, context_dim,time_dim,islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [3,4]

        if self.arch == 'archexp3':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 64, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple(64, 32, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple(32, 16, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(16, 8, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple(8, 16, context_dim,time_dim,1),
                    ScaleShiftGNMLPLayerSimple((2*16), 32, context_dim,time_dim,2),
                    ScaleShiftGNMLPLayerSimple((2*32), 64, context_dim,time_dim,4),
                    ScaleShiftGNMLPLayerSimple((2*64), self.moai_dim, context_dim,time_dim,islast=True)
            ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, islast=True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [4,5,6]

        if self.arch == 'archexp4':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 256, context_dim,time_dim,16),
                    ScaleShiftGNMLPLayerSimple(256, 512, context_dim,time_dim,32),
                    ScaleShiftGNMLPLayerSimple(512, 256, context_dim,time_dim,16),
                    ScaleShiftGNMLPLayerSimple(256, self.moai_dim, context_dim,time_dim,islast=True)
            ])
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = []

        if self.arch == 'archv0':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 200, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(200, 100, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(100, 50, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(50, 25, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(25, 10, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(10, 25, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(50, 50, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(100, 100, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(200, 200, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(400, 300, context_dim,time_dim, 5, True)
                ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [5,6,7,8]

        if self.arch == 'archv4':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 200, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(200, 100, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(100, 50, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(50, 100, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(200, 200, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(400, 300, context_dim,time_dim, 5, True)
                ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, True)
            #self.moailayers = ModuleList([
            #        GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
            #    ])
            self.skip_layers = [3,4]

        if self.arch == 'archv3':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, 200, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(200, 100, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(100, 50, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(50, 25, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(25, 50, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(100, 100, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(200, 200, context_dim,time_dim, 5),
                    ScaleShiftGNMLPLayerSimple(400, 300, context_dim,time_dim, 5, True)
                ])
            self.outmodule = ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 5, True)
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = [4,5,6]

        if self.arch == 'archv2':
            self.layers = ModuleList([
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 1),
                    ScaleShiftGNMLPLayerSimple(self.moai_dim, self.moai_dim, context_dim,time_dim, 1, True),
                ])
            self.moailayers = ModuleList([
                    GNMLPLayerSimple(self.moai_dim, self.moai_dim, islast=True),
                ])
            self.skip_layers = []
        self.time_dim = time_dim

    def forward(self, x, t, context):
        """
        Args:
            x:  Moai parameter at some timestep t, (B, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        time_emb = sinusoidalembedding(t, self.time_dim)

        x = x.view(batch_size, -1)
        unet_out = []
        out = x.clone()
        k = 1
#        if self.embedlayers is not None:
#            for i,layer in enumerate(self.embedlayers):
#                out = layer(out)

        for i, layer in enumerate(self.layers):
            out = layer(ctx=context, time=time_emb, x=out)

            if i < len(self.layers) - 1:
                unet_out.append(out.clone())
                if i in self.skip_layers:
                    out = torch.cat([out, unet_out[i-2*k]],dim=1)
                    k += 1

        if (self.arch == 'archv4'):
            out = out + self.outmodule(ctx=context, time=time_emb, x=x)
            return out.view(batch_size, -1), None
        if (self.arch == 'archexp1' or self.arch== 'archexp4' or self.arch == 'archv2'):
            moaiout = out.clone()
            moaiout = self.moailayers[0](moaiout)
            return out.view(batch_size, -1), moaiout
        out = out + self.outmodule(ctx=context, time=time_emb, x=x)
        moaiout = out.clone()
        moaiout = self.moailayers[0](moaiout)
        return out.view(batch_size, -1), moaiout.view(batch_size, -1)

class MoaiParamDiffusion(Module):
    def __init__(self, net, var_sched:VarianceScheduleMLP, device, tag):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.tag = tag
        self.device = device

    def decode(self, epoch, moaiparam_x0, context, moai=None): 
        """
        Args:
            moaiparam_x0:  Input moai parameters, (B, N, d) ==> Batch_size X Number of points X point_dim(3).
            context:  Image latent, (B, F). ==> Batch_size X Image_latent_dim 
            lossparam: NetworkLossParam object.
        """
        batch_size, _ = moaiparam_x0.size()

        t = None
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        #moaiparam_x0 *= 0.01
        #print(np.std(moaiparam_x0[0].cpu().numpy()))
        #print(np.mean(moaiparam_x0[0].cpu().numpy()))

        #print("min max {} {}".format(torch.min(moaiparam_x0,1)[0], torch.max(moaiparam_x0,1)[0]))
        #print(moaiparam_x0[0:10])
        #print(torch.std_mean(moaiparam_x0[0]))
        #print(torch.std_mean(moaiparam_x0[0,100:]))
        #print(moaiparam_x0[0,100:])
        moaiparam_xt, e_rand = self.get_train_mesh_sample(moaiparam_x0, t, epoch)
        #print("min max {} {}".format(torch.min(moaiparam_xt,1)[0], torch.max(moaiparam_xt,1)[0]))
        #exit()
#        print(moaiparam_xt.shape, flush=True)
        #meshxt,_,_= moai(shape_params=moaiparam_xt[0].unsqueeze(0).float()) 
        #trimesh.Trimesh(vertices=moaiparam_xt[0].squeeze().cpu().numpy()).export('mesh_xt_0.ply')
        #meshxt,_,_= moai(shape_params=moaiparam_xt[1].unsqueeze(0).float()) 
        #trimesh.Trimesh(vertices=moaiparam_xt[1].squeeze().cpu().numpy()).export('mesh_xt_1.ply')
        #print(torch.std_mean(moaiparam_xt[0]))
        #print(torch.std_mean(moaiparam_xt[1]), flush=True)
        #exit()
#        meshxt,_,_= moai(shape_params=moaiparam_xt[2].unsqueeze(0).float()) 
#        trimesh.Trimesh(vertices=meshxt.squeeze().cpu().numpy()).export('mesh_xt_2.ply')
#        meshxt,_,_= moai(shape_params=moaiparam_xt[3].unsqueeze(0).float()) 
#        trimesh.Trimesh(vertices=meshxt.squeeze().cpu().numpy()).export('mesh_xt_3.ply')
#        meshxt,_,_= moai(shape_params=moaiparam_xt[4].unsqueeze(0).float()) 
#        trimesh.Trimesh(vertices=meshxt.squeeze().cpu().numpy()).export('mesh_xt_4.ply')
#        meshxt,_,_= moai(shape_params=moaiparam_xt[5].unsqueeze(0).float()) 
#        trimesh.Trimesh(vertices=meshxt.squeeze().cpu().numpy()).export('mesh_xt_5.ply')
#        meshxt,_,_= moai(shape_params=moaiparam_xt[6].unsqueeze(0).float()) 
#        trimesh.Trimesh(vertices=meshxt.squeeze().cpu().numpy()).export('mesh_xt_6.ply')
#        #print("min max {} {}".format(torch.min(moaiparam_xt[6],0)[0], torch.max(moaiparam_xt[6],0)[0]))
#        exit()
        #print(np.std(moaiparam_xt[0].cpu().numpy()))
        #print(np.mean(moaiparam_xt[0].cpu().numpy()), flush=True)
        predmoaiparam_x0 = None
        getmoaiparam_x0 = False

        pred_theta, pred_moai, pred_moaiparam_x0 = self.get_network_prediction(moaiparam_xt=moaiparam_xt, t=t, context=context, prednoise=True, getmeshx0=True)
        meanxt = self.get_meanxt(moaiparam_xt, t, e_rand)
        #return pred_theta.view(batch_size,-1), e_rand.view(batch_size, -1), predmoaiparam_x0*100
        return pred_theta.view(batch_size,-1), e_rand.view(batch_size, -1), pred_moai, pred_moaiparam_x0, meanxt
        #return pred_theta.view(batch_size,-1), e_rand.view(batch_size, -1), pred_theta.view(batch_size, -1)
        #return pred_theta.view(batch_size,-1)

    def get_meshx0_from_meanpred(self, moaiparam_xt, pred_theta, t):
        mean_coeff = self.var_sched.mean_coeff[t].view(-1,1)
        sqrt_alpha = self.var_sched.sqrt_alpha[t].view(-1,1)
        moaiparam_x0 =   (moaiparam_xt * (1 - mean_coeff)) + sqrt_alpha * pred_theta * mean_coeff
        return moaiparam_x0

    def get_meshx0_from_noisepred(self, moaiparam_xt, pred_theta, t):
        sqrt_recip_alpha_cumprod = self.var_sched.sqrt_recip_alpha_cumprod[t].view(-1,1)
        sqrt_recip_minus_one_alpha_cumprod = self.var_sched.sqrt_recip_minus_one_alpha_cumprod[t].view(-1,1)
        moaiparam_x0 =  (sqrt_recip_alpha_cumprod * moaiparam_xt) - (sqrt_recip_minus_one_alpha_cumprod * pred_theta)
        return moaiparam_x0

    def get_meshx0_from_noisepred_sampling(self, moaiparam_xt, pred_theta, t, varsched):
        #print(moaiparam_xt.shape)
        t = torch.Tensor(t).long().to(self.device)
        #print(t.shape)
        sqrt_recip_alpha_cumprod = varsched.test_sqrt_recip_alpha_cumprod[t].view(-1,1).to(self.device)
        #sqrt_recip_alpha_cumprod = varsched.test_sqrt_recip_alpha_cumprod[t].to(self.device)
        #print(sqrt_recip_alpha_cumprod.shape)
        sqrt_recip_minus_one_alpha_cumprod = varsched.test_sqrt_recip_minus_one_alpha_cumprod[t].view(-1,1).to(self.device)
        #sqrt_recip_minus_one_alpha_cumprod = varsched.test_sqrt_recip_minus_one_alpha_cumprod[t].to(self.device)
        #print(sqrt_recip_minus_one_alpha_cumprod.shape, flush=True)
        moaiparam_x0 =  (sqrt_recip_alpha_cumprod * moaiparam_xt) - (sqrt_recip_minus_one_alpha_cumprod * pred_theta)
        return moaiparam_x0

    def get_network_prediction(self, moaiparam_xt, t, context=None, prednoise=True, getmeshx0=False, issampling=False, varsched=None):
        moaiparam_xt = moaiparam_xt.to(dtype=torch.float32).to(self.device)
        t = torch.Tensor(t).long().to(self.device)
        if context is not None:
            context = context.to(self.device)
        #pred_theta = self.net(moaiparam_xt, t=t, context=context)
        pred_theta, pred_moai = self.net(moaiparam_xt, t=t, context=context)

        moaiparam_x0 = None
        if prednoise:
            pred_theta = pred_theta
            if getmeshx0:
                if issampling and (varsched is not None):
                    moaiparam_x0 = self.get_meshx0_from_noisepred_sampling(moaiparam_xt, pred_theta, t, varsched)
                else:
                    moaiparam_x0 = self.get_meshx0_from_noisepred(moaiparam_xt, pred_theta, t)
                    #moaiparam_x0 = self.get_meshx0_from_meanpred(moaiparam_xt, pred_theta, t)
        else:
            moaiparam_x0 = pred_theta
            pred_theta = None
        return pred_theta, pred_moai, moaiparam_x0

    def get_meanxt(self, moaiparam_xt,t, e_rand):
        posterior_mean_coeff3 = self.var_sched.posterior_mean_coeff3[t].view(-1,1).to(self.device)
        mean = self.var_sched.sqrt_recip_alpha[t].view(-1,1) * (moaiparam_xt - posterior_mean_coeff3 * e_rand)
        return mean


    def get_train_mesh_sample(self, moaiparam_x0, t, epoch=None):
        e_rand = torch.zeros(moaiparam_x0.shape).to(self.device)  # (B, N, d)
        batch_size = moaiparam_x0.shape[0]
        e_rand = torch.randn_like(moaiparam_x0)
        sqrt_alpha_cumprod = self.var_sched.sqrt_alpha_cumprod[t].view(-1,1)
        sqrt_one_minus_alpha_cumprod = self.var_sched.sqrt_one_minus_alpha_cumprod[t].view(-1,1)
        moaiparam_xt = (sqrt_alpha_cumprod * moaiparam_x0) + (sqrt_one_minus_alpha_cumprod * e_rand)
        return moaiparam_xt, e_rand 

    def get_shapemlp_loss(self, epoch, moaiparam_x0, context): # lossparam):
        """
        Args:
            moaiparam_x0:  Input point cloud, (B, N, d) ==> Batch_size X Number of points X point_dim(3).
            context:  Image latent, (B, F). ==> Batch_size X Image_latent_dim 
            lossparam: NetworkLossParam object.
        """
        batch_size, num_points, point_dim = moaiparam_x0.size()

        t = None
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        moaiparam_xt, e_rand = self.get_train_mesh_sample(moaiparam_x0, t, epoch)
        getmoaiparam_x0 = False
        predmoaiparam_x0 = None
        if (epoch >= 1) and (np.random.rand() < 0.001):
            getmoaiparam_x0 = True
        pred_theta, predmoaiparam_x0 = self.get_network_prediction(moaiparam_xt, t, context, True, getmoaiparam_x0)

        if getmoaiparam_x0:
            indx = np.random.randint(batch_size, size=(15,))
            sampt = np.array(t)[indx]
#            if context is None: 
#                radius = torch.Tensor([lossparam.mesh_radius] * batch_size).to(self.device)
#                center = torch.Tensor([lossparam.mesh_center] * batch_size).to(self.device)
#            else:
#                radius = lossparam.mesh_radius[indx]
#                center = lossparam.mesh_center[indx]
            #visualize_kde(sampt, moaiparam_x0, moaiparam_xt, predmoaiparam_x0, self.tag, epoch)
            #visualize_prediction(sampt, predmoaiparam_x0[indx], moaiparam_x0[indx], radius, center, 1.5, self.tag, epoch)
            convertToPLY(predmoaiparam_x0[0].clone().detach().cpu().numpy(), str(epoch)+'_'+str(t[0])+'_'+self.tag+'_train_out.ply', '../results/')
            convertToPLY(moaiparam_x0[0].detach().cpu().numpy(), str(epoch)+'_'+ str(t[0])+'_'+self.tag+'_train_ref.ply', '../results/')

        loss = F.mse_loss(pred_theta.view(-1, 3), e_rand.view(-1,3), reduction='mean')
        return loss

    def get_pposterior_sample(self, pred_mesh_x0, mesh_xt, t, varsched):
        posterior_mean_coeff1 = varsched.test_posterior_mean_coeff1[t].view(-1,1).to(self.device)
        posterior_mean_coeff2 = varsched.test_posterior_mean_coeff2[t].view(-1,1).to(self.device)
        posterior_variance = varsched.test_posterior_variance[t].view(-1,1).to(self.device)
        posterior_log_variance_clipped = varsched.test_posterior_log_variance_clipped[t].view(-1,1).to(self.device)
        mean = posterior_mean_coeff1 * pred_mesh_x0 + posterior_mean_coeff2 * mesh_xt
        return mean, posterior_log_variance_clipped, posterior_variance

    def get_pposterior_sample1(self, pred_moaiparam_x0, pred_theta, t, varsched):
        mean_coeff = torch.sqrt(varsched.test_alphas_cumprod_prev[t]).view(-1,1).to(self.device)
        dir_xt = torch.sqrt(1- varsched.test_alphas_cumprod_prev[t] - (varsched.test_sigma[t] **2)).view(-1,1).to(self.device)
        mean = mean_coeff * pred_moaiparam_x0  + dir_xt * pred_theta
        return mean

    def get_mean_var(self, mesh_xt, pred_theta, t, varsched):
        sqrt_recip_alphas = torch.sqrt(1.0/ varsched.test_alphas[t]).view(-1,1).to(self.device)
        posterior_variance = torch.sqrt(varsched.test_posterior_variance[t]).view(-1,1).to(self.device)
        posterior_log_variance_clipped = ((0.5 * varsched.test_posterior_log_variance_clipped[t]).exp()).view(-1,1).to(self.device)
        c1 = ((1 - varsched.test_alphas[t])/(torch.sqrt(1 - varsched.test_alphas_cumprod[t]))).view(-1,1).to(self.device)
        mean = sqrt_recip_alphas * (mesh_xt - c1 * pred_theta)
        return mean, posterior_variance, posterior_log_variance_clipped


    def get_var(self, t, varsched):
        posterior_mean_coeff3 = varsched.test_posterior_mean_coeff3[t].view(-1,1).to(self.device)
        posterior_variance = torch.sqrt(varsched.test_posterior_variance[t]).view(-1,1).to(self.device)
        posterior_log_variance_clipped = ((0.5 * varsched.test_posterior_log_variance_clipped[t]).exp()).view(-1,1).to(self.device)
        return  posterior_variance, posterior_log_variance_clipped

    def sample(self, num_points, context, moai, batch_size=1, point_dim=3, sampling='ddpm', shapeparam=None, expparam=None): 
        #mesh_xT = torch.randn([batch_size, num_points]).to(self.device)
        #if num_points == 3:
        #    mesh_xT = torch.normal(-0.16,0.37, size=(batch_size, num_points)).to(self.device)
        #else:
        #    mesh_xT = torch.normal(-0.1,1, size=(batch_size, num_points)).to(self.device)
        mesh_xT = torch.normal(0,1, size=(batch_size, num_points)).to(self.device)
        context = context.to(self.device)
        faces = np.asarray(moai.faces_tensor.cpu())
        #varsched = VarianceScheduleTestSampling(10000, 1e-4, 0.01, 'linear').to(self.device)
        #varsched1 = VarianceScheduleTestSampling1(10000, 1e-4, 0.01, 1, 'linear').to(self.device)
        #print("shape apram", flush=True)
        #print(shapeparam, flush=True)
        #self.var_sched.num_steps = 1000
        if sampling == 'ddim':
            varsched = VarianceScheduleTestSampling(self.var_sched.num_steps, self.var_sched.beta_1, self.var_sched.beta_T, 0, self.var_sched.mode).to(self.device)
        else:
            varsched = VarianceScheduleTestSampling(self.var_sched.num_steps, self.var_sched.beta_1, self.var_sched.beta_T, 1, self.var_sched.mode).to(self.device)

        traj = {varsched.num_steps-1: mesh_xT}
        iteri = 1

        r = np.random.randint(0, batch_size)
        print("number testing samples = ", varsched.num_steps)

        iterator = [x for x in reversed(range(0, varsched.num_steps))]
        #iterator = [x for x in reversed(range(-1,varsched.num_steps,4))]
        count = 0
        traj = {varsched.num_steps-1: mesh_xT}
        for idx, t in enumerate(iterator):
            z = torch.zeros(mesh_xT.shape).to(self.device)  # (B, N, d)
            if t > 0:
                z = torch.normal(0,1, size=(mesh_xT.shape)).to(self.device)

            moaiparam_xt = traj[t]
            batch_t = ([t]*batch_size)
            #moaiparam_xt = moaiparam_xt.unsqueeze(1)
            #print(moaiparam_xt.shape)

            pred_theta, pred_moai, pred_moaiparam_x0 = self.get_network_prediction(moaiparam_xt, batch_t, context, getmeshx0=True, issampling=True, varsched=varsched)
            #print(predmoaiparam_x0.shape, flush=True)
            #if predmoaiparam_x0.ndim == 3:
            #    shape_param = predmoaiparam_x0[:,:300].squeeze(0)
            #elif predmoaiparam_x0.ndim == 4:
            #    shape_param = predmoaiparam_x0[:,:300].squeeze()
            #shape_param = predmoaiparam_x0[:,:300]
            #pred_moai_mesh_x0,_,_= moai(shape_params=shape_param.float()) 
            #predmoaiparam_x0 = pred_theta
            #print(pred_theta[0][0])
            if sampling == 'dd':
                return pred_moai
            # When we use diffusion directly from data and not the noise
            elif sampling == 'ddim1':
               #moaiparam_xt = torch.clamp(moaiparam_xt, -2,2)
               #mean, logvar, var = self.get_pposterior_sample(predmoaiparam_x0, moaiparam_xt, batch_t, varsched)
               mean, logvar, var = self.get_pposterior_sample(pred_theta, moaiparam_xt, batch_t, varsched)
               moaiparam_xprevt = mean + (0.5 * logvar).exp() * z
               #moaiparam_xprevt = torch.clamp(moaiparam_xt, -2,2)
            elif sampling == 'ddim':
                #predmoaiparam_x0 = torch.clamp(predmoaiparam_x0, -1,1)
                mean = self.get_pposterior_sample1(pred_moaiparam_x0, pred_theta, batch_t, varsched)
                moaiparam_xprevt = mean + varsched.test_sigma[t].view(-1,1) * z

                #if count % 10 == 0:
                #    trimesh.Trimesh(vertices=pred_moai_mesh_x0[0].cpu().numpy(), faces=faces, process=False).export('sampletest/ddim/sample'+str(idx)+'.ply')
            elif sampling == 'ddm':
                mean, logvar, var = self.get_pposterior_sample(pred_moaiparam_x0, moaiparam_xt, batch_t, varsched)
                moaiparam_xprevt = mean + (0.5*logvar).exp() * z
            else:
               mean, var, logvar = self.get_mean_var(moaiparam_xt, pred_theta, batch_t, varsched)
               #mean, var, logvar = self.get_mean_var(predmoaiparam_x0, pred_theta, batch_t, varsched)
               moaiparam_xprevt = mean + logvar * z
               #if shapeparam is not None:
               #    meshprevxt,_,_= moai(shape_params=shapeparam, expression_params=moaiparam_xprevt[:,:100].float(), jaw_params=moaiparam_xprevt[:,100:].float())
               #    print("mesh prev min max {} {}".format(torch.min(meshprevxt, dim=1)[0], torch.max(meshprevxt, dim=1)[0]), flush=True)
                   #moaiparam_xprevt = meshprevxt
               #if count % 10 == 0:
                    #trimesh.Trimesh(vertices=pred_moai_mesh_x0[0].cpu().numpy(), faces=faces, process=False).export('sampletest/ddpm/sample'+str(idx)+'.ply')
               #     trimesh.Trimesh(vertices=meshprevxt[0].cpu().numpy(), faces=faces, process=False).export('sampletest/ddpm/meshprevxt'+str(idx)+'.ply')
               #print("min max {} {}".format(torch.min(moaiparam_xprevt, dim=1)[0], torch.max(moaiparam_xprevt, dim=1)[0]), flush=True)
               #moaiparam_xprevt = torch.clamp(moaiparam_xprevt, -3.2, 3.2)
            if t > 0:
                traj[iterator[idx+1]] = moaiparam_xprevt.clone().detach()     # Stop gradient and save trajectory.
                del traj[t]
            else:
                traj[-1] = moaiparam_xprevt.clone().detach()
                #pred_theta, pred_moai, predmoaiparam_x0 = self.get_network_prediction(traj[-1], batch_t, context, getmeshx0=True, issampling=True, varsched=varsched)
                #return predmoaiparam_x0
                #return pred_moai
            count += 1

        return traj[-1]

