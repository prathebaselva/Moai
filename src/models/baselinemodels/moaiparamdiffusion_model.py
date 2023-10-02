# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import sys


import torch
import torch.nn.functional as F

from src.models.arcface import Arcface
from src.models.base_model import BaseModel
from src.models.network.moaiparamdiffusion_network import MoaiMLPNet, MoaiParamDiffusion, VarianceScheduleMLP
#from src.models.hseencoder import HSEEncoder
#from hsemotion.facial_emotions import HSEmotionRecognizer

from loguru import logger
import numpy as np
import trimesh
#from src.models.moai import --

class MoaiParamDiffusionModel(BaseModel):
    def __init__(self, config=None, device=None):
        super(MoaiParamDiffusionModel, self).__init__(config, device, 'MoaiParamDiffusionModel')
        self.initialize()
        self.expencoder = self.cfg.model.expencoder

    def create_model(self, model_config):
        mapping_layers = model_config.mapping_layers
        pretrained_path = None
        if not model_config.use_pretrained:
            pretrained_path = model_config.arcface_pretrained_model
        print("freeze = {}".format(self.cfg.model.with_freeze), flush=True)
        self.arcface = Arcface(pretrained_path=pretrained_path, freeze=self.cfg.model.with_freeze).to(self.device)
        #self.hseencoder = HSEEncoder(outsize=512).to(self.device)
        self.net = MoaiMLPNet(config=self.cfg.net)
        self.var_sched = VarianceScheduleMLP(config=self.cfg.varsched)
        self.diffusion = MoaiParamDiffusion(net=self.net,var_sched=self.var_sched, device=self.device, tag=self.cfg.net.tag)

    def load_model(self):
        if self.cfg.train.resume:
            #model_path = os.path.join(self.cfg.output_dir, 'model_best.tar')
            model_path = os.path.join(self.cfg.train.resume_checkpoint)
            if os.path.exists(model_path):
                logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
                checkpoint = torch.load(model_path)
                if 'arcface' in checkpoint:
                    self.arcface.load_state_dict(checkpoint['arcface'])
                if 'hseencoder' in checkpoint:
                    print('hseencoder')
                    self.hseencoder.load_state_dict(checkpoint['hseencoder'])
                if 'net' in checkpoint:
                    self.net.load_state_dict(checkpoint['net'])
                if 'var_sched' in checkpoint:
                    self.var_sched.load_state_dict(checkpoint['var_sched'])
                if 'diffusion' in checkpoint:
                    self.diffusion.load_state_dict(checkpoint['diffusion'])
            else:
                logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')
                exit()

    def model_dict(self):
        if self.expencoder == 'arcface':
            return {
                'arcface': self.arcface.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }
        elif self.expencoder == 'hseencoder':
            return {
                'hseencoder': self.hseencoder.state_dict(),
                'net': self.net.state_dict(),
                'var_sched': self.var_sched.state_dict(),
                'diffusion': self.diffusion.state_dict(),
            }

    def parameters_to_optimize(self):
        if self.expencoder == 'arcface':
            return [
                {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        elif self.expencoder == 'hseencoder':
            return [
                {'params': self.hseencoder.parameters(), 'lr': self.cfg.train.hse_lr},
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]
        else:
            return [
                {'params': self.diffusion.parameters(), 'lr': self.cfg.train.diff_lr},
            ]

    def encode(self, images, arcface_imgs):
        codedict = {}
        if self.expencoder == 'arcface':
            codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        elif self.expencoder == 'hse':
            codedict['hse'] = torch.Tensor(self.hse.extract_multi_features((images.cpu().numpy()*255).astype(np.uint8))).to(self.device)
        elif self.expencoder == 'hseencoder':
            codedict['hseencoder'] = torch.Tensor(self.hseencoder((images.cpu().numpy()*255).astype(np.uint8))).to(self.device)

        codedict['images'] = images

        return codedict

    def decode(self, codedict, epoch=0, withexp=False, shapecode=None, expcode=None, rotcode=None):
        self.epoch = epoch

        pred_theta = None
        e_rand = None

        pred_lmk2d = None
        pred_lmk3d = None
        allcode = 0


        if self.expencoder == 'arcface':
            identity_code = codedict['arcface']
        elif self.expencoder == 'hse':
            identity_code = codedict['hse']
        elif self.expencoder == 'hseencoder':
            identity_code = codedict['hseencoder']

        if not self.testing:
            moai = codedict['moai']
            gt_mesh3d = codedict['mesh3d']
            gt_shapecode = moai['shape_params'].view(-1, moai['shape_params'].shape[-1])
            gt_shapecode = gt_shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            if self.with_exp:
                gt_expcode = moai['expression_params'].view(-1, moai['expression_params'].shape[-1])
                gt_expcode = gt_expcode.to(self.device)[:, :self.cfg.model.n_exp]
                gt_rotcode = moai['rotation_params'].view(-1, moai['rotation_params'].shape[-1])
                gt_rotcode = gt_rotcode.to(self.device)[:, :self.cfg.model.n_rot]
                gt_pupilcode = moai['pupil_param'].view(-1, moai['pupil_param'].shape[-1])
                gt_pupilcode = gt_pupilcode.to(self.device)[:, :self.cfg.model.n_pupil]

                if self.cfg.net.moai_dim == 502:
                    gt_params = torch.cat([gt_shapecode, gt_expcode, gt_rotcode, gt_pupilcode], dim=1)
                elif self.cfg.net.moai_dim == 489:
                    gt_params = torch.cat([gt_shapecode, gt_expcode], dim=1)
            else:
                gt_params = gt_shapecode
            pred_noise, gt_noise, pred_params = self.diffusion.decode(self.epoch, gt_params, identity_code)

            if self.cfg.net.moai_dim == 502:
                pred_shapecode = pred_params[:, :256]
                pred_expcode = pred_params[:, 256:489]
                pred_rotcode = pred_params[:, 489:501]
                pred_pupilcode = pred_params[:, 501:502]
                
            elif self.cfg.net.moai_dim == 489:
                pred_shapecode = pred_params[:, :256]
                pred_expcode = pred_params[:, 256:489]
            else:
                pred_shapecode = pred_params

            if epoch % 100:
                pred_moai = MoaiNumpy()
                pred_moai.identity = pred_shapeparam
                pred_moai.expression = pred_expparam
                pred_mesh3d = pred_moai.vertices

                trimesh.Trimesh(vertices = pred_mesh3d, process=False).export('train.obj')

        pred_mesh3d = None

        if self.testing or self.validation:
            pred_codes = self.diffusion.sample(num_points=self.cfg.net.moai_dim, context=identity_code, batch_size=identity_code.shape[0], moai=self.moai, sampling=self.cfg.model.sampling)
            if self.with_exp:
                if self.cfg.net.moai_dim == 502:
                    pred_shapecode = pred_codes[:, :256]
                    pred_expcode = pred_codes[:, 256:489]
                    pred_rotcode = pred_codes[:, 489:501]
                    pred_pupilcode = pred_codes[:, 501:502]
                    pred_params = torch.cat([pred_shapecode, pred_expcode, pred_rotcode, pred_pupilcode], dim=1)
                elif self.cfg.net.moai_dim == 489:
                    pred_shapecode = pred_codes[:, :256]
                    pred_expcode = pred_codes[:, 256:489]
                    pred_params = torch.cat([pred_shapecode, pred_expcode], dim=1)
            else:
                pred_params = pred_shapecode
       
        output = {
            'pred_noise': pred_noise,
            'gt_noise': gt_noise,
            'pred_params': pred_params,
            'gt_params': gt_params,
            'gt_mesh3d': gt_mesh3d,
            'pred_mesh3d': pred_mesh3d,
        }

        return output

    def compute_losses(self, decoder_output, losstype='mse'):
        losses = {}

        pred_noise = decoder_output['pred_noise']
        gt_noise = decoder_output['gt_noise']

        #pred_mesh3d = decoder_output['pred_mesh3d']
        #gt_mesh3d = decoder_output['gt_mesh3d'].detach()


        #if np.random.rand() > 0.95:
        #    trimesh.Trimesh(vertices=pred_mesh3d[0].clone().detach().cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('pred_mesh3d.ply')

        noise_diff = (pred_noise - gt_noise).abs()
        losses['noise_diff'] = torch.mean(noise_diff)*1e3

        return losses
