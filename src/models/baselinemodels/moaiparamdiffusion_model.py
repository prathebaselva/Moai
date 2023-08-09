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
from src.models.diffgenerator import Generator
from src.models.base_model import BaseModel
from src.models.network.moaiparamdiffusion_network import MoaiMLPNet, MoaiParamDiffusion, VarianceScheduleMLP
from src.models.hseencoder import HSEEncoder
from hsemotion.facial_emotions import HSEmotionRecognizer

from loguru import logger
import numpy as np
import trimesh
from src.models.moai import --

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
        self.hseencoder = HSEEncoder(outsize=512).to(self.device)
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

    def decode(self, codedict, epoch=0, withpose=False, withexp=False,shapecode=None, expcode=None, rotcode=None):
        self.epoch = epoch

        moai_verts_shape = None
        pred_theta = None
        e_rand = None
        sample_canonical_vertices = None
        pred_lmk2d = None
        pred_lmk3d = None
        jawcode = torch.tensor(0).float().to(self.device)
        pred_jaw_code = torch.tensor(0).float().to(self.device)
        allcode = 0
        pred_jawparam = torch.tensor(0).float().to(self.device)
        moaiparam_x0 = torch.tensor(0).float().to(self.device)
        predx0_canonical_vertices = None
        mean_xt = None
        predx0_moaiparam = None

        if self.expencoder == 'arcface':
            identity_code = codedict['arcface']
        elif self.expencoder == 'hse':
            identity_code = codedict['hse']
        elif self.expencoder == 'hseencoder':
            identity_code = codedict['hseencoder']

        #identity_code = codedict['arcface']
        #logger.info(f'in decode')
        if not self.testing:
            moai = codedict['moai']
            #print(moai['shape_params'], flush=True)
            #print(moai['shape_params'].shape, flush=True)
            shapecode = moai['shape_params'].view(-1, moai['shape_params'].shape[-1])
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            if self.with_exp:
                expcode = moai['exp_params'].view(-1, moai['exp_params'].shape[-1])
                expcode = expcode.to(self.device)[:, :self.cfg.model.n_shape]
                with torch.no_grad():
                    moai_verts_shape, lmk2d, lmk3d = self.moai(shape_params=shapecode, expression_params=expcode)
            else:
                with torch.no_grad():
                    moai_verts_shape, lmk2d, lmk3d = self.moai(shape_params=shapecode)

        if not self.testing:
            if self.with_exp:
                if self.cfg.net.moai_dim == 115:
                    allcode = torch.cat([expcode, jawcode, eyecode, neckcode, rotcode], dim=1)
                elif self.cfg.net.moai_dim == 103:
                    allcode = torch.cat([expcode, jawcode], dim=1)
                elif self.cfg.net.moai_dim == 100:
                    allcode = expcode
                elif self.cfg.net.moai_dim == 3:
                    allcode = jawcode
                pred_theta, e_rand, pred_moaiparam, predx0_moaiparam, mean_xt = self.diffusion.decode(self.epoch, allcode, identity_code, self.moai)
            else:
                allcode = shapecode
                pred_theta, e_rand, pred_moaiparam, predx0_moaiparam, mean_xt = self.diffusion.decode(self.epoch, shapecode, identity_code, self.moai)

            if self.cfg.net.moai_dim == 115:
                pred_expparam = pred_moaiparam[:, :100]
                pred_jawparam = pred_moaiparam[:, 100:103]
                pred_eyeparam = pred_moaiparam[:, 103:109]
                pred_neckparam = pred_moaiparam[:, 109:112]
                pred_rotparam = pred_moaiparam[:, 112:115]
                pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=pred_expparam.float(), jaw_params=pred_jawparam.float(), 
                        neck_pose_params=pred_neckparam.float(), eye_pose_params=pred_eyeparam.float(), rot_params=pred_rotparam.float())
            elif self.cfg.net.moai_dim == 103:
                pred_expparam = pred_moaiparam[:, :100]
                pred_jawparam = pred_moaiparam[:, 100:103]
                predx0_expparam = predx0_moaiparam[:, :100]
                predx0_jawparam = predx0_moaiparam[:, 100:103]
                pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=pred_expparam.float(), jaw_params=pred_jawparam.float()) 
                predx0_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=predx0_expparam.float(), jaw_params=predx0_jawparam.float()) 
            elif self.cfg.net.moai_dim == 100:
                pred_expparam = pred_moaiparam[:, :100]
                predx0_expparam = predx0_moaiparam[:, :100]
                pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=pred_expparam.float(), jaw_params=jawcode) 
                predx0_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=predx0_expparam.float(), jaw_params=jawcode) 
            elif self.cfg.net.moai_dim == 3:
                pred_jawparam = pred_moaiparam
                predx0_jawparam = predx0_moaiparam
                pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=expcode, jaw_params=pred_jawparam.float())
                predx0_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=expcode, jaw_params=predx0_jawparam.float()) 
                #predx0_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=expcode, jaw_params=pred_theta.float()) 
            elif self.cfg.net.moai_dim == 300:
                shapecode = pred_moaiparam
                shapecodex0 = predx0_moaiparam
                pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode) 
                predx0_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecodex0) 

        if self.testing or self.validation:
            #logger.info(f'in testing')
            if self.with_exp:
               # print("shapeparam = ", shapecode)
                pred_moaiparam = self.diffusion.sample(num_points=self.cfg.net.moai_dim, context=identity_code, batch_size=identity_code.shape[0], moai=self.moai, sampling=self.cfg.model.sampling, shapeparam=shapecode)
                if self.cfg.net.moai_dim == 115:
                    pred_eyeparam = pred_moaiparam[:, 103:109]
                    pred_neckparam = pred_moaiparam[:, 109:112]
                    pred_rotparam = pred_moaiparam[:, 112:115]
                elif self.cfg.net.moai_dim == 103:
                    pred_expparam = pred_moaiparam[:,:100]
                    pred_jawparam = pred_moaiparam[:, 100:103]
                    #pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=pred_expparam.float(), jaw_params=pred_jawparam.float(), rot_params=rotcode) 
                    pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=pred_expparam.float(), rot_params=rotcode) 
                elif self.cfg.net.moai_dim == 100:
                    pred_expparam = pred_moaiparam
                    pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=pred_expparam.float()) 
                elif self.cfg.net.moai_dim == 3:
                    pred_jawparam = pred_moaiparam
                    pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=shapecode, expression_params=expcode, jaw_params=pred_jawparam.float()) 
            else:
                pred_moaiparam = self.diffusion.sample(num_points=300, context=identity_code, batch_size=identity_code.shape[0], moai=self.moai, sampling=self.cfg.model.sampling)
                pred_canonical_vertices, pred_lmk2d, pred_lmk3d = self.moai(shape_params=pred_moaiparam.float())
            #pred_moaiparam = pred_moaiparam.squeeze().unsqueeze(0)
       
        output = {
            'moai_verts_shape': moai_verts_shape,
            'moai_shape_code': allcode,
            'moai_jaw_code': jawcode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'predx0_canonical_shape_vertices': predx0_canonical_vertices,
            'pred_theta': pred_theta,
            'e_rand': e_rand,
            'pred_shape_code': pred_moaiparam,
            'pred_jaw_code': pred_jawparam,
            'predx0_shape_code':predx0_moaiparam,
            'mean_xt': mean_xt,
            'faceid': identity_code,
            'lmk2d': pred_lmk2d,
            'lmk3d': pred_lmk3d,
        }

        return output

    def compute_losses(self, decoder_output, losstype='mse'):
        losses = {}

        pred_theta = decoder_output['pred_theta']
        e_rand = decoder_output['e_rand']
        mean_xt = decoder_output['mean_xt']
        pred_moai = decoder_output['pred_shape_code']
        gt_moai = decoder_output['moai_shape_code'].detach()
        predx0_moai = decoder_output['predx0_shape_code']
        predx0_verts = decoder_output['predx0_canonical_shape_vertices']

        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['moai_verts_shape'].detach()

        if self.validation:
            #logger.info(f'Validation loss')
            pred_shape_diff = (pred_verts - gt_verts).abs()
            losses['pred_shape_diff'] = torch.mean(pred_shape_diff)
            trimesh.Trimesh(vertices=pred_verts[0].clone().detach().cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('pred_val.ply')
            trimesh.Trimesh(vertices=gt_verts[0].cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('gt_val.ply')
            return losses

        if np.random.rand() > 0.95:
            trimesh.Trimesh(vertices=predx0_verts[0].clone().detach().cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('pred_arc_moaidiff.ply')
            #trimesh.Trimesh(vertices=pred_verts[0].clone().detach().cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('pred_hse_archjaw2_1e3_meanx0.ply')
            trimesh.Trimesh(vertices=gt_verts[0].cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('gt_arc_moaidiff.ply')
            #trimesh.Trimesh(vertices=predx0_verts[0].clone().detach().cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('predx0_occ_archv3_1e3.ply')
            #trimesh.Trimesh(vertices=pred_verts[0].clone().detach().cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('predx0_occ_archv3_1e3.ply')
            #trimesh.Trimesh(vertices=gt_verts[0].cpu().numpy(), faces=np.asarray(self.moai.faces_tensor.cpu()), process=False).export('gt_occ_archv3_1e3.ply')

        pred_moai_param_diff = (pred_moai - gt_moai).abs()
        pred_shape_diff = (pred_verts - gt_verts).abs()
        #predx0_shape_diff = (predx0_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask
        if losstype == 'mse':
            #mse_loss = F.mse_loss(pred_verts, gt_verts, reduction='mean')
            e_loss = F.mse_loss(pred_theta, e_rand, reduction='mean')
            #losses['pred_verts_shape_canonical_diff'] = mse_loss * 1e-4
            losses['pred_theta_diff'] = e_loss*100.0
        else:

            #predx0_shapecode_diff = (predx0_moai - gt_moai).abs()
            #losses['predx0_shapecode_diff'] = torch.mean(predx0_shapecode_diff)*10

            #pred_mean_diff = (pred_theta - mean_xt).abs()
            #losses['pred_mean_diff'] = torch.mean(pred_mean_diff)

            #if self.cfg.net.moai_dim == 103:
            #    pred_jawcodex0_diff = (predx0_moai[:,100:] - gt_moai[:,100:]).abs()
            #    losses['pred_jawcodex0_diff'] = torch.mean(pred_jawcodex0_diff)*100
            #pred_jawcode_diff = F.mse_loss(pred_jaw, gt_jaw, reduction='mean')
            #losses['pred_jawcode_diff'] = torch.mean(pred_jawcode_diff)*0
            losses['pred_shape_diff'] = torch.mean(pred_shape_diff)*1e3
            #losses['pred_moai_param_diff'] = torch.mean(pred_moai_param_diff)*1e3
            #e_loss = F.l1_loss(pred_theta, e_rand, reduction='mean')
            #e_loss = (pred_theta - e_rand).abs()
            #losses['pred_theta_diff'] = torch.mean(e_loss)

        return losses
