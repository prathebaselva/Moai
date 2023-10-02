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
from glob import glob

import cv2
import numpy as np
import torch
import re
import torch.distributed as dist
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
#from pytorch3d.io import save_ply
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from src.configs.config import cfg
from src.utils import util
from src.models.flame import FLAME
import trimesh



from src.models.deca import DECA
from src.configs.deca_config import cfg as deca_cfg
from src.utils.utils import batch_orth_proj

sys.path.append("./src")
input_mean = 127.5
input_std = 127.5

class Tester(object):
    def __init__(self, model_model, config=None, device=None, args=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.render_mesh = False
        #self.embeddings = {'emb':[], 'imgname':[]}
        #self.nowimages = self.cfg.test.now_images
        #self.stirlinghqimages = self.cfg.test.stirling_hq_images
        #self.stirlinglqimages = self.cfg.test.stirling_lq_images
        #self.resnethalfimg = self.cfg.dataset.resnethalfimg
        self.args = args

        # deca model
        self.model = model_model.to(self.device)
        self.model.testing = True
        #flameModel = FLAME(self.cfg.model).to(self.device)
        #self.faces = flameModel.faces_tensor.cpu()
        #self.faces = model_model.flameGenerativeModel.generator.faces_tensor.cpu()

        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')

    def load_checkpoint(self, model_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}

        checkpoint = torch.load(model_path, map_location)
        if 'arcface' in checkpoint:
            print("arcface")
            self.model.arcface.load_state_dict(checkpoint['arcface'])
        if 'hseencoder' in checkpoint:
            print("hseencoder")
            self.model.hseencoder.load_state_dict(checkpoint['hseencoder'])
        if 'resnet' in checkpoint:
            print("resnet")
            self.model.resnet.load_state_dict(checkpoint['resnet'])
        if 'ranknet' in checkpoint:
            print("ranknet")
            self.model.ranknet.load_state_dict(checkpoint['ranknet'])
        if 'net' in checkpoint:
            print("net")
            self.model.net.load_state_dict(checkpoint['net'])
        if 'fnet' in checkpoint:
            print("fnet")
            self.model.fnet.load_state_dict(checkpoint['fnet'])
        if 'var_sched' in checkpoint:
            print("var_sched")
            self.model.var_sched.load_state_dict(checkpoint['var_sched'])
        if 'diffusion' in checkpoint:
            print("diffusion")
            self.model.diffusion.load_state_dict(checkpoint['diffusion'])

        print("done", flush=True)
        logger.info(f"[TESTER] Resume from {model_path}")

    def load_model_dict(self, model_dict):
        dist.barrier()

        self.model.canonicalModel.load_state_dict(model_dict['canonicalModel'])
        self.model.arcface.load_state_dict(model_dict['arcface'])

    def process_image(self, img, app):
        images = []
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
        if bboxes.shape[0] != 1:
            logger.error('Face not detected!')
            return images
        i = 0
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        aimg = face_align.norm_crop(img, landmark=face.kps)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)

        images.append(torch.tensor(blob[0])[None])

        return images

    def process_folder(self, folder, app):
        images = []
        image_names = []
        arcface = []
        count = 0
        files_actor = sorted(sorted(os.listdir(folder)))
        for file in files_actor:
            if file.startswith('._'):
                continue
            image_path = folder + '/' + file
            logger.info(image_path)
            image_names.append(image_path)
            count += 1

            ### NOW CROPPING
            scale = 1.6
            # scale = np.random.rand() * (1.8 - 1.1) + 1.1
            bbx_path = image_path.replace('.jpg', '.npy').replace('iphone_pictures', 'detected_face')
            bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            left = bbx_data['left']
            right = bbx_data['right']
            top = bbx_data['top']
            bottom = bbx_data['bottom']

            image = imread(image_path)[:, :, :3]

            h, w, _ = image.shape
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * scale)

            crop_size = 224
            # crop image
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
            DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.
            dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))

            arcface += self.process_image(cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR), app)

            dst_image = dst_image.transpose(2, 0, 1)
            images.append(torch.tensor(dst_image)[None])

        images = torch.cat(images, dim=0).float()
        arcface = torch.cat(arcface, dim=0).float()
        print("images = ", count)

        return images, arcface, image_names

    def get_name(self, best_model, id):
        if '_' in best_model:
            name = id if id is not None else best_model.split('_')[-1][0:-4]
        else:
            name = id if id is not None else best_model.split('/')[-1][0:-4]
        return name

    def load_cfg(self, cfg, best_model):
        self.load_checkpoint(best_model)

    def load_cfg_bkup(self, cfg, best_model):
        self.model.var_sched.num_steps = cfg.varsched.num_steps
        self.model.var_sched.beta_1 = cfg.varsched.beta_1
        self.model.var_sched.beta_T = cfg.varsched.beta_T
        self.model.net.moai_dim = cfg.net.moai_dim
        self.model.net.arch = cfg.net.arch
        self.model.expencoder = cfg.model.expencoder
        self.model.with_exp = cfg.model.with_exp
        self.model.sampling = cfg.model.sampling

        self.load_checkpoint(best_model)
        self.model.var_sched.num_steps = cfg.varsched.num_steps
        self.model.var_sched.beta_1 = cfg.varsched.beta_1
        self.model.var_sched.beta_T = cfg.varsched.beta_T
        self.model.net.flame_dim = cfg.net.flame_dim
        self.model.net.arch = cfg.net.arch
        self.model.expencoder = cfg.model.expencoder
        self.model.with_exp = cfg.model.with_exp
        self.model.sampling = cfg.model.sampling


    def test(self, imgfolder, arcfacefolder, inputfilelist, numface):
        self.model.eval()

        allimages = []
        image_names = []
        logger.info(f"[TESTER] test has begun!")
        allfiles = open(inputfilelist)

        for line in allfiles:
            line = line.strip()
            imgname = line.split('/')[-1][:-4]

            arcface = torch.tensor(np.load(os.path.join(arcfacefolder, imgname+'.npy'))).float().to('cuda')
            origimage = imread(os.path.join(imgfolder, line))
            origimage = origimage / 255.
            origimage = origimage.transpose(2, 0, 1)

            with torch.no_grad():
                if numface > 1:
                    arcface = arcface.tile(numface,1,1,1)
                    img_tensor = torch.Tensor(origimage).tile(numface,1,1,1).to('cuda')
                    codedict = self.model.encode(img_tensor, arcface)
                else:
                    codedict = self.model.encode(torch.Tensor(origimage).unsqueeze(0).to('cuda'), arcface.unsqueeze(0))
                opdict = self.model.decode(codedict, 0)# cam=deca_codedict['cam'])

            os.makedirs(os.path.join(self.cfg.output_dir, f'flamesample'), exist_ok=True)
            flame_dst_folder = os.path.join(self.cfg.output_dir, f'flamesample')
            os.makedirs(flame_dst_folder, exist_ok=True)

            pred_moai = MoaiNumpy()

            for num in range(numface):
                pred_moai.identity = opdict['pred_params'][num][:256]
                pred_moai.expression = opdict['pred_params'][num][256:489]
                savepath = os.path.join(self.cfg.output_dir, f'sample',  imgname+'.ply')
                trimesh.Trimesh(vertices=np.array(pred_moai.vertices),  process=False).export(savepath1)

