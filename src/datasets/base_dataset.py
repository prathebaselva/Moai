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
import re
from abc import ABC
from functools import reduce
from pathlib import Path
import cv2
import clip
import json

import loguru
import numpy as np
import torch
import trimesh
import scipy.io
from loguru import logger
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
#from src.model.moai import MoaiNumPy

class BaseDataset(Dataset, ABC):
    def __init__(self, name, config, device, isEval):
        self.n_images = config.n_images
        self.isEval = isEval
        self.n_train = np.Inf
        self.imagepaths = []
        self.lmk = config.lmk
        self.face_dict = {}
        self.name = name
        self.device = device
        self.min_max_K = 0
        self.cluster = False
        self.dataset_root = config.root
        self.total_images = 0
        self.config = config
        self.farlmodel, self.farlpreprocess = clip.load("ViT-B/16", device="cpu")
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list = os.path.join(self.dataset_root, 'arcface', self.name+'.npy')
        #print(image_list)
        logger.info(f'[{self.name}] Load cached file list: ' + image_list)
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        ####### MOAI NUMPY #########################
        #self.moai = MoaiNumpy().to(self.device)

        self.imagepaths = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')

        arcface_input = 'arcface_input'
        self.image_folder = arcface_input

        farl_state = torch.load(os.path.join(self.pretrained, "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"))
        self.farl_model.load_state_dict(farl_state["state_dict"], strict=False)

        self.set_smallest_numimages()

    def set_smallest_numimages(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][1])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][1]), self.imagepaths))
        loguru.logger.info(f'Dataset {self.name} with min num of images = {self.min_max_K} max num of images = {max_min_k} length = {len(self.face_dict)} total images = {self.total_images}')
        return self.min_max_K

    def compose_transforms(self, *args):
        self.transforms = transforms.Compose([t for t in args])

    def get_arcface_path(self, image_path):
        return re.sub('png|jpg', 'npy', str(image_path))

    def __len__(self):
        return len(self.imagepaths)
        #return self.total_images

    def __getitem__(self, index):
        #print(index)
        #print(self.imagepaths)
        actor = self.imagepaths[index]
        #print(actor)
        #print(self.face_dict[actor],flush=True)
    
        datas = self.face_dict[actor]
        #print("datas = ",datas)
        #print(len(datas))
        imagepaths = []
        metapaths = []
        ldmkpaths = []
        for data in datas:
            imagepaths.append(data[0])
            metapaths.append(data[1])
            ldmkpaths.append(data[2])
        


        #images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]
        images = [Path(self.dataset_root, self.name,  path) for path in imagepaths]
        metas = [Path(self.dataset_root, self.name, path) for path in metapaths]
        ldmks = [Path(self.dataset_root, self.name, path) for path in ldmkpaths]
        sample_list = np.array(np.random.choice(range(len(images)), size=self.n_images, replace=True))

        K = self.n_images
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        #params = np.load(os.path.join(self.dataset_root,self.name, params_path), allow_pickle=True)
        #shape_param = params['identity'] #geometry_identity_param
        #exp_param = params['expression'] #expression
        #eye_param =  
        #jaw_param = 
        #neck_param = 
        #trans_param = 
        # Each images share the shape moai parameter of the actor
        # Thus multiply the parameters by the number of images = K
        #moaiparams = {
        #    'shape_params': torch.cat(K * [shape_param], dim=0),
        #    'exp_params': torch.cat(K * [exp_param], dim=0),
            #'eye_params': torch.cat(K * [eye_param], dim=0),
            #'jaw_params': torch.cat(K * [jaw_param], dim=0),
            #'neck_params': torch.cat(K * [neck_param], dim=0),
            #'trans_params': torch.cat(K * [trans_param], dim=0),
        #}
        #exp = torch.ones(K)

        images_list = []
        arcface_list = []
        imagefarl_list = []
        mesh_gt_list = []
        shape_list = []
        expression_list = []
        rotation_list = []
        pupil_list = []


        for i in sample_list:
            image_path = images[i]
            meta_path = metas[i]
            ldmk_path = ldmks[i]
            #if self.occlusion:
            #    image_path = re.sub('png', 'jpg', str(image_path))
            image_name = str(image_path).split('/')[-1]
            imagebasepath = str(image_path)[:-len(image_name)]
            #print(imagebasepath, flush=True)
            image_name = image_name[:-4]
            if os.path.exists(image_path):
                imagefarl = sellf.farlpreprocess(Image.open(image_path))
                image = np.array(imread(image_path))
                image = image / 255.
                images_list.append(image)
                arcface_image = np.load(self.get_arcface_path(image_path), allow_pickle=True)
                arcface_list.append(torch.tensor(arcface_image))
                imagefarl_list.append(torch.tensor(imagefarl))
                ldmkjson = json.load(open(ldmk_path))
                mesh_3d = np.array(ldmkjson['all_verts_3d'])/100
                mesh_gt_list.append(mesh_3d)

                metajson = json.load(open(meta_path))
                identity = metajson['primary_face']
                shape = np.array(identity['identity']['geometry_identity_params'])
                expression = np.array(list(identity['expression']['blendshape_values'].values()))
                rotation = np.hstack(list(identity['expression']['bone_rotations'].values()))
                pupil = [identity['expression']['pupil_size']]
                #print(len(rotation))
                #exit()

                shape_list.append(shape)
                expression_list.append(expression)
                rotation_list.append(rotation)
                pupil_list.append(pupil)

        images_array = torch.from_numpy(np.array(images_list)).float()
        arcface_array = torch.stack(arcface_list).float()
        imagefarl_array = torch.stack(imagefarl_list).float()
        shape_array = torch.from_numpy(np.array(shape_list)).float()
        expression_array = torch.from_numpy(np.array(expression_list)).float()
        rotation_array = torch.from_numpy(np.array(rotation_list)).float()
        pupil_array = torch.from_numpy(np.array(pupil_list)).float()
        mesh3d_array = torch.from_numpy(np.array(mesh_gt_list))
        moaiparams = {
                'shape_params': shape_array,
                'expression_params': expression_array,
                'rotation_params': rotation_array,
                'pupil_param': pupil_array}

        return {
            'image': images_array,
            'arcface': arcface_array,
            'imagefarl': imagefarl_array,
            'imagename': actor,
            'dataset': self.name,
            'moai': moaiparams,
            'mesh3d': mesh3d_array,
        }
