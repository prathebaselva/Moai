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

import loguru
import numpy as np
import torch
import trimesh
import scipy.io
from loguru import logger
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
from src.model.moai import MoaiNumPy

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
    
        self.initialize()

    def initialize(self):
        logger.info(f'[{self.name}] Initialization')
        image_list = os.path.join(self.dataset_root, 'arcface', self.name+'.npy')
        #print(image_list)
        logger.info(f'[{self.name}] Load cached file list: ' + image_list)
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        ####### MOAI NUMPY #########################
        self.moai = MoaiNumpy().to(self.device)

        self.imagepaths = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')

        arcface_input = 'arcface_input'
        self.image_folder = arcface_input

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
        actor = self.imagepaths[index]
        params_path, images = self.face_dict[actor]
        images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]
        sample_list = np.array(np.random.choice(range(len(images)), size=self.n_images, replace=True))

        K = self.n_images
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        params = np.load(os.path.join(self.dataset_root,self.name, params_path), allow_pickle=True)
        shape_param = params['identity'] #geometry_identity_param
        exp_param = params['expression'] #expression
        eye_param =  
        jaw_param = 
        neck_param = 
        trans_param = 
        # Each images share the shape moai parameter of the actor
        # Thus multiply the parameters by the number of images = K
        moaiparams = {
            'shape_params': torch.cat(K * [shape_param], dim=0),
            'exp_params': torch.cat(K * [exp_param], dim=0),
            'eye_params': torch.cat(K * [eye_param], dim=0),
            'jaw_params': torch.cat(K * [jaw_param], dim=0),
            'neck_params': torch.cat(K * [neck_param], dim=0),
            'trans_params': torch.cat(K * [trans_param], dim=0),
        }
        exp = torch.ones(K)

        images_list = []
        arcface_list = []

        for i in sample_list:
            image_path = images[i]
            #if self.occlusion:
            #    image_path = re.sub('png', 'jpg', str(image_path))
            image_name = str(image_path).split('/')[-1]
            imagebasepath = str(image_path)[:-len(image_name)]
            #print(imagebasepath, flush=True)
            image_name = image_name[:-4]
            if os.path.exists(image_path):
                image = np.array(imread(image_path))
                #print(image.shape)
                #cv2.imwrite("test.jpg", image[128:224,0:224,:])
                image = image / 255.
                #image = image.transpose(2, 0, 1)
                #image = image / 255.
                #image = image.transpose(2, 0, 1)
                #print(image.shape)
                arcface_image = np.load(self.get_arcface_path(image_path), allow_pickle=True)
                #print(arcface_image, flush=True)
                pose_path = os.path.join(imagebasepath, str(image_name)+'_pose.npy')
                #print(pose_path)

                images_list.append(image)
                arcface_list.append(torch.tensor(arcface_image))

        #print(arcface_list, flush=True)

        images_array = torch.from_numpy(np.array(images_list)).float()
        arcface_array = torch.stack(arcface_list).float()


        return {
            'image': images_array,
            'arcface': arcface_array,
            'imagename': actor,
            'dataset': self.name,
            'moai': moaiparams,
            'exp': exp
        }
