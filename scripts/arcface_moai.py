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


import argparse
import os
import re
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
#import trimesh
from insightface.app import FaceAnalysis, MaskRenderer
from insightface.app.common import Face
from insightface.utils import face_align
import face_alignment
from loguru import logger
from skimage.io import imread
#import imageio
from scipy.io import loadmat
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
import math

import sys
sys.path.append('../')

from SynergyNet.synergy3DMM import SynergyNet
synmodel = SynergyNet()
facemodel = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


#from utils import util
input_mean = 127.5
input_std = 127.5

def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False

def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))

def draw_bbox(img, face, landmarks, outfile):
    box = face.bbox.astype(np.int)
    color = (0,0,255)
    cv2.rectangle(img, (box[0], box[1], box[2], box[3]), color, 2)
    cv2.imwrite(outfile+'_bbox.jpg', img)

def draw_landmark(img, face, landmarks, outfile):
    color = (0,0,255)
    for i in range(len(landmarks)):
        cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), 1, color, 2)
    cv2.imwrite(outfile+'_lmk_insight.jpg', img)

def get_center(bboxes, img):
    img_center = img.shape[0] // 2, img.shape[1] // 2
    size = bboxes.shape[0]
    distance = np.Inf
    j = 0
    for i in range(size):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx = abs(x2 - x1) / 2.0
        dy = abs(y2 - y1) / 2.0
        current = dist((x1 + dx, y1 + dy), img_center)
        if current < distance:
            distance = current
            j = i

    return j

def get_arcface_input(face,img, filepath, image_size=112):
    #aimg = face_align.norm_crop(img, landmark=face.kps)
    #M = face_align.estimate_norm(face.kps)
    #aimg = cv2.warpAffine(img, M, (image_size, image_size),0.0)

    aimg = face_align.norm_crop(img, landmark=face.kps, image_size=image_size)
    blob = cv2.dnn.blobFromImages([aimg], 1.0/input_std, (image_size, image_size), (input_mean, input_mean, input_mean), swapRB=True)
    aimg_rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filepath + '_aimg.jpg', aimg_rgb)
    np.save(filepath, blob[0])
    return blob[0], aimg_rgb


def processwithposeandlmk(img, app, path, name, aflwkpt=None):
    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    if bboxes.shape[0] == 0:
        print("no bbox found")
        print(name, flush=True)
        return 0
    i = get_center(bboxes, img)
    bbox = bboxes[i, 0:4]
    det_score = bboxes[i, 4]
    kps = None
    if kpss is not None:
        kps = kpss[i]

    filepath = str(Path(path, name))
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    blob, aimg = get_arcface_input(face, img, filepath)

    scale = 1.6
    #print(bboxes[i,0:4], flush=True)
    left, top, right, bottom = bboxes[i, 0:4]
    h, w, _ = img.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    crop_size = 224

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    img = img / 255.
    dst_image = warp(img, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image  = cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath + '.jpg', dst_image)

def processinstance(args, app, image_size=224):
    dst = Path(args.o)
    dst.mkdir(parents=True, exist_ok=True)
    processes = []
    instance = args.r
    inputdir = args.i
    outputdir = args.o
    #original_im = cv2.resize(skimage.data.astronaut(), (256,256))
   ################# LYHM #########################################
    count = 0
    if instance == 'MOAI':
        for im in sorted(glob(inputdir+'/*')):
            actorname = Path(im).stem[9:9+7]
            #print(actorname)
            #print(im)
            #if int(actorname) <= 1214:
            #    continue
            image_name = im.split('/')[-1]
            if image_name.startswith('.'):
                continue
            image_name = image_name.split('.')[0].split('_')[-1]
            #print(image_name)
            #exit()
            #actorname = im.split('/')[-2]
            im1 = cv2.imread(im)
            os.makedirs(os.path.join(outputdir, actorname), exist_ok =True)
            path = os.path.join(outputdir, actorname)
            if not processwithposeandlmk(im1, app, path, image_name):
                continue

def main( args):
    device = 'cuda:0'
    #Path(args.o).mkdir(exist_ok=True, parents=True)
    app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'landmark_3d_68'])
    app.prepare(ctx_id=0, det_size=(224, 224))

    logger.info(f'Processing has started...')
    paths = processinstance(args, app)
    #logger.info(f'Processing finished. Results has been saved in {args.o}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOAI - Towards Metrical Reconstruction of Human Faces')
    parser.add_argument('-i', default='demo/input', type=str, help='Input folder with images')
    parser.add_argument('-r', default='MOAI', type=str, help='Input instance')
    parser.add_argument('-o', default='demo/arcface', type=str, help='Processed images for MOAI input')

    args = parser.parse_args()
    main(args)
