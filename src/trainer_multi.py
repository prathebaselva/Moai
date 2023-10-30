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
import random
import sys
import math
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.datasets as datasets
from src.configs.config import cfg
from src.utils import util
import trimesh
import shutil
#from src.utils.MICA.renderer import MeshShapeRenderer

sys.path.append("./src")
#from validator import Validator


def print_info(rank):
    props = torch.cuda.get_device_properties(rank)

    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer(object):
    def __init__(self, model, pretrainedmodel=None, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
        self.withval = self.cfg.model.with_val
        print(self.cfg, flush=True)
        self.epoch = 0
        self.global_step = 0

        print(self.withval, flush=True)

        # autoencoder model
        self.model = model.to(self.device)
        if self.withval:
            self.validator = Validator(self)
        self.configure_optimizers()
        if self.cfg.train.resume:
            self.load_checkpoint()

        #if pretrainedmodel is not None:
        #    self.pretrainedmodel = pretrainedmodel
        #    self.pretrainedmodel.load_model()

        # reset optimizer if loaded from pretrained model
        if self.cfg.train.reset_optimizer:
            self.configure_optimizers()  # reset optimizer
            logger.info(f"[TRAINER] Optimizer was reset")

        if self.cfg.train.write_summary and self.device == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

        print_info(device)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            #lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
            params=self.model.parameters_to_optimize(),
            amsgrad=False)

#        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                         self.optimizer,
#                         factor=0.99,
#                         patience=1,
#                         mode='min',
#                         threshold=1e-4,
#                         eps=0,
#                         min_lr=0)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                         self.optimizer,
                         factor=0.99,
                         patience=2,
                         mode='min',
                         threshold=1e-4,
                         eps=0,
                         min_lr=0)

    def load_checkpoint(self):
        self.epoch = 0
        self.global_step = 0
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        model_path = os.path.join(self.cfg.checkpoint_dir, 'model_best.tar')

        if os.path.exists(self.cfg.train.resume_checkpoint):
            model_path = os.path.join(self.cfg.train.resume_checkpoint)
        #if os.path.exists(self.cfg.pretrained_model_path):
        #    model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f"[TRAINER] Resume training from {model_path}")
            logger.info(f"[TRAINER] Start from step {self.global_step}")
            logger.info(f"[TRAINER] Start from epoch {self.epoch}")
        else:
            logger.info('[TRAINER] Model path not found, start training from scratch')

    def save_checkpoint(self, filename):
        if self.device == 0:
            model_dict = self.model.model_dict()

            model_dict['optimizer'] = self.optimizer.state_dict()
            model_dict['scheduler'] = self.scheduler.state_dict()
            if self.withval:
                model_dict['validator'] = self.validator.state_dict()
            model_dict['epoch'] = self.epoch
            model_dict['global_step'] = self.global_step
            model_dict['batch_size'] = self.batch_size

            torch.save(model_dict, filename)

    def training_step(self, batch):
        self.model.train()
        self.model.validation = False

        #Original images
        images = batch['image'].to(self.device)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        # Moai parameters
        moai = batch['moai']
        mesh3d = batch['mesh3d']
        #exp = batch['exp']

        # arcface images
        arcface = batch['arcface']
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device)

        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }
       
        if self.cfg.net.context_dim == 0:
            encoder_output = {}
            encoder_output['arcface'] = None
            encoder_output['images'] = None
        else:
            encoder_output = self.model.encode(images, arcface)
        encoder_output['moai'] = moai
        encoder_output['mesh3d'] = mesh3d 
        #encoder_output['exp'] = exp


        #logger.info(exp)
        decoder_output = self.model.decode(encoder_output, self.epoch)
        losses = self.model.compute_losses(decoder_output, self.cfg.net.losstype)

        all_loss = 0.
        vertex_loss = 0.
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        #rank_loss = losses['rank_loss']
        #losses['vertex_loss'] = all_loss - rank_loss
        losses['all_loss'] = all_loss

        opdict = \
            {
                'images': images,
            }

        if 'pred_mesh3d' in decoder_output:
            opdict['pred_mesh3d']= decoder_output['pred_mesh3d']

        return losses, opdict

    def validation_step(self):
        return self.validator.run()

    def evaluation_step(self):
        pass

    def prepare_data(self):
        #generator = torch.Generator()
        #generator.manual_seed(self.device)

        self.train_dataset, total_images = datasets.build_train(self.cfg.dataset, self.device)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=False)
            #worker_init_fn=seed_worker,
            #generator=generator)
        self.train_iter = iter(self.train_dataloader)

        self.val_dataset, val_total_images = datasets.build_val(self.cfg.dataset, self.device)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False)
            #worker_init_fn=seed_worker,
            #generator=generator)
        self.val_iter = iter(self.val_dataloader)
        logger.info(f'[TRAINER] Training dataset is ready with {len(self.train_dataset)} actors and {total_images} images.')
        logger.info(f'[TRAINER] Validation dataset is ready with {len(self.val_dataset)} actors and {val_total_images} images.')

    def run(self):
        self.prepare_data()
        #iters_every_epoch = max(len(self.train_dataset), int(len(self.train_dataset) / self.batch_size))
        iters_every_epoch = math.ceil(len(self.train_dataset)/ self.batch_size)
        #iters_every_epoch = int(len(self.train_dataset)/ self.batch_size)
        #print(iters_every_epoch)
        #max_epochs = int(self.cfg.train.max_steps / iters_every_epoch)
        start_epoch = self.epoch
        max_epochs = self.cfg.train.max_epochs
        self.train_best_loss = np.Inf
        self.val_best_loss = np.Inf
        for epoch in range(start_epoch, max_epochs):
            epochloss = 0
            epochcount = 0
            epochtrainloss = 0
            epochtraincount = 0
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{max_epochs}]"):
                if self.global_step > self.cfg.train.max_steps:
                    break
                try:
                    #print("in try", flush=True)
                    batch = next(self.train_iter)
                except Exception as e:
                    #logger.info(f'in exception train')
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                visualizeTraining = self.global_step % self.cfg.train.vis_steps == 0

                self.optimizer.zero_grad()
                losses, opdict = self.training_step(batch)
                all_loss = losses['all_loss']
                batch_size = batch['mesh3d'].shape[0]
                #if not self.withval:
                epochtrainloss += (batch_size * all_loss.item())
                epochtraincount += batch_size

                all_loss.backward()
                self.optimizer.step()
                

                if self.global_step % self.cfg.train.log_steps == 0 and self.device == 0:
                    loss_info = f"\n" \
                                f"  Epoch: {epoch}\n" \
                                f"  Step: {self.global_step}\n" \
                                f"  Iter: {step}/{iters_every_epoch}\n" \
                                f"  Arcface LR: {self.optimizer.param_groups[0]['lr']}\n" \
                                f"  Diff LR: {self.optimizer.param_groups[1]['lr']}\n" \
                                f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                                #f"  resnet LR: {self.optimizer.param_groups[2]['lr']}\n" \
                    for k, v in losses.items():
                        loss_info = loss_info + f'  {k}: {v:.4f}\n'
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/' + k, v, global_step=self.global_step)
                    logger.info(loss_info)

                #print("output dir = ", self.cfg.output_dir, flush=True)

                if visualizeTraining and self.device == 0:
                    if 'pred_mesh3d' in opdict and opdict['pred_mesh3d'] is not None:
                        pred_mesh3d = trimesh.points.PointCloud(opdict['pred_mesh3d'][0].clone().detach().cpu().numpy())
                        pred_mesh3d.export(os.path.join(self.cfg.output_dir,  str(self.global_step)+'_'+str(self.cfg.net.tag)+ '_mesh3d.ply')) 

                if (self.global_step > 1000) and self.global_step % self.cfg.train.checkpoint_epochs_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_'+str(self.cfg.net.tag)+'.tar'))


                self.global_step += 1

            if self.withval:# and ((self.global_step % self.cfg.train.val_steps) == 0):
                logger.info("validation")
                val_loss, val_batch_size = self.validation_step()
                logger.info(val_loss)
                epochloss += (val_batch_size * val_loss)
                epochcount += val_batch_size
                epochloss = epochloss / epochcount
                if epoch > 10 and epochloss < self.val_best_loss:
                #if epochloss < self.val_best_loss:
                    os.makedirs(os.path.join(self.cfg.output_dir, 'best_models'), exist_ok=True)
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'best_models', 'model_val_'+str(self.cfg.net.tag)+'.tar'))
                    self.val_best_loss = epochloss

            epochtrainloss = epochtrainloss / epochtraincount

            if epochtrainloss < self.train_best_loss:
                logger.info(f'best train {epoch}')
                self.train_best_loss = epochtrainloss
                os.makedirs(os.path.join(self.cfg.output_dir, 'best_models'), exist_ok=True)
                self.save_checkpoint(os.path.join(self.cfg.output_dir, 'best_models', 'model_train_'+str(self.cfg.net.tag)+'_best.tar'))
                #if epoch > 50:
                #    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'best_models', 'model_train_'+str(self.cfg.net.tag)+'_'+str(epoch)+'_best.tar'))


           # if self.withval:
           #     self.scheduler.step(epochloss)
           # else:
            self.scheduler.step(epochtrainloss)
            self.epoch += 1

        self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))
        logger.info(f'[TRAINER] Fitting has ended!')
