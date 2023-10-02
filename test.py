# inspired by MICA code

import os
import sys
import torch
import re

from src.tester import Tester

from src.models.baselinemodels.moaiparamdiffusion_model import MoaiParamDiffusionModel

if __name__ == '__main__':
    from src.configs.config import parse_args
    cfg, args = parse_args()

    deviceid = torch.cuda.current_device()
    model = MoaiParamDiffusionModel(cfg, 'cuda')
    tester = Tester(model, cfg, deviceid, args)
    cfg.model.sampling = 'ddpm'
    cfg.net.arch = 'archv4'
    cfg.model.expencoder = 'arcface'
    tester.load_cfg(cfg, args.checkpoint)
    tester.test('moaiparamdiffusion_archv4_ddpm', args.testimgfolder, args.arcfacefolder, args.testfilenamepath, args.testnumsample)
