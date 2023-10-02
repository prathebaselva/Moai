import argparse
import os

from yacs.config import CfgNode as CN

cfg = CN()

cfg.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
cfg.dir_name = 'MOAI'
cfg.device = 'cuda'
cfg.pretrained_model_path = os.path.join(cfg.root_dir, cfg.dir_name, 'pretrained', 'best.pt')
cfg.output_dir = os.path.join(cfg.root_dir, cfg.dir_name, 'output')
cfg.checkpoint_dir = os.path.join(cfg.root_dir, cfg.dir_name, 'checkpoints')
cfg.pretrainwithgrad = False
cfg.modelname = ''

cfg.model = CN()
cfg.model.testing = False
cfg.model.validation = False
cfg.model.name = ''
cfg.model.pretrainname = ''

cfg.model.topology_path = os.path.join(cfg.root_dir, 'head_template.obj')
cfg.model.n_shape = 256
cfg.model.n_exp = 233
cfg.model.n_rot = 12
cfg.model.n_pupil = 1
cfg.model.n_pose = 4
cfg.model.n_tex = 200
cfg.model.n_cam = 3
cfg.model.layers = 8
cfg.model.hidden_layers_size = 256
cfg.model.mapping_layers = 3
cfg.model.use_pretrained = True
cfg.model.landmark = False
cfg.model.uniform = False
cfg.model.use_reg = False
cfg.model.with_exp = False
cfg.model.with_val = False
cfg.model.with_unpose = False
cfg.model.classfree = False
cfg.model.expencoder = 'arcface'
cfg.model.with_lmk = False
cfg.model.sampling = 'ddpm'
cfg.model.with_freeze = 'l4'
cfg.model.arcface_pretrained_model = os.path.join(cfg.root_dir, 'pretrained/arcface/glint360_r100/backbone.pth')

cfg.net = CN()
cfg.net.shape_dim = 7756*3
cfg.net.moai_dim = 502
cfg.net.context_dim = 512
cfg.net.time_dim = 512
cfg.net.arch = 'simplemoai'
cfg.net.residual = 'unet'
cfg.net.tag = ''
cfg.net.losstype = 'l1'
cfg.net.mode = 'sep'
cfg.net.numpoints = 7756

cfg.varsched = CN()
cfg.varsched.num_steps = 1000
cfg.varsched.beta_1 = 1e-4
cfg.varsched.beta_T = 0.01
cfg.varsched.mode = 'linear'

cfg.dataset = CN()
#cfg.dataset.training_data = ['Stirling', 'FaceWarehouse']
cfg.dataset.training_data = ['Lyhm', 'Stirling', 'FaceWarehouse', 'Coma']
cfg.dataset.validation_data = ['Florence']
cfg.dataset.validation_exp_data = ['AFLW2000']
cfg.dataset.test_data = ['NOW']
cfg.dataset.lmk = 'insight'
cfg.dataset.batch_size = 32
cfg.dataset.n_images = 4
cfg.dataset.tocenter = False
cfg.dataset.with100 = False
cfg.dataset.with20 = False
cfg.dataset.with_unpose = False
cfg.dataset.occlusion = 0
cfg.dataset.arc224 = 0
cfg.dataset.resnethalfimg = 0
cfg.dataset.resnetfullimg = 0
cfg.dataset.epoch = 100000
cfg.dataset.num_workers = 4
cfg.dataset.root = os.path.join(cfg.root_dir, '')
cfg.dataset.topology_path = os.path.join(cfg.root_dir, 'head_template.obj')
cfg.dataset.n_shape = 300
cfg.dataset.n_exp = 100


#---------------------------
# Training options
#-----------------------------

cfg.train = CN()
cfg.train.max_epochs = 100000
cfg.train.lr = 1e-4
cfg.train.diff_lr = 1e-4
cfg.train.diff_lr1 = 1e-4
cfg.train.varsched_lr = 1e-4
cfg.train.net_lr = 1e-4
cfg.train.fnet_lr = 1e-4
cfg.train.addnet_lr = 1e-4
cfg.train.arcface_lr = 1e-4
cfg.train.hse_lr = 1e-4
cfg.train.resnet_lr = 1e-4
cfg.train.rank_lr = 1e-4
cfg.train.weight_decay = 0.0
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.max_steps = 500000
cfg.train.val_steps = 10
cfg.train.checkpoint_steps = 1000
cfg.train.checkpoint_epochs_steps = 10000
cfg.train.val_save_img = 1200
cfg.train.vis_steps = 1200
cfg.train.val_vis_dir = 'val_images'
cfg.train.reset_optimizer = False
cfg.train.resume = False
cfg.train.resume_checkpoint = ''
cfg.train.write_summary = True


# TEST
cfg.test = CN()
cfg.test.num_points = 5023
cfg.test.batch_size = 1
cfg.test.point_dim = 3
cfg.test.cache_path = os.path.join(cfg.root_dir, 'FACEDATA/NOW/cache') 
#cfg.test.now_images = os.path.join(cfg.root_dir, 'FACEDATA/NOW/final_release_version/iphone_pictures') 
cfg.test.now_images = os.path.join(cfg.root_dir, 'FACEDATA/NOW/arcface_input') 
cfg.test.stirling_hq_images = os.path.join(cfg.root_dir, 'FACEDATA/STIRLING/arcface_input/HQ') 
cfg.test.stirling_lq_images = os.path.join(cfg.root_dir, 'FACEDATA/STIRLING/arcface_input/LQ') 
cfg.test.affectnet_images = os.path.join(cfg.root_dir, 'FACEDATA/AFFECTNET/arcface_input') 
cfg.test.aflw2000_images = os.path.join(cfg.root_dir, 'FACEDATA/AFLW2000/arcface_input') 

def get_cfg_defaults():
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint location to load', default='shape')
    parser.add_argument('--numcheckpoint', type=int, help='number of checkpoints', default=1)
    parser.add_argument('--test_dataset', type=str, help='Test dataset path', default='')
    parser.add_argument('--testfilenamepath', type=str, help='filepath', default='')
    parser.add_argument('--testimgfolder', type=str, help='imgfolder', default='')
    parser.add_argument('--testarcfacefolder', type=str, help='arcfacefolder', default='')
    parser.add_argument('--testnumsample', type=int, help='number samples', default=5)
    
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg, args
