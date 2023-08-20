import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers
import matplotlib.pyplot as plt
#import seaborn as sns

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None, tag=''):
        #tag = ''


        if step is None:
            fname = tag+'ckpt_%.6f_.pt' % float(score)
        else:
            fname = tag+'ckpt_%.6f_%d.pt' % (float(score), int(step))
        if 'best' in tag:
            fname = tag+'.pt'
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_new_results_dir(root='./results', tag=''):
    results_dir = os.path.join(root, tag)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def projectMeshToRGBImage1(ref_mesh, pred_mesh, proj_mesh, proj_mat, epoch, rgb_img, tag='', outputfolder='.'):
    pcenter, pradius = getCenterAndRadius(pred_mesh)
    print(pcenter)
    print(pradius)
    print(ref_mesh.shape)
    #print(np.max(ref_mesh, axis=0))
    #print("mean = ",torch.mean(ref_mesh, dim=0))
    #normpredx_0 = (pred_mesh - pcenter)/pradius
    normpredx_0 = (pred_mesh)#- pcenter)/pradius
    center, radius = getCenterAndRadius(proj_mesh)

    #unnorm_predx_0 =(normpredx_0 * radius) + center
    #unnorm_x_0 =(ref_mesh * radius) + center
    unnorm_predx_0 =(normpredx_0 * radius) + center
    unnorm_x_0 =(ref_mesh * radius) + center

    proj_x_0 = getoneProjection(unnorm_x_0, proj_mat[1,1])
    proj_predx_0 = getoneProjection(unnorm_predx_0, proj_mat[1,1])
    px0 = (proj_x_0.clone())
    px0 = np.array(px0.detach().cpu())
    ppx0 = (proj_predx_0.clone())
    ppx0 = np.array(ppx0.detach().cpu())
    #orig_img = rgb_img.cpu().numpy()*255.0 # inp.zeros((320,320,3), dtype=np.uint8)
    orig_img = rgb_img.cpu().numpy() # inp.zeros((320,320,3), dtype=np.uint8)
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(px0[:,0], px0[:,1], c="red")
    plt.savefig(os.path.join(outputfolder, 'orig'+tag+'_'+str(epoch)+'.png'))
    plt.close()
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
    plt.savefig(os.path.join(outputfolder, 'pred'+tag+'_'+str(epoch)+'.png'))
    plt.close()
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(px0[:,0], px0[:,1], c="red")
    plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
    plt.savefig(os.path.join(outputfolder, 'origpred'+tag+'_'+str(epoch)+'.png'))
    plt.close()


def projectMeshToRGBImage(ref_mesh, pred_mesh, radius, center, proj_mat, epoch, rgb_img, tag='', outputfolder='.'):
    normpredx_0 = (pred_mesh)#- pcenter)/pradius
    unnorm_predx_0 =(normpredx_0 * radius) + center
    unnorm_x_0 =(ref_mesh * radius) + center

    proj_x_0 = getoneProjection(unnorm_x_0, proj_mat[1,1])
    proj_predx_0 = getoneProjection(unnorm_predx_0, proj_mat[1,1])
    px0 = (proj_x_0.clone())
    px0 = np.array(px0.detach().cpu())
    ppx0 = (proj_predx_0.clone())
    ppx0 = np.array(ppx0.detach().cpu())
    #orig_img = rgb_img.cpu().numpy()*255.0 # inp.zeros((320,320,3), dtype=np.uint8)
    orig_img = rgb_img.cpu().numpy() # inp.zeros((320,320,3), dtype=np.uint8)
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(px0[:,0], px0[:,1], c="red")
    plt.savefig(os.path.join(outputfolder, 'orig'+tag+'_'+str(epoch)+'.png'))
    plt.close()
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
    plt.savefig(os.path.join(outputfolder, 'pred'+tag+'_'+str(epoch)+'.png'))
    plt.close()
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(px0[:,0], px0[:,1], c="red")
    plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
    plt.savefig(os.path.join(outputfolder, 'origpred'+tag+'_'+str(epoch)+'.png'))
    plt.close()

def projectMeshToImage(ref_mesh, pred_mesh, proj_mesh, proj_mat, epoch, tag='', outputfolder='.'):
    pcenter, pradius = getCenterAndRadius(pred_mesh)
    #print(pcenter)
    #print(pradius)
    #normpredx_0 = (pred_mesh - pcenter)/pradius
    normpredx_0 = pred_mesh
    center, radius = getCenterAndRadius(proj_mesh)

    unnorm_predx_0 =(normpredx_0 * radius) + center
    unnorm_x_0 =(ref_mesh * radius) + center

    proj_x_0 = getoneProjection(unnorm_x_0, proj_mat[1,1])
    proj_predx_0 = getoneProjection(unnorm_predx_0, proj_mat[1,1])
    px0 = (proj_x_0.clone())
    px0 = np.array(px0.detach().cpu())
    ppx0 = (proj_predx_0.clone())
    ppx0 = np.array(ppx0.detach().cpu())
    orig_img = np.zeros((320,320,3), dtype=np.uint8)
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(px0[:,0], px0[:,1], c="red")
    plt.savefig(os.path.join(outputfolder, 'orig'+tag+'_'+str(epoch)+'.png'))
    plt.close()
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
    plt.savefig(os.path.join(outputfolder, 'pred'+tag+'_'+str(epoch)+'.png'))
    plt.close()
    plt.figure(figsize=(16,16))
    plt.imshow(orig_img)
    plt.scatter(px0[:,0], px0[:,1], c="red")
    plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
    plt.savefig(os.path.join(outputfolder, 'origpred'+tag+'_'+str(epoch)+'.png'))
    plt.close()


def getAllCenterAndRadius(mesh):
    maxdim = torch.max(mesh[:,:,:3], dim=1)[0]
    mindim = torch.min(mesh[:,:,:3], dim=1)[0]
    m = (maxdim - mindim)
    center = (maxdim+mindim)/2
    radius = torch.max(m, dim = 1)[0]
    #print(center.shape)
    #print(radius.shape)
    return center,radius

def getCenterAndRadius(mesh):
    maxdim = torch.max(mesh[:,:3], dim=0)[0]
    mindim = torch.min(mesh[:,:3], dim=0)[0]

    m = maxdim - mindim
    center = (maxdim+mindim)/2
    radius = torch.max(m)
    return center,radius

def getoneProjection(x_3d, focal_length):
    x_2d = (x_3d[:,:2]/x_3d[:,2:3])[:,:3]
    f = (320 * (focal_length / 2))
    #print(f)
    x_2d *=  f
    #print(x_2d)
    x_2d += torch.tensor([[320/2, 320/2]]).to('cuda')
    x_2d[:,0] = 320 - x_2d[:,0]
    return x_2d

def projectToCamSpace(x_3d, focal_length):
    x_2d = (x_3d[:,:2]/x_3d[:,2:3])[:,:3]
    f = (320 * (focal_length / 2))
    x_2d *=  f
    x_2d += torch.tensor([[320/2, 320/2]]).to('cuda')
    x_2d[:,0] = 320 - x_2d[:,0]
    return x_2d


def visualize_kde(t, mesh_x0, mesh_xt, predmesh_x0, tag, epoch):
    plt.close()
    fig = plt.figure(figsize=(20,20))
    fig.suptitle('noisymesh_train_kde')
    if len(t) > 16:
        r = np.random.randint(0, len(t), size=(16))
        t = t[r]
    argindex = np.argsort(t)
    plt.subplot(6,3,1)
    sns.kdeplot(data = np.array(mesh_x0[0].cpu().numpy()))
    plt.ylabel('orig_data')
    for i,index in enumerate(argindex):
        plt.subplot(6,3,i+2)
        sns.kdeplot(data = np.array(mesh_xt[index].cpu().numpy()))
        plt.ylabel(str(t[index]))
    plt.savefig('./results/'+str(tag)+'noisymesh_kde'+str(epoch)+'.png')
    plt.close()

    fig = plt.figure(figsize=(20,20))
    fig.suptitle('predmesh_train_kde')
    argindex = np.argsort(t)
    plt.subplot(6,3,1)
    sns.kdeplot(data = np.array(mesh_x0[0].cpu().numpy()))
    plt.ylabel('orig_data')
    for i,index in enumerate(argindex):
        plt.subplot(6,3,i+2)
        sns.kdeplot(data = np.array(predmesh_x0[index].clone().detach().cpu().numpy()))
        plt.ylabel(str(t[index]))
    plt.savefig('/data/face_diffusion/results/'+str(tag)+'predmesh_kde'+str(epoch)+'.png')
    plt.close()

def visualize_prediction(t, predmesh_x0, mesh_x0, radius, center, focallength, tag, epoch):
    bg_img = np.zeros((320,320,3), dtype=np.uint8)
    fig = plt.figure(figsize=(80,80))
    fig.suptitle('gt_predmesh_train_proj')
    plt.imshow(bg_img)
    argindex = np.argsort(t)
    print(argindex)
    for i, index in enumerate(argindex):
        unnorm_pred_mesh = (predmesh_x0[index] * radius[index]) + center[index]
        unnorm_input_mesh = (mesh_x0[index] * radius[index]) + center[index]

        proj_input_mesh = getoneProjection(unnorm_input_mesh, focallength)
        proj_pred_mesh = getoneProjection(unnorm_pred_mesh, focallength)

        px0 = (proj_input_mesh.clone())
        px0 = np.array(px0.detach().cpu())
        ppx0 = (proj_pred_mesh.clone())
        ppx0 = np.array(ppx0.detach().cpu())

        plt.subplot(4,4,i+1)
        plt.imshow(bg_img)
        plt.scatter(px0[:,0], px0[:,1], c="red")
        plt.scatter(ppx0[:,0], ppx0[:,1], c="white")
        plt.ylabel(str(t[index]))
    plt.savefig('../results/gt_pred'+str(tag)+'_'+str(epoch)+'.png')

    plt.close()



def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn
