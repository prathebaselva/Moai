import torch
import torch.nn.functional as F


### ------------------------------------- Losses/Regularizations for vertices
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:,:,2] = weights[None,:]*real_2d_kp[:,:,2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    #print(landmarks_gt.shape)
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()
    print(real_2d.shape)
    print(predicted_landmarks.shape, flush=True)
    #pland = torch.cat([predicted_landmarks, predicted_landmarks], dim=0)
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    #loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, pland)
    return loss_lmk_2d * weight


def ConditionShapeMLPLoss(batch_size, e_theta, e_rand, seg_pair=None, uv_pair=None, knn_pair=None):
    errorloss = F.mse_loss(e_theta.view(batch_size, -1), e_rand.view(batch_size, -1), reduction='mean')
    #errorloss = F.l1_loss(e_theta.view(batch_size, -1), e_rand.view(batch_size, -1), reduction='mean')
    segloss = 0
    uvloss = 0
    knnloss = 0
    if seg_pair is not None:
        segloss =  F.mse_loss(seg_theta.view(batch_size,-1), gt_seg.view(batch_size,-1), reduction='mean')
    if uv_pair is not None:
        uvloss =  F.mse_loss(uv_theta.view(batch_size,-1), gt_uv.view(batch_size,-1), reduction='mean')
    if knn_pair is not None:
        knnloss = 1e-6* (torch.sum(torch.abs(torch.sub(knn_pair[0].to('cuda'), knn_pair[1].to('cuda')))))/batch_size
    #print(segloss)
    #print(uvloss)
    loss = errorloss + segloss + uvloss + knnloss
    return loss
    #projloss = ProjectionLoss(pred_x0)

def ConditionPwiseNetLoss(e_theta, e_rand):
    reconloss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
    projloss = ProjectionLoss()



#def ProjectionLoss(e_theta, e_rand):
