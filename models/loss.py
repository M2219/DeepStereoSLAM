import torch.nn.functional as F

def model_loss_train(pose_est, pose_gt):

    angle_loss = F.mse_loss(pose_est[:, :3], pose_gt[:, :3])
    translation_loss = F.mse_loss(pose_est[:, 3:], pose_gt[:, 3:])
    loss = (100 * angle_loss + translation_loss)

    return loss

def model_loss_test(pose_est, pose_gt):

    angle_loss = F.mse_loss(pose_est[:, :3], pose_gt[:, :3])
    translation_loss = F.mse_loss(pose_est[:, 3:], pose_gt[:, 3:])
    loss = angle_loss + translation_loss

    return loss
