from __future__ import print_function, division

import argparse
import os
import time
import yaml
import gc
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import pandas as pd
import pickle

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import *

from models import __models__, model_loss_train, model_loss_test

from dataset.dataset import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset

parser = argparse.ArgumentParser(description="DeepStereoSLAM")
parser.add_argument('--model', default='DeepStereoSLAM', help='select a model structure', choices=__models__.keys())
parser.add_argument('--pathToSettings', default="configs.yaml", help="path to settings")
parser.add_argument('--logdir', default='checkpoints', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--performance', action='store_true', help='evaluate the performance')

args = parser.parse_args()

with open(args.pathToSettings, 'r') as f:
    fsSettings = yaml.load(f, Loader=yaml.SafeLoader)

if not os.path.isfile("dataset/train_data_info_path.pkl"):
    print("generate data info")
    train_df = get_data_info("train", fsSettings, overlap=1, sample_times=1)
    valid_df = get_data_info("validation", fsSettings, overlap=1, sample_times=1)

    train_df.to_pickle("dataset/train_data_info_path.pkl")
    valid_df.to_pickle("dataset/valid_data_info_path.pkl")

with open('dataset/train_data_info_path.pkl', 'rb') as file:
    train_df = pickle.load(file)

with open('dataset/valid_data_info_path.pkl', 'rb') as file:
    valid_df = pickle.load(file)

train_sampler = SortedRandomBatchSampler(train_df, fsSettings["batch_size"], drop_last=True)
train_dataset = ImageSequenceDataset(train_df)
TrainImgLoader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

valid_sampler = SortedRandomBatchSampler(valid_df, fsSettings["batch_size"], drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df)
TestImgLoader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=0)

print('Number of samples in training dataset: ', len(train_df.index))
print('Number of samples in validation dataset: ', len(valid_df.index))
print('='*50)

model = __models__[args.model]()
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.AdamW(model.parameters(), lr=fsSettings["lr"], betas=(0.9, 0.999))

print("creating new summary file")
logger = SummaryWriter(args.logdir)

start_epoch = 0
if args.resume:
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(
        all_saved_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    start_epoch = state_dict["epoch"] + 1

elif args.loadckpt:
    cv_name = args.loadckpt.split("sceneflow_")[1].split(".")[0]
    if cv_name != args.cv:
        raise AssertionError("Please load weights compatible with " + cv_name)

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict["model"].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))

def train():
    bestepoch = 0
    error = 100

    loss_ave = AverageMeter()

    loss_ave_t = AverageMeter()

    if args.performance:
        dummy_input1 = torch.randn(1, 3, 512, 960, dtype=torch.float).cuda()
        dummy_input2 = torch.randn(1, 3, 512, 960, dtype=torch.float).cuda()
        inference_time = measure_performance(dummy_input1, dummy_input2)
        print("inference time = ", inference_time)
        return 0

    for epoch_idx in range(start_epoch, fsSettings["epochs"]):
        adjust_learning_rate(optimizer, epoch_idx, fsSettings["lr"], fsSettings["lrepochs"])

        h_t1 = None
        h_t2 = None

        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % fsSettings["summary_freq"] == 0

            #loss, scalar_outputs, h_t1, h_t2 = train_sample(sample, h_t1, h_t2, compute_metrics=do_summary)
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)

            loss_ave.update(loss)

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)

            print('Epoch {}/{} | Iter {}/{} | train loss = {:.3f}({:.3f}) | time = {:.3f}'.format(epoch_idx, fsSettings["epochs"],
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss, loss_ave.avg,
                                                                                       time.time() - start_time))
            del scalar_outputs
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        h_t1 = None
        h_t2 = None

        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            start_time = time.time()
            #loss, scalar_outputs, h_t1, h_t2 = test_sample(sample, h_t1, h_t2, compute_metrics=do_summary)
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            tt = time.time()

            loss_ave_t.update(loss)

            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            print('Epoch {}/{} | Iter {}/{} | test loss = {:.3f}({:.3f}) | time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TestImgLoader), loss, loss_ave_t.avg,
                                                                                       tt - start_time))
            del scalar_outputs
        """
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["EPE"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["EPE"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        """

        gc.collect()
    #print('MAX epoch %d total test error = %.5f' % (bestepoch, error))

#def train_sample(sample, h_t1, h_t2, compute_metrics=False):
def train_sample(sample, compute_metrics=False):
    model.train()
    img_seq, poses = sample['image_seq'], sample['pose_seq']
    print(poses)

    img_seq = img_seq.cuda()
    poses = poses.cuda()

    optimizer.zero_grad()

    #pose_est, h_t1, h_t2 = model(img_seq, h_t1, h_t2)
    pose_est = model(img_seq)
    loss = model_loss_train(pose_est, poses)

    scalar_outputs = {"loss": loss}
    #if compute_metrics:
    #    with torch.no_grad():
    #        scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
    #        scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]

    loss.backward(retain_graph=True)
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs) #, h_t1, h_t2

@make_nograd_func
#def test_sample(sample, h_t1, h_t2, compute_metrics=True):
def test_sample(sample, compute_metrics=True):
    model.eval()

    img_seq, poses = sample['image_seq'], sample['pose_sq']

    img_seq = img_seq.cuda()
    poses = poses.cuda()

    #pose_est, h_t1, h_t2 = model(img_seq, h_t1, h_t2)
    pose_est = model(img_seq)

    loss = model_loss_test(pose_est, poses)
    scalar_outputs = {"loss": loss}

    #scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    #scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    #scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    #scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    #scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    return tensor2float(loss), tensor2float(scalar_outputs) #, h_t1, h_t2

@make_nograd_func
def measure_performance(dummy_input1):
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = model(dummy_input1)

    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input1)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    np.std(timings)

    return  mean_syn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    train()
