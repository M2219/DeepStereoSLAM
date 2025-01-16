import glob
import os
import time
import yaml

import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.parallel
from PIL import Image

from models import __models__, model_loss_train, model_loss_test
from dataset.dataset import get_data_info, ImageSequenceDataset
from utils.converter import eulerAnglesToRotationMatrix


if __name__ == '__main__':


    with open('configs.yaml', 'r') as f:
        fsSettings = yaml.load(f, Loader=yaml.SafeLoader)

    videos_to_test= fsSettings["valid_video"]

    load_model_path = './checkpoint/deepslam_first.ckpt'
    save_dir = 'result/'
    seq_len = fsSettings["seq_len"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = __models__["DeepStereoSLAM"]()
    model = nn.DataParallel(model)
    model.cuda()

    state_dict = torch.load(load_model_path)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict["model"].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    print("="*50)
    print("Checkpoint loaded!")
    print("="*50)
    n_workers = 0
    overlap = 1

    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
    batch_size = 1

    fd=open('test_dump.txt', 'w')
    fd.write('\n'+'='*50 + '\n')

    for test_video in videos_to_test:
        print(test_video)
        df = get_data_info("validation", fsSettings, overlap, test_video=test_video, sample_times=1, pad_y=False, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        dataset = ImageSequenceDataset(df)
        df.to_csv('test_df.csv')

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        gt_pose = np.load('{}{}.npy'.format(fsSettings['pose_dir'], test_video))

        has_predict = False
        answer = [[0.0] * 6,]
        st_t = time.time()

        n_batch = len(dataloader)

        for i, sample in enumerate(dataloader):
            model.eval()
            print('{} / {}'.format(i, n_batch), end='\r', flush=True)
            img_seq, poses = sample['image_seq'], sample['pose_seq']


            img_seq = img_seq.cuda()
            poses = poses.cuda()
            #print("gt", poses)

            predict_pose = model(img_seq)
            #print("pred", predict_pose)

            fd.write('Batch: {}\n'.format(i))
            fd.write(' {}\n'.format(predict_pose[0]))

            predict_pose = predict_pose.data.cpu().numpy().tolist()[0]
            if i == 0:
                for i in range(len(predict_pose)):
                    predict_pose[i] += answer[-1][i] # to absolute
                answer.append(predict_pose)

            else:
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) # ?
                #ang = eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]]) # ?
                location = ang.dot(predict_pose[3:])
                predict_pose[3:] = location[:]

                # use only last predicted pose in the following prediction
                last_pose = predict_pose
                for i in range(len(last_pose)):
                    last_pose[i] += answer[-1][i]

                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose)

            #print("answer", answer)

        print('len(answer): ', len(answer))
        print('Predict use {} sec'.format(time.time() - st_t))

        # Save answer
        with open('{}/out_{}.txt'.format(save_dir, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')


        # Calculate loss
        gt_pose = np.load('{}{}.npy'.format(fsSettings['pose_dir'], test_video))  # (n_images, 6)
        loss = 0

        for t in range(len(gt_pose)):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t, :3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t, 3:6]) ** 2)
            loss = (100 * angle_loss + translation_loss)

        loss /= len(gt_pose)
        print('Loss = ', loss)
        print('='*50)
