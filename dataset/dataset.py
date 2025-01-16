import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
from utils.converter import normalize_angle_delta

def get_data_info(dataset, configs, overlap, sample_times=1, test_video=None, pad_y=False, shuffle=False, sort=True):
    X_path, Y = [], []
    X_len = []
    seq_len = configs['seq_len']
    if dataset == "train":
        folder_list = configs['train_video']
    elif dataset == "validation":
        if test_video is not None:
            folder_list = [test_video]
        else:
            folder_list = configs['valid_video']

    for folder in folder_list:

        start_t = time.time()
        poses = np.load('{}{}.npy'.format(configs['pose_dir'], folder))
        fpaths = glob.glob('{}{}/image_2/*.png'.format(configs['image_dir'], folder))
        fpaths.sort()
        if sample_times > 1:
            sample_interval = int(np.ceil(seq_len / sample_times))
            start_frames = list(range(0, seq_len, sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for st in start_frames:
            n_frames = len(fpaths) - st
            jump = seq_len - overlap
            res = n_frames % seq_len
            if res != 0:
                n_frames = n_frames - res
            x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
            y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
            Y += y_segs
            X_path += x_segs
            X_len += [len(xs) for xs in x_segs]

        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))


    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    if shuffle:
        df = df.sample(frac=1)
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df

class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, minus_point_5=False):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),])

        self.minus_point_5 = minus_point_5
        self.data_info = info_dataframe
        self.seq_len = self.data_info.seq_len
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):


        sample = {}
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        #groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0:-1]  # get relative pose w.r.t. previois frame
        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0] # get relative pose w.r.t. the first frame in the sequence

        # here we rotate the sequence relative to the first frame
        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]

        # get relative pose w.r.t. previous frame
        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

	# here we consider cases when rotation angles over Y axis go through PI -PI discontinuity
        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])

        # print('Item after transform: ' + str(index) + '   ' + str(groundtruth_sequence))

        image_path_sequence = self.image_arr[index]
        #print(image_path_sequence)

        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            w, h = img_as_img.size
            img_as_img = img_as_img.crop((w-1242, h-375, w, h))
            img_as_tensor = self.transform(img_as_img)

            #if self.minus_point_5:
            #    img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]

            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)

        image_sequence = torch.cat(image_sequence, 0)

        sample["image_seq"] = image_sequence
        sample["pose_seq"] = groundtruth_sequence[1, :]

        return sample

    def __len__(self):
        return len(self.data_info.index)

if __name__ == '__main__':

    import yaml
    start_t = time.time()
    overlap = 1
    sample_times = 1
    folder_list = ['00']

    with open("configs.yaml", 'r') as f:
        fsSettings = yaml.load(f, Loader=yaml.SafeLoader)

    df = get_data_info("train", fsSettings, overlap, sample_times)

    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))

    n_workers = 4

    dataset = ImageSequenceDataset(df)
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=1, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sorted_sampler, num_workers=n_workers)

    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        image_sequence, groundtruth_sequence = batch

        print("image_sequence,.shape = ", image_sequence.shape)
        print("groundtruth_sequence.shape =  ", groundtruth_sequence.shape)
        exit()

    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)

