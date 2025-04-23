import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomRotation, RandomHorizontalFlip
from PIL import Image
import pathlib
import numpy as np
from os import path

from src.utils.seeds import worker_init_fn, generator

class BEMSDataset(torch.utils.data.Dataset):
    def __init__(self, root="./data/BEMS_data", train=True, delay=1, time_window=10):
        super().__init__()
        self.root = root
        self.train = train
        self.delay = delay # predict delayed label
        self.time_window = time_window # fixed time frame

        if self.train:
            data_src = np.load(path.join(self.root, 'BEMS_RNN_train_data.npy'))
            label_src = np.load(path.join(self.root, 'BEMS_RNN_train_labels.npy'))
        else:
            data_src = np.load(path.join(self.root, 'BEMS_RNN_test_data.npy'))
            label_src = np.load(path.join(self.root, 'BEMS_RNN_test_labels.npy'))

        data_src = np.asarray(data_src[:-self.delay])
        label_src = np.asarray(label_src[self.delay:])

        self.data, self.label = [], []
        for frame_i in range(len(data_src) - self.time_window):
            self.data.append(data_src[frame_i:frame_i+self.time_window])
            self.label.append(label_src[frame_i:frame_i+self.time_window])

        self.data = np.asarray(self.data)
        self.label = np.asarray(self.label)
    
    def __getitem__(self, index):
        X = self.data[index, :]
        y = self.label[index, :]
        return X, y
    
    def __len__(self):
        return self.data.shape[0]
    
def create_dataset(root, batch_size, time_window=10, delay=1):
    train_dataset = BEMSDataset(root=root, train=True, delay=delay, time_window=time_window)
    test_dataset = BEMSDataset(root=root, train=False, delay=delay, time_window=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    return train_loader, test_loader