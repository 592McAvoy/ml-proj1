from torchvision.utils import make_grid
import cv2
import scipy.io
import os
from PIL import Image

import numpy as np
from torch.utils import data
from torchvision import transforms as T

from base import BaseDataLoader


class SVHNDataset(data.Dataset):
    def __init__(self, transform, mode='test', target_cls=None):
        self.transform = transform
        self.target_cls = target_cls
        self.mode = mode
        if target_cls:
            assert(target_cls in list(range(1, 11)))  # cls in 1~10

        mat = scipy.io.loadmat('data/SVHN/{}_32x32.mat'.format(mode))
        X = mat['X'].transpose(3, 0, 1, 2)
        y = mat['y'][:, 0]

        X, y = self._adjust(X, y)

        self.datalist = list(zip(X, y))

    def _adjust(self, X, y):
        if self.target_cls:
            pos_cls = self.target_cls
            neg_cls = self.target_cls%10+1
            print('Positive Number:{}\tNegative Number:{}'.format(pos_cls, neg_cls))
            new_X = []
            new_y = []
            for lab in [pos_cls, neg_cls]:
                new_X.append(X[y==lab])
                new_y.append(y[y==lab])
            print(new_X)
            new_X = np.concatenate(new_X, 0)
            new_y = np.concatenate(new_y, 0)           
            
            return new_X, new_y
        return X, y

    def __getitem__(self, index):
        data, label = self.datalist[index]
        # cvt to PIL Image
        data = Image.fromarray(np.uint8(data))
        data = self.transform(data)

        if self.target_cls:
            label = 1 if label == self.target_cls else 0
        else:
            label -= 1  # convert to 0~9

        return data, label

    def __len__(self):
        return len(self.datalist)


class SVHNLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=0, training=True, gray=True, **kwargs):

        # Normalize to -1 ~ 1
        if gray:
            trfm = T.Compose([
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            trfm = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
            ])

        self.dataset = SVHNDataset(trfm, **kwargs)
        if batch_size < 0:
            batch_size = len(self.dataset)
        self.batch_size = batch_size
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, drop_last=True)

    def get_batchsize(self):
        return self.batch_size


if __name__ == '__main__':
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])

    trfm = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    dset = SVHNDataset(trfm, target_cls=1)
    print(len(dset))
    im = make_grid([dset[i][0] for i in range(64)], nrow=8, normalize=True)
    npimg = im.numpy().transpose(1, 2, 0)*255
    print(npimg.shape)
    cv2.imwrite('data.png', npimg)

    # data, label = dset[0]
    # print(data.size(), label)

    # loader = SVHNLoader(10, mode='train')
    # print(len(loader))
    # for data, lable in loader:
    #     print(data.size())
    #     break
