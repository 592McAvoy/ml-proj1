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
        if self.target_cls and self.mode == 'train':
            n_total = y.shape[0]
            n_target = np.count_nonzero(y == self.target_cls)
            # print(n_total, n_target )
            keep_rate = n_target/(n_total-n_target)
            new_X = []
            new_y = []
            for i, (data, label) in enumerate(zip(X, y)):
                # if label == self.target_cls or np.random.rand() < keep_rate:
                if label in [self.target_cls, self.target_cls+1]:
                    new_X.append(data)
                    new_y.append(label)
            new_X, new_y = np.array(new_X), np.array(new_y)
            # print(new_y.shape[0], np.count_nonzero(new_y == self.target_cls))
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
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=True)

    def get_batchsize(self):
        return self.batch_size

from torchvision.utils import make_grid
import cv2
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
    npimg = im.numpy().transpose(1,2,0)*255
    print(npimg.shape)
    cv2.imwrite('data.png', npimg)

    # data, label = dset[0]
    # print(data.size(), label)

    # loader = SVHNLoader(10, mode='train')
    # print(len(loader))
    # for data, lable in loader:
    #     print(data.size())
    #     break
