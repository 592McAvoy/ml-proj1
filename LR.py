import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel



class LogisticRegression(BaseModel):
    def __init__(self, in_dim=32*32*3):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._weight_init()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        p = self.sigmoid(self.fc(x))
        return p

    def cal_loss(self, p, y):
        loss = -torch.sum(y*torch.log(p) + (1-y)*torch.log(1-p))
        loss /= y.size(0)
        return loss

    def _weight_init(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

import torch
if __name__ == '__main__':
    lr = LogisticRegression()
    print(lr)
    dummy = torch.rand((10, 3, 32, 32))
    p = lr(dummy)
    print(p.size())
    loss = lr.cal_loss(p, torch.rand((10,)))
    print(loss.size())
