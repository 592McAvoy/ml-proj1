import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel



class LogisticRegression(BaseModel):
    def __init__(self, in_dim=32*32*3, reg='none', lamb=0.1):
        super().__init__()
        self.reg = reg
        self.lamb = lamb
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )

        self._weight_init()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)

    def cal_loss(self, p, y):
        p = p.squeeze()
        log_likelihood = y*torch.log(p) + (1-y)*torch.log(1-p)
        # print(log_likelihood.max())
        loss_cls = -torch.sum(log_likelihood)
        loss_cls /= y.size(0)
        # print(p.size(), y.size(), log_likelihood.size())
        # loss = F.binary_cross_entropy_with_logits(log_likelihood, y)

        loss_reg = torch.Tensor([0.]).to(y.device)
        if self.reg == 'lasso':
            for param in self.parameters():
                loss_reg += torch.norm(param, p=1)
        elif self.reg == 'ridge':
            for param in self.parameters():
                loss_reg += torch.norm(param, p=2)

        loss = loss_cls+self.lamb*loss_reg

        return loss, loss_cls, loss_reg

    def _weight_init(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


if __name__ == '__main__':
    lr = LogisticRegression()
    print(lr)
    dummy = torch.rand((10, 3, 32, 32))
    p = lr(dummy)
    print(p.size())
    loss = lr.cal_loss(p, torch.rand((10,)))
    print(loss.size())
