import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class RBF(nn.Module):
    def __init__(self):
        super(RBF, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.sigma, 0, 1)

    def forward(self, x, y):
        if len(x.size()) > 2:
            x = x.contiguous().view(x.size(0), -1)
            y = y.contiguous().view(y.size(0), -1)

        x_i = x.unsqueeze(1)    # B,1,D1
        y_j = y.unsqueeze(0)    # 1,B,D2

        sqd = torch.sum(torch.pow(x_i-y_j, 2), dim=2)
        K_ij = torch.exp(-0.5*sqd/torch.pow(self.sigma, 2))  # D1, D2
        
        K_ij = K_ij / K_ij.sum(0).expand_as(K_ij)

        return K_ij


class KernelLogisticRegression(BaseModel):
    def __init__(self, N, kernel='rbf', lamb=0.1, reg='lasso'):
        super().__init__()
        self.lamb = lamb
        self.reg_type = reg
        self.fc = nn.Sequential(
            nn.Linear(N, 1, bias=False),
            nn.Sigmoid()
        )
        self.X = None

        if kernel == 'rbf':
            self.kernel = RBF()
        else:
            raise NotImplementedError(kernel)

        self._weight_init()

    def fit(self, x, y):
        """

        Args:
            x ([type]): data (B, C, H, W)
            y ([type]): label (B, )

        Returns:
            [type]: [description]
        """
        self.X = x.contiguous().view(x.size(0), -1)
        p, K = self.predict(x)
        
        p = p.squeeze()
        log_likelihood = y*torch.log(p) + (1-y)*torch.log(1-p)
        loss_cls = -torch.sum(log_likelihood)
        loss_cls /= y.size(0)
        
        c = self.fc[0].weight
        if self.reg_type == 'ridge':
            loss_reg = c.matmul(K).matmul(c.t())
        elif self.reg_type == 'lasso':
            # BUG: negative loss
            # loss_reg = c.matmul(torch.diag(K))
            loss_reg = torch.abs(torch.diag(K)*c).sum()
            loss_reg /= y.size(0)

        loss = loss_cls+self.lamb*loss_reg

        return p, (loss, loss_cls, loss_reg)

    def predict(self, x):
        x = x.contiguous().view(x.size(0), -1)
        K = self.kernel(x, self.X)
        out = self.fc(K)

        return out, K

    def _weight_init(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    m.bias.data.fill_(0.01)


if __name__ == '__main__':

    dummy = torch.rand((3, 3, 32, 32))
    dummy1 = torch.rand((5, 3, 32, 32))
    y = torch.Tensor([1, 0, 1])
    y1 = torch.Tensor([1, 0, 0, 1, 1])

    klr = KernelLogisticRegression(N=5)
    out, loss = klr.fit(dummy1, y1)
    print(out.size())
    out, _ = klr.predict(dummy)
    print(out.size())

    # rbf = RBF()
    # k = rbf(dummy, dummy1)
    # print(k.size())
    # lr = LogisticRegression()
    # print(lr)
    # p = lr(dummy)
    # print(p.size())
    # loss = lr.cal_loss(p, torch.rand((10,)))
    # print(loss.size())

    #  N = self.X.size(0)
    #     if self.reg_loss == 'ridge':
    #         # c_hat = (K+\lamb*I)^(-1)*Y
    #         c = torch.inverse(K+self.lamb*torch.eye(K.size(0))) # B * B
    #         c = c.matmul(self.y.view(N, 1)) # B*1
    #         reg_term = c.t().matmul(K).matmul(c)

    #     elif self.reg_loss == 'lasso':
    #         # c_hat = (K+\lamb*K^(-1)*Diag(K))^(-1)*Y
    #         c = torch.inverse(K+self.lamb*torch.inverse(K).matmul(torch.diag(K)))
    #         c = c.matmul(self.y.view(N, 1)) # B*1
    #         reg_term = c.t().matmul(torch.diag(K)).matmul(c)
