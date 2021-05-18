import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.component.kernel import RBF, Poly, Sigmoid, Cosine


class KernelLogisticRegression(BaseModel):
    def __init__(self, N, fea_dim=1024, kernel='rbf', lamb=0.1, reg='lasso'):
        super().__init__()
        self.lamb = lamb
        self.reg_type = reg
        self.fc = nn.Sequential(
            nn.Linear(N, 1, bias=False),
            nn.Sigmoid()
        )
        self.X = nn.Parameter(torch.zeros(N, fea_dim), requires_grad=False)

        if kernel == 'rbf':
            self.kernel = RBF()
        elif kernel.startswith('poly'):
            d = int(kernel.split('_')[-1])
            self.kernel = Poly(d)
        elif kernel == 'cosine':
            self.kernel = Cosine()
        elif kernel == 'sigmoid':
            self.kernel = Sigmoid()
        else:
            raise NotImplementedError(kernel)

        self._weight_init()

    def fit(self, x):
        """

        Args:
            x ([type]): data (B, C, H, W)
            y ([type]): label (B, )

        Returns:
            [type]: [description]
        """
        self.X.data = x.contiguous().view(x.size(0), -1)
        
        return self.predict(x)
    
    def cal_loss(self, p, y, K=None):
        p = p.squeeze()
        log_likelihood = y*torch.log(p) + (1-y)*torch.log(1-p)
        loss_cls = -torch.sum(log_likelihood)
        loss_cls /= y.size(0)
        
        loss_reg = torch.Tensor([0.]).to(y.device)
        c = self.fc[0].weight
        if self.reg_type == 'ridge'and K is not None:
            loss_reg = c.matmul(K).matmul(c.t())
            # loss_reg /= y.size(0)
            
        # elif self.reg_type == 'lasso' and K is not None:
        #     # BUG: negative loss
        #     # loss_reg = c.matmul(torch.diag(K))
        #     loss_reg = torch.abs(torch.diag(K)*c).sum()
        #     loss_reg /= y.size(0)
 
        elif self.reg_type == 'lasso':
            for param in self.parameters():
                if not param.requires_grad:
                    continue
                loss_reg += torch.norm(param, p=1)
        # elif self.reg_type == 'ridge':
        #     for param in self.parameters():
        #         if not param.requires_grad:
        #             continue
        #         loss_reg += torch.norm(param, p=2)
            

        loss = loss_cls+self.lamb*loss_reg

        return loss, loss_cls, loss_reg
    
    def predict(self, x):
        x = x.contiguous().view(x.size(0), -1)
        K = self.kernel(x, self.X)
        p = self.fc(K)

        return p, K

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
