import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import numpy as np


class GaussianMixtureModel(BaseModel):
    def __init__(self, n_fea=256, n_class=10):
        super().__init__()
        self.n_fea = n_fea  # D
        self.n_class = n_class  # K

        self.mu = nn.Parameter(torch.Tensor(
            1, n_class, n_fea), requires_grad=False)  # (1, K, D)
        # (1, K, D)
        self.var = nn.Parameter(torch.Tensor(
            1, n_class, n_fea), requires_grad=False)
        self.pi = nn.Parameter(torch.Tensor(
            1, n_class, 1), requires_grad=False)  # (1, K, 1)

        self._init_params()
        self.fited = False

    def _init_params(self):
        self.mu.data.normal_(0., 1.)
        self.var.data.fill_(1.)
        self.pi.data.fill_(1./self.n_class)  # 1/K

    def fit(self, x, eps=1e-3, max_iter=100):
        if len(x.size()) < 3:
            x = x.unsqueeze(1)  # (N, D) -> (N, 1, D)

        current_logll = -1e10  # log likelihood
        for i in range(max_iter):
            resp = self._e_step(x)
            mu, var, pi = self._m_step(x, resp)
            logll = self._cal_logll(x)
            # print(logll)
            # break
            if (logll.abs() == float('Inf')) or (logll.abs() == float('nan')):

                self._init_params()
            
            delta = logll-current_logll
            if delta < eps:
                # converge
                break

            self._update(mu, var, pi)
            current_logll = logll

        self.fited = True
        print("Stop at step {} with delta {}".format(i, delta))

    def predict(self, x):
        if not self.fited:
            raise NotImplementedError('Call fit() first')

        if len(x.size()) < 3:
            x = x.unsqueeze(1)  # (N, D) -> (N, 1, D)
        log_p = self._cal_log_p(x)
        # print(log_p)
        log_weighted_p = log_p + torch.log(self.pi)  # N, K, 1
        p = log_weighted_p.squeeze()
        return torch.softmax(p-torch.max(p), dim=1)
        
        # print(log_weighted_p)
        # p = torch.exp(log_weighted_p).squeeze()
        # print(p)
        # return p/torch.sum(p, dim=1, keepdim=True)

    def _update(self, mu, var, pi):
        self.mu.data = mu
        self.var.data = var
        self.pi.data = pi

        # print(self.mu, self.var, self.pi)

    def _cal_log_p(self, x):
        # x ~ N(mu, var)
        # log P(x) = -0.5*[log(2*np.pi) + log(var) + (x-mu)^2/var]
        mu = self.mu
        var = self.var

        term0 = self.n_fea*np.log(2*np.pi)
        term1 = torch.sum(torch.log(var), dim=2, keepdim=True)
        # print(term1)
        term2 = torch.sum((mu*mu+x*x-2*x*mu)/var, dim=2, keepdim=True)
        log_p = -0.5*(term0+term1+term2)  # N, K, 1
        # print(log_p.size())
        # print(term0, term1, term2, log_p)

        return log_p

    def _cal_logll(self, x):
        # weighted_p = pi*p
        # log(weighted_p) = log(pi) + log(p)
        log_p = self._cal_log_p(x)
        log_weighted_p = log_p + torch.log(self.pi)  # N, K, 1
        log_weighted_p_sum = torch.logsumexp(
            log_weighted_p, dim=1, keepdim=True)  # N, 1, 1

        return log_weighted_p_sum.sum()

    def _e_step(self, x):
        # weighted_p = pi*p
        # log(weighted_p) = log(pi) + log(p)
        log_p = self._cal_log_p(x)
        log_weighted_p = log_p + torch.log(self.pi)  # N, K, 1
        log_weighted_p_sum = torch.logsumexp(
            log_weighted_p, dim=1, keepdim=True)  # N, 1, 1

        # response = weighted_p / weighted_p_sum
        # log(response) = log(weighted_p) - log(weighted_p_sum)
        log_response = log_weighted_p - log_weighted_p_sum
        response = torch.exp(log_response)  # N, K, 1

        # print(response)

        return response

    def _m_step(self, x, resp):
        eps = 1e-6
        resp_sum = torch.sum(resp, dim=0, keepdim=True) + eps  # 1, K, 1

        # mu
        mu = torch.sum(resp*x, dim=0, keepdim=True)/resp_sum

        # var
        x2 = torch.sum(resp*x*x, dim=0, keepdim=True)/resp_sum
        mu2 = mu*mu
        x_mu = torch.sum(resp*mu*x, dim=0, keepdim=True)/resp_sum
        var = x2+mu2-2*x_mu+eps

        # pi
        pi = resp_sum/x.size(0)

        # print(mu, var, pi)

        return mu, var, pi


if __name__ == '__main__':
    dummy = torch.rand((10, 256))
    gmm = GaussianMixtureModel(n_class=2)
    gmm.fit(dummy)
    prob = gmm.predict(dummy)
    print(prob)
