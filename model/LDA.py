import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# Bug: misuse of dot() and matmul()
class LinearDiscriminantAnalysis(BaseModel):
    def __init__(self):
        super().__init__()
        # projection matrix
        self.A = None
        self.H_c = None
        self.n_class = 2

        self.dummy = nn.Linear(1,1)

    def predict(self, x):
        """
        according to DEEP LINEAR DISCRIMINANT ANALYSIS

        Args:
            x (B, n_dim): input data

        Returns:
            p (B, 2): classfication probability
        """
        # 
        if self.A is None:
            raise NotImplementedError('Call fit() first')

        x = x.view(x.size(0), -1)
        T = self.H_c.matmul(self.A).matmul(self.A.t())
        # print(x.size(), T.size())
        d = torch.matmul(x, T.t()) - 0.5*torch.diag(self.H_c.matmul(T.t()))

        return torch.sigmoid(d)

    def fit(self, x, y):
        device = y.device
        x = x.view(x.size(0), -1)
        n_dim = x.size(1)
        mean_all = torch.mean(x, dim=0)
        S_w = torch.zeros((n_dim, n_dim)).to(device)
        S_b = torch.zeros((n_dim, n_dim)).to(device)

        H_c = torch.zeros((self.n_class, n_dim)).to(device)
        for c in range(self.n_class):
            # Within class scatter matrix:
            # SW = sum((X_c - mean_X_c)^2 )
            x_c = x[y == c]
            mean_c = torch.mean(x_c, dim=0)
            H_c[c, :] = mean_c
            x_c_bar = x_c-mean_c
            S_w += torch.matmul(x_c_bar.t(), x_c_bar)

            # Between class scatter:
            # SB = sum( n_c * (mean_X_c - mean_overall)^2 )
            n_c = x_c.size(0)
            x_c_hat = x_c-mean_all
            S_b += n_c*torch.matmul(x_c_hat.t(), x_c_hat)

        # tmp = SW^-1 * SB
        tmp = torch.matmul(torch.pinverse(S_w), S_b)
        eigenvalues, eigenvectors = torch.eig(tmp, eigenvectors=True)
        idx = torch.argsort(eigenvalues[:, 0], descending=True)
        # print(eigenvalues.size(), eigenvectors.size())
        eigenvalues = eigenvalues[idx, 0]
        eigenvectors = eigenvectors[idx, :]
        # print(eigenvalues.size(), eigenvectors.size())
        v = eigenvectors[:, :self.n_class-1]
        # print(v.size())
        # loss = -v.t().matmul(tmp).matmul(v).sum()
        self.A = v # (n_dim, C-1)
        self.H_c = H_c # (C, n_dim)
        # return loss


if __name__ == '__main__':
    lda = LinearDiscriminantAnalysis()
    print(lda)
    dummy = torch.rand((5, 3, 32, 32))
    y = torch.Tensor([0, 1, 1, 0, 1])
    lda.fit(dummy, y)
    p = lda.predict(dummy)
    print(p)
# https://github.com/bfshi/DeepLDA/blob/master/CNN_LDA/CNN%2BLDA.py
# def fit(self, X, y):
#         n_features = X.shape[1]
#         class_labels = np.unique(y)

#         # Within class scatter matrix:
#         # SW = sum((X_c - mean_X_c)^2 )

#         # Between class scatter:
#         # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

#         mean_overall = np.mean(X, axis=0)
#         SW = np.zeros((n_features, n_features))
#         SB = np.zeros((n_features, n_features))
#         for c in class_labels:
#             X_c = X[y == c]
#             mean_c = np.mean(X_c, axis=0)
#             # (4, n_c) * (n_c, 4) = (4,4) -> transpose
#             SW += (X_c - mean_c).T.dot((X_c - mean_c))

#             # (4, 1) * (1, 4) = (4,4) -> reshape
#             n_c = X_c.shape[0]
#             mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
#             SB += n_c * (mean_diff).dot(mean_diff.T)

#         # Determine SW^-1 * SB
#         A = np.linalg.inv(SW).dot(SB)
#         # Get eigenvalues and eigenvectors of SW^-1 * SB
#         eigenvalues, eigenvectors = np.linalg.eig(A)
#         # -> eigenvector v = [:,i] column vector, transpose for easier calculations
#         # sort eigenvalues high to low
#         eigenvectors = eigenvectors.T
#         idxs = np.argsort(abs(eigenvalues))[::-1]
#         eigenvalues = eigenvalues[idxs]
#         eigenvectors = eigenvectors[idxs]
#         # store first n eigenvectors
#         self.linear_discriminants = eigenvectors[0:self.n_components]
