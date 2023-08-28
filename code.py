"""
Created on Tue Aug 15 11:14:39 2023

@author: ruohang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, proj_dim=20, H=25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(25, proj_dim),
            nn.LeakyReLU())
        self.gru = nn.GRU(proj_dim, H, batch_first=True)

    def forward(self, x):
        x = self.layers(x)
        if x.dim() == 4:
            B, Ns, T, C = x.size()
            x = x.reshape(B * Ns, T, C)
            x, _ = self.gru(x)
            x = x[:, -1, :].reshape(B, Ns, -1)
        else:
            x, _ = self.gru(x)
            x = x[:, -1, :]
        return x

batch_size = 32
Ns, T, C = 100, 20, 25 # Sample dimensions
x = torch.randn(batch_size, Ns, T, C)
y = torch.randn(batch_size, Ns)
fe = FeatureExtractor()
e = fe(x)

class FactorEncoder(nn.Module):
    def __init__(self, M=30, K=25):
        super().__init__()
        self.portfolio_layer = nn.Linear(25, M)
        self.mapping_layer_mu = nn.Linear(M, K)
        self.mapping_layer_sigma = nn.Linear(M, K)

    def forward(self, y, e):
        ap = F.softmax(self.portfolio_layer(e), dim=-1)
        if len(e.shape) == 3:
            yp = torch.einsum("bn,bnm->bm", y, ap)
        else:
            yp = torch.einsum("n,nm->m", y, ap).unsqueeze(0)
        mu_post = self.mapping_layer_mu(yp).squeeze()
        sigma_post = F.softplus(self.mapping_layer_sigma(yp)).squeeze()
        return mu_post, sigma_post

class AlphaLayer(nn.Module):
    def __init__(self, H=25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(25, H),
            nn.LeakyReLU(),
            nn.Linear(H, 1))
        self.mapping_layer_sigma = nn.Linear(H, 1)

    def forward(self, e):
        h_alpha = self.layers[:-1](e)
        mu_alpha = self.layers[-1](h_alpha).squeeze()
        sigma_alpha = F.softplus(self.mapping_layer_sigma(h_alpha)).squeeze()
        return mu_alpha, sigma_alpha


















