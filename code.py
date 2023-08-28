#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:14:39 2023

@author: ruohang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, proj_dim=20, H=25):
        super(FeatureExtractor, self).__init__()
        self.proj = nn.Linear(25, proj_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(proj_dim, H, batch_first=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.leakyrelu(x)
        if len(x.shape) == 4:
            B, Ns, T, C = x.shape
            x = x.view(B * Ns, T, C)
            x, _ = self.gru(x)
            x = x[:, -1, :]
            x = x.view(B, Ns, -1)
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
        super(FactorEncoder, self).__init__()
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
        super(AlphaLayer, self).__init__()
        self.proj = nn.Linear(25, H)
        self.leakyrelu = nn.LeakyReLU()
        self.mapping_layer_mu = nn.Linear(H, 1)
        self.mapping_layer_sigma = nn.Linear(H, 1)

    def forward(self, e):
        h_alpha = self.proj(e)
        h_alpha = self.leakyrelu(h_alpha)
        mu_alpha = self.mapping_layer_mu(h_alpha).squeeze()
        sigma_alpha = F.softplus(self.mapping_layer_sigma(h_alpha)).squeeze()
        return mu_alpha, sigma_alpha

class FactorDecoder(nn.Module):
    def __init__(self, H=25, K=25):
        super(FactorDecoder, self).__init__()
        self.alpha_layer = AlphaLayer(H)
        self.beta_layer = nn.Linear(25, K)

    def forward(self, z, e):
        mu_z, sigma_z = z[0].unsqueeze(-1), z[1].unsqueeze(-1)
        mu_alpha, sigma_alpha = self.alpha_layer(e)
        mu_alpha = mu_alpha.unsqueeze(-1)
        sigma_alpha = sigma_alpha.unsqueeze(-1)
        beta = self.beta_layer(e)
        mu_y = (mu_alpha + beta @ mu_z).squeeze(-1)
        sigma_y = torch.sqrt((sigma_alpha ** 2) + (beta ** 2) @ (sigma_z ** 2)).squeeze(-1)
        return mu_y, sigma_y


class FactorPredictor(nn.Module):
    def __init__(self, H: int = 25, K: int = 25):
        super(FactorPredictor, self).__init__()
        self.w_key = nn.Parameter(torch.randn(K, 1))
        self.w_value = nn.Parameter(torch.randn(K, 1))
        self.q = nn.Parameter(torch.randn(H,))
        
        self.mapping_layer_mu = nn.Linear(H, 1, bias=False)
        self.mapping_layer_sigma = nn.Linear(H, 1, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, e):
        if len(e.shape) == 2:
            e_ = e.unsqueeze(0)
            k = torch.einsum("kd,dnh->knh", self.w_key, e_)
            v = torch.einsum("kd,dnh->knh", self.w_value, e_)
            a_att = torch.einsum("h,knh->kn", self.q, k) / (self.q.norm(p=2) * k.norm(p=2, dim=-1))
            a_att = torch.clamp(a_att, min=0.)
            a_att = a_att / torch.clamp(a_att.sum(dim=-1, keepdim=True), min=1e-6)
            h_muti = torch.einsum("kn,knh->kh", a_att, v)
            mu_prior = self.mapping_layer_mu(h_muti).squeeze(-1)
            sigma_prior = self.softplus(self.mapping_layer_sigma(h_muti)).squeeze(-1)
        else:
            e_ = e.unsqueeze(1)
            k = torch.einsum("kd,bdnh->bknh", self.w_key, e_)
            v = torch.einsum("kd,bdnh->bknh", self.w_value, e_)
            q_norm = self.q.norm(p=2).unsqueeze(-1)
            k_norm = k.norm(p=2, dim=-1)
            a_att = torch.einsum("h,bknh->bkn", self.q, k) / (q_norm * k_norm)
            a_att = torch.clamp(a_att, min=0.)
            a_att = a_att / torch.clamp(a_att.sum(dim=-1, keepdim=True), min=1e-6)
            h_muti = torch.einsum("bkn,bknh->bkh", a_att, v)
            mu_prior = self.mapping_layer_mu(h_muti).squeeze(-1)
            sigma_prior = self.softplus(self.mapping_layer_sigma(h_muti)).squeeze(-1)
        
        return mu_prior, sigma_prior


class FactorVAE(nn.Module):
    def __init__(self, M=30, H=25, K=25, proj_dim=20):
        super(FactorVAE, self).__init__()
        self.feature_extractor = FeatureExtractor(proj_dim=proj_dim, H=H)
        self.factor_encoder = FactorEncoder(M=M, K=K)
        self.factor_predictor = FactorPredictor(H=H, K=K)
        self.factor_decoder = FactorDecoder(H=H, K=K)

    def forward(self, x, y=None, training=False):
        # x.shape should be (batch_size, Ns, T, C)
        if training:
            if y is None:
                raise ValueError("`y` must be stock future return!")
            e = self.feature_extractor(x)
            # e.shape should be (batch_size, Ns, H)
            z_post = self.factor_encoder(y, e)
            z_prior = self.factor_predictor(e)
            y_rec = self.factor_decoder(z_post, e)
            y_pred = self.factor_decoder(z_prior, e)
            return z_post, z_prior, y_rec, y_pred
        else:
            e = self.feature_extractor(x)
            z_prior = self.factor_predictor(e)
            y_pred = self.factor_decoder(z_prior, e)
            return y_pred

# Example usage
model = FactorVAE()

if model.training:
    z_post, z_prior, y_rec, y_pred = model(x, y, training=True)
else:
    y_pred = model(x)




















