# FactorVAE: Predicting Cross-Sectional Stock Returns Using Variational Autoencoders

## Introduction
FactorVAE is a machine learning model designed to analyze the most influential factors that affect asset prices, specifically stock prices, to predict future returns. The model is built on top of the Variational Autoencoder (VAE) framework and aims to identify dynamic latent factors that influence stock returns. This repository contains a PyTorch implementation of the FactorVAE model, as proposed in the paper "FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns".

## Structure
The model consists of several key components:
- **Feature Extraction**: To reduce the dimensionality of the raw features.
- **Factor Encoder**: To encode the observed data into latent factors.
- **Factor Decoder**: To decode latent factors into predicted values.
- **Factor Predictor**: To generate latent factors that capture stock return dynamics.

**Note**: The training steps and loss function are currently under development.

## Requirements
- Python 3.x
- PyTorch
- NumPy

## Installation
git clone https://github.com/harrishe1999/FactorVAE.git
cd FactorVAE


## Usage
*This section will be updated with instructions for training and inference once those features are implemented.*

## Example
\```python
from FactorVAE import FactorVAE  # Make sure to import your model correctly

## Initialize model
model = FactorVAE()

## Model Training (To be implemented)
\```


## Acknowledgments
- Paper: "FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns"
