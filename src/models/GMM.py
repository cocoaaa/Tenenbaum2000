import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import pyro
from pyro.distributions import MultivariateNormal, Normal
from typing import List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union

class GMM():
    """Mixture of Gaussians (MoG), aka. Gaussian Mixture Model (GMM) Implementation.

    Model parameters:
    - pi (vector of length K): [pi_1, ... pi_K]
    - means (matrix of shape (K, dim_x): Each kth row is a vector of length dim_x that represents a center of
     the kth Gaussian component distribution
    - covs (3dim tensor of shape (K, dim_x, dim_x): Each kth matrix is a covariance matrix for the kth gaussian component
    distribution



    Learning algorithm: EM method applied to Guassian Mixture Model. See PRML Ch 9.2
    - Estep: choose the posterior probability function of Z|X; \theta to be
    the one that lower-bounds the ELBO objective function tightest, given the observed
    data X and current iteration's model parameter values.

    - Mstep: Compute the new values for the model parameters s.t. the new values maximizes
    the ELBO, given the observed data and the posterior probabilities (ie. the soft mixture assignment
    for each data point)


    """
    def __init__(self, n_mixtures: int, n_obs: int, dim_x: int,
                 dtype=None):
        self.K = n_mixtures
        self.N = n_obs
        self.D = dim_x
        self.dtype = torch.float32 if dtype is None else dtype

        # Initialize the model parameters as random initial values
        self.pi = torch.ones(self.K, dtype=self.dtype).div(self.K) # Uniform prior on mixing coefficients
        self.means = torch.randn(self.K, self.D, dtype=self.dtype)
        self.covs = torch.tensor(self.K, self.D, self.D, dtype=self.dtype)
        for k in range(self.K):
            self.covs[k] = torch.ones(2, dtype=self.dtype) # initialize with identity mtx

        # Register model parameters as in a dictionary
        self.params = {'pi': self.pi,
                       'means': self.means,
                       'covs': self.covs}

        # Initialize posterior probabilities for each data point: R
        self.R = torch.zeros(self.N, self.K, dtype=self.dtype)


    def info(self):
        for name, param in self.params.items():
            print(f"==={name}===")
            print(param.numpy())
        print(f"===posterior probs===")
        print(self.R)

    def get_gaussian(self, mean: torch.Tensor, cov: torch.Tensor):
        return torch.distri
    def compute_loglikelihood(self, X: Union[torch.Tensor, np.ndarray]):
        assert torch.isclose(X.shape[-1], self.D)

