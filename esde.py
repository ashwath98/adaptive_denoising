import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
from itertools import product
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import seaborn as sns
from IPython.display import Video
import os
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
from torch import Tensor
from abc import ABC, abstractmethod
import sys
sys.path.append('/home/ashwathshetty/DiffusionLearning/DiffusionEquations-Implement')
from torch.nn.functional import relu
from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm
import scipy.stats as st
from src.sde.implementations import VPSDE
from src.sde.base import ItoSDE
class ESDE():
    def __init__(self, T_max: float, rate: float = 1,sigma: float=1,dim: int=2):
        self.T_max = T_max
        self.rate = rate
        self.sigma=sigma
        self.dim = dim


  

    def sample_random_levels (self, length: int):

        return torch.distributions.Exponential(rate=self.rate).sample((length,))

    @staticmethod
    def _mult_first_dim(level,X):
        """
        Helper function to multiply one-dimensional noise level vector with tensor of
        arbitrary shape.
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:
            has same shape as X_0 - inputs X_0[i] multipled with t[i]
        """
        return level.view(-1,*[1]*(X.dim()-1))*X
    
    def run_forward(self, X_0: Tensor, level: Tensor, clip_factor: float = 0.01):
        """
        Function to evolve SDE forward in time from 0 to t<=self.T_max.
        Assume that conditional distribution is Gaussian
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:
            X_t: shape as X_0 - noised input
            noise: shape as X_0 - noise converting X_0 to X_t
            score: shape as X_0 - score of conditional distribution q_t|0(X_t|X_0)
        """
        noise = torch.randn(size=X_0.shape)
        X_t = (X_0+self.sigma*self._mult_first_dim(torch.sqrt(level),noise))
        if clip_factor is not None:
            level=torch.clip(level,min=clip_factor)
        score=-self.dim*self._mult_first_dim(1/(torch.sqrt(level)*self.sigma),noise)

        return X_t, noise,score

    def run_forward_random_times(self, X_0: Tensor):
        """Function to evolve SDE forward until random times."""
        levels = self.sample_random_levels(X_0.shape[0])
        X_t, noise,score= self.run_forward(X_0,levels)
        return X_t, noise,score,levels
