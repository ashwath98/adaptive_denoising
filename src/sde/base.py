from abc import ABC, abstractmethod
import torch
from torch import Tensor

class ItoSDE(ABC):
    def __init__(self, T_max: float):
        self.T_max = T_max

    @abstractmethod
    def cond_exp(self, X_0: Tensor, t: Tensor):
        pass

    @abstractmethod
    def cond_var(self, X_0: Tensor, t: Tensor):
        pass

    @abstractmethod
    def f_drift(self, X_t: Tensor, t: Tensor):
        pass

    @abstractmethod
    def g_random(self, X_t: Tensor, t: Tensor):
        pass

    def cond_std(self, X_0: Tensor, t: Tensor):
        """Conditional standard deviation. Square root of self.cond_var."""
        return torch.sqrt(self.cond_var(X_0=X_0,t=t))

    def sample_random_times(self, length: int):
        """Sample 'length' time points uniformly in interval [0,T]"""
        return torch.rand(size=(length,))*self.T_max

    @staticmethod
    def _mult_first_dim(t,X):
        """
        Helper function to multiply one-dimensional time vector with tensor of
        arbitrary shape.
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:
            has same shape as X_0 - inputs X_0[i] multipled with t[i]
        """
        return t.view(-1,*[1]*(X.dim()-1))*X

    def run_forward(self, X_0: Tensor, t: Tensor, clip_factor: float = 0.01):
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
        cond_std = self.cond_std(X_0,t)
        cond_exp = self.cond_exp(X_0,t)
        X_t = self._mult_first_dim(cond_std,noise)+cond_exp
        if clip_factor is not None:
            cond_std = torch.clip(cond_std,min=clip_factor)
        score = -self._mult_first_dim(1/cond_std,noise)

        return X_t, noise, score

    def run_forward_random_times(self, X_0: Tensor):
        """Function to evolve SDE forward until random times."""
        t = self.sample_random_times(X_0.shape[0])
        X_t, noise, score = self.run_forward(X_0,t)
        return X_t, noise, score, t