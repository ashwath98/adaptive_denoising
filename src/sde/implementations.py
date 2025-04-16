from .base import ItoSDE
import torch
from torch import Tensor

class VPSDE(ItoSDE):
    def __init__(self,T_max: float, beta_min: float = 0.0, beta_max: float = 1.0):
       self.T_max = T_max
       self.beta_min = beta_min
       self.beta_max = beta_max

    def _beta_derivative(self, t: Tensor):
        return self.beta_min+(self.beta_max - self.beta_min)*t

    def _beta(self, t: Tensor):
        return (self.beta_min*t)+0.5*(self.beta_max - self.beta_min)*(t**2)

    def cond_exp(self, X_0: Tensor, t: Tensor):
        """
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:

        """
        assert len(t.shape) == 1, "Time must be 1-dimensional."
        assert t.shape[0] == X_0.shape[0]
        beta_t = self._beta(t)
        cond_exp_t = torch.exp(-0.5*beta_t)
        return self._mult_first_dim(cond_exp_t,X_0)

    def cond_var(self, X_0: Tensor, t: Tensor):
        """
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:

        """
        assert len(t.shape) == 1, "Time must be 1-dimensional."
        #assert t.shape[0] == X_0.shape[0]
        beta_t = self._beta(t)
        cond_var_t = 1-torch.exp(-beta_t)
        return cond_var_t


    def f_drift(self, X_t: Tensor, t: Tensor):
        """
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:

        """
        assert len(t.shape) == 1, "Time must be 1-dimensional."
        assert t.shape[0] == X_t.shape[0]
        deriv_beta_t = self._beta_derivative(t)
        return -0.5*self._mult_first_dim(deriv_beta_t,X_t)

    def g_random(self, t: Tensor):
        """
        Inputs:
            X_0: shape (n,*,*,...,*)
            t: shape (n)
        Outputs:

        """
        assert len(t.shape) == 1, "Time must be 1-dimensional."
        deriv_beta_t = self._beta_derivative(t)
        return torch.sqrt(deriv_beta_t)
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
