import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import torch
from celluloid import Camera
from typing import List, Tuple
from IPython.display import HTML
import seaborn as sns

class DiffusionVisualizer:
    def __init__(self, sde, model, device='cuda'):
        self.sde = sde
        self.model = model
        self.device = device
        
    def plot_data_manifold(self, 
                          data: torch.Tensor,
                          manifold_func=None,
                          n_samples: int = 1000,
                          title: str = "Data Manifold",
                          save_path: str = None):
        """Visualize the original data distribution and manifold."""
        plt.figure(figsize=(10, 10))
        
        # Plot samples
        plt.scatter(data[:n_samples, 0], data[:n_samples, 1], 
                   alpha=0.5, label='Data samples')
        
        # Plot manifold if provided
        if manifold_func is not None:
            x_ref = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            plt.plot(x_ref, manifold_func(x_ref), 'r-', 
                    label='True manifold', linewidth=2)
        
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def visualize_forward_process(self,
                                initial_samples: torch.Tensor,
                                n_steps: int = 8,
                                save_path: str = None):
        """Visualize the forward (noising) process."""
        time_vec = torch.linspace(0, 1, n_steps)**2
        X_0 = initial_samples.repeat(n_steps, 1, 1).transpose(1, 0)
        X_t, noise, score = self.sde.run_forward(X_0, time_vec)

        fig, axs = plt.subplots(1, n_steps, figsize=(6*n_steps, 6))
        for idx in range(n_steps):
            axs[idx].scatter(X_t[idx, :, 0], X_t[idx, :, 1], alpha=0.5)
            axs[idx].set_xlim(-2.0, 2.0)
            axs[idx].set_ylim(-2.0, 2.0)
            axs[idx].set_title(f"t = {time_vec[idx]:.2f}")
            axs[idx].grid(True, alpha=0.3)
        
        plt.suptitle("Forward Process: Data → Noise", y=1.05)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return X_t, noise, score

    def create_diffusion_animation(self,
                                 trajectory: torch.Tensor,
                                 manifold_func=None,
                                 save_path: str = "diffusion_process.gif"):
        """Create animation of the diffusion process."""
        fig, ax = plt.subplots(figsize=(10, 10))
        camera = Camera(fig)
        
        # Plot reference points if manifold function is provided
        if manifold_func is not None:
            x_ref = np.linspace(-2, 2, 100)
            y_ref = manifold_func(x_ref)
        
        for idx in range(trajectory.shape[0]):
            if manifold_func is not None:
                ax.plot(x_ref, y_ref, 'r-', alpha=0.5)
            
            ax.scatter(trajectory[idx, :, 0], 
                      trajectory[idx, :, 1],
                      color='blue', alpha=0.5)
            
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.grid(True, alpha=0.3)
            camera.snap()
        
        animation = camera.animate()
        animation.save(save_path)
        plt.close()
        
        return HTML(f'<img src="{save_path}">')

    def visualize_backward_process(self,
                                 n_samples: int = 1000,
                                 n_steps: int = 10,
                                 manifold_func=None,
                                 save_path: str = None):
        """Visualize the backward (denoising) process."""
        # Generate random initial noise
        x_start = torch.randn(n_samples, 2)
        
        # Run backward process
        self.model.eval()
        with torch.no_grad():
            trajectory = self.run_backward_process(x_start, n_steps)
        
        # Create animation
        if save_path:
            return self.create_diffusion_animation(
                trajectory, 
                manifold_func=manifold_func,
                save_path=save_path
            )
            
    def run_backward_process(self, 
                           x_start: torch.Tensor,
                           n_steps: int = 10) -> torch.Tensor:
        """Run the backward process and return the trajectory."""
        self.model = self.model.to(self.device)
        n_samples = x_start.shape[0]
        
        # Time discretization
        time_grid = torch.linspace(self.sde.T_max, 0, n_steps)
        dt = torch.abs(time_grid[0] - time_grid[1])
        
        # Initialize trajectory storage
        trajectory = [x_start]
        
        for idx, t in enumerate(time_grid[:-1]):
            x = trajectory[-1]
            time_vec = torch.full((n_samples,), t)
            
            # Get model prediction
            with torch.no_grad():
                pred = self.model(x.to(self.device), 
                                time_vec.to(self.device)).cpu()
            
            # Compute drift and diffusion terms
            drift = self.sde.f_drift(x, time_vec)
            diffusion = self.sde.g_random(x, time_vec)
            
            # Euler-Maruyama step
            noise = torch.randn_like(x)
            dx = (-drift - diffusion**2 * pred) * dt + diffusion * np.sqrt(dt) * noise
            
            # Update
            x_new = x + dx
            trajectory.append(x_new)
        
        return torch.stack(trajectory)

    def plot_samples_comparison(self,
                              real_samples: torch.Tensor,
                              generated_samples: torch.Tensor,
                              manifold_func=None,
                              save_path: str = None):
        """Compare real and generated samples."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot real samples
        ax1.scatter(real_samples[:, 0], real_samples[:, 1], 
                   alpha=0.5, label='Real samples')
        if manifold_func is not None:
            x_ref = np.linspace(-2, 2, 100)
            ax1.plot(x_ref, manifold_func(x_ref), 'r-', 
                    label='True manifold', linewidth=2)
        ax1.set_title("Real Data Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot generated samples
        ax2.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                   alpha=0.5, label='Generated samples')
        if manifold_func is not None:
            ax2.plot(x_ref, manifold_func(x_ref), 'r-', 
                    label='True manifold', linewidth=2)
        ax2.set_title("Generated Data Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 