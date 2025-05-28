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
from tqdm import tqdm
from esde import ESDE
from model import Unet
from utils.visualization import plot_score_grid
# First, let's import all the model classes
# ... [Keep all the model class definitions from the notebook: ESDE, Unet, ChannelShuffle, etc.] ...
def run_backwards(model: torch.nn.Module, sde: ItoSDE, x_start: Tensor, device, train_score, n_steps: int = 10, 
                 plot_evolution: bool = True, clip_max: float = 1.0, clip_min: float = -1.0, dim_scale: float = 1.0, 
                 div_threshold: float = 15, random_drift: bool = True, score_threshold: float = 1e-4, 
                 individual_stopping: bool = False, **kwargs):
    """Function to run stochastic differential equation. We assume a deterministic initial distribution p_0."""
    model = model.to(device)
    model.eval()
    
    # Move initial tensors to device once
    x_start = x_start.to(device)
    n_traj = x_start.shape[0]
    
    # Pre-compute and move to device
    time_grid = torch.linspace(sde.T_max, 0, n_steps).to(device)
    step_size = torch.tensor(sde.sigma**2 / n_steps).to(device)
    
    # Generate noise on device directly
    noise = torch.randn(size=(n_steps,*list(x_start.shape)), device=device)
    random_drift_grid_sample = torch.sqrt(step_size) * noise
    del noise  # Free memory

    # Use a deque instead of list for trajectory storage
    from collections import deque
    x_traj = deque([x_start], maxlen=2)  # Only keep current and previous step
    all_outputs = []  # Store only necessary outputs
    
    # Initialize active mask only if individual stopping is enabled
    if individual_stopping:
        active_mask = torch.ones(n_traj, dtype=torch.bool, device=device)
    
    if plot_evolution:
        fig, axs = plt.subplots(1, len(time_grid), figsize=(6*len(time_grid), 6))
    
    with torch.no_grad():  # Disable gradient computation
        for idx, step_number in tqdm(enumerate(time_grid), total=n_steps):
            x = x_traj[-1]  # Get last step
            random_drift_sample = random_drift_grid_sample[idx]
            
            # Compute model estimate
            model_estimate = model(x, None)
            
            if train_score:
                score_estimates = model_estimate * dim_scale
            else:
                level = torch.tensor(1.0, device=device)
                score_estimates = -sde._mult_first_dim(1 / (torch.sqrt(level) * sde.sigma), model_estimate) * sde.dim

            if individual_stopping:
                # Check score norm for each sample individually
                score_norms = torch.norm(score_estimates.view(score_estimates.shape[0], -1), dim=1)
                active_mask = active_mask & (score_norms >= score_threshold)
                
                # If no samples are active, stop early
                if not active_mask.any():
                    print(f"All samples converged at step {idx}")
                    break
                
            correction_term = step_size * score_estimates
            if random_drift:
                change = correction_term + random_drift_sample
            else:
                change = correction_term
                
            # Apply changes based on stopping mode
            if individual_stopping:
                next_step = x.clone()
                next_step[active_mask] += change[active_mask]
            else:
                next_step = x + change
            
            if plot_evolution:
                # Move to CPU only if plotting
                cpu_next = next_step.cpu()
                cpu_change = change.cpu()
                axs[idx].scatter(cpu_next[:,0], cpu_next[:,1])
                axs[idx].quiver(cpu_next[:,0], cpu_next[:,1], cpu_change[:,0], cpu_change[:,1])
                axs[idx].set_xlim(-2.0, 2.0)
                axs[idx].set_ylim(-2.0, 2.0)
                axs[idx].set_title(f"Step={idx}")
            
            # Store step and free memory
            x_traj.append(next_step)
            all_outputs.append(next_step.clone())
            
            # Print number of active samples periodically if individual stopping is enabled
            if individual_stopping and idx % 10 == 0:
                print(f"Step {idx}: {active_mask.sum().item()}/{n_traj} samples still active")
            
            # Clear unused tensors
            del change

    output = torch.stack(all_outputs)
    return output, time_grid
def load_mnist_data(image_size=28, batch_size=128):
    """Load and prepare MNIST dataset."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader

def plot_with_score_overlay(image, score, title=None):
    """Plot image with score vectors overlaid."""
    plt.figure(figsize=(10, 10))
    
    # Plot the image
    plt.imshow(image.squeeze(), cmap='gray')
    
    # Create a grid for the score vectors
    h, w = image.shape[-2:]
    step = 4  # Show score every 4 pixels
    y, x = np.mgrid[0:h:step, 0:w:step]
    
    # Get score vectors at grid points
    score_y = score[0, y, x].cpu().numpy()
    score_x = score[1, y, x].cpu().numpy() if score.shape[0] > 1 else np.zeros_like(score_y)
    
    # Plot score vectors
    plt.quiver(x, y, score_x, score_y, color='red', scale=50)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def analyze_score_estimates(model, sde, trainset, n_grid_points=16, device='cuda', settings_name='default'):
    """
    Analyze score estimates at different noise levels for a single image.
    """
    model.eval()
    
    # Exactly match your implementation
    time_vec = torch.linspace(0, 2, n_grid_points)**2
    X_0 = torch.stack([trainset.__getitem__(23420)[0].unsqueeze(0).squeeze()]*n_grid_points)
    X_t, noise, true_score = sde.run_forward(X_0, torch.linspace(0, 1.0, n_grid_points)**2)

    # Get model's score estimate
    with torch.no_grad():
        estimated_score = model(X_t.unsqueeze(1), None).squeeze(1)
    
    # Plot and save using the visualization utility
    save_path = plot_score_grid(
        X_t, 
        true_score, 
        estimated_score, 
        settings_name=settings_name,
        save_dir='results/score_grids'
    )
    
    return {
        'noised_images': X_t,
        'true_score': true_score,
        'estimated_score': estimated_score,
        'time_vec': time_vec,
        'save_path': save_path
    }

def visualize_backward_process(model, sde, x_start, n_steps=100, n_samples_to_show=8, plot_interval=10, save_dir='results/backward_process'):
    """
    Visualize the backward diffusion process and score estimates for selected samples.
    Args:
        model: The diffusion model
        sde: The SDE instance
        x_start: Starting noise tensor
        n_steps: Number of diffusion steps
        n_samples_to_show: Number of samples to visualize
        plot_interval: Show every nth step
        save_dir: Directory to save the visualization plots
    """
    model.eval()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Run backward process
    output, time_grid = run_backwards(
        model, sde, x_start=x_start, 
        n_steps=n_steps, device='cpu', 
        train_score=True, plot_evolution=False
    )
    
    # Select steps to visualize
    steps_to_show = range(0, n_steps, plot_interval)
    n_cols = len(steps_to_show)
    
    with torch.no_grad():
        for sample_idx in range(min(n_samples_to_show, x_start.shape[0])):
            fig, axes = plt.subplots(2, n_cols, figsize=(20, 8))
            fig.suptitle(f'Sample {sample_idx + 1} Evolution')
            
            for col, step in enumerate(steps_to_show):
                # Plot the image
                img = output[step][sample_idx].cpu()
                score = model(img.unsqueeze(0), None).squeeze().cpu()
                score_norm = torch.norm(score.view(-1))
                
                # Image with score norm in title
                axes[0, col].imshow(img.squeeze(), cmap='gray')
                axes[0, col].axis('off')
                axes[0, col].set_title(f'Step {step}\nScore Norm: {score_norm:.2f}')
                
                # Score estimate
                axes[1, col].imshow(score.squeeze(), cmap='RdBu')
                axes[1, col].axis('off')
                axes[1, col].set_title('Score Estimate')
            
            plt.tight_layout()
            
            # Save the figure
            save_path = os.path.join(save_dir, f'sample_{sample_idx}_evolution.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

            # Save individual timesteps
            for col, step in enumerate(steps_to_show):
                img = output[step][sample_idx].cpu()
                score = model(img.unsqueeze(0), None).squeeze().cpu()
                score_norm = torch.norm(score.view(-1))
                
                # Create individual plot for this timestep
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                
                # Image with score norm in title
                ax[0].imshow(img.squeeze(), cmap='gray')
                ax[0].axis('off')
                ax[0].set_title(f'Image at Step {step}\nScore Norm: {score_norm:.2f}')
                
                # Score
                ax[1].imshow(score.squeeze(), cmap='RdBu')
                ax[1].axis('off')
                ax[1].set_title('Score Estimate')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'sample_{sample_idx}_step_{step}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_TIMESTEPS = 100
    n_channels = 1
    image_size = 28
    model = Unet(base_dim=image_size, in_channels=n_channels, out_channels=n_channels, time_embedding_dim=256, timesteps=N_TIMESTEPS, dim_mults=[2, 4], temp=100.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.compile(model)
    TRAIN_SCORE = True
    setting="rate_1_orig_train"
    # Load model weights
    rate=1  
    model_state_dict = torch.load(f'/home/ashwathshetty/DiffusionLearning/DiffusionEquations-Implement/checkpoints/final_model_score_1_{rate}_1_1.pt')
    model.load_state_dict(model_state_dict)
    # Initialize SDE
    sde = ESDE(T_max=1, dim=1, rate=1)
    
    # Load MNIST data
    image_size = 28
    classes_by_index  = np.arange(0,10).astype('str')

    transform = transforms.Compose([transforms.Resize(image_size),\
                                transforms.ToTensor(),\
                                transforms.Normalize([0.5],[0.5])]) #Normalize to -1,1
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)

    batch_size = 128
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    
    # Get a sample image
    sample_image = next(iter(trainloader))[0][0]
    sample_image = sample_image
    # Analyze score estimates with specific settings name
    settings_name = f'mnist_temp{model.temp}_timesteps{N_TIMESTEPS}'
    results = analyze_score_estimates(
        model, 
        sde,
        trainset,  # Pass the entire trainset instead of a single image
        n_grid_points=16,
        settings_name=settings_name
    )
    print(f"Plot saved to: {results['save_path']}")
    model = model.to(device)
    torch._functorch.config.donated_buffer = False
    use_dataset_samples = True
    if use_dataset_samples:
        # Get some samples from the dataset
        dataiter = iter(trainloader)
        samples, labels = next(dataiter)
        samples = samples[:4] # Get 4 samples
        
        # Create noised versions at different levels
        noise_levels = [1.0]  # Different noise levels
        x_start = []
        
        # Apply forward process to create noised samples
        for level in noise_levels:
            # Use the SDE to generate noised samples at specific level
            noised_samples, _, _ = sde.run_forward(samples, torch.ones(samples.shape[0]) * level)
            x_start.append(noised_samples)
        
        # Combine all noised samples
        x_start = torch.cat(x_start, dim=0)
        num_samples = x_start.shape[0]
        save_dir = 'results/backward_process_dataset_samples'
    else:
        # Original approach: use random noise
        x_start = torch.clip(torch.randn(size=next(iter(trainloader))[0].shape)[:8], -1.0, 1.0)
        save_dir = 'results/backward_process'
   #x_start = torch.clip(torch.randn(size=next(iter(trainloader))[0].shape)[:8], -1.0, 1.0)
    visualize_backward_process(
        model, sde, x_start, 
        n_steps=1000, 
        n_samples_to_show=4,
        plot_interval=100,
        save_dir='results/backward_process'
    )

if __name__ == "__main__":
    main() 