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
from backward_adaptive import run_backwards, run_backwards_adaptive
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
    output, time_grid = run_backwards_adaptive(
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
    #model = torch.compile(model)
    TRAIN_SCORE = True
    setting="rate_1_orig_train"
    # Load model weights
    rate=0.5
    #path=f'/home/ashwathshetty/DiffusionLearning/DiffusionEquations-Implement/checkpoints/final_model_score_1_{rate}_1_1.pt'
    path='ckpt_res_test_my_model.pth'
    model_state_dict = torch.load(path)
    model.load_state_dict(model_state_dict)
    # Initialize SDE
    sde = ESDE(T_max=1, dim=1, rate=1)
    
    # Load MNIST data
    image_size = 28
    classes_by_index  = np.arange(0,10).astype('str')

    transform = transforms.Compose([transforms.Resize(image_size),\
                                transforms.ToTensor(),
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
    use_dataset_samples =   False
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
        n_steps=2000, 
        n_samples_to_show=4,
        plot_interval=200,
        save_dir='results/backward_process'
    )

if __name__ == "__main__":
    main() 