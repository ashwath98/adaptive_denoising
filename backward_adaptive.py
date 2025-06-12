import torch
import matplotlib.pyplot as plt
from src.sde.implementations import ItoSDE
from src.models.score_model import FullConnectedScoreModel
from tqdm import tqdm
from torch import Tensor
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

def run_backwards_adaptive(model: torch.nn.Module, sde: ItoSDE, x_start: Tensor, device, train_score, n_steps: int = 10, 
                 plot_evolution: bool = True, clip_max: float = 1.0, clip_min: float = -1.0, dim_scale: float = 1.0, 
                 div_threshold: float = 15, random_drift: bool = True, score_threshold: float = 1e-4, 
                 individual_stopping: bool = False, base_step_size: float = 1.0, min_step_size: float = 1e-4, 
                 noise_scale: float = 1.0, **kwargs):
    """Function to run stochastic differential equation with adaptive step sizing based on score norm."""
    model = model.to(device)
    model.eval()
    
    # Move initial tensors to device once
    x_start = x_start.to(device)
    n_traj = x_start.shape[0]
    
    # Pre-compute and move to device
    time_grid = torch.linspace(sde.T_max, 0, n_steps).to(device)
    base_step = torch.tensor(base_step_size , device=device)  # Scale by n_steps like in original
    
    # Pre-generate base noise like in run_backwards to avoid shape issues
    base_noise = torch.randn(size=(n_steps, *list(x_start.shape)), device=device)
    
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
            
            # Compute model estimate
            model_estimate = model(x, None)
            
            if train_score:
                score_estimates = model_estimate * dim_scale
            else:
                level = torch.tensor(1.0, device=device)
                score_estimates = -sde._mult_first_dim(1 / (torch.sqrt(level) * sde.sigma), model_estimate) * sde.dim

            # Compute score norm for adaptive step sizing
            score_norms = torch.norm(score_estimates.view(score_estimates.shape[0], -1), dim=1)
            score_norms = torch.clamp(score_norms, min=1e-5)
            
            # Compute adaptive step size (inversely proportional to score norm)
            adaptive_step_size = base_step / (1.0 + score_norms)
            adaptive_step_size = torch.clamp(adaptive_step_size, min=min_step_size)
            
            if individual_stopping:
                # Check score norm for each sample individually
                active_mask = active_mask & (score_norms >= score_threshold)
                
                # If no samples are active, stop early
                if not active_mask.any():
                    print(f"All samples converged at step {idx}")
                    break
                
            # Apply adaptive step size to correction term
            correction_term = score_estimates * adaptive_step_size.view(-1, 1, 1, 1)
            
            if random_drift:
                # Use pre-generated noise but scale it adaptively
                step_noise = base_noise[idx]  # Get pre-generated noise for this step
                # Scale the noise by the adaptive step size
                adaptive_noise = step_noise * torch.sqrt(adaptive_step_size).view(-1, 1, 1, 1) * noise_scale
                change = correction_term + adaptive_noise
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