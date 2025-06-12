import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
import functools
# Marginal Probability Standard Deviation Function
def marginal_prob_std(t, sigma):
    """
    Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Parameters:
    - t: A vector of time steps.
    - sigma: The $\sigma$ in our SDE.

    Returns:
    - The standard deviation.
    """
    # Convert time steps to a PyTorch tensor
    t = t.clone().detach().to(device)
    
    # Calculate and return the standard deviation based on the given formula
    # return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
    return torch.sqrt(t)

# Factor for changing X_0
def start_factor(t, sigma):
    """
    Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Parameters:
    - t: A vector of time steps.
    - sigma: The $\sigma$ in our SDE.

    Returns:
    - The standard deviation.
    """
    # Convert time steps to a PyTorch tensor
    t = t.clone().detach().to(device)
    
    # Calculate and return the factor for X_0
    # return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
    return (torch.exp( sigma * t))

# %%
def diffusion_coeff(t, sigmar):
    """
    Compute the diffusion coefficient of our SDE.

    Parameters:
    - t: A vector of time steps.
    - sigma: The $\sigma$ in our SDE.

    Returns:
    - The vector of diffusion coefficients.
    """
    # Calculate and return the diffusion coefficients based on the given formula
    #  return torch.tensor(sigma**t, device=device)
    return torch.tensor(sigmar, device=device)
# Ensure output folder exists
os.makedirs("temp", exist_ok=True)

def save_image(x, step):
    image = x[0].cpu().clamp(0, 1).squeeze()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig(f"temp/sample_step_{step}.png")
    plt.close()

def replace_left_half_with_noise(image, noise_std=1.0):
    B, C, H, W = image.shape
    noise = torch.randn_like(image) * noise_std
    image[:, :, :, :W] = noise[:, :, :, :W]
    return image
r =  1
d = 28**2
sigma =  2*r/d
sigmar=1

# marginal probability standard
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

# start factor
start_factor_fn = functools.partial(start_factor, sigma=sigma)

# diffusion coefficient
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigmar=sigmar)


def Euler_Maruyama_mine(score_model, marginal_prob_std, diffusion_coeff,
                           batch_size=64, x_shape=(1, 28, 28), device='cpu', eps=0.01,
                           t_threshold=0.01, T=2.0, num_steps=500, max_iterations=10000,
                           base_step=0.02, noise_scale=1.0, noise_std=0.5, y=None,
                           score_threshold=300, min_step_size: float = 1e-4, my_mode=False):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    random_index = torch.randint(0, len(mnist_data), (batch_size,))
    images, _ = zip(*[mnist_data[i] for i in random_index])
    x = torch.stack(images).to(device)
    
    save_image(x, -1)
    x = replace_left_half_with_noise(x, noise_std)
    with torch.no_grad():
        active_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        iteration_counter = torch.zeros(batch_size, device=device)
        stop_reason = torch.zeros(batch_size, device=device)  # 0 = not stopped, 1 = time, 2 = score

        i = 0
        while active_mask.any() and i < max_iterations:
            i += 1

            # Only process active images
            x_active = x[active_mask]
            if not my_mode:
                score_estimates = score_model(x_active, y=y)
            else:
                score_estimates = score_model(x_active)
            score_norms = torch.norm(score_estimates.view(score_estimates.shape[0], -1), dim=1)
            score_norms = torch.clamp(score_norms, min=1e-5)

            # Compute adaptive step size
            adaptive_step_size = base_step / (1.0 + score_norms)
            adaptive_step_size = torch.clamp(adaptive_step_size, min=min_step_size)

            # Apply adaptive step size to correction term
            correction_term = score_estimates * adaptive_step_size.view(-1, 1, 1, 1)
            random_drift=torch.sqrt(adaptive_step_size).view(-1, 1, 1, 1) * noise_scale * torch.randn_like(x_active)
            mean_x = x_active + correction_term+random_drift
            x[active_mask] = mean_x

            # Determine which samples are done
            finished_by_score = score_norms > score_threshold       
            global_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            finished_indices_score = global_indices[finished_by_score]
            iteration_counter[finished_indices_score] = i
            stop_reason[finished_indices_score] = 2  # 2 = score
            active_mask[finished_indices_score] = False

            # Optional logging
            #if finished_indices_score.numel() > 0:
            #    print(f"Step {i}: {finished_indices_score.numel()} samples stopped due to high score norm (> {score_threshold})")

            if i % 500 == 0:
                save_image(x, i)

        # Final stats
        fulfilled_count = (~active_mask).sum().item()
        fulfilled_ratio = fulfilled_count / batch_size
        finished_iters = iteration_counter[iteration_counter > 0]
        mean_steps = finished_iters.mean().item() if finished_iters.numel() > 0 else float('nan')

        score_stops = (stop_reason == 2).sum().item()
        time_stops = (stop_reason == 1).sum().item()

        print(f"\nFinished after {i} iterations")
        print(f"Percentage of images that satisfied stopping condition: {fulfilled_ratio * 100:.2f}%")
        print(f"Average number of steps until stopping (only for fulfilled images): {mean_steps:.2f}")
        print(f"Stopped due to score norm: {score_stops}/{batch_size} ({100 * score_stops / batch_size:.2f}%)")
        print(f"Stopped due to estimated_t: {time_stops}/{batch_size} ({100 * time_stops / batch_size:.2f}%)")

    return x


            # Euler-Maruyama update mit dynamischem Rauschen
def Euler_Maruyama_sampler(score_model, marginal_prob_std, diffusion_coeff,
                           batch_size=64, x_shape=(1, 28, 28), device='cpu', eps=0.01,
                           t_threshold=0.01, T=2.0, num_steps=500, max_iterations=10000,
                           base_step=0.02, noise_scale=1.0, noise_std=0.5, y=None,
                           score_threshold=300):
    # Load MNIST and prepare noisy input
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    random_index = torch.randint(0, len(mnist_data), (batch_size,))
    images, _ = zip(*[mnist_data[i] for i in random_index])
    x = torch.stack(images).to(device)

    save_image(x, -1)
    x = replace_left_half_with_noise(x, noise_std)

    with torch.no_grad():
        active_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        iteration_counter = torch.zeros(batch_size, device=device)
        stop_reason = torch.zeros(batch_size, device=device)  # 0 = not stopped, 1 = time, 2 = score

        i = 0

        while active_mask.any() and i < max_iterations:
            i += 1

            # Only process active images
            x_active = x[active_mask]

            # Predict score and estimated time
            score = score_model(x_active, y=y)
            #estimated_t = time_predictor_model(x_active, y=y).squeeze()

            # Compute score norm
            score_norm = torch.norm(score.view(score.shape[0], -1), dim=1)
            score_norm = torch.clamp(score_norm, min=1e-5)

            # Dynamisches Rauschmaß basierend auf score_norm
            normalized_score = (score_norm) / (800)
            normalized_score = torch.clamp(normalized_score, 0.0, 1.0)
            noise_factor = 1.0  

            # Compute adaptive step size
            step_size = base_step / (1.0 + score_norm)
            step_size = torch.clamp(step_size, min=1e-4)

            # Euler-Maruyama update mit dynamischem Rauschen
            mean_x = x_active + score * step_size.view(-1, 1, 1, 1)
            noise = torch.sqrt(step_size.view(-1, 1, 1, 1)) * noise_scale * noise_factor * torch.randn_like(x_active)
            x_updated = mean_x #+ noise

            # Copy back updated images
            x[active_mask] = x_updated

            # Determine which samples are done
            #finished_by_time = estimated_t !=estimated_t
            finished_by_score = score_norm > score_threshold

            global_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)

            # Handle time-based stopping
            # finished_indices_time = global_indices[finished_by_time]
            # iteration_counter[finished_indices_time] = i
            # stop_reason[finished_indices_time] = 1  # 1 = time
            # active_mask[finished_indices_time] = False

            # Handle score-based stopping
            finished_indices_score = global_indices[finished_by_score]
            iteration_counter[finished_indices_score] = i
            stop_reason[finished_indices_score] = 2  # 2 = score
            active_mask[finished_indices_score] = False

            # Optional logging
            #if finished_indices_score.numel() > 0:
            #    print(f"Step {i}: {finished_indices_score.numel()} samples stopped due to high score norm (> {score_threshold})")

            if i % 500 == 0:
                save_image(x, i)

        # Final stats
        fulfilled_count = (~active_mask).sum().item()
        fulfilled_ratio = fulfilled_count / batch_size
        finished_iters = iteration_counter[iteration_counter > 0]
        mean_steps = finished_iters.mean().item() if finished_iters.numel() > 0 else float('nan')

        score_stops = (stop_reason == 2).sum().item()
        time_stops = (stop_reason == 1).sum().item()

        print(f"\nFinished after {i} iterations")
        print(f"Percentage of images that satisfied stopping condition: {fulfilled_ratio * 100:.2f}%")
        print(f"Average number of steps until stopping (only for fulfilled images): {mean_steps:.2f}")
        print(f"Stopped due to score norm: {score_stops}/{batch_size} ({100 * score_stops / batch_size:.2f}%)")
        print(f"Stopped due to estimated_t: {time_stops}/{batch_size} ({100 * time_stops / batch_size:.2f}%)")

    return x


import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np 
from model import UNet_res,Unet
original_mode=False
if original_mode:
    score_model = UNet_res()
    score_model = torch.nn.DataParallel(UNet_res())  # Removed marginal_prob_std as it's not needed anymore

else:
    score_model = Unet(timesteps=100, time_embedding_dim=256, in_channels=1, out_channels=1, base_dim=28, dim_mults=[2,4], temp=100.0)  # Removed marginal_prob_std as it's not needed anymore
# Lade die vortrainierten Modelle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
original_mode=False
if original_mode:
    ckpt_score_model = torch.load('ckpt_res_3.pth', map_location=device)
else:
    ckpt_score_model = torch.load('ckpt_res_test_my_model.pth', map_location=device)
#score_model = torch.nn.DataParallel(UNet_res())  # Removed marginal_prob_std as it's not needed anymore
score_model = score_model.to(device)

# %%

score_model.load_state_dict(ckpt_score_model)
score_model.to(device)

# Setze Modelle in den Evaluationsmodus
score_model.eval()

# Set Sample-Batchgröße (Schritte werden nun automatisch ermittelt)
sample_batch_size = 64

# Rufe den adaptiven Sampler auf (num_steps optional für Safety, aber wird nicht gebraucht)
samples = Euler_Maruyama_mine(
    score_model,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    batch_size=sample_batch_size,
    x_shape=(1, 28, 28),
    device=device,
    eps=0,
    t_threshold=0.035,
    max_iterations = 20000,
    base_step = 0.5,
    noise_scale = 1,
    noise_std = 1,
    score_threshold=500,
    y=None,
    my_mode=True
)

# Clipping in [0, 1] für Visualisierung
samples = samples.clamp(-1.0, 1.0)

# Erzeuge ein Gitter zur Visualisierung
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

# Plot & Speichern
plt.figure(figsize=(6, 6))
plt.axis('off')
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=samples.min().item(), vmax=samples.max().item())
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), cmap="gray", norm=norm)
plt.savefig(f"temp/generated.png")
plt.show()