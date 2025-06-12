# %%
# https://levelup.gitconnected.com/building-stable-diffusion-from-scratch-using-python-f3ebc8c42da3#65d1
# Import the required libraries
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Extract a batch of unique images
unique_images, unique_labels = next(iter(train_loader))
unique_images = unique_images.numpy()

# Display a grid of unique images
fig, axes = plt.subplots(4, 16, figsize=(16, 4), sharex=True, sharey=True)  # Create a 4x16 grid of subplots with a wider figure

for i in range(4):  # Loop over rows
    for j in range(16):  # Loop over columns
        index = i * 16 + j  # Calculate the index in the batch
        axes[i, j].imshow(unique_images[index].squeeze(), cmap='gray')  # Show the image using a grayscale colormap
        axes[i, j].axis('off')  # Turn off axis labels and ticks

plt.show()  # Display the plot

# %%
# Import the PyTorch library for tensor operations.
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Extract a batch of unique images
unique_images, unique_labels = next(iter(train_loader))
unique_images = unique_images.numpy()

# Display a grid of unique images
fig, axes = plt.subplots(4, 16, figsize=(16, 4), sharex=True, sharey=True)  # Create a 4x16 grid of subplots with a wider figure

for i in range(4):  # Loop over rows
    for j in range(16):  # Loop over columns
        index = i * 16 + j  # Calculate the index in the batch
        axes[i, j].imshow(unique_images[index].squeeze(), cmap='gray')  # Show the image using a grayscale colormap
        axes[i, j].axis('off')  # Turn off axis labels and ticks

plt.show()  


# %%


# %%
# Define a module for a fully connected layer that reshapes outputs to feature maps.
class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Parameters:
        - input_dim: Dimensionality of the input features
        - output_dim: Dimensionality of the output features
        """
        super().__init__()

        # Define a fully connected layer
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor after passing through the fully connected layer
          and reshaping to a 4D tensor (feature map)
        """

        # Apply the fully connected layer and reshape the output to a 4D tensor
        return self.dense(x)[..., None, None]
        # This broadcasts the 2D tensor to a 4D tensor, adding the same value across space.

# %%


# %%
import torch
import torch.nn as nn

# Define the UNet architecture without time dependency
class UNet_res(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        """
        Parameters:
        - channels: The number of channels for feature maps of each resolution.
        """
        super().__init__()

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1)

        # Swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, y=None):
        """
        Parameters:
        - x: Input tensor
        - y: Target tensor (optional, not used in this pass)

        Returns:
        - h: Output tensor after passing through the U-Net architecture
        """

        # Encoding path
        h1 = self.conv1(x)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3)
        h4 = self.act(self.gnorm4(h4))

        # Decoding path
        h = self.tconv4(h4)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        return h


# %%
import torch
import torch.nn as nn

# Define a simpler network for predicting `random_t` in an arbitrary range
class TimePredictorNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        """
        A network for predicting random_t values based on input images.

        Parameters:
        - channels: List defining the number of channels in each layer.
        """
        super().__init__()

        # Encoding layers with decreasing resolution
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Fully connected layers to reduce to a single output for `random_t`
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels[3] * 4 * 4, 128)  # Assuming input is 28x28; adjust as needed
        self.fc2 = nn.Linear(128, 1)  # Final layer outputting a single scalar without activation

        # Activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, y=None):
        """
        Forward pass through the network.

        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor with predicted `random_t` values
        """
        # Encoding path
        h1 = self.act(self.gnorm1(self.conv1(x)))
        h2 = self.act(self.gnorm2(self.conv2(h1)))
        h3 = self.act(self.gnorm3(self.conv3(h2)))
        h4 = self.act(self.gnorm4(self.conv4(h3)))

        # Flatten and apply fully connected layers
        h_flat = self.flatten(h4)
        h_fc1 = self.act(self.fc1(h_flat))
        predicted_t = self.fc2(h_fc1)  # No activation, output can take any value

        return predicted_t.squeeze()


# %%
# Using GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

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

# %%
# Sigma Value
# sigma =  25.0

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

# %%
def loss_fn(model, x, marginal_prob_std, start_factor, eps=1e-3, K=50, lambda_param=1.0):
    
    
#def loss_fn(model, x, marginal_prob_std, start_factor, eps=1e-5):
    """
    The loss function for training score-based generative models.

    Parameters:
    - model: A PyTorch model instance
    - x: A mini-batch of training data.
    - marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    - start_factor: A function that gives the start factor for perturbing x.
    - eps: A tolerance value for numerical stability.
    """
    # Sample noise level from an Exponential distribution
    random_t = torch.distributions.Exponential(rate=lambda_param).sample((x.shape[0],)).to(x.device)
    random_t = random_t + eps

    
    # Find the noise std at the sampled time `t`
    std = marginal_prob_std(random_t)
    stf = start_factor(random_t)
    
    # Perturb the input data
    #perturbed_x = x * stf[:, None, None, None] + torch.randn_like(x) * std[:, None, None, None]
    perturbed_x = x  + torch.randn_like(x).to(device) * std[:, None, None, None]
    #import pdb; pdb.set_trace()
    # Define z based on perturbed_x and x
    diff = perturbed_x - x
    #norm_diff = torch.sqrt(torch.sum(diff**2, dim=(1, 2, 3), keepdim=True))
    # Compute the scaling factor with minimum constraint
    #scaling_factor = torch.minimum(d / (norm_diff + eps), torch.tensor(K, device=x.device))
    #z = scaling_factor * diff / (norm_diff + eps)  # Add eps to avoid division by zero
    z = diff / torch.sqrt(random_t[:, None, None, None])
    
    # Get the score from the model using the perturbed data              
    score = model(perturbed_x)
    
    # Calculate the loss based on the score and the modified z
    #loss = torch.mean(torch.sum((score * std[:, None, None, None] - z)**2, dim=(1, 2, 3)))
    loss = torch.mean(torch .sum((score + z)**2, dim=(1, 2, 3)))
    
    return loss


# %%


# %%
def loss_fn_random_t(model, x, marginal_prob_std, start_factor, eps=1e-3, lambda_param=1):
    """
    Loss function for training a model to learn the random_t directly.

    Parameters:
    - model: A PyTorch model instance that predicts `random_t` values.
    - x: A mini-batch of training data.
    - marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    - start_factor: A function that gives the start factor for perturbing x.
    - eps: A tolerance value for numerical stability.
    """
    # Sample noise level from an Exponential distribution
    random_t = torch.distributions.Exponential(rate=lambda_param).sample((x.shape[0],)).to(x.device)
    random_t = random_t + eps
    true_random_t = random_t
    
    # Find the noise std and start factor at the true_random_t
    std = marginal_prob_std(true_random_t)
    stf = start_factor(true_random_t)
    
    # Perturb the input data
    perturbed_x = x + torch.randn_like(x).to(device) * std[:, None, None, None]

    # Let the model predict `random_t` based on the perturbed input
    predicted_random_t = model(perturbed_x).squeeze()  # Ensure the output shape matches true_random_t

    # Calculate the loss as the mean squared error between predicted and true random_t
    loss = torch.mean((predicted_random_t - true_random_t) ** 2)

    return loss


# %%


# %%
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms

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

def Euler_Maruyama_sampler(score_model, time_predictor_model, marginal_prob_std, diffusion_coeff,
                           batch_size=64, x_shape=(1, 28, 28), device='cpu', eps=0.01,
                           t_threshold=0.01, T=2.0, num_steps=500, max_iterations=10000,
                           base_step=0.02, noise_scale=1.0, noise_std=0.5, y=None,
                           score_threshold=7.5):
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
            estimated_t = time_predictor_model(x_active, y=y).squeeze()

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
            x_updated = mean_x + noise

            # Copy back updated images
            x[active_mask] = x_updated

            # Determine which samples are done
            finished_by_time = estimated_t !=estimated_t
            finished_by_score = score_norm > score_threshold

            global_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)

            # Handle time-based stopping
            finished_indices_time = global_indices[finished_by_time]
            iteration_counter[finished_indices_time] = i
            stop_reason[finished_indices_time] = 1  # 1 = time
            active_mask[finished_indices_time] = False

            # Handle score-based stopping
            finished_indices_score = global_indices[finished_by_score & ~finished_by_time]
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


# %%
# Initialize the alternate U-Net model for training.
score_model = torch.nn.DataParallel(UNet_res())  # Removed marginal_prob_std as it's not needed anymore
score_model = score_model.to(device)

# %%
score_model.cuda()

# %%


# Setze die Anzahl der Trainingsepochen, Mini-Batch-Größe und Lernrate.
n_epochs = 180
batch_size = 512
lr = 3e-4
#lr = 1e-4

# Lade den MNIST-Datensatz für das Training
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Initialisiere den Adam-Optimierer mit der angegebenen Lernrate
optimizer = Adam(score_model.parameters(), lr=lr)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Lernraten-Scheduler für die Anpassung der Lernrate während des Trainings
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.995 ** epoch))

# Trainingsloop über Epochen
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    # Iteriere über Mini-Batches im Trainings-Datenloader
    for x, y in data_loader:
        x = x.to(device)
        
        # Berechne den Verlust für das aktuelle Mini-Batch
        loss = loss_fn(score_model, x, marginal_prob_std_fn, start_factor_fn)
        
        # Setze die Gradienten auf Null, backpropagiere und aktualisiere die Modellparameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Akkumuliere den Gesamtverlust und die Anzahl der verarbeiteten Items
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    
    # Passe die Lernrate mit dem Scheduler an
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    
    # Drucke den durchschnittlichen Verlust und die Lernrate für die aktuelle Epoche
    print('{} Durchschnittlicher Verlust: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    tqdm_epoch.set_description('Durchschnittlicher Verlust: {:5f}'.format(avg_loss / num_items))
    
    # Speichere den Modellcheckpoint nach jeder Epoche
    torch.save(score_model.state_dict(), 'ckpt_res_3.pth')


# %%


# %%
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import trange

# Initialize the model
time_predictor_model = torch.nn.DataParallel(TimePredictorNet()).to(device)


# %%

# Set the number of training epochs, mini-batch size, and learning rate
n_epochs = 15
batch_size = 256
lr = 3e-4

# Load the MNIST dataset for training
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize the Adam optimizer with the specified learning rate
optimizer = Adam(time_predictor_model.parameters(), lr=lr)

# Learning rate scheduler to adjust the learning rate during training
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.99 ** epoch))

# Training loop over epochs
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        
        # Calculate the loss for the current mini-batch
        loss = loss_fn_random_t(time_predictor_model, x, marginal_prob_std_fn, start_factor_fn)
        
        # Zero the gradients, backpropagate, and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate total loss and number of processed items
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    
    # Adjust the learning rate with the scheduler
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    
    # Print the average loss and learning rate for the current epoch
    #print('{} Durchschnittlicher Verlust: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    tqdm_epoch.set_description('Durchschnittlicher Verlust: {:5f}'.format(avg_loss / num_items))
    
    # Save the model checkpoint after each epoch
    torch.save(time_predictor_model.state_dict(), 'ckpt_random_t.pth')


# %%
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np 

# Lade die vortrainierten Modelle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_score_model = torch.load('ckpt_res_3.pth', map_location=device)
score_model.load_state_dict(ckpt_score_model)
score_model.to(device)
ckpt_random_t_model = torch.load('ckpt_random_t.pth', map_location=device)
time_predictor_model.load_state_dict(ckpt_random_t_model)
time_predictor_model.to(device)
# Setze Modelle in den Evaluationsmodus
score_model.eval()
time_predictor_model.eval()

# Set Sample-Batchgröße (Schritte werden nun automatisch ermittelt)
sample_batch_size = 64

# Rufe den adaptiven Sampler auf (num_steps optional für Safety, aber wird nicht gebraucht)
samples = Euler_Maruyama_sampler(
    score_model,
    time_predictor_model,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    batch_size=sample_batch_size,
    x_shape=(1, 28, 28),
    device=device,
    eps=0,
    t_threshold=0.035,
    max_iterations = 1000,
    base_step = 0.5,
    noise_scale = 1,
    noise_std = 1,
    score_threshold=30,
    y=None
)

# Clipping in [0, 1] für Visualisierung
samples = samples.clamp(0.0, 1.0)

# Erzeuge ein Gitter zur Visualisierung
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

# Plot & Speichern
plt.figure(figsize=(6, 6))
plt.axis('off')
norm = mcolors.Normalize(vmin=samples.min().item(), vmax=samples.max().item())
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), cmap="gray", norm=norm)
plt.savefig(f"temp/generated.png")
plt.show()



# %%


# %%
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

# Lade MNIST und bereite einen Batch vor
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Batchgröße und Bildauswahl
batch_size = 64
indices = torch.randint(0, len(mnist_data), (batch_size,))
images = torch.stack([mnist_data[i][0] for i in indices]).to(device).float()  # (B, 1, 28, 28)

# Liste von Noise-Levels t
t_values = torch.linspace(0.001, 1, steps=200)

t_list = []
norm_list = []

# Stelle sicher, dass das Modell auf dem gleichen Gerät ist
score_model.to(device).eval()

with torch.no_grad():
    for t in t_values:
        t_val = t.item()

        # Addiere Noise proportional zu t
        noise = torch.randn_like(images) * t_val
        x_noisy = images + noise

        # Berechne Scores
        score = score_model(x_noisy)

        # Normen berechnen (direkt über alle Pixel)
        norms = torch.norm(score.view(score.size(0), -1), dim=1)  # (B,)
        mean_norm = norms.mean().item()

        # Werte speichern
        t_list.append(t_val)
        norm_list.append(mean_norm)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_list, norm_list, label='||score|| (mean)', marker='o', color='blue')
plt.xlabel("Noise-Level t")
plt.ylabel("||score|| (mean)")
plt.title("Vergleich von t und ||score||")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%



