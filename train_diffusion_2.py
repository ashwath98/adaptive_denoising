import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import functools
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the training dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
if torch.max(train_dataset[0][0])>1:
    print("not normalized")
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
import torch
import torch.nn as nn
from model import UNet_res, Unet
# Define the UNet architecture without time dependency

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
    diff = perturbed_x - x
    z = diff / torch.sqrt(random_t[:, None, None, None])
    
    # Get the score from the model using the perturbed data              






    
    score = model(perturbed_x)
    
    # Calculate the loss based on the score and the modified z
    #loss = torch.mean(torch.sum((score * std[:, None, None, None] - z)**2, dim=(1, 2, 3)))
    loss = torch.mean(torch .sum((score + z)**2, dim=(1, 2, 3)))
    
    return loss


n_epochs = 180
batch_size = 512
lr = 3e-4
#lr = 1e-4
original_mode=False
if original_mode:   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    score_model = torch.nn.DataParallel(UNet_res())  # Removed marginal_prob_std as it's not needed anymore
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    score_model = Unet(timesteps=100, time_embedding_dim=256, in_channels=1, out_channels=1, base_dim=28, dim_mults=[2,4], temp=100.0)  # Removed marginal_prob_std as it's not needed anymore
score_model = score_model.to(device)

# %%
score_model.cuda()
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import trange
model_path = 'ckpt_res_test_my_model2.pth'
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
    torch.save(score_model.state_dict(), model_path)