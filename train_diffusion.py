import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.sde.implementations import VPSDE
from src.models.UnetOG import Unet  # Assuming you move the Unet class to a separate model.py file
import argparse
import wandb
import os

def train_diffusion_model(model, sde, dataloader, optimizer, device, n_epochs, print_every, train_score=False, scheduler=None, start_epoch=0, checkpoint_dir="checkpoints"):
    model.train()
    model = model.to(device)
    loss_function = nn.MSELoss(reduction='mean')
    running_loss_list = []

    for epoch in range(start_epoch, n_epochs):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        for idx, (x_inp, y_inp) in enumerate(dataloader):
            optimizer.zero_grad()
            
            X_t, noise, score, time = sde.run_forward_random_times(x_inp)

            X_t = X_t.to(device)
            noise = noise.to(device)
            score = score.to(device)
            time = time.to(device)

            model_pred = model(X_t, time)

            loss = loss_function(score, model_pred) if train_score else loss_function(noise, model_pred)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.detach().item()

            if (idx + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                running_loss_list.append(avg_loss)
                running_loss = 0.0
                print(f"Batch {idx + 1}, Loss: {avg_loss:.6f}")
                wandb.log({"loss": avg_loss})
    return model, running_loss_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="diffusion-training")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--train_score", action="store_true", help="Train score function instead of noise")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "image_size": 28,
            "batch_size": 128,
            "n_timesteps": 100,
            "n_channels": 1,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "n_epochs": 100,
            "train_score": args.train_score,  # Added train_score to config
        }
    )
    
    # Hyperparameters
    IMAGE_SIZE = 28
    BATCH_SIZE = 128
    N_TIMESTEPS = 100
    N_CHANNELS = 1
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 0.0
    N_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
    ])
    
    # Dataset and DataLoader
    trainset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    trainloader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=2
    )
    
    # Initialize SDE
    sde = VPSDE(T_max=1, beta_min=0.01, beta_max=10.0)
    
    # Initialize model
    model = Unet(
        base_dim=IMAGE_SIZE,
        in_channels=N_CHANNELS,
        out_channels=N_CHANNELS,
        time_embedding_dim=256,
        timesteps=N_TIMESTEPS,
        dim_mults=[2, 4],
        temp=100.0
    )
    model = torch.compile(model)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        maximize=False
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        LEARNING_RATE,
        total_steps=N_EPOCHS * len(trainloader),
        pct_start=0.25,
        anneal_strategy='cos'
    )
    
    # Train model
    model, loss_history = train_diffusion_model(
        model=model,
        sde=sde,
        dataloader=trainloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        n_epochs=N_EPOCHS,
        print_every=100,
        train_score=args.train_score,
        start_epoch=0,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save final model with train_score in filename
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    final_model_path = os.path.join(
        args.checkpoint_dir, 
        f"final_model_{'score' if args.train_score else 'noise'}.pt"
    )

    torch.save(model.state_dict(), final_model_path)
    
    wandb.finish()

if __name__ == "__main__":
    main() 