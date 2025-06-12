import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import argparse
from src.models.UnetESDE import Unet
from src.sde.implementations import ESDE

def train_2d_diffusion(model, sde, dataloader, optimizer, device, n_epochs, print_every, 
                      train_score=False, scheduler=None, start_epoch=0, checkpoint_dir="checkpoints",save_training_checkpoints=False,time_cond=False):
    """
    Training loop for 2D exponential diffusion model
    """
    model.train()
    model = model.to(device)
    loss_function = nn.MSELoss(reduction='mean')
    running_loss_list = []

    for epoch in range(start_epoch, n_epochs):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        
        for idx, (x_inp, _) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            
            # Forward pass through SDE
            X_t, noise, score, time = sde.run_forward_random_times(x_inp)

            # Move tensors to device
            X_t = X_t.to(device)
            noise = noise.to(device)
            score = score.to(device)

            # Model prediction - no time input needed
            if time_cond==False:
                time=None
            
            model_pred = model(X_t,time)

            # Compute loss based on training mode
            loss = loss_function(score, model_pred) if train_score else loss_function(noise, model_pred)

            # Backward pass
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
                wandb.log({
                    "loss": avg_loss,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # Save checkpoint after each epoch
        if save_training_checkpoints:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_epoch_{epoch}_{'score' if train_score else 'noise'}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': running_loss,
            }, checkpoint_path)

    return model, running_loss_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="2d-diffusion-training")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--train_score", action="store_true", help="Train score function instead of noise")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--T_max", type=float, default=1)
    parser.add_argument("--rate", type=float, default=1)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--save_training_checkpoints", type=bool, default=False, help="Save training checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "image_size": 28,
            "batch_size": args.batch_size,
            "n_channels": 1,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "n_epochs": args.n_epochs,
            "train_score": args.train_score,
        }
    )
    
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
    ])
    
    # Dataset and DataLoader
    IMAGE_SIZE = 28
    N_TIMESTEPS = 100
    N_CHANNELS = 1
    trainset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    # Initialize SDE
    sde = ESDE(T_max=args.T_max, rate=args.rate, sigma=args.sigma, dim=args.dim)
    
    # Initialize model - removed time embedding and timesteps parameters
    model = Unet(
        base_dim=IMAGE_SIZE,
        in_channels=N_CHANNELS,
        out_channels=N_CHANNELS,
        time_embedding_dim=256,
        timesteps=N_TIMESTEPS,
        dim_mults=[2, 4],
        temp=100.0
    )
    if torch.cuda.is_available():
        model = torch.compile(model)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.learning_rate,
        total_steps=args.n_epochs * len(trainloader),
        pct_start=0.25,
        anneal_strategy='cos'
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # Train model
    model, loss_history = train_2d_diffusion(
        model=model,
        sde=sde,
        dataloader=trainloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        n_epochs=args.n_epochs,
        print_every=100,
        train_score=args.train_score,
        start_epoch=start_epoch,
        checkpoint_dir=args.checkpoint_dir,
        save_training_checkpoints=args.save_training_checkpoints
    )
    
    # Save final model
    final_model_path = os.path.join(
        args.checkpoint_dir, 
        f"final_model_{'score' if args.train_score else 'noise'}_{args.T_max}_{args.rate}_{args.sigma}_{args.dim}.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    
    wandb.finish()

if __name__ == "__main__":
    main() 