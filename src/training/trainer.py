import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, Any
import wandb
from tqdm import tqdm

class DiffusionTrainer:
    def __init__(
        self, 
        model: nn.Module,
        sde: Any,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        config: Dict[str, Any],
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.sde = sde
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.use_wandb = use_wandb
        self.loss_fn = nn.MSELoss(reduction='mean')

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for idx, x_inp in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass
            X_t, noise, score, time = self.sde.run_forward_random_times(x_inp)
            X_t, noise, score, time = (
                X_t.to(self.device),
                noise.to(self.device),
                score.to(self.device),
                time.to(self.device)
            )
            
            # Model prediction
            pred = self.model(X_t, time)
            
            # Loss computation
            target = score if self.config['train_score'] else noise
            loss = self.loss_fn(pred, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Logging
            running_loss += loss.item()
            if (idx + 1) % self.config['log_every'] == 0:
                avg_loss = running_loss / self.config['log_every']
                pbar.set_description(f"Epoch {epoch} - Loss: {avg_loss:.6f}")
                
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'step': idx,
                        'loss': avg_loss,
                    })
                running_loss = 0.0
        
        return running_loss / len(self.dataloader) 