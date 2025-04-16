import torch
import torch.nn as nn
import torch.nn.functional as F

class FullConnectedScoreModel(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super().__init__()
        self.input_layer = nn.Linear(data_dim+1, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(n_hidden_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        x = torch.concat([x, t.unsqueeze(1)], axis=1)
        x = F.relu(self.input_batch_norm(self.input_layer(x)))
        
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            
        return self.output_layer(x) 