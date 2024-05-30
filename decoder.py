import torch
import torch.nn as nn
import torch.nn.functional as F

from workspace import *

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(latent_size + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim - latent_size - 2),            
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),            
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    
    def forward(self, g, x, latent):
        s = torch.sin(g)
        c = torch.cos(g)
        rot = torch.stack([
                                torch.stack([c, s], dim=1),
                                torch.stack([-s, c], dim=1)
                            ], dim=2).float()
        x = torch.matmul(rot, x.unsqueeze(-1)).squeeze(-1)

        inputs = torch.cat((x, latent), 1)
        y = self.block1(inputs)
        y = torch.cat((y, inputs), 1)
        return self.block2(y)