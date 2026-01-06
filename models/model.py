import torch
import torch.nn as nn

class CyberbullyDetector(nn.Module):
    def __init__(self, input_dim=1761, dropout_rate=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.network(x)
        logits = self.head(x)
        probability = torch.sigmoid(logits)
        return probability
