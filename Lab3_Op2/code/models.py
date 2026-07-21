import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim=32, num_hidden_layers=2, activation='relu'):
        super(MLP, self).__init__()
        
        def get_activation():
            if activation.lower() == 'relu':
                return nn.ReLU()
            elif activation.lower() == 'tanh':
                return nn.Tanh()
            elif activation.lower() == 'sigmoid':
                return nn.Sigmoid()
            else:
                return nn.ReLU()

        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(get_activation())
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_activation())
            
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PointCNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super(PointCNN, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, hidden_dim, kernel_size=2), nn.ReLU())
        self.fc = nn.Linear(hidden_dim, 3)
    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).squeeze(-1))

class PointTransformer(nn.Module):
    def __init__(self, hidden_dim=16, num_heads=2):
        super(PointTransformer, self).__init__()
        self.embedding = nn.Linear(1, hidden_dim) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_dim * 2, 3)
    def forward(self, x):
        x = self.transformer(self.embedding(x.unsqueeze(-1)))
        return self.fc(x.reshape(x.size(0), -1))