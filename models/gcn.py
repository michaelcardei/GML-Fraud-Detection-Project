import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudGNN, self).__init__()
        # First GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Batch Normalization layer
        self.bn1 = BatchNorm(hidden_dim)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Batch Normalization layer
        self.bn2 = BatchNorm(hidden_dim)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer with ReLU and dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # First fully connected layer with ReLU and dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        return x
