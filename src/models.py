import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, BatchNorm, GATConv, GATv2Conv


class FraudGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudGCN, self).__init__()
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


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.gat1 = GATv2Conv(in_channels=input_dim, out_channels=hidden_dim, heads=1, dropout=0.4)
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * 1, out_channels=output_dim, heads=1, concat=False, dropout=0.4
        )
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
