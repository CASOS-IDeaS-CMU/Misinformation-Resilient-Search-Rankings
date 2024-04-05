from torch_geometric.nn.conv import transformer_conv, GCNConv
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, global_add_pool, SAGEConv, GATConv, SAGPooling, GraphConv, GCN2Conv, GATv2Conv, GeneralConv, PDNConv, GMMConv 
from torch_geometric.nn import global_mean_pool, global_max_pool

import torch
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import BatchNorm

class GNN_v2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, use_weights = True):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # Batch normalization for conv1
        self.conv2 = GCNConv(hidden_channels, 64)
        self.bn2 = BatchNorm(64)  # Batch normalization for conv2
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(64, out_channels)
        self.use_weights = use_weights

    def forward(self, x, edge_index, edge_weight):
        if self.use_weights:
            x = self.conv1(x, edge_index, edge_weight)
            x = self.bn1(x).relu()  # Apply batch normalization before ReLU
            x = self.conv2(x, edge_index, edge_weight)
            x = self.bn2(x).relu()  # Apply batch normalization before ReLU
        else:
            x = self.conv1(x, edge_index)
            x = self.bn1(x).relu()  # Apply batch normalization before ReLU
            x = self.conv2(x, edge_index)
            x = self.bn2(x).relu()  # Apply batch normalization before ReLU
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class GNN_v1(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, use_weights=True):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, 128)
        #self.bn1 = BatchNorm(128)  # Batch normalization for conv1
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(128, out_channels)
        
        self.use_weights = use_weights

    def forward(self, x, edge_index, edge_weight):
        if self.use_weights:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index).relu()
        #x = self.relu()  # Apply batch normalization before ReLU
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class LinModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, use_weights=True):
        super().__init__()
        #self.conv1 = GCNConv(hidden_channels, 128)
        #self.bn1 = BatchNorm(128)  # Batch normalization for conv1
        self.lin1 = torch.nn.Linear(23, 128)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(128, out_channels)
        
        self.use_weights = use_weights

    def forward(self, x, edge_index, edge_weight):
        x = self.lin1(x).relu()
        #x = self.relu()  # Apply batch normalization before ReLU
        x = self.dropout(x)
        x = self.linear(x)
        return x