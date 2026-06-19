from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from config import cfg
from graph.layers import GCNConvTarget

class KnowledgeGCN(nn.Module):
    def __init__(self, vertices, edge_index):
        super().__init__()
        self.sensor_proj = nn.Linear(3328, cfg.embed_dim)
        self.register_buffer("base_vertices", vertices)
        self.register_buffer("edge_index", edge_index)

        num_nodes = vertices.size(0) + 2  # +2 for sensor and encoding nodes
        
        self.conv1 = GCNConvTarget(cfg.embed_dim, cfg.gnn_hidden, 1, edge_index, num_nodes)
        self.bn1   = nn.BatchNorm1d(cfg.gnn_hidden)
        self.conv2 = GCNConvTarget(cfg.gnn_hidden, cfg.gnn_hidden, 2, edge_index, num_nodes)
        self.bn2   = nn.BatchNorm1d(cfg.gnn_hidden)
        self.conv3 = GCNConvTarget(cfg.gnn_hidden, cfg.gnn_hidden, 3, edge_index, num_nodes)
        self.bn3   = nn.BatchNorm1d(cfg.gnn_hidden)

    def forward(self, sensor_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the KnowledgeGCN model."""

        projected_sensors = self.sensor_proj(sensor_batch)  # [B, 1024]

        x = torch.stack([
        torch.cat([
            self.base_vertices,                       # [V-2, 1024]
            projected_sensors[i].unsqueeze(0),        # [1, 1024]
            torch.zeros((1, 1024), device=sensor_batch.device)
        ], dim=0)
        for i in range(projected_sensors.shape[0])
    ])
    
        batch_size, num_nodes, _ = x.shape

        x = F.elu(self.bn1(
        self.conv1(x, self.edge_index).reshape(batch_size*num_nodes, -1)
    ).reshape(batch_size, num_nodes, -1))

        x = F.elu(self.bn2(
        self.conv2(x, self.edge_index).reshape(batch_size*num_nodes, -1)
    ).reshape(batch_size, num_nodes, -1))

        x = F.elu(self.bn3(
        self.conv3(x, self.edge_index).reshape(batch_size*num_nodes, -1)
    ).reshape(batch_size, num_nodes, -1))

        return x
    