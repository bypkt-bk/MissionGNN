"""Custom message-passing layer with k-hop subgraph masking."""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph

class GCNConvTarget(MessagePassing):
    def __init__(self, in_channels, out_channels, k_hops, edge_index, num_nodes):
        super().__init__(aggr="mean")
        self.k = k_hops
        self.lin = nn.Linear(in_channels, out_channels)

        sensor_idx = num_nodes - 2
        _, sub_edge_index, _, _ = k_hop_subgraph(
            sensor_idx, k_hops, edge_index,
            relabel_nodes=False, directed=True, flow='target_to_source'
        )
        self.register_buffer("sub_edge_index", sub_edge_index)

    def forward(self, x, edge_index=None):
        x = self.lin(x)
        return self.propagate(self.sub_edge_index, x=x)

    def message(self, x_i, x_j):
        return x_j
