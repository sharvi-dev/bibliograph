"""Graph Transformer architecture for cross-book link prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, HeteroConv, Linear


class BibliographGT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def predict_link(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class HeteroBibliographGT(nn.Module):
    """Heterogeneous variant using per-relation-type TransformerConv weights."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, metadata, heads: int = 4):
        super().__init__()
        self.conv1 = HeteroConv(
            {et: TransformerConv(in_channels, hidden_channels, heads=heads, dropout=0.1) for et in metadata[1]},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {et: TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False) for et in metadata[1]},
            aggr="sum",
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

    def predict_link(self, z_dict, edge_index, node_type: str = "concept") -> torch.Tensor:
        z = z_dict[node_type]
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
