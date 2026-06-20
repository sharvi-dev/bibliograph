"""Heterogeneous Graph Transformer for cross-book link prediction.

Architecture per node type:
  Linear(384 → hidden)         per-type input projection
  HeteroConv(TransformerConv)  message passing, concat=False so output = hidden
  LayerNorm + residual         stabilize training
  → repeat for layer 2 (hidden → out), residual via Linear(hidden → out)

Decoder: dot-product (symmetric, untyped) — scores edge existence, not relation type.
Cross-book discovery emerges from the shared SBERT embedding space, not from
relation-type routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, HeteroConv


class HeteroBibliographGT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        metadata: tuple,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        node_types, edge_types = metadata

        # per-node-type input projection: in_channels → hidden_channels
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, hidden_channels, bias=False) for nt in node_types
        })

        # conv 1: hidden → hidden, concat=False keeps dim = hidden_channels
        self.conv1 = HeteroConv(
            {et: TransformerConv(hidden_channels, hidden_channels, heads=heads,
                                 concat=False, dropout=dropout)
             for et in edge_types},
            aggr="sum",
        )
        # conv 2: hidden → out_channels
        self.conv2 = HeteroConv(
            {et: TransformerConv(hidden_channels, out_channels, heads=heads,
                                 concat=False, dropout=dropout)
             for et in edge_types},
            aggr="sum",
        )

        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(out_channels) for nt in node_types})

        # residual projection for the dim change hidden → out between layers
        self.res_proj = nn.ModuleDict({
            nt: nn.Linear(hidden_channels, out_channels, bias=False) for nt in node_types
        })

        self.dropout = nn.Dropout(dropout)
        self._node_types = list(node_types)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        # input projection + activation
        h = {nt: F.elu(self.input_proj[nt](x)) for nt, x in x_dict.items()}

        # conv layer 1 — residual: h stays at hidden_channels dimension
        h1 = self.conv1(h, edge_index_dict)
        h = {
            nt: self.norm1[nt](self.dropout(F.elu(h1[nt])) + h[nt])
            for nt in self._node_types
            if nt in h1
        }

        # conv layer 2 — residual via res_proj (hidden → out)
        h2 = self.conv2(h, edge_index_dict)
        out = {
            nt: self.norm2[nt](self.dropout(F.elu(h2[nt])) + self.res_proj[nt](h[nt]))
            for nt in self._node_types
            if nt in h2
        }

        return out

    def predict_link(
        self,
        z_dict: dict,
        edge_index: torch.Tensor,
        node_type: str = "concept",
    ) -> torch.Tensor:
        # L2-normalize so the decoder is cosine similarity in [-1, 1];
        # sigmoid maps this to a meaningful [0.27, 0.73] confidence range
        z = F.normalize(z_dict[node_type], p=2, dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
