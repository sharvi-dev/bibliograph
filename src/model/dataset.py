"""Link prediction dataset with positive/negative edge sampling."""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling


def split_edges(
    data: HeteroData,
    edge_type: tuple[str, str, str] = ("concept", "related_to", "concept"),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, torch.Tensor]:
    edge_index = data[edge_type].edge_index
    n_edges = edge_index.shape[1]

    perm = torch.randperm(n_edges)
    edge_index = edge_index[:, perm]

    n_train = int(n_edges * train_ratio)
    n_val = int(n_edges * val_ratio)

    return {
        "train_pos": edge_index[:, :n_train],
        "val_pos": edge_index[:, n_train: n_train + n_val],
        "test_pos": edge_index[:, n_train + n_val:],
    }


def sample_negatives(pos_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.shape[1],
    )
