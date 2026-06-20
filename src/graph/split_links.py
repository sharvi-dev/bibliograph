"""Split concept-concept semantic edges into train / val / test splits.

Uses a fixed seed so every run produces the same split.
Deduplicates symmetric edges (contradicts, related_to get both directions in
the graph — we keep only the canonical src < dst direction for splitting).

Output: data/graph/link_split.pt
  {
    "train_pos": LongTensor(2, N_train),
    "val_pos":   LongTensor(2, N_val),
    "test_pos":  LongTensor(2, N_test),
    "num_nodes": int,
  }
"""
import torch
from src.config import GRAPH_SAVE_PATH, DATA_GRAPH

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test gets the remaining 0.1


def split_links() -> dict:
    data = torch.load(GRAPH_SAVE_PATH, weights_only=False)

    # collect all concept-concept semantic edge indices
    raw_edges = []
    for src_t, rel, dst_t in data.edge_types:
        if src_t == "concept" and dst_t == "concept":
            raw_edges.append(data[src_t, rel, dst_t].edge_index)

    if not raw_edges:
        raise ValueError("No concept-concept edges in graph.")

    all_ei = torch.cat(raw_edges, dim=1)

    # deduplicate — keep canonical src < dst to collapse symmetric duplicates
    seen: set[tuple[int, int]] = set()
    src_list, dst_list = [], []
    for k in range(all_ei.shape[1]):
        s, d = int(all_ei[0, k]), int(all_ei[1, k])
        if s > d:
            s, d = d, s
        if s == d:
            continue
        if (s, d) not in seen:
            seen.add((s, d))
            src_list.append(s)
            dst_list.append(d)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    n = edge_index.shape[1]

    torch.manual_seed(SEED)
    perm = torch.randperm(n)
    edge_index = edge_index[:, perm]

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    split = {
        "train_pos": edge_index[:, :n_train],
        "val_pos":   edge_index[:, n_train: n_train + n_val],
        "test_pos":  edge_index[:, n_train + n_val:],
        "num_nodes": int(data["concept"].x.shape[0]),
    }

    out_path = DATA_GRAPH / "link_split.pt"
    torch.save(split, out_path)

    n_tr = split["train_pos"].shape[1]
    n_vl = split["val_pos"].shape[1]
    n_te = split["test_pos"].shape[1]
    print(f"  {n} unique edges → {n_tr} train / {n_vl} val / {n_te} test")
    print(f"  saved → {out_path}")
    return split


if __name__ == "__main__":
    split_links()
