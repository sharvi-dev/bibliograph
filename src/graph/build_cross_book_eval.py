"""Build an index of all purely cross-book concept pairs.

A pair (A, B) is cross-book when the source_books sets of A and B are
completely disjoint — neither concept ever appeared in the same text.
These are the latent "bridge" candidates the trained model will score.

Output: data/graph/cross_book_pairs.pt
  {
    "edge_index": LongTensor(2, N_cross),   — concept index pairs
    "num_nodes":  int,
    "num_pairs":  int,
  }
"""
import json
import torch
from src.config import DATA_GRAPH


def build_cross_book_pairs() -> dict:
    node_lookup = json.loads((DATA_GRAPH / "node_lookup.json").read_text())
    concepts = node_lookup["concept"]  # str(idx) → {canonical_name, source_books}

    n = len(concepts)
    # build list of frozensets: idx → frozenset of book slugs
    idx_to_books: list[frozenset] = [frozenset()] * n
    for idx_str, meta in concepts.items():
        idx = int(idx_str)
        idx_to_books[idx] = frozenset(meta.get("source_books", []))

    src_list, dst_list = [], []
    for i in range(n):
        books_i = idx_to_books[i]
        if not books_i:
            continue
        for j in range(i + 1, n):
            books_j = idx_to_books[j]
            if books_i.isdisjoint(books_j):
                src_list.append(i)
                dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    result = {
        "edge_index": edge_index,
        "num_nodes": n,
        "num_pairs": len(src_list),
    }

    out_path = DATA_GRAPH / "cross_book_pairs.pt"
    torch.save(result, out_path)
    print(f"  {len(src_list):,} cross-book pairs from {n} concepts")
    print(f"  saved → {out_path}")
    return result


if __name__ == "__main__":
    build_cross_book_pairs()
