"""Load the saved graph and print sanity-check statistics.

Warning signs checked:
  - Isolated book / author nodes
  - related_to dominating all edges
  - One book producing most edges
  - Relation-class imbalance
  - Self-loops
  - Duplicate edges per relation
"""
import json
import torch
from collections import Counter
from src.config import GRAPH_SAVE_PATH, DATA_GRAPH


def inspect() -> None:
    data = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    node_lookup = json.loads((DATA_GRAPH / "node_lookup.json").read_text())

    print("=" * 60)
    print("NODE COUNTS")
    print("=" * 60)
    for node_type in data.node_types:
        n = data[node_type].x.shape[0]
        dim = data[node_type].x.shape[1]
        print(f"  {node_type:12s}: {n:>5} nodes  (emb dim {dim})")

    print("\n" + "=" * 60)
    print("EDGE COUNTS BY RELATION")
    print("=" * 60)
    total_semantic = 0
    rel_counts: dict[str, int] = {}
    for edge_type in data.edge_types:
        n = data[edge_type].edge_index.shape[1]
        label = " → ".join(edge_type)
        print(f"  {label:50s}: {n:>6} edges")
        src, rel, dst = edge_type
        if src == "concept" and dst == "concept":
            rel_counts[rel] = n
            total_semantic += n

    if total_semantic:
        print(f"\n  Relation distribution (semantic edges only):")
        for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / total_semantic
            bar = "█" * int(pct / 2)
            print(f"    {rel:12s}: {cnt:>5}  ({pct:5.1f}%)  {bar}")
        dominant = max(rel_counts, key=rel_counts.get)
        if rel_counts[dominant] / total_semantic > 0.7:
            print(f"\n  ⚠  WARNING: '{dominant}' is >70% of all semantic edges — prompt may need tuning")

    print("\n" + "=" * 60)
    print("STRUCTURAL CHECKS")
    print("=" * 60)

    # isolated book nodes
    if ("concept", "appears_in", "book") in data.edge_types:
        book_with_concepts = set(data["concept", "appears_in", "book"].edge_index[1].tolist())
        n_books = data["book"].x.shape[0]
        isolated_books = n_books - len(book_with_concepts)
        print(f"  Isolated book nodes   : {isolated_books}")
        if isolated_books:
            print("  ⚠  WARNING: some book nodes have no concept edges")

    # isolated author nodes
    if ("book", "written_by", "author") in data.edge_types:
        authors_with_books = set(data["book", "written_by", "author"].edge_index[1].tolist())
        n_authors = data["author"].x.shape[0]
        isolated_authors = n_authors - len(authors_with_books)
        print(f"  Isolated author nodes : {isolated_authors}")

    # self-loops in semantic edges
    for edge_type in data.edge_types:
        src_t, rel, dst_t = edge_type
        if src_t != dst_t:
            continue
        ei = data[edge_type].edge_index
        self_loops = (ei[0] == ei[1]).sum().item()
        if self_loops:
            print(f"  ⚠  Self-loops in ({rel}): {self_loops}")

    # duplicate edges
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        pairs = list(zip(ei[0].tolist(), ei[1].tolist()))
        dupes = len(pairs) - len(set(pairs))
        if dupes:
            label = " → ".join(edge_type)
            print(f"  ⚠  Duplicate edges in {label}: {dupes}")

    # concepts per book
    print("\n" + "=" * 60)
    print("CONCEPTS PER BOOK")
    print("=" * 60)
    if ("concept", "appears_in", "book") in data.edge_types:
        ei = data["concept", "appears_in", "book"].edge_index
        book_counts = Counter(ei[1].tolist())
        books = node_lookup.get("book", {})
        for b_idx, cnt in sorted(book_counts.items(), key=lambda x: -x[1]):
            title = books.get(str(b_idx), {}).get("title", f"book_{b_idx}")
            print(f"  {title:50s}: {cnt} concepts")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    inspect()
