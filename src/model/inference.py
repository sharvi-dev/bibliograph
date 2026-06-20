"""Score all cross-book concept pairs and return the top-K bridges.

Used by the API and frontend. Can also be run standalone.

  python -m src.model.inference [--top-k 20] [--book-a SLUG] [--book-b SLUG]
"""
import argparse
import json
import torch
import numpy as np
from src.config import GRAPH_SAVE_PATH, MODEL_SAVE_PATH, DATA_GRAPH
from src.model.model import HeteroBibliographGT
from src.model.train import IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, HEADS, DROPOUT

BATCH = 4096


def load_model(data):
    model = HeteroBibliographGT(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        metadata=data.metadata(),
        heads=HEADS,
        dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    return model


def get_embeddings(model, data) -> dict:
    with torch.no_grad():
        return model(data.x_dict, data.edge_index_dict)


def top_bridges(
    top_k: int = 20,
    book_a_slug: str | None = None,
    book_b_slug: str | None = None,
) -> list[dict]:
    """Return the top-k cross-book bridges, optionally filtered to a specific book pair."""
    data    = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    cb      = torch.load(DATA_GRAPH / "cross_book_pairs.pt", weights_only=True)
    nl      = json.loads((DATA_GRAPH / "node_lookup.json").read_text())
    ev      = json.loads((DATA_GRAPH / "evidence_lookup.json").read_text())

    model  = load_model(data)
    z_dict = get_embeddings(model, data)

    cb_ei    = cb["edge_index"]
    concepts = nl["concept"]

    # optional book-pair filter
    if book_a_slug or book_b_slug:
        mask = []
        for k in range(cb_ei.shape[1]):
            i, j = int(cb_ei[0, k]), int(cb_ei[1, k])
            bi = set(concepts[str(i)]["source_books"])
            bj = set(concepts[str(j)]["source_books"])
            a_ok = (book_a_slug is None) or (book_a_slug in bi)
            b_ok = (book_b_slug is None) or (book_b_slug in bj)
            ab_ok = a_ok and b_ok
            ba_ok = (book_b_slug is None or book_b_slug in bi) and (book_a_slug is None or book_a_slug in bj)
            if ab_ok or ba_ok:
                mask.append(k)
        if not mask:
            return []
        cb_ei = cb_ei[:, torch.tensor(mask, dtype=torch.long)]

    # score in batches
    all_scores = []
    with torch.no_grad():
        for start in range(0, cb_ei.shape[1], BATCH):
            batch = cb_ei[:, start: start + BATCH]
            s = torch.sigmoid(model.predict_link(z_dict, batch))
            all_scores.append(s)
    all_scores = torch.cat(all_scores).numpy()

    top_idx = np.argsort(all_scores)[::-1][:top_k]

    results = []
    for k in top_idx:
        i, j = int(cb_ei[0, k]), int(cb_ei[1, k])
        ci, cj = concepts[str(i)], concepts[str(j)]
        name_a, name_b = ci["canonical_name"], cj["canonical_name"]
        ev_key  = f"{name_a}||{name_b}"
        evidence = ev.get(ev_key, ev.get(f"{name_b}||{name_a}", {}))
        results.append({
            "concept_a":   name_a,
            "concept_b":   name_b,
            "books_a":     ci["source_books"],
            "books_b":     cj["source_books"],
            "score":       float(all_scores[k]),
            "evidence":    evidence.get("evidence", ""),
            "relation":    evidence.get("relation", ""),
            "book_id":     evidence.get("book_id", ""),
        })
    return results


def _cli():
    parser = argparse.ArgumentParser(description="Surface top cross-book bridges.")
    parser.add_argument("--top-k",  type=int,  default=20)
    parser.add_argument("--book-a", type=str,  default=None)
    parser.add_argument("--book-b", type=str,  default=None)
    args = parser.parse_args()

    bridges = top_bridges(top_k=args.top_k, book_a_slug=args.book_a, book_b_slug=args.book_b)
    if not bridges:
        print("No bridges found.")
        return
    print(f"\nTop {len(bridges)} cross-book bridges:")
    print("-" * 90)
    for rank, b in enumerate(bridges, 1):
        print(f"{rank:>2}. [{b['score']:.3f}] {b['concept_a']!r:30s} {b['books_a']}")
        print(f"         ↔ {b['concept_b']!r:30s} {b['books_b']}")
        if b["evidence"]:
            print(f"         {b['evidence'][:120]}")
        print()


if __name__ == "__main__":
    _cli()
