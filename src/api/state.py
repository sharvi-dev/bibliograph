"""Model + graph singleton loaded once at API startup.

All route handlers call state.get(key) instead of re-loading from disk.
"""
import json
import torch
from src.config import GRAPH_SAVE_PATH, MODEL_SAVE_PATH, DATA_GRAPH
from src.model.model import HeteroBibliographGT
from src.model.train import IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, HEADS, DROPOUT

_cache: dict = {}


def load() -> None:
    data = torch.load(GRAPH_SAVE_PATH, weights_only=False)

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

    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)

    node_lookup = json.loads((DATA_GRAPH / "node_lookup.json").read_text())

    # pre-build book_slug → {concept_idx: name} for O(1) graph lookups
    book_to_concepts: dict[str, dict[int, str]] = {}
    for idx_str, meta in node_lookup["concept"].items():
        for slug in meta["source_books"]:
            book_to_concepts.setdefault(slug, {})[int(idx_str)] = meta["canonical_name"]

    _cache.update({
        "model":           model,
        "z_dict":          z_dict,
        "data":            data,
        "node_lookup":     node_lookup,
        "evidence_lookup": json.loads((DATA_GRAPH / "evidence_lookup.json").read_text()),
        "cross_book_pairs": torch.load(DATA_GRAPH / "cross_book_pairs.pt", weights_only=True),
        "book_to_concepts": book_to_concepts,
    })
    print("  BiblioGraph model loaded.")


def get(key: str):
    return _cache[key]


def clear() -> None:
    _cache.clear()


def is_loaded() -> bool:
    return bool(_cache)
