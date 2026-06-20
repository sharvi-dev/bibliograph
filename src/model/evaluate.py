"""Evaluate trained model on held-out test set and print top cross-book bridges."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from src.config import GRAPH_SAVE_PATH, MODEL_SAVE_PATH, DATA_GRAPH
from src.model.model import HeteroBibliographGT
from src.model.train import IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, HEADS, DROPOUT

TOP_K = 20


def evaluate() -> None:
    data    = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    split   = torch.load(DATA_GRAPH / "link_split.pt",        weights_only=True)
    cb      = torch.load(DATA_GRAPH / "cross_book_pairs.pt",  weights_only=True)
    nl      = json.loads((DATA_GRAPH / "node_lookup.json").read_text())
    ev      = json.loads((DATA_GRAPH / "evidence_lookup.json").read_text())

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

    x_dict          = data.x_dict
    edge_index_dict = data.edge_index_dict
    num_nodes       = int(data["concept"].x.shape[0])
    test_pos        = split["test_pos"]

    with torch.no_grad():
        z_dict = model(x_dict, edge_index_dict)

    # ── test AUC ─────────────────────────────────────────────────────────────
    test_neg = negative_sampling(test_pos, num_nodes=num_nodes, num_neg_samples=test_pos.shape[1])
    pos_s = torch.sigmoid(model.predict_link(z_dict, test_pos)).numpy()
    neg_s = torch.sigmoid(model.predict_link(z_dict, test_neg)).numpy()
    labels = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    scores = np.concatenate([pos_s, neg_s])
    print(f"Test AUC: {roc_auc_score(labels, scores):.4f}")

    # ── top cross-book bridges ────────────────────────────────────────────────
    cb_ei = cb["edge_index"]
    concepts = nl["concept"]
    books    = nl["book"]

    BATCH = 4096
    all_scores = []
    with torch.no_grad():
        for start in range(0, cb_ei.shape[1], BATCH):
            batch_ei = cb_ei[:, start: start + BATCH]
            s = torch.sigmoid(model.predict_link(z_dict, batch_ei))
            all_scores.append(s)
    all_scores = torch.cat(all_scores).numpy()

    top_idx = np.argsort(all_scores)[::-1][:TOP_K]

    print(f"\nTop {TOP_K} cross-book bridges:")
    print("-" * 90)
    for rank, k in enumerate(top_idx, 1):
        i, j = int(cb_ei[0, k]), int(cb_ei[1, k])
        ci   = concepts[str(i)]
        cj   = concepts[str(j)]
        name_a, name_b = ci["canonical_name"], cj["canonical_name"]
        books_a = ", ".join(ci["source_books"])
        books_b = ", ".join(cj["source_books"])
        conf = float(all_scores[k])
        ev_key = f"{name_a}||{name_b}"
        evidence = ev.get(ev_key, ev.get(f"{name_b}||{name_a}", {})).get("evidence", "")
        print(f"{rank:>2}. [{conf:.3f}] {name_a!r:30s} ({books_a})")
        print(f"         ↔ {name_b!r:30s} ({books_b})")
        if evidence:
            snippet = evidence[:120].replace("\n", " ")
            print(f"         evidence: {snippet}")
        print()


if __name__ == "__main__":
    evaluate()
