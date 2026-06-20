"""Evaluate trained model on test set and surface top cross-book bridges."""
import json
import torch
from sklearn.metrics import roc_auc_score
from src.config import GRAPH_SAVE_PATH, MODEL_SAVE_PATH, DATA_PROCESSED, GUTENBERG_BOOKS
from src.model.model import BibliographGT
from src.model.dataset import split_edges, sample_negatives
from src.model.train import HIDDEN_CHANNELS, OUT_CHANNELS, HEADS

TOP_K = 20


def top_cross_book_bridges(model, z, concept_to_book: dict, concept_texts: list, top_k: int = TOP_K):
    """Score all cross-book concept pairs and return the highest-confidence ones."""
    n = z.shape[0]
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                if concept_to_book[i] == concept_to_book[j]:
                    continue
                edge = torch.tensor([[i], [j]])
                score = torch.sigmoid(model.predict_link(z, edge)).item()
                results.append((score, concept_texts[i], concept_texts[j], concept_to_book[i], concept_to_book[j]))

    results.sort(reverse=True)
    return results[:top_k]


def evaluate() -> None:
    data = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    concept_x = data["concept"].x
    num_nodes = concept_x.shape[0]
    in_channels = concept_x.shape[1]

    all_edges = []
    for et in data.edge_types:
        if et[0] == "concept" and et[2] == "concept":
            all_edges.append(data[et].edge_index)
    edge_index = torch.cat(all_edges, dim=1)

    model = BibliographGT(in_channels, HIDDEN_CHANNELS, OUT_CHANNELS, HEADS)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()

    splits = split_edges(data)
    test_pos = splits["test_pos"]
    test_neg = sample_negatives(test_pos, num_nodes)

    with torch.no_grad():
        z = model(concept_x, edge_index)
        pos_scores = torch.sigmoid(model.predict_link(z, test_pos)).numpy()
        neg_scores = torch.sigmoid(model.predict_link(z, test_neg)).numpy()

    import numpy as np
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(labels, scores)
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    evaluate()
