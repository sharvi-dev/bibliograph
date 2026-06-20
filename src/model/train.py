"""Training loop for the Graph Transformer link prediction model."""
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from src.config import GRAPH_SAVE_PATH, MODEL_SAVE_PATH
from src.model.model import BibliographGT
from src.model.dataset import split_edges, sample_negatives

HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
HEADS = 4
LR = 1e-3
EPOCHS = 150
LOG_EVERY = 10


def train_epoch(model, optimizer, z, train_pos, num_nodes):
    model.train()
    optimizer.zero_grad()

    train_neg = sample_negatives(train_pos, num_nodes)
    pos_scores = model.predict_link(z, train_pos)
    neg_scores = model.predict_link(z, train_neg)

    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
    scores = torch.cat([pos_scores, neg_scores])
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, z, pos_edges, num_nodes):
    from sklearn.metrics import roc_auc_score
    model.eval()
    with torch.no_grad():
        neg_edges = sample_negatives(pos_edges, num_nodes)
        pos_scores = torch.sigmoid(model.predict_link(z, pos_edges))
        neg_scores = torch.sigmoid(model.predict_link(z, neg_edges))
        labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))]).numpy()
        scores = torch.cat([pos_scores, neg_scores]).numpy()
    return roc_auc_score(labels, scores)


def train() -> None:
    data: HeteroData = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    concept_x = data["concept"].x
    num_nodes = concept_x.shape[0]
    in_channels = concept_x.shape[1]

    # use all concept-concept edges collapsed across relation types
    all_edges = []
    for et in data.edge_types:
        if et[0] == "concept" and et[2] == "concept":
            all_edges.append(data[et].edge_index)
    if not all_edges:
        raise ValueError("No concept-concept edges found in graph.")
    edge_index = torch.cat(all_edges, dim=1)

    splits = split_edges(data)  # placeholder; real split needs concept-concept edge_index
    train_pos = splits["train_pos"]
    val_pos = splits["val_pos"]

    model = BibliographGT(in_channels, HIDDEN_CHANNELS, OUT_CHANNELS, HEADS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        z = model(concept_x, edge_index)
        loss = train_epoch(model, optimizer, z, train_pos, num_nodes)
        if epoch % LOG_EVERY == 0:
            z = model(concept_x, edge_index)
            auc = evaluate(model, z, val_pos, num_nodes)
            print(f"Epoch {epoch:>3} | loss {loss:.4f} | val AUC {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  → saved best model (AUC {best_auc:.4f})")

    print(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
