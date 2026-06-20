"""Training loop for HeteroBibliographGT link prediction."""
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from src.config import GRAPH_SAVE_PATH, MODEL_SAVE_PATH, DATA_GRAPH
from src.model.model import HeteroBibliographGT

IN_CHANNELS = 384       # SBERT all-MiniLM-L6-v2
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
HEADS = 4
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 200
LOG_EVERY = 10
PATIENCE = 20           # early-stop if val AUC doesn't improve


def _auc(model, z_dict, pos_ei, num_nodes: int) -> float:
    model.eval()
    with torch.no_grad():
        neg_ei = negative_sampling(pos_ei, num_nodes=num_nodes, num_neg_samples=pos_ei.shape[1])
        pos_s = torch.sigmoid(model.predict_link(z_dict, pos_ei)).cpu().numpy()
        neg_s = torch.sigmoid(model.predict_link(z_dict, neg_ei)).cpu().numpy()
    import numpy as np
    labels = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    scores = np.concatenate([pos_s, neg_s])
    return float(roc_auc_score(labels, scores))


def train() -> None:
    data = torch.load(GRAPH_SAVE_PATH, weights_only=False)
    split = torch.load(DATA_GRAPH / "link_split.pt", weights_only=True)

    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    num_nodes = int(data["concept"].x.shape[0])

    train_pos = split["train_pos"]
    val_pos   = split["val_pos"]

    model = HeteroBibliographGT(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        metadata=data.metadata(),
        heads=HEADS,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auc = 0.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        z_dict = model(x_dict, edge_index_dict)

        train_neg = negative_sampling(
            train_pos, num_nodes=num_nodes, num_neg_samples=train_pos.shape[1]
        )
        pos_scores = model.predict_link(z_dict, train_pos)
        neg_scores = model.predict_link(z_dict, train_neg)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(train_pos.shape[1]),
            torch.zeros(train_neg.shape[1]),
        ])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0:
            z_dict = model(x_dict, edge_index_dict)
            auc = _auc(model, z_dict, val_pos, num_nodes)
            marker = ""
            if auc > best_auc:
                best_auc = auc
                no_improve = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                marker = "  ← saved"
            else:
                no_improve += 1
            print(f"epoch {epoch:>3} | loss {loss.item():.4f} | val AUC {auc:.4f}{marker}")
            if no_improve >= PATIENCE // LOG_EVERY:
                print(f"  early stop (no improvement for {PATIENCE} epochs)")
                break

    print(f"\ntraining complete — best val AUC: {best_auc:.4f}")
    print(f"model saved → {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
