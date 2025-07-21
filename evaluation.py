import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from train import load_and_process, SteamGNN

def evaluate(args):
    # 1. Reconstruct the data and splits
    print("Loading data and splits…")
    data, train_edges, val_edges, test_edges, maps = load_and_process(
        args.users, args.games, args.recs
    )

    # 2. Initialize model and load weights
    print(f"Loading model from {args.save_model}")
    model = SteamGNN(in_feats=data.x.size(1), hid_feats=args.hidden)
    model.load_state_dict(torch.load(args.save_model))
    model.eval()

    # 3. Compute embeddings
    print("Computing node embeddings…")
    with torch.no_grad():
        h = model(data.x, data.edge_index)

    # 4. Score positives (test set)
    print("Scoring positive test edges…")
    pos_edge_index = data.edge_index[:, test_edges]
    pos_scores = model.decode(h, pos_edge_index).cpu().numpy()

    # 5. Sample negatives of the same size
    print("Sampling negative edges…")
    neg_edge_index = negative_sampling(
        edge_index   = data.edge_index,
        num_nodes    = data.num_nodes,
        num_neg_samples = pos_scores.shape[0]
    )
    with torch.no_grad():
        neg_scores = model.decode(h, neg_edge_index).cpu().numpy()

    # 6. Compute metrics
    y_true  = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)

    print(f"Evaluation results on test set:")
    print(f"  • AUC: {auc:.4f}")
    print(f"  • Average Precision (AP): {ap:.4f}")

    # 7. (Optional) compute BPR loss on test set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Steam GNN model")
    parser.add_argument("--users",      default="data/users.csv",
                        help="Path to users.csv")
    parser.add_argument("--games",      default="data/games.csv",
                        help="Path to games.csv")
    parser.add_argument("--recs",       default="data/recommendations.csv",
                        help="Path to recommendations.csv")
    parser.add_argument("--save_model", default="steam_gnn.pt",
                        help="Path where the model checkpoint was saved")
    parser.add_argument("--hidden",     type=int, default=64,
                        help="Hidden dimension (must match the training setting)")
    args = parser.parse_args()

    start = time.time()
    evaluate(args)
    print(f"Total evaluation time: {time.time() - start:.2f}s")