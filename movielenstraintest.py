import os
import random
from collections import defaultdict
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from torch_geometric.data import download_url, extract_zip
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.nn import SAGEConv, GATConv

# =============================================================
# 0. Config: choose your GNN backbone here
# =============================================================
# Options: 'lightgcn' | 'sage' | 'gat'
CONFIG = {
    'backbone': 'gat',   # <-- switch this
    'hidden_dim': 64,
    'num_layers': 3,
    'batch_size': 256,
    'num_epoch': 50,
    'epoch_size': 200,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'topK': 10,
    'lambda': 1e-6,
    'gat_heads': 1,          # only used when backbone == 'gat'
    'gat_dropout': 0.2       # only used when backbone == 'gat'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================
# 1. Data Prep (MovieLens latest-small)
# =============================================================

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
DATA_DIR = './'

zip_path = download_url(URL, DATA_DIR)
extract_zip(zip_path, DATA_DIR)

movie_path = os.path.join(DATA_DIR, 'ml-100k', 'u.item')
rating_path = os.path.join(DATA_DIR, 'ml-100k', 'u.data')


def preprocessing(movie_path: str, rating_path: str):
    """Load MovieLens (supports both ml-latest-small and ml-100k) and build user->item edges (rating>=4)."""
    # If it's the 100k dataset (u.data / u.item)
    if rating_path.endswith('u.data'):
        rating_df = pd.read_csv(rating_path, sep='	', header=None,
                                names=['userId', 'movieId', 'rating', 'timestamp'])
        movie_df = pd.read_csv(movie_path, sep='|', header=None, encoding='latin-1', usecols=[0, 1],
                               names=['movieId', 'title'])

        movie_mapping = {mid: i for i, mid in enumerate(movie_df.movieId.unique())}
        user_mapping = {uid: i for i, uid in enumerate(rating_df.userId.unique())}

        num_users = len(user_mapping)
        num_movies = len(movie_mapping)

        users = rating_df.userId.map(user_mapping).tolist()
        movies = rating_df.movieId.map(movie_mapping).add(num_users).tolist()  # offset items
        ratings = rating_df.rating.values

        mask = ratings >= 4.0
        user_list = [u for u, keep in zip(users, mask) if keep]
        item_list = [m for m, keep in zip(movies, mask) if keep]

        edge_index = torch.tensor([user_list, item_list], dtype=torch.long)
        return edge_index, num_users, num_movies, movie_mapping, user_mapping

    # Otherwise assume ml-latest-small CSV layout
    movie_df = pd.read_csv(movie_path, index_col='movieId')
    rating_df_idxed = pd.read_csv(rating_path, index_col='userId')

    movie_mapping = {mid: i for i, mid in enumerate(movie_df.index.unique())}
    user_mapping = {uid: i for i, uid in enumerate(rating_df_idxed.index.unique())}

    num_users = len(user_mapping)
    num_movies = len(movie_mapping)

    rating_df = pd.read_csv(rating_path)
    users = [user_mapping[u] for u in rating_df['userId']]
    movies = [movie_mapping[m] + num_users for m in rating_df['movieId']]
    ratings = rating_df['rating'].values

    mask = ratings >= 4.0
    user_list = [u for u, keep in zip(users, mask) if keep]
    item_list = [m for m, keep in zip(movies, mask) if keep]

    edge_index = torch.tensor([user_list, item_list], dtype=torch.long)
    return edge_index, num_users, num_movies, movie_mapping, user_mapping


edge_index, num_users, num_movies, movie_mapping, user_mapping = preprocessing(movie_path, rating_path)
num_nodes = num_users + num_movies

# =============================================================
# 2. Split
# =============================================================

num_edges = edge_index.size(1)
all_idx = np.arange(num_edges)
idx_train, idx_tmp = train_test_split(all_idx, test_size=0.2, random_state=42)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=42)

train_edge_index = edge_index[:, idx_train]
val_edge_index = edge_index[:, idx_val]
test_edge_index = edge_index[:, idx_test]

# symmetric edge_index for message passing (SAGE/GAT/LightGCN need undirected)
train_edge_sym = torch.cat([train_edge_index, train_edge_index.flip(0)], dim=1)

# =============================================================
# 3. LightGCN normalized adjacency (only used when backbone == 'lightgcn')
# =============================================================

def build_norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(edge_index, weights, (num_nodes, num_nodes))
    return adj.coalesce()

train_adj = build_norm_adj(train_edge_sym, num_nodes)

# =============================================================
# 4. Utilities: positives dict, sampling, loss
# =============================================================

def build_user_pos_dict(edge_index: torch.Tensor, num_users: int) -> dict:
    up = defaultdict(set)
    u = edge_index[0].tolist()
    it = (edge_index[1] - num_users).tolist()
    for a, b in zip(u, it):
        up[a].add(b)
    return up

user_pos_train = build_user_pos_dict(train_edge_index, num_users)
user_pos_val = build_user_pos_dict(val_edge_index, num_users)
user_pos_test = build_user_pos_dict(test_edge_index, num_users)


def mini_batch_sample(batch_size: int,
                      pos_edge_index: torch.Tensor,
                      num_users: int,
                      num_items: int,
                      user_pos_dict: dict):
    E = pos_edge_index.size(1)
    perm = torch.randperm(E)[:batch_size]
    users = pos_edge_index[0, perm]
    pos_items = pos_edge_index[1, perm] - num_users

    neg_items = torch.empty_like(pos_items)
    for i, u in enumerate(users.tolist()):
        seen = user_pos_dict[u]
        while True:
            j = random.randint(0, num_items - 1)
            if j not in seen:
                neg_items[i] = j
                break
    return users, pos_items, neg_items


def bpr_loss(users_emb, users_emb0, pos_emb, pos_emb0, neg_emb, neg_emb0, reg_lambda: float):
    pos_scores = (users_emb * pos_emb).sum(dim=1)
    neg_scores = (users_emb * neg_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    reg = (users_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2) + neg_emb0.norm(2).pow(2))
    return loss + reg_lambda * reg

# =============================================================
# 5. Backbone Implementations
# =============================================================

class RecommenderBackbone(nn.Module):
    """Abstract base class. Child classes must implement forward() returning a list of layer outputs."""
    def forward(self, x0: torch.Tensor) -> list[torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError


class LightGCNBackbone(RecommenderBackbone):
    def __init__(self, num_layers: int, norm_adj: torch.Tensor):
        super().__init__()
        self.num_layers = num_layers
        self.adj = norm_adj

    def forward(self, x0: torch.Tensor) -> list[torch.Tensor]:
        xs = [x0]
        x = x0
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.adj, x)
            xs.append(x)
        return xs


class SAGEBackbone(RecommenderBackbone):
    def __init__(self, hidden_dim: int, num_layers: int, edge_index: torch.Tensor):
        super().__init__()
        self.edge_index = edge_index
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> list[torch.Tensor]:
        xs = [x0]
        x = x0
        for conv in self.convs:
            x = conv(x, self.edge_index)
            x = F.relu(x)
            xs.append(x)
        return xs


class GATBackbone(RecommenderBackbone):
    def __init__(self, hidden_dim: int, num_layers: int, edge_index: torch.Tensor, heads: int = 1, dropout: float = 0.2):
        super().__init__()
        self.edge_index = edge_index
        self.dropout = dropout
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> list[torch.Tensor]:
        xs = [x0]
        x = x0
        for conv in self.convs:
            x = conv(x, self.edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        return xs


# =============================================================
# 6. Wrapper Model
# =============================================================

class RecommenderModel(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 hidden_dim: int,
                 num_layers: int,
                 backbone: Literal['lightgcn', 'sage', 'gat'],
                 edge_index: torch.Tensor,
                 norm_adj: Optional[torch.Tensor] = None,
                 gat_heads: int = 1,
                 gat_dropout: float = 0.2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.total_nodes = num_users + num_items

        # learnable node embeddings as input features
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        if backbone == 'lightgcn':
            assert norm_adj is not None, 'LightGCN needs normalized adjacency.'
            self.backbone = LightGCNBackbone(num_layers, norm_adj)
        elif backbone == 'sage':
            self.backbone = SAGEBackbone(hidden_dim, num_layers, edge_index)
        elif backbone == 'gat':
            self.backbone = GATBackbone(hidden_dim, num_layers, edge_index, heads=gat_heads, dropout=gat_dropout)
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

    def get_embeddings(self):
        x0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        xs = self.backbone(x0)
        x_final = torch.stack(xs, dim=1).mean(dim=1)  # mean over layers (LightGCN-style)
        u_final = x_final[:self.num_users]
        i_final = x_final[self.num_users:]
        return u_final, self.user_emb.weight, i_final, self.item_emb.weight


# =============================================================
# 7. Evaluation
# =============================================================

def evaluate(model: RecommenderModel,
             eval_edge_index: torch.Tensor,
             mask_edge_indices: list[torch.Tensor],
             k: int,
             reg_lambda: float,
             num_users: int,
             num_items: int):
    model.eval()
    with torch.no_grad():
        u_final, u0, i_final, i0 = model.get_embeddings()

        # Loss on eval edges (BPR with structured negative sampling)
        edges = structured_negative_sampling(eval_edge_index, contains_neg_self_loops=False)
        users_idx, pos_idx, neg_idx = edges[0], edges[1], edges[2]
        pos_idx = pos_idx - num_users
        neg_idx = neg_idx - num_users

        loss = bpr_loss(u_final[users_idx], u0[users_idx],
                        i_final[pos_idx], i0[pos_idx],
                        i_final[neg_idx], i0[neg_idx],
                        reg_lambda).item()

        scores = torch.matmul(u_final, i_final.t())  # [U, I]

        # mask seen items
        masked = defaultdict(set)
        for ei in mask_edge_indices:
            up = build_user_pos_dict(ei, num_users)
            for u, items in up.items():
                masked[u].update(items)
        for u, items in masked.items():
            if items:
                scores[u, torch.tensor(list(items), device=scores.device)] = float('-inf')

        gt = build_user_pos_dict(eval_edge_index, num_users)
        recalls, precisions = [], []
        for u, true_items in gt.items():
            if not true_items:
                continue
            topk = scores[u].topk(k).indices.tolist()
            hit = len(set(topk) & true_items)
            recalls.append(hit / len(true_items))
            precisions.append(hit / k)

        if len(recalls) == 0:
            return loss, 0.0, 0.0
        return loss, float(np.mean(recalls)), float(np.mean(precisions))


# =============================================================
# 8. Training
# =============================================================

model = RecommenderModel(num_users=num_users,
                         num_items=num_movies,
                         hidden_dim=CONFIG['hidden_dim'],
                         num_layers=CONFIG['num_layers'],
                         backbone=CONFIG['backbone'],
                         edge_index=train_edge_sym.to(DEVICE),
                         norm_adj=train_adj.to(DEVICE) if CONFIG['backbone'] == 'lightgcn' else None,
                         gat_heads=CONFIG['gat_heads'],
                         gat_dropout=CONFIG['gat_dropout']).to(DEVICE)

train_edge_index = train_edge_index.to(DEVICE)
val_edge_index = val_edge_index.to(DEVICE)

def run_train():
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=CONFIG['lr_decay'])

    for epoch in range(CONFIG['num_epoch']):
        model.train()
        for _ in range(CONFIG['epoch_size']):
            u_final, u0, i_final, i0 = model.get_embeddings()
            users, pos_items, neg_items = mini_batch_sample(CONFIG['batch_size'],
                                                            train_edge_index.cpu(),
                                                            num_users,
                                                            num_movies,
                                                            user_pos_train)
            users = users.to(DEVICE)
            pos_items = pos_items.to(DEVICE)
            neg_items = neg_items.to(DEVICE)

            loss = bpr_loss(u_final[users], u0[users],
                            i_final[pos_items], i0[pos_items],
                            i_final[neg_items], i0[neg_items],
                            CONFIG['lambda'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_recall, val_precision = evaluate(model,
                                                       val_edge_index,
                                                       [train_edge_index],
                                                       CONFIG['topK'],
                                                       CONFIG['lambda'],
                                                       num_users,
                                                       num_movies)
        print(f"Epoch {epoch:02d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | R@{CONFIG['topK']}={val_recall:.4f} | P@{CONFIG['topK']}={val_precision:.4f}")
        scheduler.step()

    # Test
    test_loss, test_recall, test_precision = evaluate(model,
                                                      test_edge_index.to(DEVICE),
                                                      [train_edge_index, val_edge_index],
                                                      CONFIG['topK'],
                                                      CONFIG['lambda'],
                                                      num_users,
                                                      num_movies)
    print(f"Test | loss={test_loss:.4f} | R@{CONFIG['topK']}={test_recall:.4f} | P@{CONFIG['topK']}={test_precision:.4f}")


# =============================================================
# 9. Recommendation helper
# =============================================================

def predict(user_raw_id: int, topK: int = 10):
    model.eval()
    with torch.no_grad():
        if user_raw_id not in user_mapping:
            raise ValueError('Unknown user id')
        u = user_mapping[user_raw_id]
        u_final, _, i_final, _ = model.get_embeddings()
        scores = (i_final @ u_final[u]).cpu()
        # mask seen items
        seen = set()
        for d in [user_pos_train, user_pos_val, user_pos_test]:
            seen.update(d.get(u, set()))
        if seen:
            scores[list(seen)] = float('-inf')
        top_items = scores.topk(topK).indices.tolist()

        inv_movie_map = {v: k for k, v in movie_mapping.items()}
        if movie_path.endswith('u.item'):
            df_movies = pd.read_csv(movie_path, sep='|', header=None, encoding='latin-1', usecols=[0,1],
                                     names=['movieId','title'])
            titles = pd.Series(df_movies.title.values, index=df_movies.movieId).to_dict()
            genres = defaultdict(str)  # 100k has genres as one-hot columns; skip for simplicity
        else:
            df_movies = pd.read_csv(movie_path)
            titles = pd.Series(df_movies.title.values, index=df_movies.movieId).to_dict()
            genres = pd.Series(df_movies.genres.values, index=df_movies.movieId).to_dict()

        print(f'Top {topK} recommendations for user {user_raw_id}:')
        for idx in top_items:
            ml_id = inv_movie_map[idx]
            print(f'- {titles[ml_id]} ({genres[ml_id]})')


if __name__ == '__main__':
    run_train()
    Example: predict(123, 10)
