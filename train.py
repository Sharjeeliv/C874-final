import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from SteamGNN import SteamGNN
import time
# 2. Data loading & preprocessing
def load_and_process(users_path, games_path, recs_path):

    users_path = 'data/users.csv'
    games_path = 'data/games.csv'
    recs_path = 'data/recommendations.csv'
    # ——— Load raw CSVs ———
    print("1/12: Loading CSV files")
    users = pd.read_csv(users_path)
    games = pd.read_csv(games_path)
    recs = pd.read_csv(recs_path, parse_dates=["date"])
    print(f"   Loaded {len(users)} users, {len(games)} games, {len(recs)} interactions")

    # ——— Filter & reset index so recs indices run 0…N-1 ———
    #    (ensures train/val/test splits align with edge order)
    print("2/12: Filtering interactions to valid users/games")
    recs = recs[recs.user_id.isin(users.user_id) & recs.app_id.isin(games.app_id)]
    recs = recs.reset_index(drop=True)
    print(f"   {len(recs)} interactions remain after filtering")

    # ——— ID remapping ———

    user2idx = {u: i for i, u in enumerate(users.user_id.unique())}
    game2idx = {g: i for i, g in enumerate(games.app_id.unique())}

    # ——— Build user features (products, reviews) ———
    print("4/12: Building user feature matrix")
    users_sorted = users.set_index("user_id").loc[user2idx.keys()]
    user_feats = torch.tensor(
        users_sorted[["products", "reviews"]].to_numpy(dtype=float),
        dtype=torch.float
    )  # [num_users, 2]
    print(f"   user_feats shape: {user_feats.shape}")

    # ——— Build game features (numeric + bool + year bin) ———
    print("5/12: Building game feature matrix")
    gf = games.set_index("app_id").loc[game2idx.keys()]
    years = pd.to_datetime(gf["date_release"], errors="coerce").dt.year.fillna(0).astype(int)
    year_bins = pd.cut(years,
                       bins=[0, 2009, 2014, 2019, 2025],
                       labels=False).to_numpy().reshape(-1, 1)
    numeric = gf[["positive_ratio", "user_reviews", "price_final", "price_original", "discount"]].to_numpy()
    bools   = gf[["win", "mac", "linux", "steam_deck"]].astype(int).to_numpy()
    game_feats = torch.tensor(
        np.hstack([numeric, bools, year_bins]),
        dtype=torch.float
    )  # [num_games, 10]

    # ——— Pad smaller feature matrix to match widths ———
    uf, gf_ = user_feats.size(1), game_feats.size(1)
    if uf < gf_:
        pad = torch.zeros((user_feats.size(0), gf_ - uf))
        user_feats = torch.cat([user_feats, pad], dim=1)
    elif gf_ < uf:
        pad = torch.zeros((game_feats.size(0), uf - gf_))
        game_feats = torch.cat([game_feats, pad], dim=1)

    print(f"   game_feats shape: {game_feats.shape}")

    # ——— Final node feature matrix X ———
    print("7/12: Concatenating user and game features")
    x = torch.cat([user_feats, game_feats], dim=0)
    print(f"   Combined x shape: {x.shape}")

    # ——— Build edges, edge_attr, and labels ———
    # 8. Build edge_index
    print("8/12: Constructing edge_index")
    u = recs.user_id.map(user2idx).to_numpy()
    v = recs.app_id.map(game2idx).to_numpy() + len(user2idx)
    u_v = np.hstack([u, v])
    v_u = np.hstack([v, u])
    edges_np = np.vstack([u_v, v_u])
    edge_index = torch.from_numpy(edges_np).long()
    print(f"   edge_index shape: {edge_index.shape}")

    # 9. Build edge_attr and labels
    print("9/12: Building edge attributes and labels")
    attrs = recs[["helpful", "funny", "hours"]].to_numpy()
    edge_attr = torch.from_numpy(np.vstack([attrs, attrs])).float()
    labels = torch.from_numpy(
        np.hstack([recs.is_recommended.astype(int), recs.is_recommended.astype(int)])
    ).float()
    print(f"   edge_attr shape: {edge_attr.shape}, labels shape: {labels.shape}")

    # 10. Train/val/test split
    print("10/12: Splitting data into train/val/test by time")
    recs_sorted = recs.sort_values("date")
    train_idx, val_idx, test_idx = [], [], []
    for uid, grp in recs_sorted.groupby("user_id"):
        n = len(grp)
        nt = max(1, int(n * 0.1))
        nv = max(1, int(n * 0.1))
        idx = grp.index.to_list()
        test_idx += idx[-nt:]
        val_idx += idx[-nt - nv:-nt]
        train_idx += idx[:-nt - nv]
    print(f"   #train: {len(train_idx)}, #val: {len(val_idx)}, #test: {len(test_idx)}")

    # 11. Convert indices to torch tensors
    print("11/12: Converting split indices to edge indices")
    N = len(recs)
    train_arr = np.array(train_idx, dtype=int)
    val_arr = np.array(val_idx, dtype=int)
    test_arr = np.array(test_idx, dtype=int)
    train_edges = torch.from_numpy(np.concatenate([train_arr, train_arr + N])).long()
    val_edges = torch.from_numpy(np.concatenate([val_arr, val_arr + N])).long()
    test_edges = torch.from_numpy(np.concatenate([test_arr, test_arr + N])).long()
    print("   Converted train/val/test edges")

    # 12. Create Data object
    print("12/12: Creating PyG Data object")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    print("Data object ready")
    return data, train_edges, val_edges, test_edges, (user2idx, game2idx)

# 3. Training loop
def train(args):
    data, train_edges, val_edges, test_edges, maps = load_and_process(
        args.users, args.games, args.recs)
    print("Initializing model and optimizer")
    model = SteamGNN(in_feats=data.x.size(1), hid_feats=args.hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_epochs = args.epochs
    start_time = time.time()

    print(f"Training for {total_epochs} epochs...\n")
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{total_epochs} - starting")

        model.train()
        opt.zero_grad()

        h = model(data.x, data.edge_index)
        pos_score = model.decode(h, data.edge_index[:, train_edges])

        neg_edge = negative_sampling(
            edge_index=data.edge_index[:, train_edges],
            num_nodes=data.num_nodes,
            num_neg_samples=train_edges.size(0)
        )
        neg_score = model.decode(h, neg_edge)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ], dim=0)

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        opt.step()

        # timing
        elapsed = time.time() - start_time
        avg_per_epoch = elapsed / epoch
        remaining = total_epochs - epoch
        eta = avg_per_epoch * remaining

        # format times
        def fmt(t):
            m, s = divmod(int(t), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        print(f"  -> Completed epoch {epoch}/{total_epochs}")
        print(f"     Loss: {loss.item():.4f}")
        print(f"     Elapsed: {fmt(elapsed)} | ETA: {fmt(eta)} ({remaining} epochs left)\n")

    # Save artifacts
    torch.save(model.state_dict(), args.save_model)
    torch.save({
        "train_edges": train_edges,
        "val_edges":   val_edges,
        "test_edges":  test_edges,
        "maps":        maps
    }, args.save_splits)
    print("Training complete. Model and splits saved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users",      required=False)
    p.add_argument("--games",      required=False)
    p.add_argument("--recs",       required=False)
    p.add_argument("--hidden",  type=int,   default=64)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_model", default="steam_gnn.pt")
    p.add_argument("--save_splits",default="splits.pt")
    args = p.parse_args()
    train(args)