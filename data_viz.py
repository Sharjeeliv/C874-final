import argparse
import json
import os
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations

def load_data(users_path, games_path, recs_path, meta_path):
    direc = 'data/'
    users = pd.read_csv(direc+users_path)
    games = pd.read_csv(direc+games_path)
    recs = pd.read_csv(direc+recs_path, parse_dates=['date'])

    # Try standard JSON load; if it fails, parse as JSON lines
    try:
        with open(direc+meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        meta = {}
        with open(direc+meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    meta.update(obj)
                except json.JSONDecodeError:
                    continue
    return users, games, recs, meta

"""
def plot_user_stats(users):
    # number of products per user
    plt.figure()
    plt.hist(users['products'], bins=50, log=True)
    plt.xlabel('Products per user')
    plt.ylabel('Count of users')
    plt.title('Distribution of Products Owned')
    plt.savefig("Figures/prodcutdist")
    plt.clf()

    # number of reviews per user
    plt.figure()
    plt.hist(users['reviews'], bins=20, log=True)
    plt.xlabel('Reviews written per user')
    plt.ylabel('Count of users')
    plt.title('Distribution of Reviews Written')
    plt.savefig("Figures/reviewdist")
    plt.clf()

def plot_game_metadata(games):
    # Release year histogram
    years = pd.to_datetime(games['date_release'], errors='coerce').dt.year.dropna().astype(int)
    plt.figure()
    plt.hist(years, bins=range(years.min(), years.max()+2), edgecolor='black')
    plt.xlabel('Release Year')
    plt.ylabel('Count of Games')
    plt.title('Game Release Years')
    plt.savefig("Figures/gameyears")
    plt.clf()

    # Price distribution (final)
    plt.figure()
    plt.hist(games['price_final'], bins=50, log=True)
    plt.xlabel('Final Price (USD)')
    plt.ylabel('Count of Games')
    plt.title('Distribution of Final Prices')
    plt.savefig("Figures/pricedist")
    plt.clf()

    # Discount distribution
    plt.figure()
    plt.hist(games['discount'], bins=20)
    plt.xlabel('Discount (%)')
    plt.ylabel('Count of Games')
    plt.title('Discount Percentage Distribution')
    plt.savefig("Figures/discountdist")
    plt.clf()

    # Positive review ratio
    plt.figure()
    plt.hist(games['positive_ratio'], bins=20)
    plt.xlabel('Positive Review Ratio (%)')
    plt.ylabel('Count of Games')
    plt.title('Positive Review Ratio Distribution')
    plt.savefig("Figures/postivereviewdist")
    plt.clf()

    # Platform support counts
    plt.figure()
    platforms = ['win', 'mac', 'linux', 'steam_deck']
    support = [games[plat].sum() for plat in platforms]
    plt.bar(platforms, support)
    plt.ylabel('Number of Games Supported')
    plt.title('OS / Steam Deck Support')
    plt.savefig("Figures/OSsupportdist")
    plt.clf()

def plot_recommendation_stats(recs):
    # Ensure date is datetime and set as index for time-based rolling
    recs = recs.copy()
    recs['date'] = pd.to_datetime(recs['date'], errors='coerce')
    recs.sort_values('date', inplace=True)
    recs.set_index('date', inplace=True)

    plt.figure()
    plt.hist(recs['hours'].dropna(), bins=50, log=True)
    plt.xlabel('Hours Played')
    plt.ylabel('Count of Reviews')
    plt.title('Distribution of Hours Played')
    plt.savefig("Figures/hoursplayed")
    plt.clf()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(recs['helpful'].dropna(), bins=20)
    axs[0].set_title('Helpful Votes')
    axs[0].set_xlabel('Helpful')
    axs[0].set_ylabel('Count')
    axs[1].hist(recs['funny'].dropna(), bins=20)
    axs[1].set_title('Funny Votes')
    axs[1].set_xlabel('Funny')
    axs[1].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig("Figures/funnyvshelpful")
    plt.clf()

    # Rolling 30-day mean of boolean is_recommended
    if recs['is_recommended'].dtype != 'int':
        recs['is_recommended'] = recs['is_recommended'].astype(int)
    roll = recs['is_recommended'].rolling('30D').mean()
    plt.figure()
    plt.plot(roll.index, roll.values)
    plt.xlabel('Date')
    plt.ylabel('30-day Rolling % Recommended')
    plt.title('Temporal Trend of Recommendations')
    plt.savefig("Figures/recstemporal")
    plt.clf()
"""

def plot_rating_dist(recs):
    counts = recs['is_recommended'].value_counts()
    sizes = [counts.get(True, 0), counts.get(False, 0)]
    labels = ['Recommended', 'Not Recommended']
    colors = ['tab:cyan', 'tab:red']
    explode = (0.05, 0.05)  # offset both slices slightly

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.80,
        wedgeprops=dict(width=0.3, edgecolor='white'),
        textprops=dict(color='black', fontsize=12, weight='bold'),
        shadow=True
    )

    # Draw a circle at the center to make it a donut
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    plt.gca().add_artist(centre_circle)

    # Legend outside
    plt.legend(
        wedges,
        labels,
        title="Review Label",
        loc="center",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.title('Recommendation Label Distribution', fontsize=14, weight='bold')
    plt.axis('equal')  # keep it circular

    out_path = os.path.join('Figures', "label_distribution_pie.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_bipartite_graph_stats(recs, games, sample_size=10000, top_n=30, min_user_overlap_games=2,
        sample_users=None,
        normalize=None,      # None | "jaccard" | "cosine"
        out_dir="Figures",
        max_label_len=20,
        include_id=False, wrap_labels=False, dpi=150):

    subsample = recs.sample(n=min(sample_size, len(recs)), random_state=42)

    G = nx.Graph()
    # preserve bipartite labels
    for uid in subsample['user_id'].unique():
        G.add_node(uid, bipartite='user')
    for aid in subsample['app_id'].unique():
        G.add_node(aid, bipartite='game')
    G.add_edges_from(zip(subsample['user_id'], subsample['app_id']))

    # separate nodes by attribute instead of sets()
    user_nodes = [n for n,d in G.nodes(data=True) if d.get('bipartite')=='user']
    game_nodes = [n for n,d in G.nodes(data=True) if d.get('bipartite')=='game']
    user_degs = [G.degree(u) for u in user_nodes]
    game_degs = [G.degree(i) for i in game_nodes]

    plt.figure()
    plt.hist(user_degs, bins=50, log=True)
    plt.xlabel('User Degree')
    plt.ylabel('Count')
    plt.title('User Degree Distribution (Subsampled)')
    #plt.savefig("Figures/userdist")
    plt.clf()

    plt.figure()
    plt.hist(game_degs, bins=50, log=True)
    plt.xlabel('Game Degree')
    plt.ylabel('Count')
    plt.title('Game Degree Distribution (Subsampled)')
    #plt.savefig("Figures/gamedist")
    plt.clf()

    # --- Mapping app_id -> title ---
    id2title = dict(zip(games['app_id'], games['title']))

    # 1. Positive interactions only
    pos = recs[recs['is_recommended'] == True]
    if pos.empty:
        print("[CoPlay] No positive interactions found.")
        return

    # 2. Top-N games
    game_counts = pos['app_id'].value_counts()
    top_games = game_counts.nlargest(top_n).index.tolist()
    actual_top_n = len(top_games)
    print(f"[CoPlay] Using top {actual_top_n} games for heatmap.")

    # 3. Per-user restricted lists
    user_games = defaultdict(list)
    sub = pos[pos['app_id'].isin(top_games)][['user_id', 'app_id']]
    for uid, gid in zip(sub['user_id'].values, sub['app_id'].values):
        user_games[uid].append(gid)

    # Filter users by min overlap
    filtered_users = [u for u, glist in user_games.items()
                      if len(set(glist)) >= min_user_overlap_games]

    # Optional sampling
    if sample_users is not None and len(filtered_users) > sample_users:
        rng = np.random.default_rng(42)
        filtered_users = rng.choice(filtered_users, size=sample_users, replace=False)
        print(f"[CoPlay] Sampled {len(filtered_users)} users (from {len(user_games)} candidates).")
    else:
        print(f"[CoPlay] Using {len(filtered_users)} qualifying users.")

    if len(filtered_users) == 0:
        print("[CoPlay] No users meet the overlap criterion; aborting heatmap.")
        return

    # 4. Build index map & count
    idx_map = {g: i for i, g in enumerate(top_games)}
    co_matrix = np.zeros((actual_top_n, actual_top_n), dtype=np.int32)
    game_user_count = np.zeros(actual_top_n, dtype=np.int32)

    for u in filtered_users:
        glist = list({g for g in user_games[u] if g in idx_map})
        for g in glist:
            game_user_count[idx_map[g]] += 1
        if len(glist) > 1:
            for g1, g2 in combinations(glist, 2):
                i, j = idx_map[g1], idx_map[g2]
                co_matrix[i, j] += 1
                co_matrix[j, i] += 1

    # 5. Normalization (optional)
    if normalize == "jaccard":
        a = game_user_count.reshape(-1, 1)
        denom = (a + a.T - co_matrix).astype(float)
        denom[denom == 0] = 1.0
        norm_matrix = co_matrix / denom
        metric_label = "Jaccard Similarity"
        fname_suffix = "jaccard"
    elif normalize == "cosine":
        a = game_user_count.reshape(-1, 1)
        denom = np.sqrt(a * a.T).astype(float)
        denom[denom == 0] = 1.0
        norm_matrix = co_matrix / denom
        metric_label = "Cosine Similarity"
        fname_suffix = "cosine"
    else:
        norm_matrix = co_matrix
        metric_label = "Shared Users"
        fname_suffix = "shared"

    np.fill_diagonal(norm_matrix, 0)

    # 6. Prepare labels (titles)
    def format_label(app_id):
        title = id2title.get(app_id, str(app_id))
        if wrap_labels:
            # crude wrap
            wrapped = "\n".join(textwrap.wrap(title, width=max_label_len))
            truncated = wrapped
        else:
            truncated = (title[:max_label_len] + "…") if len(title) > max_label_len else title
        if include_id:
            truncated += f" ({app_id})"
        return truncated

    labels = [format_label(g) for g in top_games]

    # 7. Plot
    plt.figure(figsize=(max(8, actual_top_n * 0.4), max(8, actual_top_n * 0.4)))
    im = plt.imshow(norm_matrix, cmap="viridis", interpolation="nearest")
    plt.xticks(range(actual_top_n), labels, rotation=90)
    plt.yticks(range(actual_top_n), labels)
    plt.colorbar(im, label=metric_label)
    plt.title(f"Co-play Heatmap (Top {actual_top_n} Games) — {metric_label}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"co_play_heatmap_top{actual_top_n}_{fname_suffix}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"[CoPlay] Saved heatmap with titles to {out_path}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--users', required=False)
    p.add_argument('--games', required=False)
    p.add_argument('--recs', required=False)
    p.add_argument('--meta', required=False)
    args = p.parse_args()

    users, games, recs, meta = load_data('users.csv', 'games.csv'
                                         , 'recommendations.csv', 'games_metadata.json')


    #plot_user_stats(users)
    #plot_game_metadata(games)
    #plot_recommendation_stats(recs)
    plot_rating_dist(recs)
    """plot_bipartite_graph_stats(recs,
        games,
        top_n=30,
        min_user_overlap_games=6,
        sample_users=3709628,  # adjust or None
        normalize=None,  # "cosine", "jaccard",  or None
        out_dir="Figures",
        max_label_len=20,
        include_id=False,
        wrap_labels=False
    )"""

if __name__ == '__main__':
    main()