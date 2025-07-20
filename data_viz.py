import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
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

def plot_bipartite_graph_stats(recs, sample_size=10000, top_n=30):
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
    plt.savefig("Figures/userdist")
    plt.clf()

    plt.figure()
    plt.hist(game_degs, bins=50, log=True)
    plt.xlabel('Game Degree')
    plt.ylabel('Count')
    plt.title('Game Degree Distribution (Subsampled)')
    plt.savefig("Figures/gamedist")
    plt.clf()

    interactions = subsample[recs['is_recommended'] == True]

    # 2. build binary user × game pivot
    pivot = (
        interactions
        .assign(interact=1)
        .pivot_table(index='user_id',
                     columns='app_id',
                     values='interact',
                     aggfunc='sum',
                     fill_value=0)
        .clip(0, 1)  # ensure binary
    )

    # 3. pick the top-N most-interacted games
    game_counts = pivot.sum(axis=0)
    top_games = game_counts.nlargest(top_n).index.tolist()
    mat = pivot[top_games]

    # 4. compute co-play matrix
    co_matrix = mat.T.dot(mat).values  # shape (top_n, top_n)
    np.fill_diagonal(co_matrix, 0)  # zero out self-cooccurrence

    # 5. order by total co-plays (optional: cluster for block structure)
    order = np.argsort(game_counts[top_games])[::-1]
    co_matrix = co_matrix[order][:, order]
    labels = [top_games[i] for i in order]

    # 6. plot
    plt.figure(figsize=(8, 8))
    im = plt.imshow(co_matrix, interpolation='nearest')
    plt.xticks(range(top_n), labels, rotation='vertical')
    plt.yticks(range(top_n), labels)
    plt.colorbar(im, label='Number of shared users')
    plt.title(f'Co-play Heatmap (Top {top_n} Games by Play Count)')
    plt.tight_layout()
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--users', required=True)
    p.add_argument('--games', required=True)
    p.add_argument('--recs', required=True)
    p.add_argument('--meta', required=True)
    args = p.parse_args()

    users, games, recs, meta = load_data(args.users, args.games, args.recs, args.meta)

    plot_user_stats(users)
    plot_game_metadata(games)
    plot_recommendation_stats(recs)
    plot_bipartite_graph_stats(recs)

if __name__ == '__main__':
    main()