import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(users_path: str, movies_path: str, ratings_path: str):
    """Load ml-100k files and return tidy DataFrames with consistent column names."""
    # u.user: user id | age | gender | occupation | zip code
    users = pd.read_csv(
        users_path,
        sep='|',
        header=None,
        names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip']
    )

    # u.item: movie id | title | release date | video release date | IMDb URL | 19 genre flags
    genre_cols = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    movies = pd.read_csv(
        movies_path,
        sep='|',
        header=None,
        encoding='latin-1',
        names=['MovieID', 'Title', 'ReleaseDate', 'VideoReleaseDate', 'IMDbURL'] + genre_cols
    )

    # Add a joined Genres string for convenience (used by a couple of plots)
    movies['Genres'] = movies[genre_cols].apply(
        lambda row: '|'.join([g for g in genre_cols if row[g] == 1]), axis=1
    )

    # u.data: user id | item id | rating | timestamp
    ratings = pd.read_csv(
        ratings_path,
        sep='\t',
        header=None,
        names=['UserID', 'MovieID', 'Rating', 'Timestamp']
    )

    return users, movies, ratings


def plot_rating_histogram(ratings: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    bins = np.arange(ratings['Rating'].min() - 0.5, ratings['Rating'].max() + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ratings['Rating'], bins=bins, color='tab:blue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Ratings', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rating_distribution.png'), dpi=150)
    plt.close(fig)


def plot_ratings_per_user(ratings: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    counts = ratings.groupby('UserID').size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=50, color='tab:orange', edgecolor='white', alpha=0.8, log=True)
    ax.set_xlabel('Number of Ratings per User', fontsize=12)
    ax.set_ylabel('Count of Users (log scale)', fontsize=12)
    ax.set_title('Ratings per User', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ratings_per_user.png'), dpi=150)
    plt.close(fig)


def plot_ratings_per_movie(ratings: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    counts = ratings.groupby('MovieID').size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=50, color='tab:green', edgecolor='white', alpha=0.8, log=True)
    ax.set_xlabel('Number of Ratings per Movie', fontsize=12)
    ax.set_ylabel('Count of Movies (log scale)', fontsize=12)
    ax.set_title('Ratings per Movie', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ratings_per_movie.png'), dpi=150)
    plt.close(fig)


def plot_user_demographics(users: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    # Age distribution (raw ages, 5-year bins)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(users['Age'], bins=range(users['Age'].min(), users['Age'].max() + 5, 5),
            color='tab:purple', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count of Users', fontsize=12)
    ax.set_title('User Age Distribution', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'age_distribution.png'), dpi=150)
    plt.close(fig)

    # Gender distribution
    fig, ax = plt.subplots(figsize=(4, 4))
    gender_counts = users['Gender'].value_counts().reindex(['M', 'F'])
    ax.bar(gender_counts.index, gender_counts.values,
           color=['tab:blue', 'tab:red'], edgecolor='white', alpha=0.9)
    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Count of Users', fontsize=12)
    ax.set_title('Gender Distribution', fontsize=14, weight='bold')
    for i, v in enumerate(gender_counts.values):
        ax.text(i, v + v * 0.01, str(v), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'gender_distribution.png'), dpi=150)
    plt.close(fig)

    # Occupation distribution (top 10) – occupations are already strings
    occ_counts = users['Occupation'].value_counts().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(occ_counts.index[::-1], occ_counts.values[::-1],
            color='tab:cyan', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Count of Users', fontsize=12)
    ax.set_title('Top 10 Occupations', fontsize=14, weight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'occupation_distribution.png'), dpi=150)
    plt.close(fig)


def plot_genre_distribution(movies: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    # Using one-hot genre columns if present
    genre_cols = [c for c in movies.columns if c not in
                  ['MovieID', 'Title', 'ReleaseDate', 'VideoReleaseDate', 'IMDbURL', 'Genres']]
    # Filter to only the 0/1 genre columns
    genre_cols = [c for c in genre_cols if movies[c].dropna().isin([0, 1]).all()]
    if genre_cols:
        genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
        labels = genre_counts.index
        values = genre_counts.values
    else:
        # fallback if something odd happened
        all_genres = movies['Genres'].str.split('|').explode()
        genre_counts = all_genres.value_counts()
        labels = genre_counts.index
        values = genre_counts.values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1], color='tab:olive', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Count of Movies', fontsize=12)
    ax.set_title('Movie Genre Distribution', fontsize=14, weight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'genre_distribution.png'), dpi=150)
    plt.close(fig)


def plot_avg_rating_by_age_gender(ratings: pd.DataFrame, users: pd.DataFrame, out_dir: str = "Figures"):
    ensure_dir(out_dir)
    df = ratings.merge(users, on='UserID')
    # Bin ages into groups similar to ML-1M but computed from raw ages
    bins = [0, 17, 24, 34, 44, 49, 55, 100]
    labels = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True, include_lowest=True)

    pivot = (df.pivot_table(index='age_group', columns='Gender', values='Rating', observed=False,
                            aggfunc='mean')
               .reindex(labels)[['M', 'F']])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot, cmap='viridis', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Avg Rating', fontsize=12)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Age Group', fontsize=12)
    ax.set_title('Average Rating by Age & Gender', fontsize=14, weight='bold')

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'avg_rating_age_gender_heatmap.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EDA for MovieLens-100K dataset")
    parser.add_argument('--users', default='ml-100k/u.user', help='Path to u.user')
    parser.add_argument('--movies', default='ml-100k/u.item', help='Path to u.item')
    parser.add_argument('--ratings', default='ml-100k/u.data', help='Path to u.data')
    parser.add_argument('--out_dir', default='Figures-ml100k', help='Directory to save figures')
    args = parser.parse_args()

    users, movies, ratings = load_data(args.users, args.movies, args.ratings)

    plot_rating_histogram(ratings, args.out_dir)
    plot_ratings_per_user(ratings, args.out_dir)
    plot_ratings_per_movie(ratings, args.out_dir)
    plot_user_demographics(users, args.out_dir)
    plot_genre_distribution(movies, args.out_dir)
    plot_avg_rating_by_age_gender(ratings, users, args.out_dir)


if __name__ == '__main__':
    main()