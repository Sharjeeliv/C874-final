import pandas as pd
from pathlib import Path

ROOT_PATH = Path.cwd()
DATASET_NAME = "ml-100k"
DATA_PATH = ROOT_PATH / 'data'
RAW_DATA_PATH = DATA_PATH / DATASET_NAME / 'raw'

# Validate if dataset is present
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset directory {DATA_PATH} does not exist. \
                            Please ensure the dataset is downloaded and placed in the correct directory.")

# Interaction data
headings = ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]
interactions = pd.read_csv(RAW_DATA_PATH / "u.data", sep="\t", header=None, names=headings)
interactions.to_csv(DATA_PATH / DATASET_NAME / f"{DATASET_NAME}.inter", sep="\t", index=False)

# Item data
headings = [
    'item_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url',
    'unknown_genre', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
items = pd.read_csv(RAW_DATA_PATH / "u.item", sep="|", header=None, names=headings, encoding="latin-1")
# Process item data
items['release_year'] = pd.to_datetime(items['release_date'], errors='coerce').dt.year.astype('Int64')
items['movie_title'] = items['movie_title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
genre_columns = headings[5:]  # Exclude the first five columns
items['class'] = items[genre_columns].apply(lambda x: ' '.join(x.index[x == 1]), axis=1)
# Select relevant columns and rename them
items = items[['item_id', 'movie_title', 'release_year', 'class']]
items.rename(columns={
    'item_id': 'item_id:token',
    'movie_title': 'item_name:token_seq',
    'release_year': 'release_year:token',
    'class': 'genres:token_seq'
}, inplace=True)
items.to_csv(DATA_PATH / DATASET_NAME / f"{DATASET_NAME}.item", sep="\t", index=False)

# User data
user_headings = ["user_id:token", "age:token", "gender:token", "occupation:token", "zip_code:token"]
users = pd.read_csv(RAW_DATA_PATH / "u.user", sep="|", header=None, names=user_headings, encoding="latin-1")
users['age:token'] = users['age:token'].astype(str)  # Ensure
# age is treated as a string
users['zip_code:token'] = users['zip_code:token'].astype(str)
users.to_csv(DATA_PATH / DATASET_NAME / f"{DATASET_NAME}.user", sep="\t", index=False)

if __name__ == "__main__":
    print("Preparing data for ML-100K")
    print(f"Save path: {DATA_PATH}")