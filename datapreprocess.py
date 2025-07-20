import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

PATH = 'data/'
user_df = pd.read_csv(f'{PATH}users.csv')
games_df = pd.read_csv(f'{PATH}games.csv')
recommendation_df = pd.read_csv(f'{PATH}recommendations.csv')

print("-----USER CSV HEAD-------")
print(user_df.head())
print("-----GAMES CSV HEAD-------")
print(games_df.head())
print("-----RECS CSV HEAD-------")
print(recommendation_df.head())

"""
user2idx = {u:i for i,u in enumerate(user_df.user_id.unique())}
game2idx = {g:i for i,g in enumerate(games_df.app_id.unique())}



src = recommendation_df.user_id.map(user2idx)
dst = recommendation_df.app_id.map(game2idx) + len(user2idx)   # offset game IDs
edge_index = torch.tensor([src.tolist() + dst.tolist(),
                           dst.tolist() + src.tolist()], dtype=torch.long)


"""