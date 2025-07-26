# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def calc_wtd_rating(x: pd.DataFrame, min_votes: int, overall_avg: int):
    
    w_min = min_votes / (min_votes + x["vote_count"])
    w_vote = x["vote_count"] / (min_votes + x["vote_count"])
    
    wtd_rating = overall_avg * w_min + x["vote_average"] * w_vote
    return wtd_rating

def norm_rating(x: pd.DataFrame):
    
    avg_rating = x.groupby("userId")["rating"].transform(np.mean)
    
    return round(x["rating"] - avg_rating, 2)

def clean_imbd_id(x):
    
    return "tt" + x["imdbId"].astype(str).str.zfill(7)
    
# Data 
min_votes = 1000

credits_df = pd.read_csv("./data/credits.csv")
ratings_small_df = pd.read_csv("./data/ratings_small.csv")
movies_metadata_df = pd.read_csv("./data/movies_metadata.csv")
links_small_df = pd.read_csv("./data/links_small.csv")

links_small_df["imdb_id"] = clean_imbd_id(links_small_df)
movies_metadata_df["release_date"] = pd.to_datetime(movies_metadata_df["release_date"], errors="coerce")

ratings_small_df["rating_date"] = pd.to_datetime(ratings_small_df["timestamp"], unit='s')
ratings_small_df = ratings_small_df.merge(links_small_df, how="left", on="movieId")

ratings_df_merged = ratings_small_df.merge(movies_metadata_df, how="left", on="imdb_id")

outliers_df = ratings_df_merged[(ratings_df_merged["rating_date"] - ratings_df_merged["release_date"]).dt.days < 0]

overall_avg = movies_metadata_df["vote_average"].mean()
movies_metadata_df["wtd_rating"] =( movies_metadata_df
                                   .apply(lambda x: 
                                          calc_wtd_rating(x, 
                                                          min_votes = min_votes, 
                                                          overall_avg=overall_avg
                                                          ), axis=1))
# Modeling
movies_small_df.head()
movies_small_df["norm_rating"] = norm_rating(movies_small_df)

movies_df = movies_small_df.pivot(index="userId", columns="movieId", values="norm_rating")


# Evaluation



# Serving