import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


class MovieLensDataLoader:    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.ml_dir = os.path.join(self.raw_dir, "ml-1m")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download_and_extract(self):
        
        DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(self.raw_dir, "ml-1m.zip")
        
        if os.path.exists(self.ml_dir):
            return
        
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(DATA_URL, zip_path)
            print("Download complete")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        print("Extraction complete")
    
    def load_raw_data(self):
        self.download_and_extract()
        
        ratings = pd.read_csv(
            os.path.join(self.ml_dir, "ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1"
        )
        
        movies = pd.read_csv(
            os.path.join(self.ml_dir, "movies.dat"),
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1"
        )
        
        users = pd.read_csv(
            os.path.join(self.ml_dir, "users.dat"),
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"],
            encoding="latin-1"
        )
        
        print(f"Loaded {len(ratings):,} ratings")
        print(f"Loaded {len(movies):,} movies")
        print(f"Loaded {len(users):,} users")
        
        return ratings, movies, users
    
    def load_splits(self):
        train_path = os.path.join(self.processed_dir, "train.csv")
        val_path = os.path.join(self.processed_dir, "val.csv")
        test_path = os.path.join(self.processed_dir, "test.csv")
        
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        
        print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
        
        return train, val, test
    
    def save_splits(self, train, val, test):
        train.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.processed_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
        print(f"Saved splits to {self.processed_dir}")


def build_user_item_matrix(df, n_users=None, n_items=None):
    if n_users is None:
        n_users = df['user_id'].max() + 1
    if n_items is None:
        n_items = df['item_id'].max() + 1
    
    rows = df['user_id'].values
    cols = df['item_id'].values
    data = df['rating'].values
    
    matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Built {matrix.shape} matrix, sparsity: {sparsity:.4f}")
    
    return matrix


def per_user_temporal_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    
    train_data = []
    val_data = []
    test_data = []
    
    for user_id, user_df in df.groupby('user_id'):
        user_sorted = user_df.sort_values('timestamp')
        n = len(user_sorted)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.append(user_sorted.iloc[:train_end])
        val_data.append(user_sorted.iloc[train_end:val_end])
        test_data.append(user_sorted.iloc[val_end:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def load_data():
    loader = MovieLensDataLoader()
    train, val, test = loader.load_splits()
    _, movies, _ = loader.load_raw_data()
    return train, val, test, movies


def load_full_data():
    loader = MovieLensDataLoader()
    return loader.load_raw_data()