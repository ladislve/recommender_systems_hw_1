import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict

class PopularityRanker:
    def __init__(self, train_df, min_ratings=5):
        counts = train_df.groupby('item_id').size()
        self.scores = counts.to_dict()
        self.ranked = counts[counts >= min_ratings].sort_values(ascending=False).index.tolist()
        self._user_seen = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    def predict_for_user(self, user_id, k=10, train_df=None):
        seen = self._user_seen.get(user_id, set())
        recs = [(iid, self.scores[iid]) for iid in self.ranked if iid not in seen]
        return recs[:k]


class RecencyRanker:
    def __init__(self, train_df, half_life_days=90, min_ratings=5):
        df = train_df.copy()
        max_ts = df['timestamp'].max()
        days_ago = (max_ts - df['timestamp']) / 86400
        df['weight'] = np.exp(-np.log(2) * days_ago / half_life_days)

        item_scores = df.groupby('item_id')['weight'].sum()
        item_counts = df.groupby('item_id').size()
        item_scores = item_scores[item_counts >= min_ratings]

        self.scores = item_scores.to_dict()
        self.ranked = item_scores.sort_values(ascending=False).index.tolist()
        self._user_seen = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    def predict_for_user(self, user_id, k=10, train_df=None):
        seen = self._user_seen.get(user_id, set())
        recs = [(iid, self.scores[iid]) for iid in self.ranked if iid not in seen]
        return recs[:k]


class PersonalizedPageRankRanker:
    def __init__(self, train_df, alpha=0.15, n_iterations=20, threshold=4.0):
        self.alpha = alpha
        self.n_iter = n_iterations

        df = train_df[train_df['rating'] >= threshold]
        self.n_users = train_df['user_id'].max() + 1
        self.n_items = train_df['item_id'].max() + 1
        n = self.n_users + self.n_items

        rows = np.concatenate([df['user_id'].values, self.n_users + df['item_id'].values])
        cols = np.concatenate([self.n_users + df['item_id'].values, df['user_id'].values])
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))

        degree = np.array(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1
        inv_deg = 1.0 / degree
        self.transition = csr_matrix(
            (inv_deg[adj.nonzero()[0]] * adj.data, adj.indices, adj.indptr), shape=(n, n)
        )

        self._user_seen = {}
        for uid, g in train_df.groupby('user_id'):
            self._user_seen[uid] = set(g['item_id'])

    def run_ppr(self, user_id):
        n = self.n_users + self.n_items
        teleport = np.zeros(n, dtype=np.float32)
        teleport[user_id] = 1.0
        scores = teleport.copy()
        for _ in range(self.n_iter):
            scores = (1 - self.alpha) * (self.transition.T @ scores) + self.alpha * teleport
        return scores[self.n_users:]

    def predict_for_user(self, user_id, k=10, train_df=None):
        if user_id >= self.n_users:
            return []
        item_scores = self.run_ppr(user_id)
        seen = self._user_seen.get(user_id, set())
        for iid in seen:
            if iid < len(item_scores):
                item_scores[iid] = -np.inf
        top_k = np.argpartition(item_scores, -k)[-k:]
        top_k = top_k[np.argsort(item_scores[top_k])[::-1]]
        return [(int(i), float(item_scores[i])) for i in top_k if item_scores[i] > -np.inf]