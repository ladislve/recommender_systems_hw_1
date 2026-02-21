import numpy as np
import pandas as pd
from numba import njit, prange

@njit(fastmath=True)
def bpr_epoch(user_ids, pos_items_flat, pos_offsets, pos_counts,
              n_items, user_factors, item_factors, user_bias, item_bias,
              lr, reg, n_samples):
    loss = 0.0
    n_users = len(pos_offsets)

    for _ in range(n_samples):
        u = np.random.randint(0, n_users)
        uid = user_ids[u]
        count = pos_counts[u]
        if count == 0:
            continue
        offset = pos_offsets[u]

        pi = pos_items_flat[offset + np.random.randint(0, count)]

        ni = np.random.randint(0, n_items)
        attempts = 0
        while attempts < 10:
            found = False
            for k in range(count):
                if pos_items_flat[offset + k] == ni:
                    found = True
                    break
            if not found:
                break
            ni = np.random.randint(0, n_items)
            attempts += 1

        x_ui = user_bias[uid] + item_bias[pi] + np.dot(user_factors[uid], item_factors[pi])
        x_uj = user_bias[uid] + item_bias[ni] + np.dot(user_factors[uid], item_factors[ni])
        x_uij = x_ui - x_uj

        sig = 1.0 / (1.0 + np.exp(x_uij))
        loss += np.log(1.0 / (1.0 + np.exp(-x_uij)) + 1e-10)

        user_factors[uid] += lr * (sig * (item_factors[pi] - item_factors[ni]) - reg * user_factors[uid])
        item_factors[pi] += lr * (sig * user_factors[uid] - reg * item_factors[pi])
        item_factors[ni] += lr * (-sig * user_factors[uid] - reg * item_factors[ni])

        item_bias[pi] += lr * (sig - reg * item_bias[pi])
        item_bias[ni] += lr * (-sig - reg * item_bias[ni])

    return loss / n_samples


class BPRMF:
    def __init__(self, n_users, n_items, n_factors=64,
                 lr=0.01, reg=0.001, n_epochs=40,
                 samples_per_epoch=None, verbose=True):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.samples_per_epoch = samples_per_epoch
        self.verbose = verbose

        np.random.seed(42)
        self.user_factors = np.random.normal(0, 0.01, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.01, (n_items, n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.loss_history = []

    def fit(self, user_pos_items):
        user_ids, pos_flat, offsets, counts = [], [], [], []
        offset = 0
        for uid in range(self.n_users):
            items = list(user_pos_items.get(uid, []))
            user_ids.append(uid)
            offsets.append(offset)
            counts.append(len(items))
            pos_flat.extend(items)
            offset += len(items)

        user_ids = np.array(user_ids, dtype=np.int64)
        pos_flat = np.array(pos_flat, dtype=np.int64)
        offsets = np.array(offsets, dtype=np.int64)
        counts = np.array(counts, dtype=np.int64)

        n_samples = self.samples_per_epoch or len(pos_flat) * 5

        for epoch in range(self.n_epochs):
            loss = bpr_epoch(user_ids, pos_flat, offsets, counts,
                             self.n_items, self.user_factors, self.item_factors,
                             self.user_bias, self.item_bias,
                             self.lr, self.reg, n_samples)
            self.loss_history.append(loss)
            if self.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: loss = {loss:.4f}")

        if self.verbose:
            print(f"Epoch {self.n_epochs - 1:3d}: loss = {self.loss_history[-1]:.4f}")

    def score_all_items(self, user_id):
        if user_id >= self.n_users:
            return np.zeros(self.n_items)
        return self.item_bias + self.item_factors @ self.user_factors[user_id]

    def predict_for_user(self, user_id, k=10, train_df=None):
        scores = self.score_all_items(user_id)
        if user_id >= self.n_users:
            return []

        if train_df is not None:
            seen = train_df[train_df['user_id'] == user_id]['item_id'].values
            scores[seen] = -np.inf

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(int(i), float(scores[i])) for i in top_k if scores[i] > -np.inf]