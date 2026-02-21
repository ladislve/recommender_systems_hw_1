import numpy as np
import pandas as pd

class WeightedBlendHybrid:
    def __init__(self, bpr_model, cb_model, alpha=0.7, n_items=None):
        self.bpr = bpr_model
        self.cb = cb_model
        self.alpha = alpha
        self.n_items = n_items

    @staticmethod
    def _normalize(scores):
        mask = scores > -np.inf
        if mask.sum() == 0:
            return scores
        valid = scores[mask]
        mn, mx = valid.min(), valid.max()
        if mx - mn < 1e-10:
            scores[mask] = 0.5
        else:
            scores[mask] = (valid - mn) / (mx - mn)
        return scores

    def predict_for_user(self, user_id, k=10, train_df=None):
        bpr_scores = self.bpr.score_all_items(user_id).copy()

        if user_id in self.cb.user_profiles:
            profile = self.cb.user_profiles[user_id]
            cb_scores = self.cb.normalized_features @ profile
            cb_scores = ((1 - self.cb.popularity_weight) * cb_scores +
                         self.cb.popularity_weight * self.cb.popularity_scores)
        else:
            cb_scores = np.zeros(self.n_items)

        if train_df is not None:
            seen = train_df[train_df['user_id'] == user_id]['item_id'].values
            bpr_scores[seen] = -np.inf
            cb_scores[seen] = -np.inf

        bpr_norm = self._normalize(bpr_scores)
        cb_norm = self._normalize(cb_scores)

        final = self.alpha * bpr_norm + (1 - self.alpha) * cb_norm
        final[bpr_scores == -np.inf] = -np.inf

        top_k = np.argpartition(final, -k)[-k:]
        top_k = top_k[np.argsort(final[top_k])[::-1]]
        return [(int(i), float(final[i])) for i in top_k if final[i] > -np.inf]


class CandidateRerankHybrid:
    def __init__(self, bpr_model, cb_model, n_candidates=100,
                 blend_alpha=0.6, n_items=None):
        self.bpr = bpr_model
        self.cb = cb_model
        self.n_candidates = n_candidates
        self.blend_alpha = blend_alpha
        self.n_items = n_items

    def predict_for_user(self, user_id, k=10, train_df=None):
        candidates = self.bpr.predict_for_user(user_id, k=self.n_candidates, train_df=train_df)
        if not candidates:
            return []

        if user_id not in self.cb.user_profiles:
            return candidates[:k]

        profile = self.cb.user_profiles[user_id]

        reranked = []
        for iid, bpr_score in candidates:
            if iid < len(self.cb.normalized_features):
                cb_score = float(self.cb.normalized_features[iid] @ profile)
            else:
                cb_score = 0.0
            combined = self.blend_alpha * bpr_score + (1 - self.blend_alpha) * cb_score
            reranked.append((iid, combined))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]