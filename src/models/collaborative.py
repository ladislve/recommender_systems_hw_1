import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd


def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int,
                            binary: bool = False, threshold: float = 4.0) -> csr_matrix:
    """
    Build sparse user-item interaction matrix.
    
    Args:
        df: DataFrame with user_id, item_id, rating columns
        n_users: Number of users
        n_items: Number of items
        binary: If True, convert to binary matrix using threshold
        threshold: Rating threshold for binary conversion
        
    Returns:
        Sparse CSR matrix of shape (n_users, n_items)
    """
    rows = df['user_id'].values
    cols = df['item_id'].values
    
    if binary:
        # binary interactions (rating >= threshold counts as 1)
        data = (df['rating'].values >= threshold).astype(np.float32)
    else:
        data = df['rating'].values.astype(np.float32)
    
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def compute_jaccard_similarity(binary_matrix: csr_matrix, 
                               min_cooccurrence: int = 2) -> csr_matrix:
    """
    Compute item-item Jaccard similarity from binary user-item matrix.
    
    Jaccard(i, j) = |users who liked both i and j| / |users who liked i or j|
    
    This metric is inspired by Microsoft SAR (Smart Adaptive Recommendations).
    It treats interactions as binary (liked/not liked) and measures set overlap.
    
    Args:
        binary_matrix: Sparse user-item matrix (1 = positive interaction)
        min_cooccurrence: Minimum co-occurrence count to compute similarity
        
    Returns:
        Sparse item-item similarity matrix
    """
    # transpose to get item-user matrix
    item_user = binary_matrix.T.tocsr()  # shape: (n_items, n_users)
    n_items = item_user.shape[0]
    
    # compute item popularity (number of users who liked each item)
    item_counts = np.array(item_user.sum(axis=1)).flatten()
    
    # co-occurrence matrix: C[i,j] = number of users who liked both i and j
    print("Computing co-occurrence matrix")
    cooccurrence = item_user @ item_user.T  # shape: (n_items, n_items)
    cooccurrence = cooccurrence.tocsr()
    
    # compute Jaccard: |intersection| / |union|
    print("Computing Jaccard coefficients")
    similarity = lil_matrix((n_items, n_items), dtype=np.float32)
    
    for i in range(n_items):
        if i % 500 == 0:
            print(f"  Processing item {i}/{n_items}")
        
        row = cooccurrence.getrow(i)
        indices = row.indices
        data = row.data
        
        for idx, j in enumerate(indices):
            if j <= i:  # skip diagonal and lower triangle
                continue
            
            co_count = data[idx]
            if co_count < min_cooccurrence:
                continue
            
            # Jaccard = intersection / union
            union = item_counts[i] + item_counts[j] - co_count
            if union > 0:
                jaccard = co_count / union
                similarity[i, j] = jaccard
                similarity[j, i] = jaccard  # symmetric
    
    similarity = similarity.tocsr()
    print(f"Similarity matrix: {similarity.shape}, nnz: {similarity.nnz:,}")
    
    return similarity


def compute_adjusted_cosine_similarity(rating_matrix: csr_matrix,
                                       min_common_users: int = 2) -> csr_matrix:
    """
    Compute item-item adjusted cosine similarity.
    
    Adjusts for user rating bias by centering ratings around user mean.
    This handles the case where some users rate everything high (lenient)
    while others rate everything low (strict).
    
    sim(i, j) = sum_u (r_ui - mean_u)(r_uj - mean_u) / 
                (sqrt(sum_u (r_ui - mean_u)^2) * sqrt(sum_u (r_uj - mean_u)^2))
    
    Args:
        rating_matrix: Sparse user-item rating matrix
        min_common_users: Minimum number of common users to compute similarity
        
    Returns:
        Sparse item-item similarity matrix
    """
    # compute user means (only over rated items)
    print("Computing user means")
    user_sums = np.array(rating_matrix.sum(axis=1)).flatten()
    user_counts = np.array((rating_matrix > 0).sum(axis=1)).flatten()
    user_means = np.divide(user_sums, user_counts, where=user_counts > 0)
    user_means[user_counts == 0] = 0
    
    # center ratings by subtracting user mean
    print("Centering ratings")
    rating_coo = rating_matrix.tocoo()
    centered_data = rating_coo.data - user_means[rating_coo.row]
    centered_matrix = csr_matrix(
        (centered_data, (rating_coo.row, rating_coo.col)),
        shape=rating_matrix.shape
    )
    
    # transpose to get item-user matrix
    item_user = centered_matrix.T.tocsr()  # shape: (n_items, n_users)
    n_items = item_user.shape[0]
    
    # compute item norms for denominator
    item_norms = np.sqrt(np.array(item_user.power(2).sum(axis=1)).flatten())
    
    # compute co-occurrence count matrix
    binary_item_user = (item_user != 0).astype(np.float32)
    cooccurrence_count = binary_item_user @ binary_item_user.T
    
    # dot product matrix (numerator of cosine)
    print("Computing dot products")
    dot_product = item_user @ item_user.T
    dot_product = dot_product.tocsr()
    cooccurrence_count = cooccurrence_count.tocsr()
    
    # compute similarity
    print("Computing adjusted cosine coefficients")
    similarity = lil_matrix((n_items, n_items), dtype=np.float32)
    
    for i in range(n_items):
        if i % 500 == 0:
            print(f"  Processing item {i}/{n_items}")
        
        if item_norms[i] == 0:
            continue
        
        row = dot_product.getrow(i)
        count_row = cooccurrence_count.getrow(i)
        
        for idx, j in enumerate(row.indices):
            if j <= i:  # skip diagonal and lower triangle
                continue
            
            # check minimum common users
            common_count = count_row[0, j]
            if common_count < min_common_users:
                continue
            
            if item_norms[j] == 0:
                continue
            
            # adjusted cosine = dot_product / (norm_i * norm_j)
            cos_sim = row.data[idx] / (item_norms[i] * item_norms[j])
            
            # clip to [-1, 1] range (numerical stability)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
            if cos_sim > 0:  # only keep positive similarities
                similarity[i, j] = cos_sim
                similarity[j, i] = cos_sim
    
    similarity = similarity.tocsr()
    print(f"Similarity matrix: {similarity.shape}, nnz: {similarity.nnz:,}")
    
    return similarity


class ItemItemCF:
    """
    Item-Item Collaborative Filtering Recommender.
    
    For each candidate item, computes a weighted sum of similarities
    to items the user has already rated highly.
    
    Prediction formula:
        score(u, i) = sum_{j in rated(u)} sim(i, j) * rating(u, j)
    
    Args:
        train_df: Training data with user_id, item_id, rating columns
        similarity_matrix: Precomputed item-item similarity (sparse)
        k_neighbors: Number of similar items to consider for each item
    """
    
    def __init__(self, train_df: pd.DataFrame, similarity_matrix: csr_matrix,
                 k_neighbors: int = 50):
        self.k_neighbors = k_neighbors
        self.similarity_matrix = similarity_matrix
        
        # build user-item rating matrix
        n_users = train_df['user_id'].max() + 1
        n_items = train_df['item_id'].max() + 1
        
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        data = train_df['rating'].values
        
        self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.n_items = n_items
        
        # precompute user rated items for filtering
        self.user_rated_items: Dict[int, Set[int]] = {}
        for user_id in range(n_users):
            self.user_rated_items[user_id] = set(self.user_item_matrix[user_id].indices)
        
        # precompute top-K neighbors for each item (for efficiency)
        print(f"Precomputing top-{k_neighbors} neighbors per item")
        self.item_neighbors: Dict[int, Dict[int, float]] = {}
        
        for item_id in range(n_items):
            sim_row = similarity_matrix.getrow(item_id)
            indices = sim_row.indices
            values = sim_row.data
            
            if len(values) > 0:
                # get top-K by similarity
                if len(values) > k_neighbors:
                    top_k_idx = np.argpartition(values, -k_neighbors)[-k_neighbors:]
                    self.item_neighbors[item_id] = dict(zip(indices[top_k_idx], values[top_k_idx]))
                else:
                    self.item_neighbors[item_id] = dict(zip(indices, values))
            else:
                self.item_neighbors[item_id] = {}
        
        print(f"ItemItemCF initialized: k={k_neighbors}")
    
    def predict_for_user(self, user_id: int, k: int = 10,
                        train_df: Optional[pd.DataFrame] = None) -> List[Tuple[int, float]]:
        """
        Generate top-K recommendations for a user.
        
        Score for item i = sum over user's rated items j of: sim(i,j) * rating(j)
        
        Args:
            user_id: Target user ID
            k: Number of recommendations
            train_df: Not used (kept for interface compatibility)
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        # handle unknown users
        if user_id >= self.user_item_matrix.shape[0]:
            return []
        
        if user_id not in self.user_rated_items:
            return []
        
        # get user's ratings
        user_row = self.user_item_matrix[user_id]
        rated_items = user_row.indices
        ratings = user_row.data
        
        if len(rated_items) == 0:
            return []
        
        # score each candidate item
        scores = np.zeros(self.n_items, dtype=np.float32)
        
        for rated_item, rating in zip(rated_items, ratings):
            # get neighbors of this rated item
            neighbors = self.item_neighbors.get(rated_item, {})
            
            for neighbor_item, sim in neighbors.items():
                # accumulate weighted score
                scores[neighbor_item] += sim * rating
        
        # filter out already rated items
        already_rated = self.user_rated_items[user_id]
        for item_id in already_rated:
            scores[item_id] = -np.inf
        
        # get top-K
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        
        # filter out invalid scores
        recommendations = [
            (int(idx), float(scores[idx]))
            for idx in top_k_idx
            if scores[idx] > -np.inf and scores[idx] > 0
        ]
        
        return recommendations[:k]
