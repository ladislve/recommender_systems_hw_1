import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def jaccard_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Jaccard similarity (for binary vectors)."""
    a_binary = (vec_a > 0).astype(int)
    b_binary = (vec_b > 0).astype(int)
    
    intersection = np.sum(a_binary & b_binary)
    union = np.sum(a_binary | b_binary)
    
    if union == 0:
        return 0.0
    return intersection / union


def pearson_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return 0.0
    return np.corrcoef(vec_a, vec_b)[0, 1]


class ContentBasedRecommender:
    """
    Content-based recommender with configurable similarity function.
    
    Builds user preference profiles by aggregating item feature vectors
    weighted by user ratings, then recommends items with similar features.
    
    User Profile Construction:
        u = sum_{i in I_u} (r_{ui} - mean_r_u) * v_i / |I_u|
    
    Args:
        item_features: numpy array (n_items x n_features) - item feature matrix
        similarity: 'cosine', 'jaccard', or 'pearson'
        use_rating_weights: whether to weight by (rating - mean_rating)
    """
    
    def __init__(self, item_features: np.ndarray, similarity: str = 'cosine', 
                 use_rating_weights: bool = True):
        self.item_features = item_features
        self.similarity = similarity
        self.use_rating_weights = use_rating_weights
        
        # precompute normalized features for cosine similarity
        self.item_norms = np.linalg.norm(item_features, axis=1, keepdims=True)
        self.item_norms[self.item_norms == 0] = 1  # avoid division by zero
        self.normalized_features = item_features / self.item_norms
        
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.user_mean_ratings: Dict[int, float] = {}
        
    def fit(self, train_df: pd.DataFrame) -> 'ContentBasedRecommender':
        """
        Build user profiles from training data.
        
        Args:
            train_df: DataFrame with user_id, item_id, rating columns
            
        Returns:
            self for method chaining
        """
        # compute user mean ratings
        self.user_mean_ratings = train_df.groupby('user_id')['rating'].mean().to_dict()
        
        # build user profiles
        for user_id, group in train_df.groupby('user_id'):
            items = group['item_id'].values
            ratings = group['rating'].values
            
            if self.use_rating_weights:
                mean_rating = self.user_mean_ratings[user_id]
                weights = ratings - mean_rating
            else:
                weights = np.ones(len(ratings))
            
            # aggregate item features weighted by preference
            profile = np.zeros(self.item_features.shape[1])
            for item_id, weight in zip(items, weights):
                if item_id < len(self.item_features):
                    profile += weight * self.item_features[item_id]
            
            # normalize profile
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            
            self.user_profiles[user_id] = profile
            
        return self
    
    def _compute_similarity(self, user_profile: np.ndarray, 
                           item_vector: np.ndarray) -> float:
        """Compute similarity between user profile and item."""
        if self.similarity == 'cosine':
            return cosine_sim(user_profile, item_vector)
        elif self.similarity == 'jaccard':
            return jaccard_sim(user_profile, item_vector)
        elif self.similarity == 'pearson':
            return pearson_sim(user_profile, item_vector)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
    
    def predict_for_user(self, user_id: int, k: int = 10, 
                        train_df: Optional[pd.DataFrame] = None) -> List[Tuple[int, float]]:
        """
        Generate top-k recommendations for a user.
        
        Args:
            user_id: Target user ID
            k: Number of recommendations
            train_df: Training data to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # get items already rated by user
        rated_items = set()
        if train_df is not None:
            rated_items = set(train_df[train_df['user_id'] == user_id]['item_id'])
        
        # compute scores for all items
        if self.similarity == 'cosine':
            # efficient batch computation for cosine
            scores = self.normalized_features @ user_profile
        else:
            # compute individually for other similarities
            scores = np.array([
                self._compute_similarity(user_profile, self.item_features[i])
                for i in range(len(self.item_features))
            ])
        
        # mask already rated items
        for item_id in rated_items:
            if item_id < len(scores):
                scores[item_id] = -np.inf
        
        # get top-k items
        top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
        
        return [(int(idx), float(scores[idx])) for idx in top_k_indices 
                if scores[idx] > -np.inf]


class EnhancedContentBasedRecommender:
    """
    Enhanced CB recommender with TF-IDF features and popularity smoothing.
    
    Improvements over basic CB:
    1. TF-IDF features: Downweight common genres, upweight rare ones
    2. Popularity boost: Slight preference for items with more ratings
    3. Quality filtering: Exclude items with too few ratings
    
    Args:
        item_features: TF-IDF weighted feature matrix
        popularity_weight: Blend weight for popularity score (0-1)
        min_ratings_threshold: Minimum ratings for item to be recommended
    """
    
    def __init__(self, item_features: np.ndarray, popularity_weight: float = 0.1,
                 min_ratings_threshold: int = 5):
        self.item_features = item_features
        self.popularity_weight = popularity_weight
        self.min_ratings_threshold = min_ratings_threshold
        
        # normalize features for cosine similarity
        norms = np.linalg.norm(item_features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.normalized_features = item_features / norms
        
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.user_mean_ratings: Dict[int, float] = {}
        self.item_popularity: Dict[int, int] = {}
        self.popularity_scores: Optional[np.ndarray] = None
    
    def fit(self, train_df: pd.DataFrame) -> 'EnhancedContentBasedRecommender':
        """
        Build user profiles and compute popularity scores.
        
        Args:
            train_df: DataFrame with user_id, item_id, rating columns
            
        Returns:
            self for method chaining
        """
        # compute item popularity (log-scaled to reduce skew)
        item_counts = train_df.groupby('item_id').size()
        self.item_popularity = item_counts.to_dict()
        
        # create normalized popularity scores
        max_pop = item_counts.max()
        self.popularity_scores = np.zeros(len(self.item_features))
        for item_id, count in self.item_popularity.items():
            if item_id < len(self.popularity_scores):
                # log scale to reduce dominance of blockbusters
                self.popularity_scores[item_id] = np.log1p(count) / np.log1p(max_pop)
        
        # compute user mean ratings
        self.user_mean_ratings = train_df.groupby('user_id')['rating'].mean().to_dict()
        
        # build user profiles with rating-weighted aggregation
        for user_id, group in train_df.groupby('user_id'):
            items = group['item_id'].values
            ratings = group['rating'].values
            
            # center ratings and use as weights
            mean_rating = self.user_mean_ratings[user_id]
            weights = ratings - mean_rating
            
            # only use positively weighted items (above user's mean)
            positive_mask = weights > 0
            
            profile = np.zeros(self.item_features.shape[1])
            
            if positive_mask.any():
                for item_id, weight in zip(items[positive_mask], weights[positive_mask]):
                    if item_id < len(self.item_features):
                        profile += weight * self.item_features[item_id]
            else:
                # fallback: use all items with equal weight if no positive
                for item_id in items:
                    if item_id < len(self.item_features):
                        profile += self.item_features[item_id]
            
            # normalize
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            
            self.user_profiles[user_id] = profile
            
        return self
    
    def predict_for_user(self, user_id: int, k: int = 10,
                        train_df: Optional[pd.DataFrame] = None) -> List[Tuple[int, float]]:
        """
        Generate top-k recommendations with popularity blending.
        
        Final score = (1 - alpha) * content_score + alpha * popularity_score
        
        Args:
            user_id: Target user ID
            k: Number of recommendations
            train_df: Training data to exclude already rated items
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # compute content similarity scores
        content_scores = self.normalized_features @ user_profile
        
        # blend with popularity
        final_scores = ((1 - self.popularity_weight) * content_scores + 
                       self.popularity_weight * self.popularity_scores)
        
        # filter items with too few ratings (cold items)
        for item_id in range(len(final_scores)):
            if self.item_popularity.get(item_id, 0) < self.min_ratings_threshold:
                final_scores[item_id] = -np.inf
        
        # mask already rated items
        if train_df is not None:
            rated_items = train_df[train_df['user_id'] == user_id]['item_id'].values
            final_scores[rated_items] = -np.inf
        
        # get top-k
        top_k_indices = np.argpartition(final_scores, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(final_scores[top_k_indices])[::-1]]
        
        return [(int(idx), float(final_scores[idx])) for idx in top_k_indices 
                if final_scores[idx] > -np.inf]


def build_binary_genre_matrix(movies_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build binary genre feature matrix from movies DataFrame.
    
    Args:
        movies_df: DataFrame with item_id and genres columns (pipe-separated)
        
    Returns:
        Tuple of (genre_matrix, genre_list)
    """
    movies_df = movies_df.copy()
    movies_df['genres_list'] = movies_df['genres'].str.split('|')
    
    all_genres = sorted(set(g for genres in movies_df['genres_list'] for g in genres))
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    
    n_items = movies_df['item_id'].max() + 1
    n_genres = len(all_genres)
    
    genre_matrix = np.zeros((n_items, n_genres))
    
    for _, row in movies_df.iterrows():
        item_id = row['item_id']
        for genre in row['genres_list']:
            genre_matrix[item_id, genre_to_idx[genre]] = 1
    
    return genre_matrix, all_genres


def build_tfidf_features(movies_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build TF-IDF weighted feature matrix including genres and decade.
    
    Args:
        movies_df: DataFrame with item_id, title, and genres columns
        
    Returns:
        Tuple of (tfidf_matrix, feature_names)
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    movies_df = movies_df.copy()
    
    # extract year from title
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    
    movies_df['year'] = movies_df['title'].apply(extract_year)
    movies_df['decade'] = (movies_df['year'] // 10 * 10).fillna(0).astype(int)
    
    # create text representation: genres + decade
    def create_feature_text(row):
        features = row['genres'].replace('|', ' ')
        if row['decade'] > 0:
            features += f" decade_{row['decade']}"
        return features
    
    movies_df['features'] = movies_df.apply(create_feature_text, axis=1)
    
    # fit TF-IDF vectorizer
    tfidf = TfidfVectorizer(token_pattern=r'\b\w+\b')
    
    item_ids = movies_df['item_id'].values
    feature_texts = movies_df['features'].values
    
    tfidf_matrix = tfidf.fit_transform(feature_texts)
    
    # create full matrix with item_id as index
    n_items = movies_df['item_id'].max() + 1
    full_matrix = np.zeros((n_items, tfidf_matrix.shape[1]))
    
    for i, item_id in enumerate(item_ids):
        full_matrix[item_id] = tfidf_matrix[i].toarray().flatten()
    
    return full_matrix, tfidf.get_feature_names_out()
