import numpy as np
from scipy.sparse import csr_matrix

class ALS:
    def __init__(self, n_factors=20, regularization=0.1, n_iterations=15, verbose=True):

        # hyperparameters
        self.n_factors = n_factors # latent dimension (k)
        self.reg = regularization # L2 regularization (lambda)
        self.n_iterations = n_iterations
        self.verbose = verbose
        
        # params to learn
        self.user_factors = None # Matrix P (n_users x k)
        self.item_factors = None # Matrix Q (n_items x k)
        self.global_mean = 0 # Mu

        # for metrics
        self.train_loss_history = []
        self.val_loss_history = []
    
    def fit(self, train_df, val_df=None): # training ALS 
        n_users = train_df['user_id'].max() + 1
        n_items = train_df['item_id'].max() + 1
        
        # center ratings - deviations from the mean
        # r_ui = mu + P_u . Q_i  -->  (r_ui - mu) = P_u . Q_i
        self.global_mean = train_df['rating'].mean()
        centered_ratings = train_df['rating'].values - self.global_mean
        
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        
        # sparse matrix construction

        # R - all items rated by user u
        R = csr_matrix((centered_ratings, (rows, cols)), shape=(n_users, n_items))
        # R_T - all users who rated item i
        R_T = R.T.tocsr()
        
        np.random.seed(42)

        # initialize factors with random noise
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # pre-compute identity matrix * lambda for regularization term (Lambda * I)
        eye = np.eye(self.n_factors)
        
        # optimization loop
        for iteration in range(self.n_iterations):
            # fix items, solve for users
            # for each user u, solve: (Q^T Q + lambda*I) * P_u = Q^T * r_u
            for u in range(n_users):
                # get indices of items rated by user u
                item_indices = R[u].indices
                if len(item_indices) == 0:
                    continue
                
                # get ratings and item factors
                r_u = R[u].data 
                V_u = self.item_factors[item_indices]
                
                # Ax = b
                lambda_I = self.reg * len(item_indices) * eye
                A = V_u.T @ V_u + lambda_I
                b = V_u.T @ r_u
                
                # solving linear system
                try:
                    self.user_factors[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
            
            # fix users, solve for items
            # for each item i, solve: (P^T P + lambda*I) * Q_i = P^T * r_i
            for i in range(n_items):

                # get indices of users who rated item i
                user_indices = R_T[i].indices
                if len(user_indices) == 0:
                    continue
                
                # get ratings and user factors
                r_i = R_T[i].data
                U_i = self.user_factors[user_indices]
                
                # construct linear system
                lambda_I = self.reg * len(user_indices) * eye
                A = U_i.T @ U_i + lambda_I
                b = U_i.T @ r_i
                
                # solve
                try:
                    self.item_factors[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
            
            # metrics
            train_rmse = self.compute_rmse(train_df)
            self.train_loss_history.append(train_rmse)
            
            if val_df is not None:
                val_rmse = self.compute_rmse(val_df)
                self.val_loss_history.append(val_rmse)
                
                if self.verbose:
                    print(f"iteration {iteration+1:2d}: train RMSE = {train_rmse:.4f}, val RMSE = {val_rmse:.4f}")

    def predict_rating(self, user_id, item_id):
        # cold start
        if user_id >= len(self.user_factors) or item_id >= len(self.item_factors):
            return self.global_mean
        
        # prediction = mean + dot product
        pred = self.global_mean + np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return np.clip(pred, 1.0, 5.0)

    def predict_for_user(self, user_id, k=10, train_df=None):
        if user_id >= len(self.user_factors):
            return []
        # calculating scores: Mean + P_u * Q_all^T
        scores = self.global_mean + self.item_factors @ self.user_factors[user_id]
        
        # masking items already seen in training set
        if train_df is not None:
            rated_items = train_df[train_df['user_id'] == user_id]['item_id'].values
            scores[rated_items] = -np.inf
        
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(i, scores[i]) for i in top_indices]

    def compute_rmse(self, df):
        users = df['user_id'].values
        items = df['item_id'].values
        ratings = df['rating'].values
        
        valid_mask = (users < len(self.user_factors)) & (items < len(self.item_factors))
        
        u_valid = users[valid_mask]
        i_valid = items[valid_mask]
        r_valid = ratings[valid_mask]
        
        interactions = np.sum(self.user_factors[u_valid] * self.item_factors[i_valid], axis=1)
        preds = self.global_mean + interactions
        preds = np.clip(preds, 1.0, 5.0)
        
        mse = np.mean((preds - r_valid) ** 2)
        return np.sqrt(mse)