from numba import njit, prange
import numpy as np

# compiling function for higher performance
@njit(parallel=True, fastmath=True)
def funksvd_epoch(users, items, ratings, user_factors, item_factors, user_bias, item_bias, global_bias, lr, reg):

    n = len(users)
    # shuffling indices to perform SGD (random sampling) to prevent model from learning patterns based on order of data
    indices = np.random.permutation(n)
    loss = 0.0
    
    # loop over every interaction in the dataset
    for i in prange(n):
        idx = indices[i]
        u, it, r = users[idx], items[idx], ratings[idx]
        
        # forward pass - predicting rating
        # r_hat = mu + b_u + b_i + (P_u . Q_i)
        pred = global_bias + user_bias[u] + item_bias[it] + np.dot(user_factors[u], item_factors[it])
        
        # calculate error
        err = r - pred
        loss += err * err
        
        # backward pass - update biases
        # b_new = b_old + lr * (error - reg * b_old)
        user_bias[u] += lr * (err - reg * user_bias[u])
        item_bias[it] += lr * (err - reg * item_bias[it])
        
        # update latent factors. perform coordinate descent step
        uf_old = user_factors[u].copy()

        # P_u += lr * (error * Q_i - reg * P_u)
        user_factors[u] += lr * (err * item_factors[it] - reg * user_factors[u])
        # Q_i += lr * (error * P_u_old - reg * Q_i)
        item_factors[it] += lr * (err * uf_old - reg * item_factors[it])
    
    # Return RMSE for this epoch to track convergence
    return np.sqrt(loss / n)


class FunkSVD:
    def __init__(self, n_factors=50, learning_rate=0.005, regularization=0.02, n_epochs=20, verbose=True):

        # hyperparameters
        self.n_factors = n_factors # num of latent vectors
        self.lr = learning_rate # step size for SGD
        self.reg = regularization # L2 penalty

        self.n_epochs = n_epochs
        self.verbose = verbose

        # model weights
        self.user_factors = None # Matrix P (n_users x k)
        self.item_factors = None # Matrix Q (n_items x k)
        self.user_bias = None # Vector b_u
        self.item_bias = None # Vector b_i
        self.global_bias = None # Scalar mu

        # for metrics
        self.train_loss_history = []
        self.val_loss_history = []

    def fit(self, train_df, val_df=None): # training factorization model 

        # matrix shape
        n_users = train_df['user_id'].max() + 1
        n_items = train_df['item_id'].max() + 1

        np.random.seed(42)
        
        # initialize latent factors using normal distribution. Not zero as in this case gradients would be also zero
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # biases set to zero
        self.user_bias, self.item_bias = np.zeros(n_users), np.zeros(n_items)
        # global bias is just the average rating of the training set
        self.global_bias = train_df['rating'].mean()
        
        # data prep for numba
        users = train_df['user_id'].values.astype(np.int32)
        items = train_df['item_id'].values.astype(np.int32)
        ratings = train_df['rating'].values.astype(np.float64)
        
        # training loop
        for epoch in range(self.n_epochs):
            train_rmse = funksvd_epoch(users, items, ratings, self.user_factors, 
                                       self.item_factors, self.user_bias, self.item_bias,
                                       self.global_bias, self.lr, self.reg)
            self.train_loss_history.append(train_rmse)
            
            # validation
            if val_df is not None:
                val_rmse = self.compute_rmse(val_df)
                self.val_loss_history.append(val_rmse)
                if self.verbose and epoch % 5 == 0:
                    print(f"epoch {epoch:2d}: train={train_rmse:.4f}, val={val_rmse:.4f}")
    

    def compute_rmse(self, df): # RMSE calculation 
        u = df['user_id'].values
        i = df['item_id'].values
        r = df['rating'].values

        # Global + User Bias + Item Bias + Dot Product
        pred = self.global_bias + self.user_bias[u] + self.item_bias[i] + np.sum(self.user_factors[u] * self.item_factors[i], axis=1)

        # clip predictions to valid range [1, 5] before calculating error
        return np.sqrt(np.mean((r - np.clip(pred, 1, 5)) ** 2))
    
    def predict_rating(self, user_id, item_id): # predict single rating 

        # cold start check. if user or item not present in training, return global average
        if user_id >= len(self.user_factors) or item_id >= len(self.item_factors):
            return self.global_bias
        
        # matrix factorization formula
        pred = self.global_bias + self.user_bias[user_id] + self.item_bias[item_id] + np.dot(self.user_factors[user_id], self.item_factors[item_id])
        
        return np.clip(pred, 1.0, 5.0)
    
    # generating top-K recommendations for a user
    def predict_for_user(self, user_id, k=10, train_df=None):

        # empty if unknown
        if user_id >= len(self.user_factors):
            return []
        
        # identify items the user has already rated to exclude them later
        rated = set(train_df[train_df['user_id'] == user_id]['item_id']) if train_df is not None else set()

        # scoring 
        scores = self.global_bias + self.user_bias[user_id] + self.item_bias + self.item_factors @ self.user_factors[user_id]

        # masking 
        # set scores of already watched movies to -infinity so they aren't recommended
        scores[list(rated)] = -np.inf

        # top K retrieval
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]

        return [(i, scores[i]) for i in top_k]