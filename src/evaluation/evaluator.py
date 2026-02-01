import pandas as pd
from collections import defaultdict
from metrics import *

class RecommenderEvaluator:    
    def __init__(self, train_df, test_df, k_values=[5, 10, 20], relevance_threshold=4.0):
        self.train_df = train_df
        self.test_df = test_df
        self.k_values = k_values
        self.relevance_threshold = relevance_threshold
        
        self.ground_truth = self.build_ground_truth()
        
        self.item_catalog = set(train_df['item_id'].unique())
        self.item_popularity = train_df.groupby('item_id').size().to_dict()

        self.history = []
        
    def build_ground_truth(self):
        ground_truth = defaultdict(dict)
        for _, row in self.test_df.iterrows():
            ground_truth[row['user_id']][row['item_id']] = row['rating']
        return dict(ground_truth)
    
    def evaluate_model(self, model, model_name="Model"):
        results = defaultdict(list)
        all_predictions = []
        
        test_users = list(self.ground_truth.keys())
        
        for user_id in test_users:
            gt = self.ground_truth[user_id]
            
            max_k = max(self.k_values)
            predictions = model.predict_for_user(user_id, k=max_k, train_df=self.train_df)
            all_predictions.append(predictions)
            
            for k in self.k_values:
                results[f'NDCG@{k}'].append(ndcg_at_k(predictions, gt, k))
                results[f'Recall@{k}'].append(recall_at_k(predictions, gt, k, self.relevance_threshold))
                results[f'Precision@{k}'].append(precision_at_k(predictions, gt, k, self.relevance_threshold))
        
        metrics = {}
        for metric_name, values in results.items():
            metrics[metric_name] = np.mean(values)
        
        metrics['Coverage'] = coverage(all_predictions, self.item_catalog)
        metrics['Popularity_Bias'] = popularity_bias(all_predictions, self.item_popularity)
        metrics['Model'] = model_name
        self.history.append(metrics)
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        print(f"{model_name} - Evaluation results")
        
        print("Ranking metrics:")
        for k in self.k_values:
            print(f"NDCG@{k:2d}: {metrics[f'NDCG@{k}']:.4f}")
        
        print("\n")
        
        print("Relevance metrics (threshold={:.1f}):".format(self.relevance_threshold))
        for k in self.k_values:
            print(f"Recall@{k:2d}: {metrics[f'Recall@{k}']:.4f}")
            print(f"Precision@{k:2d}: {metrics[f'Precision@{k}']:.4f}")
        
        print("\n")
        
        print(f"Diversity metrics:")
        print(f"Coverage: {metrics['Coverage']:.4f}")
        print(f"Popularity bias: {metrics['Popularity_Bias']:.2f}")

    def save_results(self, filepath):
        if not self.history:
            print("No history to save")
            return

        df = pd.DataFrame(self.history)
        
        cols = ['Model'] + [c for c in df.columns if c != 'Model']
        df = df[cols]
        
        df.to_csv(filepath, index=False)