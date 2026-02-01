import numpy as np

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores)[:k]
    if relevance_scores.size == 0:
        return 0.0
    gains = 2 ** relevance_scores - 1
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(predictions, ground_truth, k=10):
    if not predictions or not ground_truth:
        return 0.0
    
    relevance = [ground_truth.get(item_id, 0) for item_id, _ in predictions[:k]]
    
    ideal_relevance = sorted(ground_truth.values(), reverse=True)
    
    dcg = dcg_at_k(relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(predictions, ground_truth, k=10, threshold=4.0):
    if not predictions or not ground_truth:
        return 0.0
    
    relevant_items = {item for item, rating in ground_truth.items() if rating >= threshold}
    
    if len(relevant_items) == 0:
        return 0.0
    
    predicted_items = {item_id for item_id, _ in predictions[:k]}
    
    hits = len(predicted_items & relevant_items)
    return hits / len(relevant_items)

def precision_at_k(predictions, ground_truth, k=10, threshold=4.0):
    if not predictions or not ground_truth:
        return 0.0
    
    relevant_items = {item for item, rating in ground_truth.items() if rating >= threshold}
    predicted_items = {item_id for item_id, _ in predictions[:k]}
    
    if len(predicted_items) == 0:
        return 0.0
    
    hits = len(predicted_items & relevant_items)
    return hits / len(predicted_items)

def coverage(all_predictions, item_catalog):
    recommended_items = set()
    for preds in all_predictions:
        recommended_items.update([item_id for item_id, _ in preds])
    
    return len(recommended_items) / len(item_catalog) if len(item_catalog) > 0 else 0.0

def popularity_bias(all_predictions, item_popularity):
    all_recommended = []
    for preds in all_predictions:
        all_recommended.extend([item_id for item_id, _ in preds])
    
    if not all_recommended:
        return 0.0
    
    avg_pop = np.mean([item_popularity.get(item, 0) for item in all_recommended])
    return avg_pop