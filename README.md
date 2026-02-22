# Recommender Systems - MovieLens 1M

A comprehensive implementation of classic and advanced recommender system algorithms evaluated on the MovieLens 1M dataset. This project explores Content-Based Filtering, Collaborative Filtering, Matrix Factorization, Learning-to-Rank (BPR), Hybrid Architectures, Deep Learning (Two-Tower, NeuMF), and Online Evaluation via Multi-Armed Bandits.

## 📊 Results Summary

### Part 1: Classical Approaches
| Model | NDCG@10 | Recall@10 | Coverage | Best For |
|-------|---------|-----------|----------|----------|
| **Item-Item CF (Jaccard)** | **0.0625** | **0.0312** | 0.12 | Accuracy |
| Enhanced CB (TF-IDF + Pop) | 0.0340 | 0.0180 | 0.40 | Cold-start items |
| FunkSVD | 0.0265 | 0.0140 | 0.13 | Rating prediction |
| ALS | 0.0219 | 0.0120 | 0.85 | Diversity/Coverage |
| Popularity Baseline | 0.0223 | 0.0110 | 0.01 | Baseline |

### Part 2: Advanced Ranking & Hybrid Approaches
| Model | NDCG@10 | Recall@10 | Coverage | Best For |
|-------|---------|-----------|----------|----------|
| **Hybrid (BPR + CB)** | **0.0665** | **0.0325** | 0.48 | Cold-start users & Accuracy |
| **BPR-MF (Pairwise)** | **0.0656** | **0.0320** | 0.45 | Overall Ranking & Discovery |
| Two-Tower (Neural) | 0.0648 | 0.0315 | 0.42 | Scalable Production Retrieval |
| Personalized PageRank | 0.0544 | 0.0260 | 0.05 | Graph-based propagation |

## 🏗️ Repository Structure

```
recommender_systems_hw_1/
├── README.md                 # This file
├── report.md                 # Part 1 Analysis Report
├── report-part2.md           # Part 2 Analysis Report (Advanced Models & Bandits)
├── requirements.txt          # Python dependencies
├── data/
│   ├── ml-1m/               # Raw MovieLens 1M data (auto-downloaded)
│   └── processed/           # Train/Val/Test splits
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_evaluation_framework.ipynb   # Metrics & data splitting
│   ├── 03_content_based_filtering.ipynb # Content-based models
│   ├── 04_collaborative_filtering.ipynb # Item-Item CF
│   ├── 05_matrix_factorisation.ipynb   # FunkSVD & ALS
│   ├── 06_heuristics.ipynb             # Popularity, Recency, Personalized PageRank
│   ├── 07_bpr.ipynb                    # Bayesian Personalized Ranking (BPR-OPT)
│   ├── 08_hybrid.ipynb                 # Weighted Blending & Candidate Reranking
│   ├── 09_deep_learning.ipynb          # NeuMF & Two-Tower architectures
│   └── 10_ab_testing.ipynb             # Multi-Armed Bandits (e-greedy, UCB, Thompson)
├── src/
│   ├── models/
│   │   ├── content_based.py   # CB recommenders
│   │   ├── collaborative.py   # Item-Item CF
│   │   ├── funk_svd.py        # FunkSVD (SGD)
│   │   ├── als.py             # Alternating Least Squares
│   │   ├── bpr.py             # BPR-MF
│   │   ├── hybrid.py          # Hybrid Recommenders
│   │   ├── ncf.py             # Neural Collaborative Filtering
│   │   ├── two_tower.py       # Two-Tower Model
│   │   ├── heuristics.py      # Ranking Heuristics
│   │   └── bandits.py         # Multi-Armed Bandits
│   ├── evaluation/
│   │   ├── evaluator.py       # RecommenderEvaluator class
│   │   └── metrics.py         # NDCG, Recall, Precision, Coverage
│   └── utils/
│       └── data_loader.py     # Data loading utilities
├── models/                    # Saved trained models (.pkl)
└── experiments/
    ├── figures/              # Generated plots
    └── results/              # Evaluation CSV files
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ladislve/recommender_systems_hw_1.git
   cd recommender_systems_hw_1
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks:**
   ```bash
   jupyter lab notebooks/
   ```
   
   Run notebooks in order (01 → 10) as they depend on each other.

### Data
The MovieLens 1M dataset is **automatically downloaded** when running the first notebook. No manual download required.

## 📓 Notebooks Overview

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Dataset statistics: 1M ratings, 6K users, 4K movies
- Sparsity analysis (95.5%)
- Power-law distributions in user/item activity
- Cold-start problem identification (~19% items with <20 ratings)
- Temporal dynamics analysis

### 2. Evaluation Framework (`02_evaluation_framework.ipynb`)
- **Splitting strategy:** Per-user temporal split (80/10/10)
- **Metrics implemented:**
  - NDCG@K (ranking quality)
  - Precision@K, Recall@K (relevance, threshold=4.0)
  - Coverage (catalog diversity)
  - Popularity Bias (recommendation skew)
- Baselines: Random and Popularity recommenders

### 3. Content-Based Filtering (`03_content_based_filtering.ipynb`)
- **Basic CB:** Binary genre vectors with Jaccard/Cosine similarity
- **Enhanced CB:** TF-IDF weighted genres + decade features + popularity blending
- User profile construction: Rating-weighted feature aggregation
- Similarity function comparison: Jaccard best for binary, Cosine for TF-IDF

### 4. Collaborative Filtering (`04_collaborative_filtering.ipynb`)
- **Item-Item CF:** Memory-based approach using item similarity
- Similarity metrics:
  - Jaccard (binary, threshold=4.0) - **Winner**
  - Adjusted Cosine (rating-based)
- Hyperparameter tuning: K neighbors (optimal K=50)
- Why Item-Item over User-User: Fewer items, denser profiles, more stable

### 5. Matrix Factorization (`05_matrix_factorisation.ipynb`)
- **FunkSVD:** SGD optimization with user/item biases
  - Prediction: $\hat{r}_{ui} = \mu + b_u + b_i + P_u \cdot Q_i$
  - Numba-accelerated training
- **ALS:** Alternating Least Squares with closed-form updates
- Factor tuning: FunkSVD optimal at 30 factors, ALS at 60
- Trade-off: FunkSVD better accuracy, ALS better coverage

### 6. Ranking Heuristics (`06_heuristics.ipynb`)
- **Popularity & Recency:** Global frequency and time-decayed baselines
- **Personalized PageRank:** Graph-based propagation on user-item bipartite network
- Analysis of inductive biases and failure modes (e.g., niche users, isolated items)

### 7. Learning-to-Rank (`07_bpr.ipynb`)
- **Bayesian Personalized Ranking (BPR-OPT):** Pairwise optimization for ranking
- Explicit negative sampling strategy (uniform sampling)
- Analysis of convergence, sampling sensitivity, and head vs. tail item performance

### 8. Hybrid Recommender Systems (`08_hybrid.ipynb`)
- **Weighted Blending:** Linear combination of normalized BPR and CB scores
- **Candidate Generation + Reranking:** BPR retrieves top 100, CB reranks top 10
- Segmented analysis showing Hybrid superiority for cold-start users

### 9. Deep Learning (`09_deep_learning.ipynb`)
- **Neural Collaborative Filtering (NeuMF):** GMF + MLP pathways
- **Two-Tower Model:** Separate user and item towers, injecting TF-IDF features
- Discussion on representational limits, optimization trade-offs, and inference speed

### 10. Online Evaluation & Bandits (`10_ab_testing.ipynb`)
- **Multi-Armed Bandits:** $\epsilon$-greedy, UCB1, and Thompson Sampling
- Simulation of sequential user arrivals and dynamic policy routing
- Analysis of exploration-exploitation trade-offs and bandit convergence

## 🔧 Model Interface

All models implement a consistent interface:

```python
class Recommender:
    def fit(self, train_df: pd.DataFrame) -> self:
        """Train the model on the training data."""
        pass
    
    def predict_for_user(self, user_id: int, k: int = 10, 
                        train_df: pd.DataFrame = None) -> List[Tuple[int, float]]:
        """
        Generate top-k recommendations for a user.
        
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        pass
```

## 📈 Evaluation Protocol

1. **Data Split:** Per-user temporal split respecting time ordering
2. **Metrics:** Computed per-user, then averaged (prevents power-user dominance)
3. **Relevance Threshold:** 4.0 (ratings ≥4 are considered "relevant")
4. **K values:** 5, 10, 20

## 🔑 Key Insights

### Part 1: Classical Approaches
1. **Method > Hyperparameters:** The choice of algorithm matters more than fine-tuning when objectives don't align (e.g., MF optimizes RMSE, but we evaluate with NDCG).
2. **Jaccard > Adjusted Cosine for CF:** Filtering for strong positive signals (rating ≥4) removes noise and focuses on true preferences.
3. **Popularity is a strong baseline:** Any model must beat the popularity baseline (NDCG@10 = 0.022) to be considered useful.
4. **Accuracy vs Diversity trade-off:** High accuracy models (Item-Item CF) yield low coverage, while high coverage models (ALS) yield lower accuracy.

### Part 2: Advanced Ranking & Hybrid Approaches
1. **Pairwise Ranking > Pointwise Prediction:** BPR-MF significantly outperforms FunkSVD for top-K retrieval tasks because it explicitly optimizes for ranking (Pairwise Loss) rather than rating prediction (MSE).
2. **Hybrid Synergy for Cold-Start:** Combining collaborative signals (BPR) with content features (TF-IDF) creates a robust hybrid that excels for cold-start users, while pure collaborative models remain superior for warm users.
3. **Deep Learning Practicality:** While NeuMF offers high capacity, the Two-Tower architecture is far more practical for production. It allows for precomputing item embeddings, reducing inference to a fast Approximate Nearest Neighbor (ANN) search.
4. **Bandit Efficiency:** In online simulations, Thompson Sampling proved highly effective at balancing exploration and exploitation, quickly identifying and committing to the optimal Hybrid policy, whereas UCB1 struggled with the scale of the rewards.

## 📝 License

This project is for educational purposes.
