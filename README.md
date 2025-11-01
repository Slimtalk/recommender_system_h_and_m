# H&M Fashion Recommender System

A hybrid recommender system built with LightFM for the H&M dataset, implementing both collaborative filtering and hybrid approaches to handle warm-start and cold-start scenarios in fashion recommendations.

## Overview

This project implements a production-ready recommender system using the H&M Personalized Fashion Recommendations dataset from Kaggle. The system addresses the challenge of providing personalized article recommendations by combining:

- **Collaborative Filtering Model**: For users and items with sufficient historical interactions (warm-start)
- **Hybrid Model**: Leveraging both interactions and metadata features to tackle cold-start problems

The implementation uses **LightFM**, a hybrid recommendation algorithm that seamlessly combines collaborative and content-based approaches through a unified latent representation.

## Key Features

- **Comprehensive Data Exploration**: In-depth analysis of user behavior, item popularity, temporal patterns, and dataset sparsity
- **Dual-Model Architecture**: Separate models optimized for warm-start and cold-start scenarios
- **Hyperparameter Optimization**: Systematic grid search with cross-validation for optimal model configuration
- **Robust Evaluation**: Multiple metrics including Precision@K, Recall@K, NDCG, and AUC-ROC
- **Scalable Implementation**: Multi-threaded training leveraging all available CPU cores
- **Production-Ready**: Complete pipeline from data ingestion to model evaluation with saved artifacts

## Dataset Statistics

The H&M dataset comprises fashion retail transactions with rich metadata:

| Metric | Value |
|--------|-------|
| **Users (Customers)** | 1,362,281 |
| **Items (Articles)** | 104,547 |
| **Interactions (Transactions)** | 31,788,324 |
| **Time Period** | Sept 2018 - Sept 2020 |
| **Sparsity** | 99.98% |
| **Items for 80% Coverage** | 21,659 (20.7%) |

The dataset includes:
- `transactions_train.csv`: User-item interactions with timestamps
- `articles.csv`: Article metadata (type, color, department, etc.)
- `customers.csv`: Customer metadata (age, club membership, etc.)

## 🏗️ Project Structure

```
Recommender Systems/
├── README.md                                    # This file
├── .gitignore                                   # Git ignore rules
│
├── Rendu/LastVersion/
│   └── recommender_system_h_and_m_lightfm.ipynb # Main analysis notebook
│
├── h_and_m_data/                                # Dataset (not in repo)
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│
└── hm-env/                                       # Python virtual environment
```

## Quick Start

### Prerequisites

- Python 3.10+
- 4GB+ RAM (16GB+ recommended)
- H&M dataset from [Kaggle Competition](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations)

### Installation

1. **Clone the repository**

2. **Activate the virtual environment**
   ```bash
   source hm-env/bin/activate
   ```
   
   Or create a new environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install lightfm pandas numpy scipy scikit-learn matplotlib seaborn tqdm itables
   ```

3. **Download the H&M dataset**
   - Visit the [Kaggle competition page](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations)
   - Download the dataset files
   - Place `articles.csv`, `customers.csv`, and `transactions_train.csv` in `h_and_m_data/`

4. **Launch Jupyter Lab**
   ```bash
   jupyter lab "recommender_system_h_and_m_lightfm.ipynb"
   ```

## Usage

### Running the Complete Pipeline

The notebook is structured in sequential steps:

1. **Data Exploration & Understanding** (Step 1)
   - Load and inspect the three CSV files
   - Analyze dataset characteristics, sparsity, and distributions
   - Visualize temporal patterns and user/item behavior

2. **Data Preprocessing** (Step 2)
   - Clean and transform raw data
   - Engineer features from metadata
   - Create user-item interaction matrices
   - Split data into train/validation/test sets

3. **Model Building** (Step 3)
   - Build LightFM dataset with features
   - Implement collaborative filtering baseline
   - Implement hybrid model with item and user features
   - Configure multi-threaded training

4. **Hyperparameter Tuning** (Step 4)
   - Define parameter grid
   - Run systematic grid search
   - Evaluate with cross-validation
   - Save best model configuration

5. **Evaluation & Analysis** (Step 5)
   - Test final models on held-out test set
   - Calculate Precision@K, Recall@K, NDCG, AUC
   - Analyze recommendation quality
   - Generate example recommendations


## Results & Performance

The system achieves competitive performance on standard recommendation metrics:

- **Precision@10**: Measures relevance of top recommendations
- **Recall@10**: Measures coverage of relevant items
- **NDCG**: Normalized discounted cumulative gain for ranking quality
- **AUC**: Area under ROC curve for binary relevance

Detailed results and performance comparisons between the collaborative and hybrid models are documented in the notebook output cells.

## Technical Details

### Algorithm: LightFM

LightFM is a hybrid latent representation model that:
- Represents users and items as linear combinations of their feature latent vectors
- Supports both implicit and explicit feedback
- Scales to millions of users and items
- Handles cold-start through content features

### Training Configuration

- **Loss Function**: WARP (Weighted Approximate-Rank Pairwise) for implicit feedback
- **Optimization**: Stochastic Gradient Descent with AdaGrad
- **Parallelization**: Multi-threaded training across all CPU cores
- **Evaluation**: Train/validation/test split with temporal awareness

### Key Dependencies

- `lightfm`: Hybrid recommendation algorithm
- `pandas`, `numpy`: Data manipulation
- `scipy.sparse`: Efficient sparse matrix operations
- `scikit-learn`: Evaluation metrics and utilities
- `matplotlib`, `seaborn`: Visualization
- `itables`: Interactive data tables in notebooks

## Data Privacy & Git Considerations

**Important**: The raw H&M dataset (`h_and_m_data/`) is excluded from version control via `.gitignore` due to:
- Large file sizes (~2GB+ uncompressed)
- Kaggle dataset licensing terms
- Privacy considerations


## References & Attribution

- **Dataset**: [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) (Kaggle)
- **Algorithm**: [LightFM Documentation](https://making.lyst.com/lightfm/docs/home.html)
- **Paper**: Kula, M. (2015). "Metadata Embeddings for User and Item Cold-start Recommendations"

---

**Author**: Slim MEDDEB

For questions, issues, or collaboration inquiries, please open an issue or contact the author.
