# Credit Card Fraud Detection Using Random Forest Classifier

A production-quality machine learning project for detecting fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

## Project Overview

This project classifies transactions as:

- `0` = Legitimate Transaction
- `1` = Fraudulent Transaction

The dataset is highly imbalanced (~0.17% fraud). Instead of oversampling/undersampling techniques, all models use:

```python
class_weight="balanced"
```

The primary model is a **Random Forest Classifier** tuned with `RandomizedSearchCV`.

## Project Structure

```text
├── data/
│   ├── creditcard.csv
│   └── split_indices.pkl
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── results/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── amount_by_class.png
│   ├── correlation_heatmap.png
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_comparison.csv
├── src/
│   ├── check_data.py
│   ├── eda.py
│   ├── split_data.py
│   ├── preprocess.py
│   ├── train_logistic.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── evaluate_models.py
│   ├── save_model.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Project dataset
data/creditcard.csv (not included in repository)
Download from Kaggle and place inside data


## Setup

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Ensure `creditcard.csv` is located at:

```text
data/creditcard.csv
```

## Execution Workflow

Run scripts from the `src/` directory:

```bash
cd src
```

### Phase 1: Data Understanding

```bash
python check_data.py
```

Prints dataset shape, columns, data types, missing values, class distribution, fraud percentage, and amount statistics.

### Phase 2: Exploratory Data Analysis

```bash
python eda.py
```

Generates visualizations in `results/`:

- Class distribution
- Amount distribution
- Amount by class
- Correlation heatmap
- Top positive/negative fraud correlations (printed to terminal)

### Phase 3: Data Preprocessing & Split

```bash
python split_data.py
```

Creates an 80/20 stratified train-test split and saves indices to `data/split_indices.pkl`.

Scaling of `Amount` and `Time` is handled automatically during model training using `StandardScaler`.

### Phase 4: Model Training

```bash
python train_logistic.py
python train_decision_tree.py
python train_random_forest.py
```

Trains three classifiers with balanced class weights. Random Forest also runs hyperparameter tuning via `RandomizedSearchCV`.

### Phase 5–7: Evaluation & Comparison

```bash
python evaluate_models.py
```

Evaluates all models on the **test set only** and saves:

- Classification report
- Confusion matrix
- ROC curve
- Model comparison CSV
- Feature importance chart (Random Forest)

### Phase 9: Save Production Artifacts

```bash
python save_model.py
```

Saves:

- `models/fraud_model.pkl`
- `models/scaler.pkl`

### Phase 10: Interactive Prediction

```bash
python predict.py
```

Terminal menu options:

1. Test a random transaction from the held-out test set
2. Enter transaction features manually (`Time`, `V1–V28`, `Amount`)
3. Exit

## Full Pipeline (One Command Sequence)

```bash
cd src
python check_data.py
python eda.py
python split_data.py
python train_logistic.py
python train_decision_tree.py
python train_random_forest.py
python evaluate_models.py
python save_model.py
python predict.py
```

## Models Used

| Model | Configuration |
|-------|---------------|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, tuned with `RandomizedSearchCV`, `scoring="f1"` |

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Key Design Decisions

- **No SMOTE/ADASYN/oversampling/undersampling** — imbalance handled via `class_weight="balanced"`
- **Stratified split** — preserves fraud ratio in train and test sets
- **Scaler fit on train only** — prevents data leakage
- **Modular scripts** — each phase is independently runnable

## Requirements

- Python 3.11+
- See `requirements.txt` for package dependencies

## Author Notes

This project is designed as a final-year machine learning capstone with clean modular code, reproducible workflows, and production-style artifact saving for inference.
