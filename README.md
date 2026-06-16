# Fraud Detection - E-commerce and Bank Transactions

## Project Overview

This project improves fraud detection across two transaction streams for **Adey Innovations Inc.**:

- **E-commerce transactions** (Fraud_Data.csv) — rich user, device, and behavioral context
- **Bank credit card transactions** (creditcard.csv) — PCA-anonymized features

Both datasets are highly imbalanced, requiring specialized preprocessing, resampling, and evaluation strategies.

## Repository Structure

```
fraud-detection/
├── .github/workflows/unittests.yml
├── .vscode/settings.json
├── data/
│   ├── raw/                    # Original datasets (gitignored)
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb          # EDA for e-commerce fraud data
│   ├── eda-creditcard.ipynb          # EDA for credit card data
│   ├── feature-engineering.ipynb     # Feature engineering, scaling, SMOTE
│   ├── modeling.ipynb                # Model training (Task 2)
│   ├── shap-explainability.ipynb     # SHAP analysis (Task 3)
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Raw data loading functions
│   ├── preprocessing.py              # Full preprocessing pipeline
│   └── modeling.py                   # Model training and evaluation
├── scripts/
│   ├── __init__.py
│   └── generate_notebooks.py         # Regenerates .ipynb from Python
├── tests/
│   └── __init__.py
├── models/                           # Saved model artifacts (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, modeling, metrics
- `imbalanced-learn` — SMOTE resampling
- `xgboost` — ensemble model
- `shap` — model explainability
- `matplotlib`, `seaborn` — visualization
- `jupyter` — notebooks

## Task 1: Data Analysis and Preprocessing

### Data Cleaning
- **Missing values**: No missing values found in either dataset
- **Duplicates**: Exact duplicate rows removed from both datasets
- **Data types**: Timestamps parsed to datetime, categorical columns set to `category` dtype, target columns cast to `int8`

### Exploratory Data Analysis

**Fraud_Data.csv (~151K transactions)**
- Univariate distributions: age, purchase_value, gender, browser, source
- Bivariate analysis: features vs fraud target, fraud rates by browser/source/gender
- Class imbalance: ~9% fraud (approximately 10:1 ratio)

**creditcard.csv (~285K transactions)**
- Univariate distributions: Time, Amount, PCA features V1-V28
- Bivariate analysis: Amount and top PCA features by fraud class
- Class imbalance: ~0.17% fraud (approximately 578:1 ratio)

### Geolocation Integration (Fraud_Data only)
1. `ip_address` column is already numeric (float) — cast to integer
2. Range-based merge with `IpAddress_to_Country.csv` using `pd.merge_asof`
3. Validated each IP falls within the matched range; unmatched IPs labeled "Unknown"
4. Analyzed fraud rates by country

### Feature Engineering (Fraud_Data)

| Feature | Description |
|---|---|
| `time_since_signup` | Hours between signup and purchase (clipped to >= 0) |
| `hour_of_day` | Hour of purchase (0-23) |
| `day_of_week` | Day of purchase (0=Monday, 6=Sunday) |
| `user_transaction_count` | Total transactions per user |
| `txn_in_24h` | Transactions by same user in preceding 24 hours |
| `country` | Geolocation from IP-to-country lookup |

### Data Transformation
- **Numerical features**: StandardScaler
- **Categorical features**: OneHotEncoder (with `handle_unknown='ignore'`)
- Applied via `ColumnTransformer` on train/test splits

### Class Imbalance Handling

**Technique: SMOTE** (Synthetic Minority Over-sampling Technique)

**Justification:**
1. SMOTE generates synthetic minority samples along the feature-space convex hull, preserving decision boundaries better than random oversampling
2. It avoids the information loss of undersampling the majority class
3. Applied **only on the training set** to prevent data leakage into test evaluation
4. Both datasets balanced to 50/50 after SMOTE

| Dataset | Before SMOTE | After SMOTE |
|---|---|---|
| Fraud_Data | ~9% fraud | 50% fraud |
| creditcard | ~0.17% fraud | 50% fraud |

### Deliverables
- `notebooks/eda-fraud-data.ipynb` — Full EDA report for Fraud_Data.csv
- `notebooks/eda-creditcard.ipynb` — Full EDA report for creditcard.csv
- `notebooks/feature-engineering.ipynb` — Feature engineering, encoding, scaling, SMOTE
- `data/processed/` — Cleaned datasets and train/test splits
