# Fraud Detection - E-commerce and Bank Transactions

This repository contains a production-grade machine learning system to identify and prevent fraudulent transactions across two data streams: E-commerce transactions (`Fraud_Data.csv`) and Credit Card transactions (`creditcard.csv`).

## 1. Project Overview & Architecture
This pipeline addresses two distinct transaction data patterns:
- **E-Commerce Transactions**: Feature engineering based on signup-to-purchase latency, temporal attributes, geolocation, and transaction frequency.
- **Credit Card Transactions**: Highly anonymized PCA-transformed features.

The repository follows a clean, modular structure:
```text
├── data/
│   ├── raw/             # Raw datasets (gitignored)
│   └── processed/       # Preprocessed arrays and generated plots
├── models/              # PERSISTED trained model objects (pkl)
├── notebooks/           # Jupyter notebooks for EDA, Modeling, and Explainability
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/                 # Reusable production source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── modeling.py
├── reports/             # Business intelligence and analysis reports
│   └── final_report.md
├── requirements.txt     # Virtual environment requirements
└── README.md            # Global project documentation
```

## 2. Installation & Setup
To run the notebooks or modular code locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Simbogj/fraud-detection
   cd fraud-detection
   ```

2. **Set up the Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Modeling Results & Evaluation
We trained two models for each dataset: **Logistic Regression** (baseline) and **XGBoost** (ensemble). Models were validated using 5-fold Stratified Cross-Validation.

### E-Commerce Fraud Data Results
- **Logistic Regression**: AUC-PR: `0.394` | F1-Score: `0.512`
- **XGBoost**: AUC-PR: `0.626` | F1-Score: `0.704`

### Credit Card PCA Data Results
- **Logistic Regression**: AUC-PR: `0.713` | F1-Score: `0.724`
- **XGBoost**: AUC-PR: `0.803` | F1-Score: `0.825`

**Conclusion**: **XGBoost** was selected as the final production model for both datasets due to its significantly higher AUC-PR and F1 scores, which translate to a better balance between catching fraud (recall) and minimizing false alarms (precision).

## 4. Key SHAP Explainability Insights
- **`time_since_signup`**: The single strongest driver of fraud. Instant transactions following signup (typically under 1 hour) are highly suspicious.
- **`user_txn_count` & `txn_in_24h`**: Velocity markers show that high frequencies of transactions are strong fraud flags.
- **`purchase_value`**: Higher-value transactions systematically increase the fraud probability score.

## 5. Actionable Recommendations
1. Require multi-factor authentication (MFA) for transactions occurring within 1 hour of user signup.
2. Trigger temporary account locks or CAPTCHAs if an account exceeds 3 transactions within 24 hours.
3. Automatically escalate high-ticket purchases (>$300) to secondary verification if the purchase IP differs from the signup IP.
