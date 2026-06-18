# Task 1: Preprocessing and Feature Engineering Pipeline

This documentation details the design, rationale, and implementation details of the data preprocessing, enrichment, and feature engineering pipeline for the E-commerce transaction data (`Fraud_Data.csv`).

## 1. Geolocation Integration
- **The Challenge**: The IP addresses in the raw transaction log are stored as large numeric float values. The geolocation mappings in `IpAddress_to_Country.csv` contain lower and upper integer IP boundaries for each country.
- **The Solution**:
  1. We round and cast the `ip_address` in the transaction data to `int64`.
  2. We drop any NaNs and sort both datasets by the IP bounds.
  3. We perform a fast, range-based lookup using `pd.merge_asof` with `direction='backward'` on the sorted IP bounds.
  4. We validate the range match by checking if the user's IP address is less than or equal to the upper bound of the matched range (`ip_address_int <= upper_bound_ip_address`). If not, or if no match is found, the country is set to `'Unknown'`.

## 2. Feature Engineering
We engineered two primary families of features to capture the temporal and behavioural patterns of fraud:
- **Temporal Features**:
  - `time_since_signup`: The duration (in hours) between the user's signup time and the transaction purchase time. Short signup-to-purchase times are a strong indicator of automated fraud scripts.
  - `hour_of_day`: The hour the purchase occurred, capturing diurnal patterns (e.g., higher fraud rates during typical sleeping hours).
  - `day_of_week`: The day the purchase occurred.
- **Velocity Features**:
  - `user_txn_count`: The total count of transactions associated with the same `user_id`.
  - `txn_in_24h`: The transaction count within a rolling 24-hour window, indicating high-frequency checkout attempts.

## 3. Data Transformation (Scaling & Encoding)
- **Categorical Columns**: Features such as `source`, `browser`, `sex`, and `country` are encoded using `OneHotEncoder` with `handle_unknown='ignore'` and `sparse_output=False` to produce a dense representation suitable for tree-based and linear models.
- **Numerical Columns**: Features like `purchase_value`, `age`, `time_since_signup`, `hour_of_day`, `day_of_week`, `user_txn_count`, and `txn_in_24h` are scaled using `StandardScaler` to normalize the distribution (mean = 0, variance = 1), preventing features with larger scales (like IP or purchase values) from dominating the learning process.

## 4. Class Imbalance Handling
- **The Challenge**: E-commerce transaction fraud is heavily imbalanced (~9.36% fraud cases, ~90.64% legitimate). Left unaddressed, classifiers will bias toward the majority class and fail to detect fraud.
- **The Solution (SMOTE)**:
  - We apply **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set to a 1:1 ratio. SMOTE works by creating synthetic examples of the minority class along the line segments joining k-nearest neighbors, preventing overfitting.
  - **CRITICAL**: SMOTE is applied **only to the training split**. The test split is left in its natural, imbalanced state to guarantee realistic evaluation metrics.

## 5. Defensive Error Handling & Pipeline Robustness
- The module `src/preprocessing.py` and the notebook `eda-fraud-data.ipynb` implement:
  - Input file verification to raise clear `FileNotFoundError` exceptions.
  - Duplication removal (`drop_duplicates`) and missing-value imputation (medians for numeric, modes for categorical columns).
  - Type-safe casting of merge keys to `'int64'` to prevent pandas merge-asof type conflicts.
  - Structured logging of row counts, shapes, and match ratios at each step.
