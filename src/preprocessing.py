import logging
import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from intervaltree import IntervalTree

# Configure logger for preprocessing
logger = logging.getLogger('preprocess')
logger.setLevel(logging.INFO)
# Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(parents=True, exist_ok=True)
handler_console = logging.StreamHandler()
handler_file = logging.FileHandler(log_dir / 'preprocess.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
for handler in (handler_console, handler_file):
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file, raising a clear error if the file does not exist."""
    if not path.is_file():
        logger.error(f"CSV file not found: {path}")
        raise FileNotFoundError(f"CSV file not found: {path}")
    logger.info(f"Loading CSV file: {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, drop duplicates, and enforce dtypes where possible."""
    logger.info("Starting data cleaning")
    initial_shape = df.shape
    # Drop exact duplicate rows
    df = df.drop_duplicates()
    logger.info(f"Dropped duplicates: {initial_shape[0] - df.shape[0]}")

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Impute numeric with median, categorical with mode
    for col in num_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            logger.info(f"Imputed missing numeric column '{col}' with median={median}")
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            logger.info(f"Imputed missing categorical column '{col}' with mode='{mode}'")
    logger.info("Data cleaning completed")
    return df

def ip_to_int(ip_val) -> int:
    """Convert IP address (either numeric float/int or dotted IPv4 string) to integer.
    Returns -1 for malformed entries or NaN.
    """
    if pd.isna(ip_val):
        return -1
    if isinstance(ip_val, (int, float, np.integer, np.floating)):
        return int(round(ip_val))
    try:
        ip_str = str(ip_val).strip()
        if '.' in ip_str:
            parts = ip_str.split('.')
            if len(parts) == 4:
                return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        return int(float(ip_str))
    except Exception:
        logger.warning(f"Failed to convert IP '{ip_val}' to integer")
        return -1

def build_ip_lookup(df_ip: pd.DataFrame) -> IntervalTree:
    """Build an IntervalTree for fast IP‑range to country lookup.
    Expected columns: 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'.
    """
    tree = IntervalTree()
    for _, row in df_ip.iterrows():
        lo = int(row['lower_bound_ip_address'])
        hi = int(row['upper_bound_ip_address'])
        country = row['country']
        tree[lo:hi + 1] = country
    logger.info("IP lookup interval tree constructed")
    return tree

def enrich_geolocation(df: pd.DataFrame, ip_lookup: IntervalTree) -> pd.DataFrame:
    """Add a 'country' column based on the IP address of each transaction."""
    logger.info("Starting geolocation enrichment")
    if 'ip_address' not in df.columns:
        logger.error("Column 'ip_address' not found for geolocation enrichment")
        raise KeyError("Column 'ip_address' not found for geolocation enrichment")
    df = df.copy()
    df['ip_int'] = df['ip_address'].apply(ip_to_int)
    df['country'] = df['ip_int'].apply(lambda x: next(iter(ip_lookup[x]), np.nan) if x != -1 else np.nan)
    missing = df['country'].isnull().sum()
    if missing:
        logger.warning(f"{missing} rows could not be mapped to a country")
    df.drop(columns=['ip_int'], inplace=True)
    logger.info("Geolocation enrichment completed")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and velocity features for fraud detection.
    Expected timestamp columns: 'signup_time', 'purchase_time'.
    """
    logger.info("Starting feature engineering")
    df = df.copy()
    # Ensure timestamps are datetime
    for col in ['signup_time', 'purchase_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    # Transaction frequency per user in the last 24h (simple approximation)
    if 'user_id' in df.columns and 'purchase_time' in df.columns:
        df.sort_values(['user_id', 'purchase_time'], inplace=True)
        df['transactions_last_24h'] = (
            df.groupby('user_id')['purchase_time']
            .transform(lambda s: s.diff().dt.total_seconds().fillna(0).lt(24 * 3600).cumsum())
        )
    logger.info("Feature engineering completed")
    return df

def encode_and_scale(df: pd.DataFrame, target_column: str):
    """One‑hot encode categorical columns and scale numeric columns.
    Returns transformed feature matrix (numpy array), the fitted ColumnTransformer, and the target series.
    """
    logger.info("Starting encoding and scaling")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # Identify column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    # Define transformers
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    logger.info("Encoding and scaling completed")
    return X_processed, y.values, preprocessor

def apply_resampling(X, y, method: str = None):
    """Optionally resample the training data.
    If method is None, the data is returned unchanged.
    Currently supports 'smote' (requires imbalanced-learn).
    """
    if method is None:
        return X, y
    if method.lower() == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as e:
            logger.error("imbalanced-learn is not installed. Install it to use SMOTE.")
            raise e
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(f"Applied SMOTE: original {len(y)} -> resampled {len(y_res)}")
        return X_res, y_res
    else:
        raise ValueError(f"Unsupported resampling method: {method}")

def preprocess_fraud_data(raw_dir: Path, processed_dir: Path, resample_method: str = None):
    """Orchestrate the full preprocessing pipeline for the fraud‑detection datasets.
    Parameters
    ----------
    raw_dir: Path to the directory containing raw CSV files.
    processed_dir: Path where the cleaned CSVs will be saved.
    resample_method: Optional resampling technique (e.g., 'smote').
    """
    logger.info("Starting full preprocessing pipeline")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    fraud_path = raw_dir / 'Fraud_Data.csv'
    ip_lookup_path = raw_dir / 'IpAddress_to_Country.csv'
    creditcard_path = raw_dir / 'creditcard.csv'

    fraud_df = load_csv(fraud_path)
    ip_df = load_csv(ip_lookup_path)
    creditcard_df = load_csv(creditcard_path)

    # Clean
    fraud_df = clean_data(fraud_df)
    creditcard_df = clean_data(creditcard_df)

    # Geolocation enrichment for fraud data
    ip_tree = build_ip_lookup(ip_df)
    fraud_df = enrich_geolocation(fraud_df, ip_tree)

    # Feature engineering
    fraud_df = engineer_features(fraud_df)
    # Credit card dataset already has engineered features; we only ensure correct types
    if 'Time' in creditcard_df.columns:
        creditcard_df.rename(columns={'Time': 'time_seconds'}, inplace=True)
    # No timestamp columns in creditcard, so we skip temporal features

    # Encode & scale – separate for each dataset (they have different target column names)
    fraud_X, fraud_y, fraud_preproc = encode_and_scale(fraud_df, target_column='class')
    cc_X, cc_y, cc_preproc = encode_and_scale(creditcard_df, target_column='Class')

    # Optional resampling (deferred by default)
    if resample_method:
        fraud_X, fraud_y = apply_resampling(fraud_X, fraud_y, method=resample_method)
        cc_X, cc_y = apply_resampling(cc_X, cc_y, method=resample_method)

    # Save processed data as NumPy npz to preserve matrix shape efficiently
    np.savez_compressed(processed_dir / 'fraud_preprocessed.npz', X=fraud_X, y=fraud_y)
    np.savez_compressed(processed_dir / 'creditcard_preprocessed.npz', X=cc_X, y=cc_y)
    # Also persist the preprocessors for future inference
    import joblib
    joblib.dump(fraud_preproc, processed_dir / 'fraud_preprocessor.pkl')
    joblib.dump(cc_preproc, processed_dir / 'creditcard_preprocessor.pkl')
    logger.info("Preprocessing pipeline completed successfully")
    return (fraud_X, fraud_y), (cc_X, cc_y)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run preprocessing for fraud detection project')
    parser.add_argument('--raw_dir', type=str, default='data/raw', help='Directory with raw CSV files')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--resample', type=str, default=None, help='Resampling method (e.g., smote)')
    args = parser.parse_args()
    preprocess_fraud_data(Path(args.raw_dir), Path(args.processed_dir), resample_method=args.resample)
