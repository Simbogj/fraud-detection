import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix, classification_report
import joblib

# XGBoost import
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception as e:
    _HAS_XGB = False
    logging.getLogger('modeling').warning('XGBoost not available: %s', e)

# ------------------------------------------------------------------
# Logger configuration
# ------------------------------------------------------------------
logger = logging.getLogger('modeling')
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'modeling.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for h in (console_handler, file_handler):
        h.setFormatter(formatter)
        logger.addHandler(h)

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def load_processed_data(processed_dir: Path, dataset: str = 'fraud'):
    """Load pre-processed train/test splits from CSV files produced by the
    feature-engineering notebook, or from a cached .npz bundle.

    Parameters
    ----------
    processed_dir : Path
        Directory containing the processed files.
    dataset : str
        'fraud' for Fraud_Data or 'cc' for creditcard.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Feature matrices and target vectors.
    """
    prefix = 'fraud' if dataset == 'fraud' else 'cc'
    npz_path = processed_dir / f'{prefix}_preprocessed.npz'

    # Try loading from cached .npz first
    if npz_path.exists():
        data = np.load(npz_path)
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        logger.info('Loaded %s from .npz: train %s, test %s',
                    dataset, X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

    # Fall back to CSV splits from feature-engineering notebook
    x_train_path = processed_dir / f'{prefix}_X_train_smote.csv'
    y_train_path = processed_dir / f'{prefix}_y_train_smote.csv'
    x_test_path  = processed_dir / f'{prefix}_X_test.csv'
    y_test_path  = processed_dir / f'{prefix}_y_test.csv'

    for p in (x_train_path, y_train_path, x_test_path, y_test_path):
        if not p.exists():
            raise FileNotFoundError(
                f'{p} not found. Run the feature-engineering notebook first '
                f'to generate processed data splits.'
            )

    X_train = pd.read_csv(x_train_path).values
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test  = pd.read_csv(x_test_path).values
    y_test  = pd.read_csv(y_test_path).values.ravel()

    logger.info('Loaded %s from CSVs: train %s, test %s',
                dataset, X_train.shape, X_test.shape)

    # Cache as .npz for faster subsequent loads
    np.savez(npz_path,
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
    logger.info('Cached %s to %s', dataset, npz_path)

    return X_train, X_test, y_train, y_test

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train‑test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info('Train‑test split: %d train, %d test', len(y_train), len(y_test))
    return X_train, X_test, y_train, y_test

# ------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------
def train_logistic_regression(X_train, y_train, **kwargs):
    """Train a Logistic Regression model with balanced class weight.

    Parameters
    ----------
    X_train, y_train : array‑like
        Training data.
    kwargs : dict
        Additional arguments passed to ``LogisticRegression``.
    """
    model = LogisticRegression(class_weight='balanced', max_iter=1000, **kwargs)
    model.fit(X_train, y_train)
    logger.info('Trained Logistic Regression')
    return model

def train_xgboost(X_train, y_train, **kwargs):
    """Train an XGBoost classifier.

    Raises
    ------
    RuntimeError
        If XGBoost is not installed.
    """
    if not _HAS_XGB:
        raise RuntimeError('XGBoost is not available in the environment.')
    params = dict(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
    )
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    logger.info('Trained XGBoost')
    return model

# ------------------------------------------------------------------
# Evaluation utilities
# ------------------------------------------------------------------
def evaluate_model(model, X, y, threshold=0.5):
    """Compute a suite of metrics for a binary classifier.

    Returns a dict with keys: ``precision``, ``recall``, ``auc_pr``, ``f1``, ``confusion``.
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        # XGBoost's ``predict`` returns raw probabilities when ``output_margin=False``
        probs = model.predict(X)
    preds = (probs >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y, probs)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, output_dict=True)
    logger.info('Evaluation – AUC‑PR: %.4f, F1: %.4f', auc_pr, f1)
    return {
        'precision': precision.mean(),
        'recall': recall.mean(),
        'auc_pr': auc_pr,
        'f1': f1,
        'confusion': cm,
        'report': report,
    }

def cross_validate(model_cls, X, y, cv=5, **model_kwargs):
    """Perform stratified K‑fold cross‑validation and return mean/std of AUC‑PR.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        scores.append(auc(recall, precision))
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info('Cross‑validation – mean AUC‑PR: %.4f ± %.4f', mean_score, std_score)
    return {'mean_auc_pr': mean_score, 'std_auc_pr': std_score}

# ------------------------------------------------------------------
# Persistence utilities
# ------------------------------------------------------------------
def save_model(model, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info('Model saved to %s', output_path)

def load_model(model_path: Path):
    model = joblib.load(model_path)
    logger.info('Model loaded from %s', model_path)
    return model

# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run modeling pipeline for fraud detection')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Directory with processed files')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to store trained models')
    parser.add_argument('--dataset', type=str, default='fraud', choices=['fraud', 'cc'],
                        help='Which dataset: fraud (Fraud_Data) or cc (creditcard)')
    parser.add_argument('--run_xgb', action='store_true', help='Train XGBoost model (if available)')
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_dir = Path(args.model_dir)

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir, dataset=args.dataset)

    # Logistic Regression
    lr = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test)
    save_model(lr, model_dir / 'logistic_regression.pkl')
    print('\nLogistic Regression metrics:')
    for k, v in lr_metrics.items():
        if k != 'confusion' and k != 'report':
            print(f'{k}: {v:.4f}')
    print('Confusion matrix:\n', lr_metrics['confusion'])

    # XGBoost (optional)
    if args.run_xgb:
        if not _HAS_XGB:
            raise RuntimeError('XGBoost not installed – cannot run XGBoost training.')
        xgb = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb, X_test, y_test)
        save_model(xgb, model_dir / 'xgboost_model.pkl')
        print('\nXGBoost metrics:')
        for k, v in xgb_metrics.items():
            if k != 'confusion' and k != 'report':
                print(f'{k}: {v:.4f}')
        print('Confusion matrix:\n', xgb_metrics['confusion'])
