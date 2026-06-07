"""
Data loading module for the Fraud Detection project.
Provides functions to load raw CSV datasets with appropriate dtypes.
"""

import os
import pandas as pd

# Resolve paths relative to this file's location (src/ -> project root)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(_BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(_BASE_DIR, "data", "processed")


def load_fraud_data() -> pd.DataFrame:
    """
    Load Fraud_Data.csv with correct dtypes.

    Returns
    -------
    pd.DataFrame
        E-commerce transaction data with parsed datetime columns.
    """
    path = os.path.join(RAW_DATA_DIR, "Fraud_Data.csv")
    df = pd.read_csv(
        path,
        parse_dates=["signup_time", "purchase_time"],
    )
    print(f"Loaded Fraud_Data.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_ip_country() -> pd.DataFrame:
    """
    Load IpAddress_to_Country.csv.

    Returns
    -------
    pd.DataFrame
        IP range-to-country mapping table.
    """
    path = os.path.join(RAW_DATA_DIR, "IpAddress_to_Country.csv")
    df = pd.read_csv(path)
    print(
        f"Loaded IpAddress_to_Country.csv: {df.shape[0]} rows, {df.shape[1]} columns"
    )
    return df


def load_creditcard() -> pd.DataFrame:
    """
    Load creditcard.csv.

    Returns
    -------
    pd.DataFrame
        Bank credit card transaction data.
    """
    path = os.path.join(RAW_DATA_DIR, "creditcard.csv")
    df = pd.read_csv(path)
    print(f"Loaded creditcard.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
