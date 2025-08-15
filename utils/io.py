"""
Data processing and I/O utilities for the BTC forecasting system.

This module provides functions for loading, processing, and saving time series data
with proper UTC timestamp handling and NeuralForecast schema compliance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def aggregate_1min_to_15min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute OHLCV data to 15-minute bars with proper UTC EOB alignment.
    
    Args:
        df_1min: DataFrame with 1-minute OHLCV data
        
    Returns:
        DataFrame with 15-minute OHLCV bars, UTC EOB timestamps
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Handle different column name cases
    df_work = df_1min.copy()
    
    # Standardize column names to lowercase
    df_work.columns = df_work.columns.str.lower()
    
    # Check for timestamp column
    if 'timestamp' not in df_work.columns and 'ds' not in df_work.columns:
        raise ValueError("No timestamp column found in 1-minute data")
    
    # Check for required OHLCV columns
    missing_cols = [col for col in required_cols if col not in df_work.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert timestamp to UTC datetime
    ts_col = 'timestamp' if 'timestamp' in df_work.columns else 'ds'
    
    # Handle Unix timestamp conversion
    if df_work[ts_col].dtype in ['int64', 'float64']:
        df_work['ds'] = pd.to_datetime(df_work[ts_col], unit='s', utc=True)
    else:
        df_work['ds'] = pd.to_datetime(df_work[ts_col], utc=True)
    
    # Set ds as index for resampling
    df_work = df_work.set_index('ds')
    
    # Resample to 15-minute bars with EOB alignment
    df_15min = df_work.resample('15min', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Reset index to get ds column back
    df_15min = df_15min.reset_index()
    
    # Ensure EOB timestamps (:00, :15, :30, :45)
    minutes = df_15min['ds'].dt.minute
    if not all(minutes.isin([0, 15, 30, 45])):
        raise ValueError("Aggregation produced non-EOB timestamps")
    
    return df_15min


def load_raw_1min_data(file_path: str = "data/raw/btcusd_1-min_data.csv") -> pd.DataFrame:
    """
    Load raw 1-minute BTC OHLCV data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with 1-minute OHLCV data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the file format is invalid
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load raw data from {file_path}: {e}")


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to parquet with proper directory management.
    
    Args:
        df: DataFrame to save
        path: Output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def timestamped_path(base_dir: str, stem: str, ext: str = "parquet") -> str:
    """
    Generate timestamped file paths for versioning.
    
    Args:
        base_dir: Base directory for the file
        stem: Base filename stem
        ext: File extension (default: "parquet")
        
    Returns:
        Timestamped file path
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{stem}_{timestamp}.{ext}"
    return str(Path(base_dir) / filename)


def load_canonical_frame(path: str) -> pd.DataFrame:
    """
    Load and validate stored canonical frames.
    
    Args:
        path: Path to the parquet file
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Canonical frame file not found: {path}")
    
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load canonical frame from {path}: {e}")


def regularize_to_grid_utc(df_ohlcv: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Create a complete UTC time grid with proper end-of-bar alignment.
    
    Args:
        df_ohlcv: DataFrame with OHLCV data and 'ds' timestamp column
        freq: Frequency for the grid (default: "15min")
        
    Returns:
        DataFrame with complete UTC grid, gaps filled with NaN
        
    Raises:
        ValueError: If no timestamp column found or conversion fails
    """
    if 'ds' not in df_ohlcv.columns:
        raise ValueError("DataFrame must have 'ds' column for grid regularization")
    
    df_work = df_ohlcv.copy()
    
    # Convert timestamps to UTC timezone with proper datetime64[ns] format
    df_work['ds'] = pd.to_datetime(df_work['ds'], utc=True)
    
    # Snap timestamps to frequency boundaries (floor to nearest boundary)
    df_work['ds'] = df_work['ds'].dt.floor(freq)
    
    # Create complete date range from min to max timestamp
    date_range = pd.date_range(
        start=df_work['ds'].min(),
        end=df_work['ds'].max(),
        freq=freq,
        tz='UTC'
    )
    
    # Create complete grid DataFrame
    complete_grid = pd.DataFrame({'ds': date_range})
    
    # Merge original data with complete grid, filling gaps with NaN
    # Use left join to preserve all timestamps in the grid
    df_regularized = complete_grid.merge(df_work, on='ds', how='left')
    
    # Preserve OHLCV columns order
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [col for col in ohlcv_cols if col in df_regularized.columns]
    
    # Reorder columns: ds first, then available OHLCV columns, then others
    other_cols = [col for col in df_regularized.columns if col not in ['ds'] + available_cols]
    column_order = ['ds'] + available_cols + other_cols
    
    return df_regularized[column_order]


def make_nf_canonical(df_ohlcv_15m: pd.DataFrame, unique_id: str = "BTC-USD") -> pd.DataFrame:
    """
    Transform OHLCV data into NeuralForecast canonical schema.
    
    Args:
        df_ohlcv_15m: DataFrame with 15-minute OHLCV data
        unique_id: Unique identifier for the time series (default: "BTC-USD")
        
    Returns:
        DataFrame in NF canonical format with log returns as target
        
    Raises:
        ValueError: If required columns are missing
    """
    if 'close' not in df_ohlcv_15m.columns:
        raise ValueError("DataFrame must have 'close' column for log return computation")
    
    # Ensure proper time grid first
    df_regularized = regularize_to_grid_utc(df_ohlcv_15m, "15min")
    
    # Compute log returns: y = log(close_t) - log(close_t-1)
    df_regularized['y'] = np.log(df_regularized['close']).diff()
    
    # Create canonical schema: ["unique_id", "ds", "y", "open", "high", "low", "close", "volume"]
    df_canonical = df_regularized.copy()
    df_canonical['unique_id'] = unique_id
    
    # Define canonical column order
    canonical_cols = ['unique_id', 'ds', 'y']
    
    # Add OHLCV columns if they exist
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_cols:
        if col in df_canonical.columns:
            canonical_cols.append(col)
    
    # Add any other columns
    other_cols = [col for col in df_canonical.columns if col not in canonical_cols]
    canonical_cols.extend(other_cols)
    
    return df_canonical[canonical_cols]


def drop_train_nans_and_winsorize(nf_df: pd.DataFrame, lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    """
    Apply winsorization for training stability while preserving evaluation data.
    
    Args:
        nf_df: NeuralForecast DataFrame with 'y' column
        lower_q: Lower quantile for winsorization (default: 0.001)
        upper_q: Upper quantile for winsorization (default: 0.999)
        
    Returns:
        DataFrame with both 'y' (original) and 'y_train' (winsorized) columns
        
    Raises:
        ValueError: If 'y' column is missing
    """
    if 'y' not in nf_df.columns:
        raise ValueError("DataFrame must have 'y' column for winsorization")
    
    df_result = nf_df.copy()
    
    # Compute quantiles on non-NaN y values only
    y_values = df_result['y'].dropna()
    
    if len(y_values) == 0:
        # If all y values are NaN, just copy y to y_train
        df_result['y_train'] = df_result['y']
        return df_result
    
    # Calculate quantile bounds
    lower_bound = y_values.quantile(lower_q)
    upper_bound = y_values.quantile(upper_q)
    
    # Create y_train column with clipped values
    df_result['y_train'] = df_result['y'].clip(lower=lower_bound, upper=upper_bound)
    
    # Preserve original y column for evaluation
    # y_train will have winsorized values, y will have original values
    
    return df_result