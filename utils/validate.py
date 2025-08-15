"""
Data validation utilities for the BTC forecasting system.

This module provides assertion functions to validate data quality, prevent leakage,
and ensure proper time series structure for NeuralForecast compatibility.
"""

import pandas as pd
import numpy as np
from typing import Sequence


def assert_regular_grid(df: pd.DataFrame, freq: str = "15min") -> None:
    """
    Validate 15-minute time grid completeness and monotonicity.
    
    Args:
        df: DataFrame with 'ds' column containing timestamps
        freq: Expected frequency (default: "15min")
        
    Raises:
        AssertionError: If grid is irregular, non-monotonic, or has wrong timezone
    """
    if 'ds' not in df.columns:
        raise AssertionError("DataFrame must have 'ds' column for time validation")
    
    # Extract DatetimeIndex - convert to DatetimeIndex for proper methods
    ds = pd.DatetimeIndex(pd.to_datetime(df['ds']))
    
    # Check if empty
    if len(ds) == 0:
        raise AssertionError("Empty DataFrame cannot have regular grid")
    
    # Assert monotonic increasing
    if not ds.is_monotonic_increasing:
        raise AssertionError("Timestamps are not monotonic increasing")
    
    # Assert UTC timezone
    if ds.tz is None:
        raise AssertionError("Timestamps must be timezone-aware (UTC required)")
    
    if str(ds.tz) != "UTC":
        raise AssertionError(f"Timestamps must be in UTC timezone, got: {ds.tz}")
    
    # Create expected date range
    expected = pd.date_range(ds.min(), ds.max(), freq=freq, tz="UTC")
    
    # Assert length match
    if len(ds) != len(expected):
        raise AssertionError(
            f"Missing or extra bars vs regular grid: expected {len(expected)} bars, got {len(ds)}"
        )
    
    # Check for exact timestamp matches
    if not ds.equals(expected):
        # Find differences using set operations
        ds_set = set(ds)
        expected_set = set(expected)
        
        missing = expected_set - ds_set
        extra = ds_set - expected_set
        
        error_msg = "Grid irregularities detected:"
        if len(missing) > 0:
            error_msg += f" Missing {len(missing)} timestamps"
        if len(extra) > 0:
            error_msg += f" Extra {len(extra)} timestamps"
            
        raise AssertionError(error_msg)


def assert_utc_eob(df: pd.DataFrame, freq: str = "15min") -> None:
    """
    Validate UTC end-of-bar timestamp alignment.
    
    Args:
        df: DataFrame with 'ds' column containing timestamps
        freq: Expected frequency (default: "15min")
        
    Raises:
        AssertionError: If timestamps are not aligned to EOB boundaries
    """
    if 'ds' not in df.columns:
        raise AssertionError("DataFrame must have 'ds' column for EOB validation")
    
    # Extract DatetimeIndex
    ds = pd.DatetimeIndex(pd.to_datetime(df['ds']))
    
    if len(ds) == 0:
        return  # Empty DataFrame is valid
    
    # Assert UTC timezone
    if ds.tz is None:
        raise AssertionError("Timestamps must be timezone-aware for EOB validation")
    
    if str(ds.tz) != "UTC":
        raise AssertionError(f"EOB validation requires UTC timezone, got: {ds.tz}")
    
    # Check minute alignment for 15-minute frequency
    if freq == "15min":
        minutes = ds.minute
        valid_minutes = [0, 15, 30, 45]
        
        invalid_minutes = minutes[~minutes.isin(valid_minutes)]
        if len(invalid_minutes) > 0:
            unique_invalid = invalid_minutes.unique()
            raise AssertionError(
                f"Timestamps not aligned to 15-minute EOB boundaries. "
                f"Found invalid minutes: {sorted(unique_invalid)}, expected: {valid_minutes}"
            )
    
    # Check second and microsecond alignment (should be 0 for EOB)
    if not all(ds.second == 0):
        raise AssertionError("EOB timestamps must have seconds = 0")
    
    if not all(ds.microsecond == 0):
        raise AssertionError("EOB timestamps must have microseconds = 0")


def assert_shifted(df: pd.DataFrame, hist_cols: Sequence[str]) -> None:
    """
    Detect potential data leakage using correlation analysis.
    
    Args:
        df: DataFrame with 'y' column and historical feature columns
        hist_cols: List of historical column names to validate
        
    Raises:
        AssertionError: If leakage is detected in any historical column
    """
    if 'y' not in df.columns:
        raise AssertionError("DataFrame must have 'y' column for leakage detection")
    
    # Extract y values, removing NaN for correlation computation
    y = df['y'].values
    valid_mask = ~np.isnan(y)
    
    if np.sum(valid_mask) < 2:
        raise AssertionError("Need at least 2 non-NaN y values for correlation analysis")
    
    # Create lead-1 series (shift y forward by 1)
    y_lead1 = np.roll(y, -1)
    # Exclude last value since it's rolled from first
    y_lead1 = y_lead1[:-1]
    y_current = y[:-1]
    valid_mask = valid_mask[:-1]
    
    for col in hist_cols:
        if col not in df.columns:
            raise AssertionError(f"Historical column '{col}' not found in DataFrame")
        
        x = df[col].values[:-1]  # Exclude last value to match y_lead1 length
        
        # Skip if all values are NaN
        col_valid_mask = valid_mask & ~np.isnan(x)
        if np.sum(col_valid_mask) < 2:
            continue  # Skip columns with insufficient data
        
        # Compute correlations using only valid data points
        x_valid = x[col_valid_mask]
        y_current_valid = y_current[col_valid_mask]
        y_lead1_valid = y_lead1[col_valid_mask]
        
        # Compute correlations
        corr_current = np.corrcoef(x_valid, y_current_valid)[0, 1]
        corr_lead1 = np.corrcoef(x_valid, y_lead1_valid)[0, 1]
        
        # Handle NaN correlations (can happen with constant values)
        if np.isnan(corr_current):
            corr_current = 0.0
        if np.isnan(corr_lead1):
            corr_lead1 = 0.0
        
        # Check for leakage: if correlation with future y is stronger than current y
        # This suggests the feature contains future information
        if abs(corr_lead1) > abs(corr_current) + 0.1:  # Small tolerance for numerical precision
            raise AssertionError(
                f"Potential leakage in {col}: fix shift(1) - "
                f"corr(x,y)={corr_current:.3f} < corr(x,y_lead1)={corr_lead1:.3f}"
            )


def assert_no_forward_fill_y(df: pd.DataFrame) -> None:
    """
    Prevent target variable forward-filling.
    
    Args:
        df: DataFrame with 'y' and 'close' columns
        
    Raises:
        AssertionError: If forward-fill violations are detected
    """
    if 'y' not in df.columns:
        raise AssertionError("DataFrame must have 'y' column for forward-fill validation")
    
    if 'close' not in df.columns:
        raise AssertionError("DataFrame must have 'close' column for forward-fill validation")
    
    # Check rows where 'close' is NaN should have y as NaN
    close_nan_mask = df['close'].isna()
    y_notnan_mask = df['y'].notna()
    
    # Find violations: close is NaN but y is not NaN
    violations = close_nan_mask & y_notnan_mask
    
    if violations.any():
        num_violations = violations.sum()
        violation_indices = df.index[violations].tolist()
        
        raise AssertionError(
            f"Detected non-NaN y where close is NaN (forbidden forward-fill of target). "
            f"Found {num_violations} violations at indices: {violation_indices[:10]}"
            + ("..." if num_violations > 10 else "")
        )