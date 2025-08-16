"""
Data processing and I/O utilities for the BTC forecasting system.

This module provides functions for loading, processing, and saving time series data
with proper UTC timestamp handling and NeuralForecast schema compliance.

Enhanced with comprehensive artifact management for:
- Cross-validation results persistence (Task 14)
- Model saving integration (Task 15)
- Atomic writes and error handling
"""

import pandas as pd
import numpy as np
import json
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Union


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


def save_cv_artifacts(cv_df: pd.DataFrame,
                      metrics_dict: Union[Dict, pd.DataFrame],
                      leaderboard_df: pd.DataFrame,
                      horizon: int,
                      output_dir: Optional[str] = None,
                      timestamp: Optional[str] = None) -> Dict[str, str]:
    """
    Save cross-validation artifacts with proper naming convention (Task 14).
    
    Implements comprehensive artifact persistence with:
    - Atomic writes (write to temp file, then rename)
    - Proper directory structure (experiments/h{horizon}/)
    - Timestamped filenames for versioning
    - Multiple format support (parquet, JSON, CSV)
    
    Args:
        cv_df: Raw CV predictions DataFrame
        metrics_dict: Metrics as dictionary or DataFrame (saved as JSON)
        leaderboard_df: Model leaderboard DataFrame
        horizon: Forecast horizon (4, 8, 16, or 32)
        output_dir: Base directory (default: experiments/h{horizon})
        timestamp: Optional timestamp string (default: auto-generated)
        
    Returns:
        Dictionary with saved file paths
        
    Raises:
        OSError: If directory creation or file writing fails
        ValueError: If input data is invalid
    """
    # Validate horizon
    if horizon not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of [4, 8, 16, 32]")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"experiments/h{horizon}"
    
    output_path = Path(output_dir)
    
    # Create directory structure with error handling
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created/verified directory: {output_path}")
    except OSError as e:
        logging.error(f"Failed to create directory {output_path}: {e}")
        raise OSError(f"Cannot create output directory {output_path}: {e}")
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    saved_paths = {}
    
    # 1. Save raw CV predictions as parquet (with timestamp)
    try:
        cv_filename = f"cv_results_{timestamp}.parquet"
        cv_path = output_path / cv_filename
        _atomic_write_parquet(cv_df, cv_path)
        saved_paths['cv_results'] = str(cv_path)
        logging.info(f"Saved CV results to {cv_path}")
    except Exception as e:
        logging.error(f"Failed to save CV results: {e}")
        raise
    
    # 2. Save metrics as JSON (not CSV as per spec)
    try:
        metrics_path = output_path / "metrics.json"
        
        # Convert DataFrame to dict if needed
        if isinstance(metrics_dict, pd.DataFrame):
            metrics_data = metrics_dict.to_dict(orient='records')
        else:
            metrics_data = metrics_dict
            
        _atomic_write_json(metrics_data, metrics_path)
        saved_paths['metrics'] = str(metrics_path)
        logging.info(f"Saved metrics to {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise
    
    # 3. Save leaderboard as CSV
    try:
        leaderboard_path = output_path / "leaderboard.csv"
        _atomic_write_csv(leaderboard_df, leaderboard_path)
        saved_paths['leaderboard'] = str(leaderboard_path)
        logging.info(f"Saved leaderboard to {leaderboard_path}")
    except Exception as e:
        logging.error(f"Failed to save leaderboard: {e}")
        raise
    
    return saved_paths


def load_cv_artifacts(horizon: int,
                      output_dir: Optional[str] = None,
                      timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Load saved CV artifacts from disk.
    
    Enhanced to support new artifact structure with:
    - JSON metrics loading
    - Timestamp-specific artifact retrieval
    - Better error handling and logging
    
    Args:
        horizon: Forecast horizon (4, 8, 16, or 32)
        output_dir: Base directory (default: experiments/h{horizon})
        timestamp: Specific timestamp to load, or None for latest
        
    Returns:
        Dictionary with loaded artifacts:
        - 'cv_results': DataFrame with CV predictions
        - 'metrics': Dict/List with metrics data
        - 'leaderboard': DataFrame with model rankings
        
    Raises:
        FileNotFoundError: If required artifacts don't exist
        ValueError: If data format is invalid
    """
    import glob
    
    # Validate horizon
    if horizon not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of [4, 8, 16, 32]")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"experiments/h{horizon}"
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory {output_path} does not exist")
    
    artifacts = {}
    
    # 1. Load metrics from JSON
    metrics_path = output_path / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                artifacts['metrics'] = json.load(f)
            logging.info(f"Loaded metrics from {metrics_path}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse metrics JSON: {e}")
            raise ValueError(f"Invalid metrics JSON format: {e}")
    else:
        logging.warning(f"Metrics file not found: {metrics_path}")
    
    # 2. Load leaderboard from CSV
    leaderboard_path = output_path / "leaderboard.csv"
    if leaderboard_path.exists():
        try:
            artifacts['leaderboard'] = pd.read_csv(leaderboard_path)
            logging.info(f"Loaded leaderboard from {leaderboard_path}")
        except Exception as e:
            logging.error(f"Failed to load leaderboard: {e}")
            raise ValueError(f"Invalid leaderboard CSV: {e}")
    else:
        logging.warning(f"Leaderboard file not found: {leaderboard_path}")
    
    # 3. Load CV results (with timestamp support)
    if timestamp:
        # Load specific timestamp
        cv_path = output_path / f"cv_results_{timestamp}.parquet"
        if cv_path.exists():
            try:
                artifacts['cv_results'] = pd.read_parquet(cv_path)
                logging.info(f"Loaded CV results from {cv_path}")
            except Exception as e:
                logging.error(f"Failed to load CV results: {e}")
                raise ValueError(f"Invalid CV results parquet: {e}")
        else:
            raise FileNotFoundError(f"CV results not found for timestamp {timestamp}")
    else:
        # Load latest CV results
        cv_pattern = str(output_path / "cv_results_*.parquet")
        cv_files = glob.glob(cv_pattern)
        if cv_files:
            latest_cv = sorted(cv_files)[-1]  # Get most recent by filename
            try:
                artifacts['cv_results'] = pd.read_parquet(latest_cv)
                artifacts['cv_timestamp'] = Path(latest_cv).stem.replace('cv_results_', '')
                logging.info(f"Loaded latest CV results from {latest_cv}")
            except Exception as e:
                logging.error(f"Failed to load CV results: {e}")
                raise ValueError(f"Invalid CV results parquet: {e}")
        else:
            logging.warning("No CV results files found")
    
    return artifacts


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


# ============================================================================
# Atomic Write Helpers (for safe file operations)
# ============================================================================

def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Atomically write DataFrame to parquet file.
    
    Uses temporary file + rename to ensure atomic operation.
    
    Args:
        df: DataFrame to save
        path: Target file path
        
    Raises:
        OSError: If write or rename fails
    """
    # Create temp file in same directory (for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp.parquet',
        dir=path.parent,
        prefix=f'.{path.stem}_'
    )
    
    try:
        # Close the file descriptor (we just need the path)
        import os
        os.close(temp_fd)
        
        # Write to temp file
        df.to_parquet(temp_path, index=False)
        
        # Atomic rename (on same filesystem)
        Path(temp_path).replace(path)
        
    except Exception as e:
        # Clean up temp file on error
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise OSError(f"Failed to atomically write parquet to {path}: {e}")


def _atomic_write_json(data: Any, path: Path) -> None:
    """
    Atomically write data to JSON file.
    
    Uses temporary file + rename to ensure atomic operation.
    
    Args:
        data: Data to serialize to JSON
        path: Target file path
        
    Raises:
        OSError: If write or rename fails
    """
    # Create temp file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp.json',
        dir=path.parent,
        prefix=f'.{path.stem}_'
    )
    
    try:
        # Write JSON to temp file
        with open(temp_fd, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Atomic rename
        Path(temp_path).replace(path)
        
    except Exception as e:
        # Clean up temp file on error
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise OSError(f"Failed to atomically write JSON to {path}: {e}")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Atomically write DataFrame to CSV file.
    
    Uses temporary file + rename to ensure atomic operation.
    
    Args:
        df: DataFrame to save
        path: Target file path
        
    Raises:
        OSError: If write or rename fails
    """
    # Create temp file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp.csv',
        dir=path.parent,
        prefix=f'.{path.stem}_'
    )
    
    try:
        # Close the file descriptor
        import os
        os.close(temp_fd)
        
        # Write to temp file
        df.to_csv(temp_path, index=False)
        
        # Atomic rename
        Path(temp_path).replace(path)
        
    except Exception as e:
        # Clean up temp file on error
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise OSError(f"Failed to atomically write CSV to {path}: {e}")


# ============================================================================
# Model Saving Functions (Task 15)
# ============================================================================

def save_nf_models(nf_instance,
                   horizon: int,
                   model_names: Optional[List[str]] = None,
                   output_dir: Optional[str] = None,
                   timestamp: Optional[str] = None,
                   save_best_only: bool = False,
                   best_metric: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """
    Save NeuralForecast models using NF's native save() method (Task 15).
    
    Implements model persistence with:
    - NF's native save() method for each model
    - Metadata-rich filenames (model_type, loss, timestamp)
    - Selective saving of best models only
    - Proper directory structure (experiments/h{horizon}/models/)
    
    Args:
        nf_instance: Fitted NeuralForecast instance with models
        horizon: Forecast horizon (4, 8, 16, or 32)
        model_names: Optional list of specific models to save (default: all)
        output_dir: Base directory (default: experiments/h{horizon}/models)
        timestamp: Optional timestamp string (default: auto-generated)
        save_best_only: If True, only save models in best_metric dict
        best_metric: Dict mapping model names to their performance metric
        
    Returns:
        Dictionary mapping model names to saved file paths
        
    Raises:
        ValueError: If nf_instance is not fitted or horizon invalid
        OSError: If save operation fails
        
    Example:
        >>> # Save all models
        >>> paths = save_nf_models(nf, horizon=4)
        
        >>> # Save only best models based on sCRPS
        >>> best = {'NHITS': 0.012, 'TiDE': 0.015}
        >>> paths = save_nf_models(nf, horizon=4, save_best_only=True, best_metric=best)
    """
    # Validate horizon
    if horizon not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of [4, 8, 16, 32]")
    
    # Check if NF instance is fitted
    if not hasattr(nf_instance, 'models') or not nf_instance.models:
        raise ValueError("NeuralForecast instance has no fitted models")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"experiments/h{horizon}/models"
    
    output_path = Path(output_dir)
    
    # Create models directory
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created/verified models directory: {output_path}")
    except OSError as e:
        logging.error(f"Failed to create models directory {output_path}: {e}")
        raise OSError(f"Cannot create models directory {output_path}: {e}")
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    # Determine which models to save
    models_to_save = []
    
    if save_best_only and best_metric:
        # Only save models in best_metric dict
        for model in nf_instance.models:
            model_name = model.__class__.__name__
            if model_name in best_metric:
                models_to_save.append(model)
                logging.info(f"Will save best model {model_name} (metric: {best_metric[model_name]})")
    elif model_names:
        # Save specific models by name
        for model in nf_instance.models:
            if model.__class__.__name__ in model_names:
                models_to_save.append(model)
    else:
        # Save all models
        models_to_save = nf_instance.models
    
    saved_paths = {}
    
    # Save each model using NF's native save
    for model in models_to_save:
        model_class = model.__class__.__name__
        
        # Extract loss type if available
        loss_type = "unknown"
        if hasattr(model, 'loss'):
            if hasattr(model.loss, '__class__'):
                loss_type = model.loss.__class__.__name__.replace('Loss', '')
            elif isinstance(model.loss, str):
                loss_type = model.loss
        
        # Build metadata-rich filename
        # Format: {ModelClass}_{LossType}_{timestamp}.pkl
        filename = f"{model_class}_{loss_type}_{timestamp}.pkl"
        model_path = output_path / filename
        
        try:
            # Use NF's native save method
            # Note: NF saves the entire NF instance, not individual models
            # So we need to save the entire instance with proper naming
            temp_nf = nf_instance.__class__(models=[model], freq='15min')
            temp_nf.models = [model]  # Ensure only this model is saved
            
            # Save using NF's native method
            temp_nf.save(path=str(model_path.parent), 
                        model_index=None,  # Save all models in the instance
                        overwrite=True,
                        save_dataset=False)  # Don't save training data
            
            # The above saves to a standard NF format
            # For individual model files, we need a different approach
            # Let's save the model state directly
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            saved_paths[model_class] = str(model_path)
            logging.info(f"Saved model {model_class} to {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to save model {model_class}: {e}")
            # Continue with other models even if one fails
            continue
    
    # Also save the complete NF instance for convenience
    if len(models_to_save) > 1:
        try:
            full_path = output_path / f"nf_ensemble_{timestamp}.pkl"
            nf_instance.save(path=str(output_path),
                           model_index=None,
                           overwrite=True,
                           save_dataset=False)
            saved_paths['ensemble'] = str(full_path)
            logging.info(f"Saved complete NF ensemble to {full_path}")
        except Exception as e:
            logging.error(f"Failed to save NF ensemble: {e}")
    
    return saved_paths


def load_nf_models(horizon: int,
                   model_name: Optional[str] = None,
                   timestamp: Optional[str] = None,
                   output_dir: Optional[str] = None) -> Any:
    """
    Load saved NeuralForecast models from disk.
    
    Args:
        horizon: Forecast horizon (4, 8, 16, or 32)
        model_name: Specific model to load (e.g., 'NHITS'), or None for ensemble
        timestamp: Specific timestamp to load, or None for latest
        output_dir: Base directory (default: experiments/h{horizon}/models)
        
    Returns:
        Loaded model or NeuralForecast instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If loading fails
    """
    import glob
    import pickle
    
    # Validate horizon
    if horizon not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of [4, 8, 16, 32]")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"experiments/h{horizon}/models"
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Models directory {output_path} does not exist")
    
    if model_name:
        # Load specific model
        if timestamp:
            # Load specific timestamp
            pattern = str(output_path / f"{model_name}_*_{timestamp}.pkl")
        else:
            # Load latest
            pattern = str(output_path / f"{model_name}_*.pkl")
        
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
        
        # Get most recent file
        model_file = sorted(files)[-1]
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Loaded model from {model_file}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {model_file}: {e}")
            raise ValueError(f"Failed to load model: {e}")
    else:
        # Load ensemble or use NF's native load
        try:
            # Try NF's native load first
            from neuralforecast import NeuralForecast
            nf = NeuralForecast.load(path=str(output_path))
            logging.info(f"Loaded NF ensemble from {output_path}")
            return nf
        except Exception as e:
            logging.warning(f"Failed to load with NF.load, trying pickle: {e}")
            
            # Fallback to pickle
            if timestamp:
                ensemble_file = output_path / f"nf_ensemble_{timestamp}.pkl"
            else:
                pattern = str(output_path / "nf_ensemble_*.pkl")
                files = glob.glob(pattern)
                if not files:
                    raise FileNotFoundError("No ensemble files found")
                ensemble_file = sorted(files)[-1]
            
            with open(ensemble_file, 'rb') as f:
                nf = pickle.load(f)
            logging.info(f"Loaded ensemble from {ensemble_file}")
            return nf


def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a saved model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model metadata:
        - model_class: Model class name
        - loss_type: Loss function type
        - timestamp: Save timestamp
        - file_size: File size in bytes
        - horizon: Inferred horizon from path
    """
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    metadata = {
        'file_path': str(path),
        'file_size': path.stat().st_size,
    }
    
    # Parse filename for metadata
    # Format: {ModelClass}_{LossType}_{timestamp}.pkl
    stem = path.stem
    parts = stem.split('_')
    
    if len(parts) >= 3:
        metadata['model_class'] = parts[0]
        metadata['loss_type'] = parts[1]
        metadata['timestamp'] = '_'.join(parts[2:])  # Handle timestamps with underscores
    
    # Infer horizon from path
    if 'experiments/h' in str(path):
        import re
        match = re.search(r'experiments/h(\d+)', str(path))
        if match:
            metadata['horizon'] = int(match.group(1))
    
    return metadata


def list_saved_artifacts(horizon: int,
                         output_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all saved artifacts for a given horizon.
    
    Args:
        horizon: Forecast horizon (4, 8, 16, or 32)
        output_dir: Base directory (default: experiments/h{horizon})
        
    Returns:
        Dictionary with lists of artifacts:
        - 'cv_results': List of CV result files with timestamps
        - 'models': List of saved model files
        - 'metrics': Path to metrics.json if exists
        - 'leaderboard': Path to leaderboard.csv if exists
        
    Example:
        >>> artifacts = list_saved_artifacts(horizon=4)
        >>> print(f"Found {len(artifacts['cv_results'])} CV result files")
        >>> print(f"Found {len(artifacts['models'])} saved models")
    """
    import glob
    
    # Validate horizon
    if horizon not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of [4, 8, 16, 32]")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"experiments/h{horizon}"
    
    output_path = Path(output_dir)
    
    artifacts = {
        'cv_results': [],
        'models': [],
        'metrics': None,
        'leaderboard': None
    }
    
    if not output_path.exists():
        logging.warning(f"Output directory {output_path} does not exist")
        return artifacts
    
    # List CV result files
    cv_pattern = str(output_path / "cv_results_*.parquet")
    cv_files = glob.glob(cv_pattern)
    artifacts['cv_results'] = sorted(cv_files, reverse=True)  # Most recent first
    
    # List model files
    models_dir = output_path / "models"
    if models_dir.exists():
        model_pattern = str(models_dir / "*.pkl")
        model_files = glob.glob(model_pattern)
        artifacts['models'] = sorted(model_files, reverse=True)
    
    # Check for metrics.json
    metrics_path = output_path / "metrics.json"
    if metrics_path.exists():
        artifacts['metrics'] = str(metrics_path)
    
    # Check for leaderboard.csv
    leaderboard_path = output_path / "leaderboard.csv"
    if leaderboard_path.exists():
        artifacts['leaderboard'] = str(leaderboard_path)
    
    return artifacts


def cleanup_old_artifacts(horizon: int,
                         keep_latest: int = 5,
                         output_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Clean up old CV result files, keeping only the most recent ones.
    
    Args:
        horizon: Forecast horizon (4, 8, 16, or 32)
        keep_latest: Number of latest CV result files to keep (default: 5)
        output_dir: Base directory (default: experiments/h{horizon})
        
    Returns:
        Dictionary with cleanup statistics:
        - 'cv_removed': Number of CV files removed
        - 'models_removed': Number of model files removed
        
    Example:
        >>> stats = cleanup_old_artifacts(horizon=4, keep_latest=3)
        >>> print(f"Removed {stats['cv_removed']} old CV result files")
    """
    import glob
    
    # Validate horizon
    if horizon not in [4, 8, 16, 32]:
        raise ValueError(f"Invalid horizon {horizon}. Must be one of [4, 8, 16, 32]")
    
    # Set default output directory
    if output_dir is None:
        output_dir = f"experiments/h{horizon}"
    
    output_path = Path(output_dir)
    
    stats = {
        'cv_removed': 0,
        'models_removed': 0
    }
    
    if not output_path.exists():
        return stats
    
    # Clean up old CV result files
    cv_pattern = str(output_path / "cv_results_*.parquet")
    cv_files = sorted(glob.glob(cv_pattern), reverse=True)
    
    if len(cv_files) > keep_latest:
        files_to_remove = cv_files[keep_latest:]
        for file_path in files_to_remove:
            try:
                Path(file_path).unlink()
                stats['cv_removed'] += 1
                logging.info(f"Removed old CV file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to remove {file_path}: {e}")
    
    # Note: We don't automatically clean up models as they might be important
    # Add model cleanup logic here if needed
    
    return stats