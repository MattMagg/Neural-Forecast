# %% [markdown]
# # Neural-Forecast Training Pipeline
# 
# This notebook implements the complete data assembly and training path for the BTC forecasting system:
# 1. Load raw OHLCV → aggregate → regularize → make canonical → validate
# 2. Integrate features with leakage prevention
# 3. Instantiate NeuralForecast models from configuration
# 4. Run cross-validation with progress logging
# 5. Generate comprehensive metrics and save artifacts

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime
import time
from typing import Dict, Any, List, Optional, Tuple

print("Loading imports...")

# Import data processing functions
from utils.io import (
    load_raw_1min_data,
    aggregate_1min_to_15min,
    regularize_to_grid_utc,
    make_nf_canonical,
    drop_train_nans_and_winsorize,
    save_parquet,
    timestamped_path
)

# Import validation functions
from utils.validate import (
    assert_regular_grid,
    assert_utc_eob,
    assert_shifted,
    assert_no_forward_fill_y
)

# Import CV runner functions for cross-validation
from cv.runner import run_cv, summarize_cv, save_cv_results, generate_cv_summary_report

# Import model factory for NF model instantiation
from nf_models.factory import instantiate_models

# Import NeuralForecast
from neuralforecast import NeuralForecast

print("Imports complete")

# %% [markdown]
# ## Configuration

# %%
print("Setting up configuration...")

# Path configuration
DATA_PATH = "data/raw/btcusd_1-min_data.csv"
PROCESSED_PATH = "data/processed"
EXPERIMENT_PATH = lambda h: f"experiments/h{h}"
REPORTS_PATH = "reports"

# Configuration - Default values (was argparse)
RAW_DATA = "data/raw/btcusd_1-min_data.csv"
CONFIG = "experiments/h4.yaml"
OUTPUT_DIR = "data/processed"
SAVE_PROCESSED = True
SKIP_CV = False
USE_CONFORMAL = False
SAVE_MODELS = True

# Configure logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

print(f"Config loaded - using {CONFIG}")

# %% [markdown]
# ## Helper Functions

# %%
def load_and_process_data(raw_data_path: str = "data/raw/btcusd_1-min_data.csv") -> pd.DataFrame:
    """
    Execute the complete data assembly path as specified in requirements.
    
    Assembly sequence:
    1. Load raw 1-minute OHLCV data
    2. Aggregate to 15-minute bars with proper UTC EOB alignment
    3. Regularize to complete UTC grid
    4. Create NeuralForecast canonical format with log returns
    5. Apply winsorization for training stability
    6. Validate all quality gates
    
    Args:
        raw_data_path: Path to raw 1-minute CSV data
        
    Returns:
        Processed DataFrame ready for NeuralForecast training
        
    Raises:
        AssertionError: If any validation gate fails
        ValueError: If data processing fails
    """
    print("🔄 Starting data assembly path...")
    
    # Step 1: Load raw 1-minute OHLCV data
    print("📥 Loading raw 1-minute data...")
    df_1min = load_raw_1min_data(raw_data_path)
    print(f"   Loaded {len(df_1min):,} 1-minute bars")
    print(f"   Shape: {df_1min.shape}")
    
    # Step 2: Aggregate to 15-minute bars
    print("📊 Aggregating 1-minute to 15-minute bars...")
    df_15min = aggregate_1min_to_15min(df_1min)
    print(f"   Aggregated to {len(df_15min):,} 15-minute bars")
    print(f"   Shape: {df_15min.shape}")
    
    # Step 3: Regularize to complete UTC grid
    print("🕐 Regularizing to UTC grid...")
    df_regularized = regularize_to_grid_utc(df_15min, "15min")
    print(f"   Regularized grid: {len(df_regularized):,} bars")
    print(f"   Shape: {df_regularized.shape}")
    
    # Step 4: Create NeuralForecast canonical format
    print("🎯 Creating NF canonical format...")
    df_canonical = make_nf_canonical(df_regularized, unique_id="BTC-USD")
    print(f"   Canonical format: {len(df_canonical):,} rows")
    print(f"   Columns: {list(df_canonical.columns)}")
    print(f"   Shape: {df_canonical.shape}")
    
    # Step 5: Apply winsorization
    print("✂️  Applying winsorization...")
    df_winsorized = drop_train_nans_and_winsorize(df_canonical, lower_q=0.001, upper_q=0.999)
    
    # Count non-NaN values
    y_count = df_winsorized['y'].notna().sum()
    y_train_count = df_winsorized['y_train'].notna().sum()
    print(f"   y values: {y_count:,} non-NaN")
    print(f"   y_train values: {y_train_count:,} non-NaN")
    print(f"   Final shape: {df_winsorized.shape}")
    
    # Step 6: Validate all quality gates
    print("✅ Running validation gates...")
    
    # Grid validation
    print("   Checking regular grid...")
    assert_regular_grid(df_winsorized, "15min")
    print("   ✓ Regular grid validation passed")
    
    # UTC EOB validation
    print("   Checking UTC EOB...")
    assert_utc_eob(df_winsorized, "15min")
    print("   ✓ UTC EOB validation passed")
    
    # Forward-fill validation
    print("   Checking forward-fill...")
    assert_no_forward_fill_y(df_winsorized)
    print("   ✓ No forward-fill validation passed")
    
    # Note: assert_shifted will be called later when exogenous features are added
    print("   ⏳ Leakage validation (assert_shifted) will be applied after feature integration")
    
    print("🎉 Data assembly path completed successfully!")
    print(f"   Final output shape: {df_winsorized.shape}")
    return df_winsorized

# %%
def integrate_features(nf_base: pd.DataFrame) -> tuple:
    """
    Integrate feature engineering pipeline exactly as specified in Section 3.4.
    
    This implements the exact sequence from docs/forecasting_sf_plan.md:
    - Build base indicators
    - Apply multi-timeframe features
    - Merge and postprocess with shift(1)
    - Select features with hard cap
    - Validate leakage discipline
    
    Args:
        nf_base: Canonical NF DataFrame from make_nf_canonical
        
    Returns:
        Tuple of (nf_df, hist_cols, futr_cols, stat_cols)
    """
    print("🔗 Integrating feature engineering pipeline...")
    print(f"   Input shape: {nf_base.shape}")
    
    # Import feature engineering functions
    print("   Importing feature modules...")
    from features.registry import REGISTRY
    from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features
    print(f"   Registry has {len(REGISTRY)} indicators")
    
    # FIX 1: Ensure OHLCV columns exist and are properly typed
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in nf_base.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # FIX 2: Create a clean copy with only OHLCV data, ensure float64 dtype
    ohlcv_df = nf_base[["ds"] + required_cols].copy()
    for col in required_cols:
        ohlcv_df[col] = pd.to_numeric(ohlcv_df[col], errors='coerce').astype('float64')
    
    # FIX 3: Handle NaN values - forward fill then drop remaining NaNs
    print("   Handling NaN values in OHLCV data...")
    nan_counts = ohlcv_df[required_cols].isna().sum()
    if nan_counts.any():
        print(f"   Found NaNs: {nan_counts[nan_counts > 0].to_dict()}")
        # Forward fill for continuity in technical indicators
        ohlcv_df[required_cols] = ohlcv_df[required_cols].fillna(method='ffill')
        # Drop any remaining NaN rows at the beginning
        ohlcv_df = ohlcv_df.dropna(subset=required_cols)
        print(f"   After cleaning: {ohlcv_df.shape}")
    
    # Exact sequence from Section 3.4 of docs/forecasting_sf_plan.md
    print("   📊 Building base indicators...")
    # FIX 4: Pass the cleaned OHLCV data without redundant rename
    base_feats = build_indicators(ohlcv_df)
    print(f"   Base features shape: {base_feats.shape}")
    
    print("   🕐 Computing multi-timeframe features...")
    mtf_feats = apply_mtf(ohlcv_df)
    print(f"   MTF features shape: {mtf_feats.shape}")
    
    print("   🔄 Merging feature sets...")
    exo_raw = base_feats.merge(mtf_feats, on="ds", how="left")
    print(f"   Merged shape: {exo_raw.shape}")
    
    print("   ⚡ Applying shift(1) and pruning...")
    exo = postprocess_shift_and_prune(exo_raw, rules={})
    print(f"   After shift shape: {exo.shape}")
    
    print("   🎯 Selecting features with hard cap...")
    hist_cols, futr_cols, stat_cols = select_features(exo, policy={})
    print(f"   Selected: H={len(hist_cols)}, F={len(futr_cols)}, S={len(stat_cols)}")
    
    print("   🔀 Merging features with canonical frame...")
    nf_df = nf_base.merge(exo, on="ds", how="left")
    print(f"   Final merged shape: {nf_df.shape}")
    
    # Validate leakage discipline before any NF call (Section 2.3)
    print("   ✅ Validating leakage prevention with assert_shifted...")
    assert_shifted(nf_df, hist_cols)
    print("   ✓ Leakage validation passed!")
    
    # Print feature summary
    print(f"\n📊 Feature Summary:")
    print(f"   Historical features: {len(hist_cols)}")
    print(f"   Future features: {len(futr_cols)}")
    print(f"   Static features: {len(stat_cols)}")
    print(f"   Total features: {len(hist_cols) + len(futr_cols) + len(stat_cols)} (cap: 256)")
    print(f"   Final output shape: {nf_df.shape}")
    
    return nf_df, hist_cols, futr_cols, stat_cols

# %%
def save_processed_data(df: pd.DataFrame, output_dir: str = "data/processed") -> str:
    """
    Save processed data with timestamped filename.
    
    Args:
        df: Processed DataFrame to save
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    print(f"💾 Saving processed data...")
    print(f"   Data shape: {df.shape}")
    output_path = timestamped_path(output_dir, "btc_canonical", "parquet")
    save_parquet(df, output_path)
    print(f"   Saved to: {output_path}")
    return output_path

# %%
def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to experiment YAML config (e.g., experiments/h4.yaml)
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    print(f"Loading config from {config_path}")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading experiment configuration from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Config has {len(cfg.get('models', []))} models")
    
    # Validate required CV parameters
    required_keys = ['h', 'n_windows', 'step_size', 'val_size', 'models']
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        raise ValueError(f"Configuration missing required keys: {missing_keys}")
    
    # Validate CV parameter relationships
    if cfg['step_size'] != cfg['h']:
        logger.warning(f"step_size ({cfg['step_size']}) != h ({cfg['h']}). Setting step_size=h to prevent overlap.")
        cfg['step_size'] = cfg['h']
    
    expected_val_size = 4 * cfg['h']
    if cfg['val_size'] != expected_val_size:
        logger.warning(f"val_size ({cfg['val_size']}) != 4*h ({expected_val_size}). Setting val_size=4*h.")
        cfg['val_size'] = expected_val_size
    
    logger.info(f"Configuration loaded: h={cfg['h']}, n_windows={cfg['n_windows']}, "
                f"step_size={cfg['step_size']}, val_size={cfg['val_size']}")
    print(f"h={cfg['h']}, windows={cfg['n_windows']}")
    return cfg

# %%
def train_and_evaluate(
    nf_df: pd.DataFrame,
    hist_cols: List[str],
    futr_cols: List[str],
    stat_cols: List[str],
    cfg: Dict[str, Any],
    output_dir: Path,
    use_conformal: bool = False,
    save_models: bool = True
) -> Dict[str, Any]:
    """Train models and run cross-validation with comprehensive evaluation.
    
    This function implements the exact integration pattern from docs/forecasting_sf_plan.md
    lines 1770-1801, with additional error handling and progress logging.
    
    Args:
        nf_df: Canonical NF DataFrame with features
        hist_cols: List of historical feature columns
        futr_cols: List of future feature columns
        stat_cols: List of static feature columns
        cfg: Experiment configuration
        output_dir: Directory for saving outputs
        use_conformal: Whether to use conformal prediction
        save_models: Whether to save the best models
        
    Returns:
        Dictionary with training results including CV metrics and saved paths
    """
    print("Starting training...")
    results = {}
    start_time = time.time()
    
    # Update configuration with feature lists
    cfg['hist_exog_list'] = hist_cols
    cfg['futr_exog_list'] = futr_cols
    cfg['stat_exog_list'] = stat_cols
    
    # Step 1: Instantiate NeuralForecast models from configuration
    logger.info("="*70)
    logger.info("STEP 1: Model Instantiation")
    logger.info("="*70)
    print("Creating models...")
    
    try:
        models = instantiate_models(cfg, verbose=True)
        logger.info(f"Successfully instantiated {len(models)} models")
        print(f"Created {len(models)} models")
        
        # Create NeuralForecast instance
        nf = NeuralForecast(
            models=models,
            freq=cfg.get('freq', '15min')
        )
        
    except Exception as e:
        logger.error(f"Model instantiation failed: {str(e)}")
        raise RuntimeError(f"Failed to instantiate models: {str(e)}")
    
    # Step 2: Fit models once to store dataset (following spec lines 1777-1778)
    logger.info("="*70)
    logger.info("STEP 2: Initial Model Fitting")
    logger.info("="*70)
    print("Fitting models...")
    
    try:
        fit_start = time.time()
        logger.info(f"Fitting {len(models)} models with val_size={cfg['val_size']}...")
        
        nf.fit(df=nf_df, val_size=cfg['val_size'])
        
        fit_time = time.time() - fit_start
        logger.info(f"✅ Model fitting completed in {fit_time:.1f} seconds")
        print(f"Fitted in {fit_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Model fitting failed: {str(e)}")
        
        # Provide recovery suggestions
        if "memory" in str(e).lower() or "oom" in str(e).lower():
            logger.error("Suggestion: Reduce batch_size or input_size in configuration")
        elif "cuda" in str(e).lower():
            logger.error("Suggestion: Check GPU availability or use CPU by setting accelerator='cpu'")
        
        raise RuntimeError(f"Model fitting failed: {str(e)}")
    
    # Step 3: Optional insample predictions for diagnostics (spec lines 1780-1784)
    logger.info("="*70)
    logger.info("STEP 3: Insample Predictions (Optional)")
    logger.info("="*70)
    print("Getting insample predictions...")
    
    try:
        logger.info("Generating insample predictions for diagnostics...")
        ins = nf.predict_insample(step_size=1, level=[10,20,30,40,50,60,70,80,90])
        
        # Merge y if needed
        if "y" not in ins.columns:
            ins = ins.merge(nf_df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")
        
        logger.info(f"Generated {len(ins)} insample predictions")
        print(f"Got {len(ins)} predictions")
        results['insample_predictions'] = ins
        
    except Exception as e:
        logger.warning(f"Insample prediction failed (non-critical): {str(e)}")
        results['insample_predictions'] = None
    
    # Step 4: Run NF-native temporal cross-validation (spec lines 1789-1790)
    logger.info("="*70)
    logger.info("STEP 4: Cross-Validation")
    logger.info("="*70)
    print(f"Running CV with {cfg['n_windows']} windows...")
    
    try:
        cv_start = time.time()
        logger.info(f"Starting {cfg['n_windows']}-window cross-validation...")
        
        # Run CV with progress logging
        cv_df = run_cv_with_progress(
            nf=nf,
            df=nf_df,
            cfg=cfg,
            use_conformal=use_conformal
        )
        
        cv_time = time.time() - cv_start
        logger.info(f"✅ Cross-validation completed in {cv_time:.1f} seconds")
        print(f"CV done in {cv_time:.1f}s")
        
        # Merge y for metrics if needed (spec lines 1792-1794)
        if "y" not in cv_df.columns:
            cv_df = cv_df.merge(nf_df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")
        
        results['cv_raw'] = cv_df
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {str(e)}")
        
        # Save partial results if available
        if 'cv_df' in locals() and cv_df is not None and len(cv_df) > 0:
            logger.info("Attempting to save partial CV results...")
            partial_path = output_dir / f"cv_partial_h{cfg['h']}.parquet"
            cv_df.to_parquet(partial_path)
            logger.info(f"Partial results saved to {partial_path}")
        
        raise RuntimeError(f"Cross-validation failed: {str(e)}")
    
    # Step 5: Summarize CV results (spec line 1796)
    logger.info("="*70)
    logger.info("STEP 5: Results Summarization")
    logger.info("="*70)
    print("Summarizing results...")
    
    try:
        # Get model names for summarization
        model_names = [model.__class__.__name__ for model in models]
        
        # Create model metadata for enhanced leaderboard
        model_metadata = {}
        for model in models:
            model_name = model.__class__.__name__
            model_metadata[model_name] = {
                'loss_type': _get_loss_type(model),
                'alias': getattr(model, 'alias', model_name)
            }
        
        # Run comprehensive summarization
        summary_results = summarize_cv(
            cv_df=cv_df,
            models=model_names,
            h=cfg['h'],
            output_dir=output_dir,
            generate_plots=True,
            model_metadata=model_metadata,
            nf_instance=nf if save_models else None,
            save_best_models=save_models
        )
        
        results.update(summary_results)
        print("Results summarized")
        
        # Log leaderboard
        if 'leaderboard' in summary_results and not summary_results['leaderboard'].empty:
            logger.info("\n" + "="*70)
            logger.info("MODEL LEADERBOARD")
            logger.info("="*70)
            for idx, row in summary_results['leaderboard'].head(5).iterrows():
                logger.info(f"Rank {row['rank']}: {row['model']} - sCRPS: {row['sCRPS_mean']:.4f}")
                if idx == 0:
                    print(f"Best: {row['model']} - sCRPS: {row['sCRPS_mean']:.4f}")
        
    except Exception as e:
        logger.error(f"Results summarization failed: {str(e)}")
        raise RuntimeError(f"Failed to summarize results: {str(e)}")
    
    # Step 6: Persist artifacts (spec lines 1798-1800)
    logger.info("="*70)
    logger.info("STEP 6: Saving Artifacts")
    logger.info("="*70)
    print("Saving artifacts...")
    
    try:
        # Create experiment directory
        exp_dir = output_dir / f"experiments/h{cfg['h']}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CV raw results
        cv_path = exp_dir / "cv_raw.parquet"
        cv_df.to_parquet(cv_path)
        logger.info(f"Saved CV results to {cv_path}")
        
        # Save leaderboard
        if 'leaderboard' in results and not results['leaderboard'].empty:
            leaderboard_path = exp_dir / "leaderboard.parquet"
            results['leaderboard'].to_parquet(leaderboard_path)
            logger.info(f"Saved leaderboard to {leaderboard_path}")
        
        # Save all other results
        save_cv_results(results, exp_dir, cfg['h'])
        
        # Generate comprehensive markdown report
        report_path = exp_dir / f"cv_summary_h{cfg['h']}.md"
        generate_cv_summary_report(
            results=results,
            output_path=report_path,
            horizon=cfg['h'],
            use_conformal=use_conformal
        )
        logger.info(f"Generated summary report at {report_path}")
        
        # Save configuration used
        config_path = exp_dir / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
        print("Artifacts saved")
        
    except Exception as e:
        logger.error(f"Failed to save artifacts: {str(e)}")
        # Non-critical, continue
    
    # Calculate total training time
    total_time = time.time() - start_time
    results['training_time_seconds'] = total_time
    
    logger.info("="*70)
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info("="*70)
    print(f"Training done in {total_time:.1f}s")
    
    return results

# %%
def run_cv_with_progress(
    nf: NeuralForecast,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    use_conformal: bool = False,
    max_retries: int = 3
) -> pd.DataFrame:
    """Run CV with progress logging and retry logic for transient failures.
    
    Args:
        nf: Fitted NeuralForecast instance
        df: Canonical DataFrame
        cfg: Configuration
        use_conformal: Whether to use conformal prediction
        max_retries: Maximum retry attempts for transient failures
        
    Returns:
        CV results DataFrame
    """
    print(f"Running CV with max {max_retries} retries...")
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            attempt += 1
            if attempt > 1:
                logger.info(f"Retry attempt {attempt}/{max_retries}...")
                print(f"   Retry {attempt}/{max_retries}")
            
            print(f"   Attempt {attempt}: Running CV...")
            # Run CV with the runner module
            cv_df = run_cv(nf, df, cfg, use_conformal)
            
            # Success - return results
            print(f"   CV succeeded on attempt {attempt}")
            return cv_df
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            print(f"   Attempt {attempt} failed")
            
            # Check if error is retryable
            retryable_errors = ['timeout', 'connection', 'temporary', 'transient']
            is_retryable = any(err in error_str for err in retryable_errors)
            
            if is_retryable and attempt < max_retries:
                wait_time = attempt * 5  # Exponential backoff
                logger.warning(f"Transient error encountered, retrying in {wait_time} seconds...")
                print(f"   Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                # Non-retryable error or max retries reached
                break
    
    # All retries exhausted
    print(f"   CV failed after {max_retries} attempts")
    raise RuntimeError(f"CV failed after {max_retries} attempts. Last error: {str(last_error)}")

# %%
def _get_loss_type(model) -> str:
    """Extract loss type from model configuration.
    
    Args:
        model: NeuralForecast model instance
        
    Returns:
        String describing the loss type
    """
    print(f"   Getting loss type for {model.__class__.__name__}")
    if hasattr(model, 'loss'):
        loss = model.loss
        if hasattr(loss, '__class__'):
            loss_name = loss.__class__.__name__
            if 'Distribution' in loss_name:
                return 'StudentT'
            elif 'MQ' in loss_name:
                return 'MQLoss'
            elif 'IQ' in loss_name:
                return 'IQLoss'
    return 'unknown'

# %% [markdown]
# ## Main Execution

# %%
print("Starting main execution...")

# Load experiment configuration
cfg = load_experiment_config(CONFIG)

# Execute complete data assembly path
logger.info("Starting data processing pipeline...")
print("Processing data...")
df_processed = load_and_process_data(RAW_DATA)
print(f"Processed {len(df_processed)} rows")

# Integrate feature engineering pipeline
print("Integrating features...")
nf_df, hist_cols, futr_cols, stat_cols = integrate_features(df_processed)
print(f"Data shape: {nf_df.shape}")

# Save processed data if requested
if SAVE_PROCESSED:
    saved_path = save_processed_data(nf_df, OUTPUT_DIR)
    logger.info(f"✅ Data saved to: {saved_path}")

# Show summary statistics
logger.info("\n" + "="*70)
logger.info("DATA SUMMARY")
logger.info("="*70)
logger.info(f"Total rows: {len(nf_df):,}")
logger.info(f"Date range: {nf_df['ds'].min()} to {nf_df['ds'].max()}")
logger.info(f"Non-NaN y values: {nf_df['y'].notna().sum():,}")
logger.info(f"Non-NaN y_train values: {nf_df['y_train'].notna().sum():,}")
logger.info(f"Total columns: {len(nf_df.columns)}")
logger.info(f"Historical features: {len(hist_cols)}")
logger.info(f"Future features: {len(futr_cols)}")
logger.info(f"Static features: {len(stat_cols)}")

print(f"Total features: {len(hist_cols) + len(futr_cols) + len(stat_cols)}")

# Skip CV if requested (for testing data processing only)
if SKIP_CV:
    logger.info("Skipping cross-validation as requested.")
    print("CV skipped - data processing only")
else:
    # Run training and cross-validation
    print("Running training pipeline...")
    output_dir = Path(OUTPUT_DIR)
    results = train_and_evaluate(
        nf_df=nf_df,
        hist_cols=hist_cols,
        futr_cols=futr_cols,
        stat_cols=stat_cols,
        cfg=cfg,
        output_dir=output_dir,
        use_conformal=USE_CONFORMAL,
        save_models=SAVE_MODELS
    )
    
    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    
    if 'leaderboard' in results and not results['leaderboard'].empty:
        best_model = results['leaderboard'].iloc[0]
        logger.info(f"Best Model: {best_model['model']}")
        logger.info(f"Best sCRPS: {best_model['sCRPS_mean']:.4f}")
    
    if 'saved_models' in results:
        logger.info(f"\nSaved Models:")
        for key, path in results['saved_models'].items():
            logger.info(f"  {key}: {path}")
    
    logger.info(f"\nTotal Training Time: {results.get('training_time_seconds', 0):.1f} seconds")
    logger.info("\n✅ Training pipeline completed successfully!")
    print("Pipeline complete!")

# %%



