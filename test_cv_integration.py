#!/usr/bin/env python3
"""
Test script for CV integration in the training pipeline.

This script demonstrates the complete CV integration by:
1. Creating a small synthetic dataset for quick testing
2. Running the full training pipeline with CV
3. Verifying all outputs are generated correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(n_periods: int = 2000, freq: str = "15min") -> pd.DataFrame:
    """Create a small synthetic BTC-like dataset for testing.
    
    Args:
        n_periods: Number of time periods to generate
        freq: Frequency of the time series
        
    Returns:
        DataFrame in canonical NF format with synthetic OHLCV data
    """
    logger.info(f"Creating synthetic dataset with {n_periods} periods...")
    
    # Generate timestamps
    end_time = pd.Timestamp.now(tz='UTC').floor('15min')
    timestamps = pd.date_range(end=end_time, periods=n_periods, freq=freq)
    
    # Generate synthetic price data with realistic BTC-like behavior
    np.random.seed(42)
    
    # Base price with trend and noise
    trend = np.linspace(30000, 35000, n_periods)
    seasonal = 2000 * np.sin(np.linspace(0, 8*np.pi, n_periods))
    noise = np.random.normal(0, 500, n_periods)
    close_prices = trend + seasonal + noise
    
    # Ensure positive prices
    close_prices = np.maximum(close_prices, 1000)
    
    # Generate OHLC from close
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_periods)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_periods)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Generate volume
    base_volume = 100 + np.random.exponential(50, n_periods)
    volume = base_volume * (1 + 0.5 * np.random.randn(n_periods))
    
    # Create DataFrame
    df = pd.DataFrame({
        'unique_id': 'BTC-USD',
        'ds': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # Calculate log returns as target
    df['y'] = np.log(df['close'] / df['close'].shift(1))
    df['y_train'] = df['y'].copy()
    
    # Handle first row
    df.loc[0, 'y'] = 0
    df.loc[0, 'y_train'] = 0
    
    # Add some simple features for testing
    # These would normally come from the feature engineering pipeline
    df['rsi_14'] = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, n_periods))
    df['ma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['volume_ma_10'] = df['volume'].rolling(10, min_periods=1).mean()
    
    # Shift historical features to prevent leakage
    hist_cols = ['rsi_14', 'ma_20', 'volume_ma_10']
    for col in hist_cols:
        df[col] = df[col].shift(1)
    
    # Fill first row NaNs
    df[hist_cols] = df[hist_cols].fillna(method='bfill')
    
    logger.info(f"Created synthetic dataset with shape {df.shape}")
    return df, hist_cols

def create_test_config(output_dir: Path, h: int = 4) -> Path:
    """Create a minimal test configuration for CV.
    
    Args:
        output_dir: Directory to save the config
        h: Forecast horizon
        
    Returns:
        Path to the created config file
    """
    logger.info(f"Creating test configuration for h={h}...")
    
    config = {
        'seed': 42,
        'freq': '15min',
        'h': h,
        
        # CV settings - smaller for testing
        'n_windows': 3,  # Reduced from 6 for faster testing
        'step_size': h,
        'val_size': 4 * h,
        'refit': True,
        
        # Feature lists (will be populated at runtime)
        'hist_exog_list': [],
        'futr_exog_list': [],
        'stat_exog_list': [],
        
        # Simplified model portfolio for testing
        'models': [
            {
                'NHITS': {
                    'alias': 'NHITS_test',
                    'input_size': 96,  # Smaller for testing
                    'h': h,
                    'loss': {'kind': 'studentt'},
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'max_steps': 100,  # Much fewer steps for testing
                    'val_check_steps': 10,
                    'early_stop_patience_steps': 20,
                }
            },
            {
                'TIDE': {
                    'alias': 'TiDE_test',
                    'input_size': 96,
                    'h': h,
                    'loss': {'kind': 'mqloss', 'level': [80, 90, 95]},
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'hidden_size': 128,  # Smaller for testing
                    'num_encoder_layers': 1,
                    'num_decoder_layers': 1,
                    'max_steps': 100,
                    'val_check_steps': 10,
                    'early_stop_patience_steps': 20,
                }
            }
        ]
    }
    
    # Save config
    config_path = output_dir / f"test_config_h{h}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Test configuration saved to {config_path}")
    return config_path

def test_cv_integration():
    """Test the complete CV integration in the training pipeline."""
    
    logger.info("="*70)
    logger.info("TESTING CV INTEGRATION")
    logger.info("="*70)
    
    # Setup test directory
    test_dir = Path("test_cv_integration")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Create synthetic data
        logger.info("\nStep 1: Creating synthetic data...")
        df, hist_cols = create_synthetic_data(n_periods=500)  # Small dataset
        
        # Save synthetic data
        data_path = test_dir / "synthetic_data.parquet"
        df.to_parquet(data_path)
        logger.info(f"Saved synthetic data to {data_path}")
        
        # Step 2: Create test configuration
        logger.info("\nStep 2: Creating test configuration...")
        config_path = create_test_config(test_dir, h=4)
        
        # Step 3: Load modules and run CV
        logger.info("\nStep 3: Running CV integration test...")
        
        # Import the necessary modules
        from cv.runner import run_cv, summarize_cv
        from nf_models.factory import instantiate_models
        from neuralforecast import NeuralForecast
        
        # Load configuration
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Update config with feature lists
        cfg['hist_exog_list'] = hist_cols
        cfg['futr_exog_list'] = []
        cfg['stat_exog_list'] = []
        
        # Instantiate models
        logger.info("Instantiating models...")
        models = instantiate_models(cfg, verbose=False)
        logger.info(f"Created {len(models)} models: {[m.__class__.__name__ for m in models]}")
        
        # Create NeuralForecast instance
        nf = NeuralForecast(
            models=models,
            freq=cfg['freq']
        )
        
        # Fit models
        logger.info("Fitting models...")
        nf.fit(df=df, val_size=cfg['val_size'])
        logger.info("Models fitted successfully")
        
        # Run cross-validation
        logger.info("Running cross-validation...")
        cv_df = run_cv(nf, df, cfg, use_conformal=False)
        logger.info(f"CV completed: {len(cv_df)} predictions generated")
        
        # Merge y for metrics
        if "y" not in cv_df.columns:
            cv_df = cv_df.merge(df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")
        
        # Summarize results
        logger.info("Summarizing CV results...")
        model_names = [m.__class__.__name__ for m in models]
        results = summarize_cv(
            cv_df=cv_df,
            models=model_names,
            h=cfg['h'],
            output_dir=test_dir,
            generate_plots=False  # Skip plots for testing
        )
        
        # Step 4: Verify outputs
        logger.info("\nStep 4: Verifying outputs...")
        
        # Check that key results exist
        assert 'metrics' in results, "Missing metrics in results"
        assert 'leaderboard' in results, "Missing leaderboard in results"
        assert 'window_metrics' in results, "Missing window_metrics in results"
        
        # Check leaderboard structure
        leaderboard = results['leaderboard']
        assert not leaderboard.empty, "Leaderboard is empty"
        assert 'rank' in leaderboard.columns, "Missing rank column"
        assert 'model' in leaderboard.columns, "Missing model column"
        assert 'sCRPS_mean' in leaderboard.columns, "Missing sCRPS_mean column"
        
        # Check metrics structure
        metrics = results['metrics']
        assert not metrics.empty, "Metrics DataFrame is empty"
        assert 'sCRPS_mean' in metrics.columns, "Missing sCRPS_mean in metrics"
        assert 'sCRPS_ci_lower' in metrics.columns, "Missing confidence intervals"
        
        # Display results
        logger.info("\n" + "="*70)
        logger.info("TEST RESULTS")
        logger.info("="*70)
        
        logger.info("\nLeaderboard:")
        for _, row in leaderboard.iterrows():
            logger.info(f"  Rank {row['rank']}: {row['model']} - sCRPS: {row['sCRPS_mean']:.4f}")
        
        logger.info("\nMetrics Summary:")
        for _, row in metrics.iterrows():
            logger.info(f"  {row['model']}:")
            logger.info(f"    sCRPS: {row['sCRPS_mean']:.4f} [{row['sCRPS_ci_lower']:.4f}, {row['sCRPS_ci_upper']:.4f}]")
            if 'MAE_mean' in row:
                logger.info(f"    MAE: {row['MAE_mean']:.4f}")
        
        # Save test results
        results_path = test_dir / "test_results.yaml"
        test_summary = {
            'test_date': datetime.now().isoformat(),
            'n_models': len(models),
            'n_cv_windows': cfg['n_windows'],
            'n_predictions': len(cv_df),
            'best_model': leaderboard.iloc[0]['model'] if not leaderboard.empty else None,
            'best_scrps': float(leaderboard.iloc[0]['sCRPS_mean']) if not leaderboard.empty else None,
            'test_passed': True
        }
        
        with open(results_path, 'w') as f:
            yaml.dump(test_summary, f)
        
        logger.info(f"\nTest results saved to {results_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ CV INTEGRATION TEST PASSED")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Cleanup (optional)
        logger.info(f"\nTest artifacts saved in {test_dir}/")

def main():
    """Main entry point for the test script."""
    success = test_cv_integration()
    
    if success:
        logger.info("\n✅ All CV integration tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ CV integration tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()