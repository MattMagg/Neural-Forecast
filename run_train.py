#!/usr/bin/env python3
"""
Training pipeline for the BTC forecasting system.

This script implements the complete data assembly path:
load raw OHLCV → aggregate_1min_to_15min → regularize_to_grid_utc → make_nf_canonical → validate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

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
    
    # Step 2: Aggregate to 15-minute bars
    print("📊 Aggregating 1-minute to 15-minute bars...")
    df_15min = aggregate_1min_to_15min(df_1min)
    print(f"   Aggregated to {len(df_15min):,} 15-minute bars")
    
    # Step 3: Regularize to complete UTC grid
    print("🕐 Regularizing to UTC grid...")
    df_regularized = regularize_to_grid_utc(df_15min, "15min")
    print(f"   Regularized grid: {len(df_regularized):,} bars")
    
    # Step 4: Create NeuralForecast canonical format
    print("🎯 Creating NF canonical format...")
    df_canonical = make_nf_canonical(df_regularized, unique_id="BTC-USD")
    print(f"   Canonical format: {len(df_canonical):,} rows")
    print(f"   Columns: {list(df_canonical.columns)}")
    
    # Step 5: Apply winsorization
    print("✂️  Applying winsorization...")
    df_winsorized = drop_train_nans_and_winsorize(df_canonical, lower_q=0.001, upper_q=0.999)
    
    # Count non-NaN values
    y_count = df_winsorized['y'].notna().sum()
    y_train_count = df_winsorized['y_train'].notna().sum()
    print(f"   y values: {y_count:,} non-NaN")
    print(f"   y_train values: {y_train_count:,} non-NaN")
    
    # Step 6: Validate all quality gates
    print("✅ Running validation gates...")
    
    # Grid validation
    assert_regular_grid(df_winsorized, "15min")
    print("   ✓ Regular grid validation passed")
    
    # UTC EOB validation
    assert_utc_eob(df_winsorized, "15min")
    print("   ✓ UTC EOB validation passed")
    
    # Forward-fill validation
    assert_no_forward_fill_y(df_winsorized)
    print("   ✓ No forward-fill validation passed")
    
    # Note: assert_shifted will be called later when exogenous features are added
    print("   ⏳ Leakage validation (assert_shifted) will be applied after feature integration")
    
    print("🎉 Data assembly path completed successfully!")
    return df_winsorized


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
    
    # Import feature engineering functions
    from features.registry import REGISTRY
    from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features
    
    # Exact sequence from Section 3.4 of docs/forecasting_sf_plan.md
    print("   📊 Building base indicators...")
    base_feats = build_indicators(nf_base.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}))
    
    print("   🕐 Computing multi-timeframe features...")
    mtf_feats = apply_mtf(nf_base[["ds","open","high","low","close","volume"]])
    
    print("   🔄 Merging feature sets...")
    exo_raw = base_feats.merge(mtf_feats, on="ds", how="left")
    
    print("   ⚡ Applying shift(1) and pruning...")
    exo = postprocess_shift_and_prune(exo_raw, rules={})
    
    print("   🎯 Selecting features with hard cap...")
    hist_cols, futr_cols, stat_cols = select_features(exo, policy={})
    
    print("   🔀 Merging features with canonical frame...")
    nf_df = nf_base.merge(exo, on="ds", how="left")
    
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
    
    return nf_df, hist_cols, futr_cols, stat_cols


def save_processed_data(df: pd.DataFrame, output_dir: str = "data/processed") -> str:
    """
    Save processed data with timestamped filename.
    
    Args:
        df: Processed DataFrame to save
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_path = timestamped_path(output_dir, "btc_canonical", "parquet")
    save_parquet(df, output_path)
    print(f"💾 Saved processed data to: {output_path}")
    return output_path


def main():
    """Main training pipeline entry point."""
    parser = argparse.ArgumentParser(description="BTC Forecasting Training Pipeline")
    parser.add_argument(
        "--raw-data", 
        default="data/raw/btcusd_1-min_data.csv",
        help="Path to raw 1-minute CSV data"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save processed data to disk"
    )
    
    args = parser.parse_args()
    
    try:
        # Execute complete data assembly path
        df_processed = load_and_process_data(args.raw_data)
        
        # Integrate feature engineering pipeline
        nf_df, hist_cols, futr_cols, stat_cols = integrate_features(df_processed)
        
        # Save processed data if requested
        if args.save_processed:
            saved_path = save_processed_data(nf_df, args.output_dir)
            print(f"✅ Processing complete. Data saved to: {saved_path}")
        else:
            print("✅ Processing complete. Use --save-processed to save data.")
        
        # Show summary statistics
        print("\n📊 Summary Statistics:")
        print(f"   Total rows: {len(nf_df):,}")
        print(f"   Date range: {nf_df['ds'].min()} to {nf_df['ds'].max()}")
        print(f"   Non-NaN y values: {nf_df['y'].notna().sum():,}")
        print(f"   Non-NaN y_train values: {nf_df['y_train'].notna().sum():,}")
        print(f"   Total columns: {len(nf_df.columns)}")
        
        # Show sample of the data
        print("\n🔍 Sample Data (first 5 rows):")
        print(nf_df.head())
        
        # Store feature lists for model instantiation
        # These would be passed to instantiate_models() in the next phase
        print("\n📝 Feature lists ready for model instantiation:")
        print(f"   hist_cols: {len(hist_cols)} features")
        print(f"   futr_cols: {len(futr_cols)} features")
        print(f"   stat_cols: {len(stat_cols)} features")
        
        return nf_df, hist_cols, futr_cols, stat_cols
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()