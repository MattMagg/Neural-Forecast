#!/usr/bin/env python3
"""
Minimal testing suite for feature engineering pipeline.

Tests:
1. No-leak verification - historical features don't leak future info
2. MTF alignment - multi-timeframe values align correctly with shift
3. Feature cap enforcement - max 256 features
4. Deterministic stability - same input produces same output

Fast, not fluffy - uses minimal synthetic data for speed.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from features.builder import (
    build_indicators, 
    apply_mtf, 
    postprocess_shift_and_prune, 
    select_features
)
from features.registry import REGISTRY


def test_no_leak():
    """Test that historical features at time t only use data ≤ t-15m (not t)."""
    # Create sample data with specific patterns to verify no-leak
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    
    # Create OHLCV data with a known pattern: linearly increasing close
    sample_data = pd.DataFrame({
        'ds': dates,
        'open': 50000 + np.arange(200) * 10,
        'high': 50100 + np.arange(200) * 10,
        'low': 49900 + np.arange(200) * 10,
        'close': 50000 + np.arange(200) * 10,  # Linearly increasing
        'volume': 1000 + np.random.randn(200) * 10
    })
    
    # Build features
    base_feats = build_indicators(sample_data)
    exo_raw = base_feats
    exo = postprocess_shift_and_prune(exo_raw, rules={})
    
    # Get historical columns (those that should be shifted)
    hist_cols, _, _ = select_features(exo, policy={})
    
    # Pick a random timestamp to check (not first few due to shift)
    test_idx = 50
    test_time = dates[test_idx]
    
    # Verify shift(1) was applied: feature at row t should equal raw computation at row t-1
    # We'll check a specific indicator that we know exists: SMA
    sma_cols = [c for c in hist_cols if 'sma' in c.lower()]
    
    if sma_cols:  # If we have SMA columns
        col = sma_cols[0]
        
        # Recompute the feature without shift for comparison
        base_no_shift = build_indicators(sample_data)
        
        # The shifted value at index 50 should equal unshifted value at index 49
        shifted_val = exo.loc[exo['ds'] == test_time, col].values[0]
        prev_time = dates[test_idx - 1]
        unshifted_prev = base_no_shift.loc[base_no_shift['ds'] == prev_time, col].values[0]
        
        # They should be equal (accounting for floating point precision)
        assert np.abs(shifted_val - unshifted_prev) < 1e-10, \
            f"Shift not applied correctly: {col} at {test_time} = {shifted_val}, but prev unshifted = {unshifted_prev}"
    
    # Also verify that the first row has NaN for historical features (due to shift)
    for col in hist_cols:
        if col in exo.columns:
            first_val = exo.loc[0, col]
            assert pd.isna(first_val), f"Historical column {col} should be NaN at first row due to shift(1)"
    
    print("✅ No-leak test passed: historical features properly shifted by 1 bar")


def test_mtf_alignment():
    """Test MTF values align correctly: 09:00-09:45 holds 08:00-09:00 value."""
    # Create a tiny 1-day series (96 bars of 15-minute data)
    dates = pd.date_range('2024-01-01 00:00', periods=96, freq='15min')
    
    # Create simple data with known pattern for easy verification
    # Use a pattern where each hour has a distinct value
    sample_data = pd.DataFrame({
        'ds': dates,
        'open': [100 * (i // 4 + 1) for i in range(96)],  # Changes every hour
        'high': [100 * (i // 4 + 1) + 10 for i in range(96)],
        'low': [100 * (i // 4 + 1) - 10 for i in range(96)],
        'close': [100 * (i // 4 + 1) for i in range(96)],  # 100, 100, 100, 100, 200, 200, ...
        'volume': [1000] * 96
    })
    
    # Apply MTF to get 1h features
    mtf_feats = apply_mtf(sample_data)
    
    # Apply shift
    exo = postprocess_shift_and_prune(mtf_feats, rules={})
    
    # Find 1h SMA column (or any 1h indicator)
    cols_1h = [c for c in exo.columns if '_1h' in c]
    
    if cols_1h:
        test_col = cols_1h[0]
        
        # Check specific times as per requirement:
        # 09:00 to 09:45 should hold the 08:00-09:00 value
        # After shift(1), this becomes: 09:15 to 10:00 hold the 08:00-09:00 value
        
        time_0900 = pd.Timestamp('2024-01-01 09:00')
        time_0915 = pd.Timestamp('2024-01-01 09:15')
        time_0930 = pd.Timestamp('2024-01-01 09:30')
        time_0945 = pd.Timestamp('2024-01-01 09:45')
        time_1000 = pd.Timestamp('2024-01-01 10:00')
        time_1015 = pd.Timestamp('2024-01-01 10:15')
        
        # Get values at these times
        val_0915 = exo.loc[exo['ds'] == time_0915, test_col].values[0] if time_0915 in exo['ds'].values else None
        val_0930 = exo.loc[exo['ds'] == time_0930, test_col].values[0] if time_0930 in exo['ds'].values else None
        val_0945 = exo.loc[exo['ds'] == time_0945, test_col].values[0] if time_0945 in exo['ds'].values else None
        val_1000 = exo.loc[exo['ds'] == time_1000, test_col].values[0] if time_1000 in exo['ds'].values else None
        val_1015 = exo.loc[exo['ds'] == time_1015, test_col].values[0] if time_1015 in exo['ds'].values else None
        
        # Due to shift(1), values at 09:15-10:00 should be the same (from 08:00-09:00 hour)
        # And should update at 10:15 (getting 09:00-10:00 value)
        if val_0915 is not None and val_1000 is not None:
            assert val_0915 == val_0930 == val_0945 == val_1000, \
                f"MTF alignment failed: 09:15-10:00 should have same value after shift(1)"
            
            if val_1015 is not None:
                assert val_1015 != val_1000, \
                    f"MTF should update at 10:15 after shift(1), but got same value"
    
    print("✅ MTF alignment test passed: values align correctly with shift(1)")


def test_feature_cap():
    """Test that total feature columns (excluding 'ds') ≤ 256."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='15min')
    
    sample_data = pd.DataFrame({
        'ds': dates,
        'open': 50000 + np.random.randn(500) * 100,
        'high': 50100 + np.random.randn(500) * 100,
        'low': 49900 + np.random.randn(500) * 100,
        'close': 50000 + np.random.randn(500) * 100,
        'volume': 1000 + np.random.randn(500) * 10
    })
    
    # Build all features
    base_feats = build_indicators(sample_data)
    mtf_feats = apply_mtf(sample_data)
    exo_raw = base_feats.merge(mtf_feats, on="ds", how="left")
    exo = postprocess_shift_and_prune(exo_raw, rules={})
    
    # Select features with cap enforcement
    hist_cols, futr_cols, stat_cols = select_features(exo, policy={})
    
    # Total features should not exceed 256
    total_features = len(hist_cols) + len(futr_cols) + len(stat_cols)
    
    assert total_features <= 256, \
        f"Feature cap exceeded: {total_features} features (hist={len(hist_cols)}, " \
        f"futr={len(futr_cols)}, stat={len(stat_cols)}) > 256"
    
    # Also verify that the function enforces this even if we have more columns
    # Create a scenario with way too many features
    many_cols = ['feat_' + str(i) for i in range(300)]
    exo_many = exo.copy()
    for col in many_cols:
        exo_many[col] = np.random.randn(len(exo_many))
    
    hist_cols2, futr_cols2, stat_cols2 = select_features(exo_many, policy={})
    total_features2 = len(hist_cols2) + len(futr_cols2) + len(stat_cols2)
    
    assert total_features2 <= 256, \
        f"Feature cap not enforced with many columns: {total_features2} > 256"
    
    print(f"✅ Feature cap test passed: {total_features} features ≤ 256")


def test_deterministic():
    """Test that repeated runs on the same data yield identical exog matrices."""
    # Fix random seed for data generation
    np.random.seed(123)
    dates = pd.date_range('2024-01-01', periods=300, freq='15min')
    
    sample_data = pd.DataFrame({
        'ds': dates,
        'open': 50000 + np.random.randn(300) * 100,
        'high': 50100 + np.random.randn(300) * 100,
        'low': 49900 + np.random.randn(300) * 100,
        'close': 50000 + np.random.randn(300) * 100,
        'volume': 1000 + np.random.randn(300) * 10
    })
    
    # Run the full pipeline twice
    def run_pipeline(data):
        """Run the complete feature engineering pipeline."""
        base_feats = build_indicators(data)
        mtf_feats = apply_mtf(data)
        exo_raw = base_feats.merge(mtf_feats, on="ds", how="left")
        exo = postprocess_shift_and_prune(exo_raw, rules={})
        hist_cols, futr_cols, stat_cols = select_features(exo, policy={})
        return exo, hist_cols, futr_cols, stat_cols
    
    # Run 1
    exo1, hist1, futr1, stat1 = run_pipeline(sample_data)
    
    # Run 2 (with same input)
    exo2, hist2, futr2, stat2 = run_pipeline(sample_data)
    
    # Check that outputs are identical
    # 1. Same columns
    assert list(exo1.columns) == list(exo2.columns), \
        "Column names differ between runs"
    
    # 2. Same feature lists
    assert hist1 == hist2, f"Historical features differ: {set(hist1) ^ set(hist2)}"
    assert futr1 == futr2, f"Future features differ: {set(futr1) ^ set(futr2)}"
    assert stat1 == stat2, f"Static features differ: {set(stat1) ^ set(stat2)}"
    
    # 3. Same values (accounting for NaN equality)
    for col in exo1.columns:
        if col == 'ds':
            assert (exo1[col] == exo2[col]).all(), f"Timestamps differ"
        else:
            # Use pandas equals which handles NaN properly
            assert exo1[col].equals(exo2[col]), \
                f"Values differ in column {col}"
    
    print("✅ Deterministic test passed: pipeline produces identical results")


if __name__ == "__main__":
    # Run tests directly if executed as script
    print("Running feature engineering tests...\n")
    
    test_no_leak()
    test_mtf_alignment()
    test_feature_cap()
    test_deterministic()
    
    print("\n🎉 All tests passed!")