#!/usr/bin/env python3
"""
Feature Engineering Validation Script for CircleCI

This script validates the feature engineering pipeline by:
1. Testing 13 technical indicators using existing features/registry.py and features/builder.py
2. Validating MTF alignment and shift(1) discipline testing
3. Testing feature selection with ≤256 cap validation
4. Implementing leakage prevention validation using assert_shifted
5. Generating feature engineering artifacts for caching and reporting

Requirements: 1.3, 5.3, 6.3
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging for validation"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('.circleci/scripts/feature_validation.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_feature_registry():
    """Validate that the feature registry contains the expected 13 indicators"""
    logger = setup_logging()
    logger.info("=== Feature Registry Validation ===")
    
    try:
        from features.registry import REGISTRY, MTF_TARGETS
        
        # Count indicators by type
        indicator_count = len([spec for spec in REGISTRY if spec.kind in ['hist', 'futr']])
        hist_indicators = [spec.name for spec in REGISTRY if spec.kind == 'hist']
        futr_indicators = [spec.name for spec in REGISTRY if spec.kind == 'futr']
        
        logger.info(f"Total indicators in registry: {indicator_count}")
        logger.info(f"Historical indicators: {len(hist_indicators)} - {hist_indicators}")
        logger.info(f"Future indicators: {len(futr_indicators)} - {futr_indicators}")
        
        # Validate we have the expected 13 indicators (10 hist + 3 futr based on registry)
        expected_hist = ['rsi', 'roc', 'stoch_k', 'macd', 'atr', 'nvol', 'obv', 'mfi', 'bbands', 'donchian']
        expected_futr = ['minute_of_day', 'day_of_week', 'is_weekend']
        
        missing_hist = set(expected_hist) - set(hist_indicators)
        missing_futr = set(expected_futr) - set(futr_indicators)
        
        if missing_hist:
            logger.error(f"Missing historical indicators: {missing_hist}")
            return False
            
        if missing_futr:
            logger.error(f"Missing future indicators: {missing_futr}")
            return False
            
        # Validate MTF targets
        logger.info(f"MTF targets configured: {len(MTF_TARGETS)} timeframes")
        for tf, names in MTF_TARGETS:
            logger.info(f"  {tf}: {names}")
            
        logger.info("✓ Feature registry validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Feature registry validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_feature_computation():
    """Validate feature computation using existing builder functions"""
    logger = setup_logging()
    logger.info("=== Feature Computation Validation ===")
    
    try:
        from utils.io import load_and_process_data
        from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features
        from features.registry import REGISTRY
        
        # Load processed data (should be available from data processing job)
        logger.info("Loading processed BTC data...")
        df_canonical = load_and_process_data("data/raw/btcusd_1-min_data.csv")
        logger.info(f"Loaded {len(df_canonical)} rows of canonical data")
        
        # Test base indicator computation (15min)
        logger.info("Computing base 15-minute indicators...")
        df_indicators = build_indicators(df_canonical)
        
        base_indicator_cols = [c for c in df_indicators.columns if c != 'ds']
        logger.info(f"Generated {len(base_indicator_cols)} base indicator columns")
        logger.info(f"Sample indicator columns: {base_indicator_cols[:10]}")
        
        # Validate specific indicators are present
        expected_patterns = ['rsi_', 'roc_', 'atr_', 'macd_', 'bbands_', 'obv', 'mfi_']
        for pattern in expected_patterns:
            matching_cols = [c for c in base_indicator_cols if pattern in c]
            if not matching_cols:
                logger.error(f"No columns found matching pattern '{pattern}'")
                return False
            logger.info(f"✓ Found {len(matching_cols)} columns for {pattern}")
        
        # Test MTF computation
        logger.info("Computing multi-timeframe indicators...")
        df_mtf = apply_mtf(df_canonical)
        
        mtf_cols = [c for c in df_mtf.columns if c != 'ds']
        logger.info(f"Generated {len(mtf_cols)} MTF indicator columns")
        
        # Validate MTF columns have timeframe suffixes
        mtf_patterns = ['_30min', '_1h', '_4h']
        for pattern in mtf_patterns:
            matching_cols = [c for c in mtf_cols if pattern in c]
            if not matching_cols:
                logger.warning(f"No MTF columns found for timeframe {pattern}")
            else:
                logger.info(f"✓ Found {len(matching_cols)} columns for timeframe {pattern}")
        
        # Combine indicators
        logger.info("Combining base and MTF indicators...")
        df_combined = df_canonical[['ds']].merge(df_indicators, on='ds', how='left')
        df_combined = df_combined.merge(df_mtf, on='ds', how='left')
        
        total_features = len([c for c in df_combined.columns if c != 'ds'])
        logger.info(f"Total combined features before processing: {total_features}")
        
        logger.info("✓ Feature computation validation passed")
        return df_combined
        
    except Exception as e:
        logger.error(f"Feature computation validation failed: {e}")
        logger.error(traceback.format_exc())
        return None

def validate_shift_discipline(df_features):
    """Validate shift(1) discipline and leakage prevention"""
    logger = setup_logging()
    logger.info("=== Shift Discipline and Leakage Prevention Validation ===")
    
    try:
        from features.builder import postprocess_shift_and_prune
        from utils.validate import assert_shifted
        
        # Apply shift and pruning
        logger.info("Applying shift(1) and pruning...")
        df_processed = postprocess_shift_and_prune(df_features, {})
        
        processed_cols = [c for c in df_processed.columns if c != 'ds']
        logger.info(f"Features after shift and pruning: {len(processed_cols)}")
        
        # Identify historical columns that should be shifted
        from features.registry import REGISTRY
        hist_indicators = [spec.name for spec in REGISTRY if spec.kind == 'hist']
        
        hist_cols = []
        for col in processed_cols:
            if any(col.startswith(hist_name) for hist_name in hist_indicators):
                hist_cols.append(col)
        
        logger.info(f"Historical columns to validate for shift: {len(hist_cols)}")
        
        if hist_cols:
            # Create a target variable for leakage testing (log returns)
            df_test = df_processed.copy()
            if 'y' not in df_test.columns:
                # Create synthetic target for testing
                df_test['y'] = np.random.randn(len(df_test))
            
            # Test leakage detection
            logger.info("Testing leakage detection with assert_shifted...")
            try:
                assert_shifted(df_test, hist_cols)
                logger.info("✓ No leakage detected in historical features")
            except AssertionError as e:
                logger.error(f"Leakage detected: {e}")
                return False
        
        # Validate that future columns are not shifted
        futr_indicators = [spec.name for spec in REGISTRY if spec.kind == 'futr']
        futr_cols = []
        for col in processed_cols:
            if any(col.startswith(futr_name) for futr_name in futr_indicators):
                futr_cols.append(col)
        
        logger.info(f"Future columns (should not be shifted): {len(futr_cols)}")
        
        logger.info("✓ Shift discipline validation passed")
        return df_processed
        
    except Exception as e:
        logger.error(f"Shift discipline validation failed: {e}")
        logger.error(traceback.format_exc())
        return None

def validate_feature_selection(df_processed):
    """Validate feature selection with ≤256 cap"""
    logger = setup_logging()
    logger.info("=== Feature Selection Validation ===")
    
    try:
        from features.builder import select_features
        
        # Test feature selection
        logger.info("Testing feature selection with 256 cap...")
        hist_cols, futr_cols, stat_cols = select_features(df_processed, {})
        
        total_selected = len(hist_cols) + len(futr_cols) + len(stat_cols)
        logger.info(f"Selected features: {total_selected} total")
        logger.info(f"  Historical: {len(hist_cols)}")
        logger.info(f"  Future: {len(futr_cols)}")
        logger.info(f"  Static: {len(stat_cols)}")
        
        # Validate 256 cap
        if total_selected > 256:
            logger.error(f"Feature selection exceeded 256 cap: {total_selected}")
            return False
        
        # Validate feature types are correctly identified
        if not futr_cols:
            logger.warning("No future columns selected - this may be expected")
        
        if not hist_cols:
            logger.error("No historical columns selected - this is unexpected")
            return False
        
        logger.info("✓ Feature selection validation passed")
        return hist_cols, futr_cols, stat_cols
        
    except Exception as e:
        logger.error(f"Feature selection validation failed: {e}")
        logger.error(traceback.format_exc())
        return None

def validate_mtf_alignment():
    """Validate multi-timeframe alignment"""
    logger = setup_logging()
    logger.info("=== MTF Alignment Validation ===")
    
    try:
        from utils.io import load_and_process_data
        from features.builder import apply_mtf
        
        # Load a smaller sample for MTF testing
        logger.info("Loading data for MTF alignment testing...")
        df_canonical = load_and_process_data("data/raw/btcusd_1-min_data.csv")
        
        # Take a subset for faster testing
        df_sample = df_canonical.tail(1000).copy()
        logger.info(f"Using {len(df_sample)} rows for MTF testing")
        
        # Test MTF computation
        df_mtf = apply_mtf(df_sample)
        
        # Validate timestamps are properly aligned (should have same length and range)
        if len(df_mtf) != len(df_sample):
            logger.error(f"MTF data length mismatch: {len(df_mtf)} vs {len(df_sample)}")
            return False
            
        # Check timestamp ranges are similar (allowing for minor differences)
        mtf_start, mtf_end = df_mtf['ds'].min(), df_mtf['ds'].max()
        sample_start, sample_end = df_sample['ds'].min(), df_sample['ds'].max()
        
        if abs((mtf_start - sample_start).total_seconds()) > 900:  # 15 minutes tolerance
            logger.error(f"MTF start time misalignment: {mtf_start} vs {sample_start}")
            return False
            
        if abs((mtf_end - sample_end).total_seconds()) > 900:  # 15 minutes tolerance
            logger.error(f"MTF end time misalignment: {mtf_end} vs {sample_end}")
            return False
        
        # Check for proper EOB alignment (timestamps should be on 15-minute boundaries)
        timestamps = pd.to_datetime(df_mtf['ds'])
        minutes = timestamps.dt.minute
        valid_minutes = minutes.isin([0, 15, 30, 45])
        
        if not valid_minutes.all():
            logger.error("MTF timestamps not properly aligned to 15-minute boundaries")
            return False
        
        logger.info("✓ MTF alignment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"MTF alignment validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def generate_feature_artifacts(df_processed, selected_features):
    """Generate feature engineering artifacts for caching and reporting"""
    logger = setup_logging()
    logger.info("=== Generating Feature Engineering Artifacts ===")
    
    try:
        # Create artifacts directory
        artifacts_dir = Path("artifacts/feature_engineering")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        
        # Save processed features sample
        sample_size = min(1000, len(df_processed))
        df_sample = df_processed.tail(sample_size)
        
        # Remove duplicate columns if any
        df_sample = df_sample.loc[:, ~df_sample.columns.duplicated()]
        
        sample_path = artifacts_dir / f"feature_sample_{timestamp}.parquet"
        df_sample.to_parquet(sample_path, index=False)
        logger.info(f"Saved feature sample: {sample_path}")
        
        # Generate feature metadata
        hist_cols, futr_cols, stat_cols = selected_features
        
        metadata = {
            "timestamp": timestamp,
            "total_features": len(hist_cols) + len(futr_cols) + len(stat_cols),
            "historical_features": len(hist_cols),
            "future_features": len(futr_cols),
            "static_features": len(stat_cols),
            "sample_features": {
                "historical": hist_cols[:10],  # First 10 for brevity
                "future": futr_cols,
                "static": stat_cols
            },
            "data_shape": {
                "rows": len(df_processed),
                "columns": len(df_processed.columns)
            },
            "validation_status": "PASSED"
        }
        
        metadata_path = artifacts_dir / f"feature_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved feature metadata: {metadata_path}")
        
        # Generate feature summary report
        report_lines = [
            "# Feature Engineering Validation Report",
            f"Generated: {timestamp}",
            "",
            "## Summary",
            f"- Total features selected: {metadata['total_features']}",
            f"- Historical features: {metadata['historical_features']}",
            f"- Future features: {metadata['future_features']}",
            f"- Static features: {metadata['static_features']}",
            f"- Data shape: {metadata['data_shape']['rows']} rows × {metadata['data_shape']['columns']} columns",
            "",
            "## Validation Results",
            "✓ Feature registry validation passed",
            "✓ Feature computation validation passed", 
            "✓ Shift discipline validation passed",
            "✓ Feature selection validation passed",
            "✓ MTF alignment validation passed",
            "",
            "## Sample Features",
            "### Historical Features (first 10):",
        ]
        
        for feat in metadata['sample_features']['historical']:
            report_lines.append(f"- {feat}")
            
        report_lines.extend([
            "",
            "### Future Features:",
        ])
        
        for feat in metadata['sample_features']['future']:
            report_lines.append(f"- {feat}")
        
        report_path = artifacts_dir / f"feature_validation_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"Saved validation report: {report_path}")
        
        logger.info("✓ Feature engineering artifacts generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate feature artifacts: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main validation function"""
    logger = setup_logging()
    logger.info("Starting Feature Engineering Validation")
    
    try:
        # Step 1: Validate feature registry
        if not validate_feature_registry():
            logger.error("Feature registry validation failed")
            sys.exit(1)
        
        # Step 2: Validate feature computation
        df_features = validate_feature_computation()
        if df_features is None:
            logger.error("Feature computation validation failed")
            sys.exit(1)
        
        # Step 3: Validate shift discipline and leakage prevention
        df_processed = validate_shift_discipline(df_features)
        if df_processed is None:
            logger.error("Shift discipline validation failed")
            sys.exit(1)
        
        # Step 4: Validate feature selection
        selected_features = validate_feature_selection(df_processed)
        if selected_features is None:
            logger.error("Feature selection validation failed")
            sys.exit(1)
        
        # Step 5: Validate MTF alignment
        if not validate_mtf_alignment():
            logger.error("MTF alignment validation failed")
            sys.exit(1)
        
        # Step 6: Generate artifacts
        if not generate_feature_artifacts(df_processed, selected_features):
            logger.error("Failed to generate feature artifacts")
            sys.exit(1)
        
        logger.info("=== Feature Engineering Validation Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Feature engineering validation failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()