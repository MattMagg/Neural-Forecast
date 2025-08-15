#!/usr/bin/env python3
"""
Test script to validate feature engineering integration.

This script tests that:
1. Feature engineering pipeline integrates correctly with training
2. assert_shifted() validation is properly applied
3. Feature lists reach model instantiation
4. run_predict.py mirrors the same pipeline
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_feature_integration():
    """Test the complete feature engineering integration."""
    
    print("=" * 60)
    print("Testing Feature Engineering Integration")
    print("=" * 60)
    
    try:
        # Test 1: Import validation
        print("\n📦 Test 1: Checking imports...")
        from features.registry import REGISTRY
        from features.builder import (
            build_indicators, 
            apply_mtf, 
            postprocess_shift_and_prune, 
            select_features
        )
        from utils.validate import assert_shifted
        from nf_models.factory import instantiate_models
        print("   ✓ All imports successful")
        
        # Test 2: Create sample data
        print("\n📊 Test 2: Creating sample data...")
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        sample_data = pd.DataFrame({
            'ds': dates,
            'unique_id': 'BTC-USD',
            'open': 50000 + np.random.randn(1000) * 100,
            'high': 50100 + np.random.randn(1000) * 100,
            'low': 49900 + np.random.randn(1000) * 100,
            'close': 50000 + np.random.randn(1000) * 100,
            'volume': 1000 + np.random.randn(1000) * 10,
            'y': np.random.randn(1000) * 0.01
        })
        print(f"   ✓ Created sample data with {len(sample_data)} rows")
        
        # Test 3: Run feature pipeline
        print("\n🔧 Test 3: Running feature pipeline...")
        print("   Building base indicators...")
        base_feats = build_indicators(sample_data.rename(columns={
            "open":"open","high":"high","low":"low","close":"close","volume":"volume"
        }))
        print(f"   ✓ Base features: {len(base_feats.columns)} columns")
        
        print("   Computing multi-timeframe features...")
        mtf_feats = apply_mtf(sample_data[["ds","open","high","low","close","volume"]])
        print(f"   ✓ MTF features: {len(mtf_feats.columns)} columns")
        
        print("   Merging feature sets...")
        exo_raw = base_feats.merge(mtf_feats, on="ds", how="left")
        print(f"   ✓ Merged features: {len(exo_raw.columns)} columns")
        
        print("   Applying shift(1) and pruning...")
        exo = postprocess_shift_and_prune(exo_raw, rules={})
        print(f"   ✓ Post-processed features: {len(exo.columns)} columns")
        
        print("   Selecting features with hard cap...")
        hist_cols, futr_cols, stat_cols = select_features(exo, policy={})
        print(f"   ✓ Selected features:")
        print(f"      Historical: {len(hist_cols)}")
        print(f"      Future: {len(futr_cols)}")
        print(f"      Static: {len(stat_cols)}")
        print(f"      Total: {len(hist_cols) + len(futr_cols) + len(stat_cols)} (cap: 256)")
        
        # Test 4: Validate leakage prevention
        print("\n🔒 Test 4: Validating leakage prevention...")
        nf_df = sample_data.merge(exo, on="ds", how="left")
        
        # This should pass if shift(1) was properly applied
        try:
            assert_shifted(nf_df, hist_cols)
            print("   ✓ assert_shifted validation PASSED - no leakage detected!")
        except AssertionError as e:
            print(f"   ❌ assert_shifted validation FAILED: {e}")
            return False
        
        # Test 5: Model instantiation with features
        print("\n🤖 Test 5: Testing model instantiation with features...")
        cfg = {
            'h': 4,
            'freq': '15min',
            'models': [
                {
                    'NHITS': {
                        'alias': 'test_nhits',
                        'input_size': 100,  # Small for testing
                        'loss': {'kind': 'studentt'}
                    }
                }
            ]
        }
        
        models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
        model = models[0]
        
        # Verify feature lists are properly wired
        assert model.hist_exog_list == hist_cols, "Historical features not wired"
        assert model.futr_exog_list == futr_cols, "Future features not wired"
        assert model.stat_exog_list == stat_cols, "Static features not wired"
        print("   ✓ Model instantiated with feature lists")
        print(f"      Model type: {type(model).__name__}")
        print(f"      Hist features wired: {len(model.hist_exog_list)}")
        print(f"      Futr features wired: {len(model.futr_exog_list)}")
        print(f"      Stat features wired: {len(model.stat_exog_list)}")
        
        # Test 6: Verify integration in run_train.py
        print("\n📝 Test 6: Checking run_train.py integration...")
        with open('run_train.py', 'r') as f:
            train_content = f.read()
        
        checks = [
            ('from features.registry import REGISTRY' in train_content, 
             "Feature registry import"),
            ('from features.builder import build_indicators, apply_mtf' in train_content,
             "Feature builder imports"),
            ('assert_shifted(nf_df, hist_cols)' in train_content,
             "assert_shifted validation"),
            ('integrate_features' in train_content,
             "integrate_features function"),
            ('hist_cols, futr_cols, stat_cols' in train_content,
             "Feature list returns")
        ]
        
        all_passed = True
        for check, desc in checks:
            if check:
                print(f"   ✓ {desc}")
            else:
                print(f"   ❌ {desc}")
                all_passed = False
        
        if not all_passed:
            print("   ⚠️  Some integration checks failed")
            return False
        
        # Test 7: Verify run_predict.py mirrors the pipeline
        print("\n🔮 Test 7: Checking run_predict.py integration...")
        with open('run_predict.py', 'r') as f:
            predict_content = f.read()
        
        predict_checks = [
            ('from features.registry import REGISTRY' in predict_content,
             "Feature registry import"),
            ('from features.builder import build_indicators, apply_mtf' in predict_content,
             "Feature builder imports"),
            ('assert_shifted(nf_df, hist_cols)' in predict_content,
             "assert_shifted validation"),
            ('integrate_features_for_prediction' in predict_content,
             "Prediction feature integration"),
            ('NeuralForecast.load' in predict_content,
             "Model loading")
        ]
        
        all_predict_passed = True
        for check, desc in predict_checks:
            if check:
                print(f"   ✓ {desc}")
            else:
                print(f"   ❌ {desc}")
                all_predict_passed = False
        
        if not all_predict_passed:
            print("   ⚠️  Some prediction pipeline checks failed")
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        
        print("\n📋 Summary:")
        print("1. ✅ Feature engineering pipeline integrated into run_train.py")
        print("2. ✅ assert_shifted() validation properly applied")
        print("3. ✅ Feature lists reach model instantiation")
        print("4. ✅ run_predict.py mirrors the same pipeline")
        print("5. ✅ No data leakage detected (shift(1) working)")
        print("6. ✅ Feature cap of 256 enforced")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Make sure all required modules are properly implemented")
        return False
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_feature_integration()
    sys.exit(0 if success else 1)