"""
Basic tests for CV and Metrics foundation.

This test file validates the core functionality of the CV runner and metrics modules.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cv.runner import run_cv, summarize_cv, _validate_cv_results
from uq.metrics import compute_scrps, compute_mae, compute_rmse, compute_bias
from uq.calibration import compute_coverage, compute_pit
from utils.io import save_cv_artifacts, load_cv_artifacts


def test_imports():
    """Test that all required modules import successfully."""
    print("✓ All imports successful")
    return True


def test_metrics_functions():
    """Test basic metrics computation."""
    # Create sample data
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    # Test MAE
    mae = compute_mae(y_true, y_pred)
    assert mae > 0, "MAE should be positive"
    print(f"✓ MAE computation: {mae:.4f}")
    
    # Test RMSE
    rmse = compute_rmse(y_true, y_pred)
    assert rmse > 0, "RMSE should be positive"
    assert rmse >= mae, "RMSE should be >= MAE"
    print(f"✓ RMSE computation: {rmse:.4f}")
    
    # Test Bias
    bias = compute_bias(y_true, y_pred)
    print(f"✓ Bias computation: {bias:.4f}")
    
    # Test sCRPS (simple version)
    scrps = compute_scrps(y_true, y_pred)
    assert not np.isnan(scrps), "sCRPS should not be NaN"
    print(f"✓ sCRPS computation: {scrps:.4f}")
    
    return True


def test_coverage_computation():
    """Test coverage analysis functionality."""
    # Create sample CV results
    n_samples = 100
    np.random.seed(42)
    
    cv_df = pd.DataFrame({
        'unique_id': ['BTC'] * n_samples,
        'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
        'cutoff': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
        'y': np.random.randn(n_samples),
        'TestModel': np.random.randn(n_samples),
        'TestModel-lo-80': np.random.randn(n_samples) - 1.28,
        'TestModel-hi-80': np.random.randn(n_samples) + 1.28,
        'TestModel-lo-90': np.random.randn(n_samples) - 1.645,
        'TestModel-hi-90': np.random.randn(n_samples) + 1.645,
        'TestModel-lo-95': np.random.randn(n_samples) - 1.96,
        'TestModel-hi-95': np.random.randn(n_samples) + 1.96,
    })
    
    # Test coverage computation
    coverage_df = compute_coverage(cv_df, ['TestModel'], levels=[80, 90, 95])
    
    assert not coverage_df.empty, "Coverage DataFrame should not be empty"
    assert len(coverage_df) == 3, "Should have 3 coverage levels"
    assert all(col in coverage_df.columns for col in ['model', 'level', 'empirical_coverage']), \
        "Coverage DataFrame should have required columns"
    
    print(f"✓ Coverage computation successful for {len(coverage_df)} levels")
    
    return True


def test_cv_validation():
    """Test CV results validation."""
    # Create sample CV results
    cv_df = pd.DataFrame({
        'unique_id': ['BTC'] * 10,
        'ds': pd.date_range('2024-01-01', periods=10, freq='15min'),
        'cutoff': pd.date_range('2024-01-01', periods=10, freq='15min'),
        'y': np.random.randn(10),
        'TestModel': np.random.randn(10)
    })
    
    # Mock model
    class MockModel:
        def __init__(self):
            self.__class__.__name__ = 'TestModel'
    
    models = [MockModel()]
    
    # Should not raise an error
    try:
        _validate_cv_results(cv_df, models, level=None)
        print("✓ CV validation passed")
        return True
    except Exception as e:
        print(f"✗ CV validation failed: {e}")
        return False


def test_io_functions():
    """Test I/O functions for CV artifacts."""
    # Create sample data
    cv_df = pd.DataFrame({'test': [1, 2, 3]})
    metrics_df = pd.DataFrame({'metric': ['sCRPS'], 'value': [0.5]})
    leaderboard_df = pd.DataFrame({'model': ['TestModel'], 'rank': [1]})
    
    # Test saving
    output_dir = 'tests/temp_cv_test'
    horizon = 4
    
    saved_paths = save_cv_artifacts(
        cv_df=cv_df,
        metrics_df=metrics_df,
        leaderboard_df=leaderboard_df,
        output_dir=output_dir,
        horizon=horizon
    )
    
    assert 'cv_results' in saved_paths, "Should save CV results"
    assert 'metrics' in saved_paths, "Should save metrics"
    assert 'leaderboard' in saved_paths, "Should save leaderboard"
    
    # Test loading
    loaded = load_cv_artifacts(output_dir, horizon)
    assert 'metrics' in loaded, "Should load metrics"
    assert 'leaderboard' in loaded, "Should load leaderboard"
    
    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    
    print("✓ I/O functions working correctly")
    return True


def run_all_tests():
    """Run all foundation tests."""
    print("\n" + "="*50)
    print("TESTING CV AND METRICS FOUNDATION")
    print("="*50 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Metrics Functions", test_metrics_functions),
        ("Coverage Computation", test_coverage_computation),
        ("CV Validation", test_cv_validation),
        ("I/O Functions", test_io_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)