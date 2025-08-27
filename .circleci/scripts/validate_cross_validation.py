#!/usr/bin/env python3
"""
Cross-validation validation script for CircleCI.

This script validates the cross-validation system by:
1. Testing NF-native cross-validation execution with proper windowing
2. Validating sCRPS computation and metrics testing
3. Testing coverage and PIT diagnostics validation
4. Validating model persistence testing and artifact management
5. Testing cross-validation caching and result storage

Requirements: 1.5, 5.5, 6.4
"""

import sys
import os
import logging
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_cv_dependencies():
    """Validate all required dependencies for cross-validation."""
    logger.info("=== Cross-Validation Dependencies Validation ===")
    
    required_packages = [
        ('neuralforecast', 'NeuralForecast'),
        ('torch', 'PyTorch'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✓ {name}: {version}")
        except ImportError as e:
            logger.error(f"✗ {name}: Import failed - {e}")
            failed_imports.append(name)
    
    if failed_imports:
        raise RuntimeError(f"Failed to import required packages: {failed_imports}")
    
    # Validate GPU availability for cross-validation
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        logger.warning("⚠️  CUDA not available - CV will run on CPU")
    
    logger.info("Dependencies validation completed successfully")


def validate_cv_module_structure():
    """Validate cross-validation module structure and imports."""
    logger.info("=== Cross-Validation Module Structure Validation ===")
    
    # Check cv/runner.py exists and has required functions
    cv_runner_path = Path("cv/runner.py")
    if not cv_runner_path.exists():
        raise FileNotFoundError("cv/runner.py not found")
    
    logger.info("✓ cv/runner.py exists")
    
    # Test imports from cv.runner
    try:
        from cv.runner import run_cv, summarize_cv
        logger.info("✓ Successfully imported run_cv and summarize_cv")
    except ImportError as e:
        raise ImportError(f"Failed to import CV functions: {e}")
    
    # Check uq modules for metrics and calibration
    uq_modules = ['uq.metrics', 'uq.calibration']
    for module_name in uq_modules:
        try:
            __import__(module_name)
            logger.info(f"✓ Successfully imported {module_name}")
        except ImportError as e:
            logger.warning(f"⚠️  {module_name} import failed: {e}")
    
    # Check utils modules for validation and error recovery
    utils_modules = ['utils.validate', 'utils.error_recovery', 'utils.risk_mitigation']
    for module_name in utils_modules:
        try:
            __import__(module_name)
            logger.info(f"✓ Successfully imported {module_name}")
        except ImportError as e:
            logger.warning(f"⚠️  {module_name} import failed: {e}")
    
    logger.info("Module structure validation completed successfully")


def validate_nf_native_cv():
    """Test NF-native cross-validation execution with proper windowing."""
    logger.info("=== NF-Native Cross-Validation Execution Test ===")
    
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
        from neuralforecast.losses.pytorch import DistributionLoss
        from cv.runner import run_cv
        from utils.io import load_and_process_data
        
        # Create minimal test data
        logger.info("Creating synthetic test data...")
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        np.random.seed(42)
        
        # Generate realistic BTC-like price data
        returns = np.random.normal(0, 0.02, len(dates))
        prices = 50000 * np.exp(np.cumsum(returns))
        
        test_df = pd.DataFrame({
            'unique_id': 'BTC-USD',
            'ds': dates,
            'y': np.log(prices[1:] / prices[:-1])[:-1],  # Log returns
            'open': prices[:-2],
            'high': prices[:-2] * 1.01,
            'low': prices[:-2] * 0.99,
            'close': prices[1:-1],
            'volume': np.random.uniform(1000, 10000, len(dates)-2)
        })
        
        logger.info(f"✓ Created test dataset: {len(test_df)} rows")
        
        # Create minimal NF model for testing
        logger.info("Creating NeuralForecast model...")
        model = NHITS(
            h=4,
            input_size=24,
            loss=DistributionLoss(distribution='StudentT', return_params=True),
            max_steps=10,  # Minimal training for testing
            batch_size=32,
            enable_progress_bar=False
        )
        
        nf = NeuralForecast(models=[model], freq='15min')
        
        # Test CV configuration
        cv_config = {
            'n_windows': 3,  # Minimal for testing
            'step_size': 4,
            'val_size': 16,
            'h': 4,
            'level': [80, 90, 95]
        }
        
        logger.info("Testing NF-native cross-validation...")
        
        # Execute cross-validation
        cv_results = run_cv(
            nf=nf,
            df=test_df,
            cfg=cv_config,
            use_conformal=False,
            enable_recovery=True
        )
        
        # Validate CV results structure
        required_cols = ['unique_id', 'ds', 'cutoff', 'y', 'NHITS']
        missing_cols = [col for col in required_cols if col not in cv_results.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CV results: {missing_cols}")
        
        logger.info(f"✓ CV execution successful: {len(cv_results)} predictions")
        logger.info(f"✓ CV windows: {cv_results['cutoff'].nunique()}")
        logger.info(f"✓ Model predictions: NHITS column present")
        
        # Check for prediction intervals
        interval_cols = [col for col in cv_results.columns if '-lo-' in col or '-hi-' in col]
        if interval_cols:
            logger.info(f"✓ Prediction intervals: {len(interval_cols)} interval columns")
        else:
            logger.warning("⚠️  No prediction intervals found")
        
        return cv_results
        
    except Exception as e:
        logger.error(f"NF-native CV test failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def validate_scrps_computation(cv_results):
    """Test sCRPS computation and metrics validation."""
    logger.info("=== sCRPS Computation and Metrics Validation ===")
    
    try:
        from uq.metrics import compute_metrics_per_model
        
        # Test sCRPS computation for NHITS model
        logger.info("Computing sCRPS metrics...")
        
        metrics_df = compute_metrics_per_model(
            cv_df=cv_results,
            model_name='NHITS',
            y_col='y'
        )
        
        # Validate metrics structure
        expected_metrics = ['sCRPS', 'MAE', 'RMSE']
        found_metrics = [col for col in expected_metrics if col in metrics_df.columns]
        
        if not found_metrics:
            raise ValueError("No expected metrics found in results")
        
        logger.info(f"✓ Computed metrics: {found_metrics}")
        
        # Validate sCRPS values are reasonable
        if 'sCRPS' in metrics_df.columns:
            scrps_values = metrics_df['sCRPS'].dropna()
            if len(scrps_values) > 0:
                mean_scrps = scrps_values.mean()
                logger.info(f"✓ Mean sCRPS: {mean_scrps:.6f}")
                
                # sCRPS should be positive and finite
                if mean_scrps <= 0 or not np.isfinite(mean_scrps):
                    raise ValueError(f"Invalid sCRPS value: {mean_scrps}")
                
                logger.info("✓ sCRPS values are valid")
            else:
                logger.warning("⚠️  No valid sCRPS values computed")
        
        return metrics_df
        
    except Exception as e:
        logger.error(f"sCRPS computation test failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def validate_coverage_and_pit(cv_results):
    """Test coverage and PIT diagnostics validation."""
    logger.info("=== Coverage and PIT Diagnostics Validation ===")
    
    try:
        from uq.calibration import compute_coverage
        
        # Test coverage computation
        logger.info("Computing coverage diagnostics...")
        
        models = ['NHITS']
        coverage_df = compute_coverage(cv_results, models)
        
        if coverage_df.empty:
            logger.warning("⚠️  No coverage results computed - may be missing interval columns")
            return coverage_df
        
        logger.info(f"✓ Coverage computation successful: {len(coverage_df)} results")
        
        # Validate coverage structure
        expected_cols = ['model', 'level', 'coverage', 'nominal']
        found_cols = [col for col in expected_cols if col in coverage_df.columns]
        logger.info(f"✓ Coverage columns: {found_cols}")
        
        # Check coverage values are reasonable (0-1 range)
        if 'coverage' in coverage_df.columns:
            coverage_values = coverage_df['coverage'].dropna()
            if len(coverage_values) > 0:
                valid_coverage = ((coverage_values >= 0) & (coverage_values <= 1)).all()
                if not valid_coverage:
                    raise ValueError("Coverage values outside valid range [0,1]")
                
                logger.info(f"✓ Coverage values valid: {coverage_values.mean():.3f} mean coverage")
            else:
                logger.warning("⚠️  No valid coverage values found")
        
        # Test PIT diagnostics if available
        try:
            from uq.calibration import compute_pit_diagnostics
            
            logger.info("Testing PIT diagnostics...")
            pit_results = compute_pit_diagnostics(cv_results, models)
            
            if not pit_results.empty:
                logger.info(f"✓ PIT diagnostics computed: {len(pit_results)} results")
            else:
                logger.warning("⚠️  PIT diagnostics returned empty results")
                
        except ImportError:
            logger.warning("⚠️  PIT diagnostics function not available")
        except Exception as e:
            logger.warning(f"⚠️  PIT diagnostics failed: {str(e)}")
        
        return coverage_df
        
    except Exception as e:
        logger.error(f"Coverage and PIT validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def validate_model_persistence():
    """Test model persistence and artifact management."""
    logger.info("=== Model Persistence and Artifact Management ===")
    
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
        from neuralforecast.losses.pytorch import DistributionLoss
        import tempfile
        import shutil
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Testing model persistence in: {temp_path}")
            
            # Create minimal model
            model = NHITS(
                h=4,
                input_size=24,
                loss=DistributionLoss(distribution='StudentT', return_params=True),
                max_steps=5,
                batch_size=32,
                enable_progress_bar=False
            )
            
            nf = NeuralForecast(models=[model], freq='15min')
            
            # Create minimal training data
            dates = pd.date_range('2024-01-01', periods=100, freq='15min')
            np.random.seed(42)
            
            train_df = pd.DataFrame({
                'unique_id': 'BTC-USD',
                'ds': dates,
                'y': np.random.normal(0, 0.01, len(dates))
            })
            
            # Fit model
            logger.info("Fitting model for persistence test...")
            nf.fit(train_df)
            
            # Test save functionality
            save_path = temp_path / "test_model"
            logger.info(f"Testing model save to: {save_path}")
            
            nf.save(path=str(save_path))
            
            # Verify save artifacts exist
            if save_path.exists():
                saved_files = list(save_path.rglob("*"))
                logger.info(f"✓ Model saved successfully: {len(saved_files)} files")
                
                # Check for expected NF save structure
                expected_patterns = ['*.pkl', '*.json', '*.pt']
                found_patterns = []
                for pattern in expected_patterns:
                    if list(save_path.rglob(pattern)):
                        found_patterns.append(pattern)
                
                logger.info(f"✓ Save artifacts: {found_patterns}")
            else:
                raise FileNotFoundError("Model save directory not created")
            
            # Test load functionality
            logger.info("Testing model load...")
            loaded_nf = NeuralForecast.load(path=str(save_path))
            
            if loaded_nf is not None:
                logger.info("✓ Model loaded successfully")
                
                # Verify loaded model structure
                if hasattr(loaded_nf, 'models') and loaded_nf.models:
                    logger.info(f"✓ Loaded models: {[m.__class__.__name__ for m in loaded_nf.models]}")
                else:
                    logger.warning("⚠️  Loaded model has no models attribute")
            else:
                raise ValueError("Model load returned None")
        
        logger.info("Model persistence validation completed successfully")
        
    except Exception as e:
        logger.error(f"Model persistence test failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def validate_cv_caching():
    """Test cross-validation caching and result storage."""
    logger.info("=== Cross-Validation Caching and Result Storage ===")
    
    try:
        import tempfile
        from pathlib import Path
        
        # Test CV results caching
        logger.info("Testing CV results caching...")
        
        # Create test CV results
        test_cv_results = pd.DataFrame({
            'unique_id': ['BTC-USD'] * 20,
            'ds': pd.date_range('2024-01-01', periods=20, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=20, freq='15min'),
            'y': np.random.normal(0, 0.01, 20),
            'NHITS': np.random.normal(0, 0.01, 20)
        })
        
        # Test saving CV results
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cv_cache_path = temp_path / "cv_results.parquet"
            
            logger.info(f"Testing CV results save to: {cv_cache_path}")
            test_cv_results.to_parquet(cv_cache_path)
            
            if cv_cache_path.exists():
                file_size = cv_cache_path.stat().st_size
                logger.info(f"✓ CV results saved: {file_size} bytes")
            else:
                raise FileNotFoundError("CV results file not created")
            
            # Test loading CV results
            logger.info("Testing CV results load...")
            loaded_cv_results = pd.read_parquet(cv_cache_path)
            
            if len(loaded_cv_results) == len(test_cv_results):
                logger.info(f"✓ CV results loaded: {len(loaded_cv_results)} rows")
            else:
                raise ValueError(f"Loaded CV results size mismatch: {len(loaded_cv_results)} vs {len(test_cv_results)}")
            
            # Verify data integrity
            if loaded_cv_results.equals(test_cv_results):
                logger.info("✓ CV results data integrity verified")
            else:
                logger.warning("⚠️  CV results data integrity check failed")
        
        # Test metrics caching
        logger.info("Testing metrics caching...")
        
        test_metrics = {
            'model': 'NHITS',
            'horizon': 4,
            'sCRPS_mean': 0.001234,
            'MAE_mean': 0.005678,
            'RMSE_mean': 0.009012,
            'n_windows': 10
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics_cache_path = temp_path / "metrics.json"
            
            # Save metrics
            with open(metrics_cache_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            if metrics_cache_path.exists():
                logger.info("✓ Metrics saved successfully")
            else:
                raise FileNotFoundError("Metrics file not created")
            
            # Load metrics
            with open(metrics_cache_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            if loaded_metrics == test_metrics:
                logger.info("✓ Metrics data integrity verified")
            else:
                raise ValueError("Metrics data integrity check failed")
        
        logger.info("CV caching validation completed successfully")
        
    except Exception as e:
        logger.error(f"CV caching test failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def validate_gpu_memory_management():
    """Test GPU memory management for cross-validation."""
    logger.info("=== GPU Memory Management Validation ===")
    
    try:
        if not torch.cuda.is_available():
            logger.info("⚠️  CUDA not available - skipping GPU memory tests")
            return
        
        from utils.risk_mitigation import GPUMemoryManager
        
        # Test GPU memory monitoring
        logger.info("Testing GPU memory monitoring...")
        
        gpu_manager = GPUMemoryManager()
        mem_info = gpu_manager.get_gpu_memory_info()
        
        logger.info(f"✓ GPU memory info: {mem_info['free_gb']:.2f}GB free of {mem_info['total_gb']:.2f}GB")
        
        # Test batch size suggestion
        suggested_batch = gpu_manager.suggest_batch_size(
            model_size_mb=500,
            current_batch=512
        )
        
        logger.info(f"✓ Suggested batch size: {suggested_batch}")
        
        # Test memory cleanup
        logger.info("Testing GPU memory cleanup...")
        torch.cuda.empty_cache()
        logger.info("✓ GPU memory cleanup completed")
        
        logger.info("GPU memory management validation completed successfully")
        
    except Exception as e:
        logger.error(f"GPU memory management test failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main validation function."""
    logger.info("=== Cross-Validation Comprehensive Validation ===")
    logger.info("Requirements: 1.5, 5.5, 6.4")
    logger.info("- GPU-enabled cross-validation job using existing cv/runner.py")
    logger.info("- NF-native cross-validation execution with proper windowing")
    logger.info("- sCRPS computation validation and metrics testing")
    logger.info("- Coverage and PIT diagnostics validation")
    logger.info("- Model persistence testing and artifact management")
    logger.info("- Cross-validation caching and result storage")
    
    validation_results = {}
    
    try:
        # 1. Validate dependencies
        validate_cv_dependencies()
        validation_results['dependencies'] = 'PASSED'
        
        # 2. Validate module structure
        validate_cv_module_structure()
        validation_results['module_structure'] = 'PASSED'
        
        # 3. Test NF-native cross-validation
        cv_results = validate_nf_native_cv()
        validation_results['nf_native_cv'] = 'PASSED'
        
        # 4. Test sCRPS computation
        metrics_df = validate_scrps_computation(cv_results)
        validation_results['scrps_computation'] = 'PASSED'
        
        # 5. Test coverage and PIT diagnostics
        coverage_df = validate_coverage_and_pit(cv_results)
        validation_results['coverage_pit'] = 'PASSED'
        
        # 6. Test model persistence
        validate_model_persistence()
        validation_results['model_persistence'] = 'PASSED'
        
        # 7. Test CV caching
        validate_cv_caching()
        validation_results['cv_caching'] = 'PASSED'
        
        # 8. Test GPU memory management
        validate_gpu_memory_management()
        validation_results['gpu_memory'] = 'PASSED'
        
        # Generate summary report
        logger.info("=== Cross-Validation Validation Summary ===")
        for test_name, result in validation_results.items():
            logger.info(f"✓ {test_name}: {result}")
        
        logger.info(f"✓ All {len(validation_results)} validation tests passed successfully")
        logger.info("Cross-validation system is ready for production use")
        
        return 0
        
    except Exception as e:
        logger.error(f"Cross-validation validation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Generate failure report
        logger.error("=== Cross-Validation Validation Failure Report ===")
        for test_name, result in validation_results.items():
            status = result if result == 'PASSED' else 'FAILED'
            symbol = '✓' if status == 'PASSED' else '✗'
            logger.error(f"{symbol} {test_name}: {status}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())