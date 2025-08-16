"""
Comprehensive integration tests for CV and metrics system.

This test suite covers end-to-end CV execution, artifact persistence,
error recovery, and performance benchmarks with realistic data.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import tempfile
import shutil
import time
import json
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss

from cv.runner import run_cv, summarize_cv, _validate_cv_results, _compute_cv_metrics
from cv.hpo import optimize_hyperparameters
from uq.metrics import compute_metrics_from_cv, aggregate_metrics, rank_models_by_scrps
from uq.calibration import compute_coverage, compute_pit, check_pit_uniformity
from uq.ensembles import create_equal_weight_ensemble, ensemble_predictions
from utils.io import save_cv_artifacts, load_cv_artifacts
from utils.validate import (
    assert_regular_grid,
    assert_utc_eob,
    assert_shifted,
    assert_no_forward_fill_y
)


class TestEndToEndCVExecution:
    """Test complete CV execution with realistic data."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic BTC-like data."""
        np.random.seed(42)
        n_samples = 1000  # ~10 days of 15-min data
        
        # Generate realistic price data with trend and volatility
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min', tz='UTC')
        trend = np.linspace(40000, 42000, n_samples)
        noise = np.random.randn(n_samples) * 100
        seasonal = 500 * np.sin(2 * np.pi * np.arange(n_samples) / 96)  # Daily pattern
        prices = trend + noise + seasonal
        
        # Convert to log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': dates[1:],
            'y': log_returns
        })
        
        # Add some realistic features
        df['volume'] = np.abs(np.random.randn(len(df)) * 1000000)
        df['volatility'] = pd.Series(log_returns).rolling(20).std().fillna(0.001)
        df['momentum'] = pd.Series(log_returns).rolling(10).mean().fillna(0)
        
        # Ensure all features are shifted
        df['volume_lag1'] = df['volume'].shift(1)
        df['volatility_lag1'] = df['volatility'].shift(1)
        df['momentum_lag1'] = df['momentum'].shift(1)
        df = df.iloc[1:].reset_index(drop=True)  # Remove first row with NaN
        
        return df
        
    @pytest.fixture
    def models_config(self):
        """Create model configurations."""
        return {
            'NHITS': {
                'h': 4,
                'input_size': 48,
                'n_blocks': [1, 1, 1],
                'mlp_units': [[64, 64], [64, 64], [64, 64]],
                'n_pool_kernel_size': [2, 2, 1],
                'n_freq_downsample': [2, 2, 1],
                'pooling_mode': 'MaxPool1d',
                'interpolation_mode': 'linear',
                'batch_size': 32,
                'learning_rate': 1e-3,
                'max_steps': 100,  # Small for testing
                'val_check_steps': 10,
                'random_seed': 42,
                'loss': DistributionLoss('StudentT', return_params=True),
                'scaler_type': 'robust'
            },
            'NBEATSx': {
                'h': 4,
                'input_size': 48,
                'output_size': 4,
                'n_blocks': [1, 1],
                'n_layers': [2, 2],
                'n_hidden': [[64, 64], [64, 64]],
                'n_harmonics': 0,
                'n_polynomials': 2,
                'stack_types': ['trend', 'seasonality'],
                'batch_size': 32,
                'learning_rate': 1e-3,
                'max_steps': 100,
                'val_check_steps': 10,
                'random_seed': 42,
                'loss': MQLoss(quantiles=[0.1, 0.5, 0.9]),
                'scaler_type': 'robust'
            }
        }
        
    def test_complete_cv_workflow(self, realistic_data, models_config):
        """Test complete CV workflow from data to leaderboard."""
        # 1. Validate input data
        assert_regular_grid(realistic_data, freq='15min')
        assert_utc_eob(realistic_data)
        assert_shifted(realistic_data, ['volume_lag1', 'volatility_lag1', 'momentum_lag1'])
        
        # 2. Create models
        models = []
        for model_name, config in models_config.items():
            if model_name == 'NHITS':
                model = NHITS(**config)
            elif model_name == 'NBEATSx':
                model = NBEATSx(**config)
            models.append(model)
            
        # 3. Create NeuralForecast instance
        nf = NeuralForecast(
            models=models,
            freq='15min'
        )
        
        # 4. Prepare data for NF
        train_df = realistic_data[['unique_id', 'ds', 'y']].copy()
        
        # 5. Fit models
        nf.fit(df=train_df, val_size=48)
        
        # 6. Run cross-validation
        cv_config = {
            'n_windows': 3,  # Small for testing
            'h': 4,
            'step_size': 4,
            'val_size': 16,
            'refit': False,  # Faster for testing
            'level': [80, 90],
            'verbose': False
        }
        
        cv_results = run_cv(nf, train_df, cv_config, use_conformal=False)
        
        # 7. Validate CV results
        assert isinstance(cv_results, pd.DataFrame)
        assert 'NHITS' in cv_results.columns
        assert 'NBEATSx' in cv_results.columns
        assert len(cv_results) > 0
        
        # 8. Compute metrics and create leaderboard
        summary = summarize_cv(cv_results, horizon=4, levels=[80, 90])
        
        assert 'metrics' in summary
        assert 'leaderboard' in summary
        assert 'best_model' in summary
        
        # 9. Validate leaderboard
        leaderboard = summary['leaderboard']
        assert len(leaderboard) == 2  # Two models
        assert leaderboard['rank'].tolist() == [1, 2]
        assert all(col in leaderboard.columns for col in ['model', 'sCRPS', 'MAE', 'RMSE'])
        
        # 10. Check best model selection
        best_model = summary['best_model']
        assert best_model in ['NHITS', 'NBEATSx']
        assert best_model == leaderboard.iloc[0]['model']
        
    def test_cv_with_ensemble(self, realistic_data, models_config):
        """Test CV with ensemble creation."""
        # Create and fit models (simplified)
        models = [
            NHITS(h=4, input_size=24, max_steps=50, random_seed=42),
            NBEATSx(h=4, input_size=24, max_steps=50, random_seed=42)
        ]
        
        nf = NeuralForecast(models=models, freq='15min')
        train_df = realistic_data[['unique_id', 'ds', 'y']].head(200)  # Small subset
        nf.fit(df=train_df, val_size=24)
        
        # Run CV
        cv_config = {
            'n_windows': 2,
            'h': 4,
            'step_size': 4,
            'val_size': 8,
            'refit': False,
            'level': None,
            'verbose': False
        }
        
        cv_results = run_cv(nf, train_df, cv_config)
        
        # Create ensemble from top models
        summary = summarize_cv(cv_results, horizon=4)
        top_models = summary['leaderboard'].head(2)['model'].tolist()
        
        # Create equal-weight ensemble
        ensemble_df = create_equal_weight_ensemble(cv_results, top_models, name='Ensemble')
        
        assert 'Ensemble' in ensemble_df.columns
        assert len(ensemble_df) == len(cv_results)
        
        # Verify ensemble is average of components
        expected = cv_results[top_models].mean(axis=1)
        actual = ensemble_df['Ensemble']
        np.testing.assert_array_almost_equal(actual.values, expected.values)


class TestArtifactPersistence:
    """Test saving and loading of CV artifacts."""
    
    @pytest.fixture
    def cv_artifacts(self):
        """Create sample CV artifacts."""
        np.random.seed(42)
        
        # CV results
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min', tz='UTC'),
            'cutoff': pd.date_range('2024-01-01', periods=100, freq='15min', tz='UTC')[::10].repeat(10),
            'y': np.random.randn(100),
            'Model1': np.random.randn(100),
            'Model2': np.random.randn(100)
        })
        
        # Add prediction intervals
        for model in ['Model1', 'Model2']:
            for level in [80, 90, 95]:
                z = stats.norm.ppf((1 + level/100) / 2)
                cv_df[f'{model}-lo-{level}'] = cv_df[model] - z * 0.1
                cv_df[f'{model}-hi-{level}'] = cv_df[model] + z * 0.1
                
        # Metrics
        metrics_df = pd.DataFrame({
            'model': ['Model1', 'Model1', 'Model2', 'Model2'],
            'window': [0, 1, 0, 1],
            'sCRPS': [0.12, 0.13, 0.11, 0.14],
            'MAE': [0.5, 0.52, 0.48, 0.53],
            'RMSE': [0.6, 0.62, 0.58, 0.63],
            'coverage_80': [0.79, 0.81, 0.82, 0.78],
            'coverage_90': [0.89, 0.91, 0.92, 0.88],
            'coverage_95': [0.94, 0.96, 0.95, 0.93]
        })
        
        # Leaderboard
        leaderboard_df = pd.DataFrame({
            'rank': [1, 2],
            'model': ['Model2', 'Model1'],
            'sCRPS': [0.125, 0.125],
            'MAE': [0.505, 0.51],
            'RMSE': [0.605, 0.61],
            'coverage_80': [0.80, 0.80],
            'coverage_90': [0.90, 0.90],
            'coverage_95': [0.945, 0.95]
        })
        
        return {
            'cv_df': cv_df,
            'metrics_df': metrics_df,
            'leaderboard_df': leaderboard_df
        }
        
    def test_save_and_load_artifacts(self, cv_artifacts):
        """Test saving and loading CV artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save artifacts
            saved_paths = save_cv_artifacts(
                cv_df=cv_artifacts['cv_df'],
                metrics_df=cv_artifacts['metrics_df'],
                leaderboard_df=cv_artifacts['leaderboard_df'],
                output_dir=tmpdir,
                horizon=4
            )
            
            # Verify files were created
            for key in ['cv_results', 'metrics', 'leaderboard']:
                assert Path(saved_paths[key]).exists()
                
            # Load artifacts
            loaded = load_cv_artifacts(tmpdir, horizon=4)
            
            # Verify loaded data matches original
            pd.testing.assert_frame_equal(
                loaded['cv_results'].reset_index(drop=True),
                cv_artifacts['cv_df'].reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                loaded['metrics'].reset_index(drop=True),
                cv_artifacts['metrics_df'].reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                loaded['leaderboard'].reset_index(drop=True),
                cv_artifacts['leaderboard_df'].reset_index(drop=True)
            )
            
    def test_versioned_artifacts(self, cv_artifacts):
        """Test versioned artifact saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple versions
            for version in ['v1', 'v2', 'v3']:
                output_dir = Path(tmpdir) / version
                output_dir.mkdir(exist_ok=True)
                
                # Modify metrics slightly for each version
                cv_artifacts['metrics_df']['sCRPS'] *= (1 + 0.1 * int(version[-1]))
                
                save_cv_artifacts(
                    cv_df=cv_artifacts['cv_df'],
                    metrics_df=cv_artifacts['metrics_df'],
                    leaderboard_df=cv_artifacts['leaderboard_df'],
                    output_dir=str(output_dir),
                    horizon=4
                )
                
            # Load each version and verify
            for version in ['v1', 'v2', 'v3']:
                loaded = load_cv_artifacts(str(Path(tmpdir) / version), horizon=4)
                assert loaded is not None
                assert 'metrics' in loaded
                
    def test_partial_artifact_loading(self, cv_artifacts):
        """Test loading when some artifacts are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save only metrics and leaderboard (no CV results)
            metrics_path = Path(tmpdir) / 'h4_metrics.csv'
            leaderboard_path = Path(tmpdir) / 'h4_leaderboard.csv'
            
            cv_artifacts['metrics_df'].to_csv(metrics_path, index=False)
            cv_artifacts['leaderboard_df'].to_csv(leaderboard_path, index=False)
            
            # Try to load
            loaded = load_cv_artifacts(tmpdir, horizon=4)
            
            # Should load what's available
            assert 'metrics' in loaded
            assert 'leaderboard' in loaded
            # CV results might be None or missing
            

class TestErrorRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_cv_partial_failure_recovery(self):
        """Test recovery from partial CV failures."""
        # Create mock NF that fails on second window
        mock_nf = Mock()
        
        call_count = 0
        def cross_val_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated failure on window 2")
            return pd.DataFrame({
                'unique_id': ['BTC'] * 10,
                'ds': pd.date_range('2024-01-01', periods=10, freq='15min'),
                'y': np.random.randn(10),
                'TestModel': np.random.randn(10)
            })
            
        mock_nf.cross_validation.side_effect = cross_val_side_effect
        mock_nf.models = [Mock(__class__=Mock(__name__='TestModel'))]
        
        df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100)
        })
        
        config = {
            'n_windows': 3,
            'h': 4,
            'step_size': 4,
            'val_size': 8,
            'refit': False,
            'verbose': False
        }
        
        # Should handle the error gracefully
        with pytest.raises(RuntimeError):
            run_cv(mock_nf, df, config)
            
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        mock_nf = Mock()
        mock_nf.models = [Mock(__class__=Mock(__name__='TestModel'))]
        
        df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100)
        })
        
        # Invalid config: step_size > data length
        invalid_config = {
            'n_windows': 10,
            'h': 4,
            'step_size': 200,  # Larger than data
            'val_size': 8
        }
        
        # Should raise or handle gracefully
        with pytest.raises((ValueError, AssertionError)):
            run_cv(mock_nf, df, invalid_config)
            
    def test_nan_handling_in_metrics(self):
        """Test that metrics handle NaN values properly."""
        # Create CV results with NaN values
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 20,
            'ds': pd.date_range('2024-01-01', periods=20, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=20, freq='15min')[::10].repeat(10),
            'y': np.concatenate([np.random.randn(10), [np.nan] * 10]),
            'Model1': np.concatenate([np.random.randn(10), [np.nan] * 10])
        })
        
        # Add intervals
        cv_df['Model1-lo-80'] = cv_df['Model1'] - 1.28
        cv_df['Model1-hi-80'] = cv_df['Model1'] + 1.28
        
        # Should handle NaN gracefully
        metrics = _compute_cv_metrics(cv_df, ['Model1'], levels=[80])
        
        assert not metrics.empty
        # First window should have valid metrics
        assert not np.isnan(metrics.iloc[0]['MAE'])
        
    def test_empty_window_handling(self):
        """Test handling of empty CV windows."""
        # Create CV results with an empty window
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 10,
            'ds': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'cutoff': ['2024-01-01'] * 5 + ['2024-01-02'] * 0 + ['2024-01-03'] * 5,
            'y': np.random.randn(10),
            'Model1': np.random.randn(10)
        })
        
        # Should handle empty window
        metrics = _compute_cv_metrics(cv_df, ['Model1'], levels=None)
        assert len(metrics) == 2  # Only two windows with data


class TestConfigurationValidation:
    """Test configuration validation and parameter checking."""
    
    def test_cv_config_validation(self):
        """Test CV configuration validation."""
        valid_config = {
            'n_windows': 6,
            'h': 4,
            'step_size': 4,
            'val_size': 16,
            'refit': True,
            'level': [80, 90],
            'verbose': False
        }
        
        # Test with missing required keys
        invalid_config = valid_config.copy()
        del invalid_config['n_windows']
        
        mock_nf = Mock()
        df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100)
        })
        
        with pytest.raises((KeyError, TypeError)):
            run_cv(mock_nf, df, invalid_config)
            
    def test_model_configuration_validation(self):
        """Test model configuration validation."""
        # Invalid model config: negative input_size
        with pytest.raises((ValueError, AssertionError)):
            model = NHITS(
                h=4,
                input_size=-10,  # Invalid
                max_steps=100
            )
            
        # Invalid loss configuration
        with pytest.raises((ValueError, TypeError)):
            model = NHITS(
                h=4,
                input_size=24,
                loss="InvalidLoss"  # Should be a loss object
            )
            
    def test_horizon_consistency(self):
        """Test that horizon is consistent across configurations."""
        models = [
            NHITS(h=4, input_size=24, max_steps=50),
            NBEATSx(h=8, input_size=24, max_steps=50)  # Different horizon
        ]
        
        # NF should handle or warn about inconsistent horizons
        nf = NeuralForecast(models=models, freq='15min')
        
        # This might raise or handle differently
        # The behavior depends on NF implementation


class TestPerformanceBenchmarks:
    """Test performance benchmarks and optimization."""
    
    def test_cv_execution_time(self):
        """Test that CV execution meets time targets."""
        # Create small realistic dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': pd.date_range('2024-01-01', periods=200, freq='15min'),
            'y': np.random.randn(200)
        })
        
        # Simple model for speed
        models = [
            NHITS(h=4, input_size=24, max_steps=10, random_seed=42)
        ]
        
        nf = NeuralForecast(models=models, freq='15min')
        nf.fit(df=df, val_size=24)
        
        config = {
            'n_windows': 2,
            'h': 4,
            'step_size': 4,
            'val_size': 8,
            'refit': False,
            'verbose': False
        }
        
        start = time.time()
        cv_results = run_cv(nf, df, config)
        cv_time = time.time() - start
        
        # Should complete quickly for small dataset
        assert cv_time < 30, f"CV took too long: {cv_time:.2f}s"
        
    def test_metrics_computation_performance(self):
        """Test metrics computation performance on large datasets."""
        # Create large CV results
        n_samples = 100000
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'cutoff': np.repeat(pd.date_range('2024-01-01', periods=10, freq='D'), n_samples // 10),
            'y': np.random.randn(n_samples)
        })
        
        # Add 5 models with intervals
        for i in range(5):
            model_name = f'Model{i+1}'
            cv_df[model_name] = cv_df['y'] + np.random.randn(n_samples) * 0.1
            for level in [80, 90, 95]:
                z = stats.norm.ppf((1 + level/100) / 2)
                cv_df[f'{model_name}-lo-{level}'] = cv_df[model_name] - z * 0.1
                cv_df[f'{model_name}-hi-{level}'] = cv_df[model_name] + z * 0.1
                
        models = [f'Model{i+1}' for i in range(5)]
        
        start = time.time()
        metrics = _compute_cv_metrics(cv_df, models, levels=[80, 90, 95])
        metrics_time = time.time() - start
        
        # Should handle large datasets efficiently
        assert metrics_time < 5, f"Metrics computation too slow: {metrics_time:.2f}s"
        
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        n_samples = 50000
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples)
        })
        
        # Add multiple models
        for i in range(10):
            cv_df[f'Model{i+1}'] = np.random.randn(n_samples)
            
        # Compute metrics
        models = [f'Model{i+1}' for i in range(10)]
        metrics = compute_metrics_from_cv(cv_df, models)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"


class TestRealDataScenarios:
    """Test with realistic data scenarios and edge cases."""
    
    def test_with_missing_data(self):
        """Test handling of missing data points."""
        # Create data with gaps
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        # Remove some dates to create gaps
        dates = dates[~dates.isin(dates[40:45])]
        
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': dates,
            'y': np.random.randn(len(dates))
        })
        
        # This should be caught by validation
        with pytest.raises(AssertionError):
            assert_regular_grid(df, freq='15min')
            
    def test_with_extreme_values(self):
        """Test handling of extreme values and outliers."""
        np.random.seed(42)
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100)
        })
        
        # Add extreme outliers
        df.loc[10, 'y'] = 100  # Extreme positive
        df.loc[50, 'y'] = -100  # Extreme negative
        
        # Models should handle outliers (robust scaler)
        model = NHITS(
            h=4,
            input_size=24,
            max_steps=10,
            scaler_type='robust',
            random_seed=42
        )
        
        nf = NeuralForecast(models=[model], freq='15min')
        
        # Should not crash
        nf.fit(df=df, val_size=10)
        predictions = nf.predict()
        
        assert not predictions['NHITS'].isna().any()
        
    def test_with_different_horizons(self):
        """Test CV with different horizon configurations."""
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': pd.date_range('2024-01-01', periods=500, freq='15min'),
            'y': np.random.randn(500)
        })
        
        horizons = [4, 8, 16, 32]
        results = {}
        
        for h in horizons:
            model = NHITS(
                h=h,
                input_size=h * 6,  # 6x horizon
                max_steps=10,
                random_seed=42
            )
            
            nf = NeuralForecast(models=[model], freq='15min')
            nf.fit(df=df, val_size=h * 2)
            
            config = {
                'n_windows': 2,
                'h': h,
                'step_size': h,
                'val_size': h * 4,
                'refit': False,
                'verbose': False
            }
            
            cv_results = run_cv(nf, df, config)
            results[h] = cv_results
            
        # Verify all horizons completed
        assert all(h in results for h in horizons)
        assert all(len(results[h]) > 0 for h in horizons)


class TestIntegrationWithOtherModules:
    """Test integration with other system modules."""
    
    @patch('utils.io.Path.mkdir')
    @patch('pandas.DataFrame.to_csv')
    def test_integration_with_io_module(self, mock_to_csv, mock_mkdir):
        """Test integration with I/O utilities."""
        cv_df = pd.DataFrame({'test': [1, 2, 3]})
        metrics_df = pd.DataFrame({'metric': ['sCRPS'], 'value': [0.5]})
        leaderboard_df = pd.DataFrame({'model': ['TestModel'], 'rank': [1]})
        
        paths = save_cv_artifacts(
            cv_df=cv_df,
            metrics_df=metrics_df,
            leaderboard_df=leaderboard_df,
            output_dir='/tmp/test',
            horizon=4
        )
        
        # Verify save was called
        assert mock_to_csv.called
        assert 'cv_results' in paths
        
    def test_integration_with_validation_module(self):
        """Test integration with validation utilities."""
        # Create data that violates validation
        df = pd.DataFrame({
            'unique_id': 'BTC',
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100),
            'feature': np.random.randn(100)  # Not shifted!
        })
        
        # Should fail validation
        with pytest.raises(AssertionError):
            assert_shifted(df, ['feature'])
            
    def test_integration_with_ensemble_module(self):
        """Test integration with ensemble creation."""
        # Create CV results
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100),
            'Model1': np.random.randn(100),
            'Model2': np.random.randn(100),
            'Model3': np.random.randn(100)
        })
        
        # Create ensemble
        ensemble_df = create_equal_weight_ensemble(
            cv_df,
            models=['Model1', 'Model2'],
            name='TopEnsemble'
        )
        
        assert 'TopEnsemble' in ensemble_df.columns
        
        # Ensemble predictions
        pred_df = pd.DataFrame({
            'Model1': np.random.randn(10),
            'Model2': np.random.randn(10),
            'Model3': np.random.randn(10)
        })
        
        ensemble_pred = ensemble_predictions(
            pred_df,
            models=['Model1', 'Model3'],
            weights=[0.6, 0.4]
        )
        
        assert len(ensemble_pred) == 10


# Test runner with coverage reporting
if __name__ == "__main__":
    # Run with coverage if available
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()
        
        pytest.main([__file__, "-v", "--tb=short"])
        
        cov.stop()
        cov.save()
        
        # Generate coverage report
        print("\nCoverage Report:")
        cov.report()
        
    except ImportError:
        # Run without coverage
        pytest.main([__file__, "-v", "--tb=short"])