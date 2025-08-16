"""
Comprehensive unit tests for CV and metrics modules.

This test suite covers all metrics computation, coverage analysis, PIT uniformity,
and model selection functionality with >90% coverage target.
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
from scipy import stats

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from cv.runner import run_cv, summarize_cv, _validate_cv_results, _compute_cv_metrics
from uq.metrics import (
    compute_scrps, compute_mae, compute_rmse, compute_bias,
    compute_metrics_from_cv, aggregate_metrics, rank_models_by_scrps
)
from uq.calibration import (
    compute_coverage, compute_pit, check_pit_uniformity,
    _compute_coverage_for_level
)
from utils.io import save_cv_artifacts, load_cv_artifacts


class TestMetricsComputation:
    """Test suite for metrics computation functions."""
    
    def test_compute_mae_basic(self):
        """Test MAE computation with simple cases."""
        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mae = compute_mae(y_true, y_pred)
        assert mae == pytest.approx(0.0), "MAE should be 0 for perfect predictions"
        
        # Known error case
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        mae = compute_mae(y_true, y_pred)
        assert mae == pytest.approx(1.0), "MAE should be 1.0 for constant +1 error"
        
    def test_compute_mae_with_nan(self):
        """Test MAE handling of NaN values."""
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.0, np.nan, 5.1])
        mae = compute_mae(y_true, y_pred)
        # Should ignore NaN pairs
        assert not np.isnan(mae), "MAE should handle NaN values"
        assert mae == pytest.approx(0.1, abs=0.01), "MAE should ignore NaN pairs"
        
    def test_compute_rmse_basic(self):
        """Test RMSE computation with simple cases."""
        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmse = compute_rmse(y_true, y_pred)
        assert rmse == pytest.approx(0.0), "RMSE should be 0 for perfect predictions"
        
        # Known error case
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        rmse = compute_rmse(y_true, y_pred)
        assert rmse == pytest.approx(1.0), "RMSE should be 1.0 for constant unit error"
        
    def test_rmse_ge_mae(self):
        """Test that RMSE >= MAE always holds."""
        np.random.seed(42)
        for _ in range(10):
            y_true = np.random.randn(100)
            y_pred = y_true + np.random.randn(100) * 0.5
            mae = compute_mae(y_true, y_pred)
            rmse = compute_rmse(y_true, y_pred)
            assert rmse >= mae - 1e-10, "RMSE should always be >= MAE"
            
    def test_compute_bias_basic(self):
        """Test bias computation."""
        # No bias
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bias = compute_bias(y_true, y_pred)
        assert bias == pytest.approx(0.0), "Bias should be 0 for perfect predictions"
        
        # Systematic overestimation
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        bias = compute_bias(y_true, y_pred)
        assert bias == pytest.approx(1.0), "Bias should be 1.0 for constant +1 predictions"
        
        # Systematic underestimation
        y_true = np.array([2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        bias = compute_bias(y_true, y_pred)
        assert bias == pytest.approx(-1.0), "Bias should be -1.0 for constant -1 predictions"
        
    def test_compute_scrps_point_predictions(self):
        """Test sCRPS computation with point predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        scrps = compute_scrps(y_true, y_pred)
        assert scrps > 0, "sCRPS should be positive for imperfect predictions"
        assert scrps < 1.0, "sCRPS should be scaled (< 1 for reasonable errors)"
        
    def test_compute_scrps_quantile_predictions(self):
        """Test sCRPS computation with quantile predictions."""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randn(n_samples)
        # Create quantile predictions (10%, 50%, 90%)
        quantiles = np.array([0.1, 0.5, 0.9])
        y_pred = np.zeros((n_samples, 3))
        y_pred[:, 0] = y_true - 1.28  # 10th percentile
        y_pred[:, 1] = y_true + np.random.randn(n_samples) * 0.1  # median with small error
        y_pred[:, 2] = y_true + 1.28  # 90th percentile
        
        scrps = compute_scrps(y_true, y_pred, quantiles=quantiles)
        assert scrps > 0, "sCRPS should be positive"
        assert not np.isnan(scrps), "sCRPS should not be NaN"
        
    def test_compute_scrps_distributional(self):
        """Test sCRPS computation with distributional predictions."""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randn(n_samples)
        y_pred = y_true + np.random.randn(n_samples) * 0.1  # Small error
        
        # StudentT distribution parameters
        distribution_params = {
            "loc": y_pred,
            "scale": np.ones(n_samples) * 0.5,
            "df": np.ones(n_samples) * 5.0
        }
        
        scrps = compute_scrps(
            y_true, y_pred,
            distribution="StudentT",
            distribution_params=distribution_params
        )
        
        assert scrps > 0, "sCRPS should be positive"
        assert not np.isnan(scrps), "sCRPS should not be NaN"
        assert scrps < 1.0, "sCRPS should be reasonable for small errors"


class TestCoverageAnalysis:
    """Test suite for coverage and calibration analysis."""
    
    def test_compute_coverage_basic(self):
        """Test basic coverage computation."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create CV results with known coverage
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples),
            'Model1': np.random.randn(n_samples)
        })
        
        # Add prediction intervals with approximately correct coverage
        # 80% interval: z-score ≈ 1.28
        cv_df['Model1-lo-80'] = cv_df['y'] - 1.28
        cv_df['Model1-hi-80'] = cv_df['y'] + 1.28
        
        # 90% interval: z-score ≈ 1.645
        cv_df['Model1-lo-90'] = cv_df['y'] - 1.645
        cv_df['Model1-hi-90'] = cv_df['y'] + 1.645
        
        # 95% interval: z-score ≈ 1.96
        cv_df['Model1-lo-95'] = cv_df['y'] - 1.96
        cv_df['Model1-hi-95'] = cv_df['y'] + 1.96
        
        coverage_df = compute_coverage(cv_df, ['Model1'], levels=[80, 90, 95])
        
        assert len(coverage_df) == 3, "Should have 3 coverage levels"
        
        # Check that empirical coverage is close to nominal
        for _, row in coverage_df.iterrows():
            nominal = row['level'] / 100
            empirical = row['empirical_coverage']
            assert abs(empirical - nominal) < 0.05, \
                f"Coverage should be close to nominal (got {empirical:.3f} vs {nominal:.3f})"
                
    def test_compute_coverage_multiple_models(self):
        """Test coverage computation with multiple models."""
        np.random.seed(42)
        n_samples = 500
        
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples)
        })
        
        # Add two models with different coverage characteristics
        for model in ['Model1', 'Model2']:
            cv_df[model] = cv_df['y'] + np.random.randn(n_samples) * 0.1
            
            # Model1: correct coverage, Model2: too narrow intervals
            multiplier = 1.0 if model == 'Model1' else 0.5
            cv_df[f'{model}-lo-80'] = cv_df[model] - 1.28 * multiplier
            cv_df[f'{model}-hi-80'] = cv_df[model] + 1.28 * multiplier
            cv_df[f'{model}-lo-90'] = cv_df[model] - 1.645 * multiplier
            cv_df[f'{model}-hi-90'] = cv_df[model] + 1.645 * multiplier
            
        coverage_df = compute_coverage(cv_df, ['Model1', 'Model2'], levels=[80, 90])
        
        assert len(coverage_df) == 4, "Should have 2 models × 2 levels"
        
        # Model1 should have better coverage than Model2
        model1_cov = coverage_df[coverage_df['model'] == 'Model1']['empirical_coverage'].mean()
        model2_cov = coverage_df[coverage_df['model'] == 'Model2']['empirical_coverage'].mean()
        assert model1_cov > model2_cov, "Model1 should have better coverage"
        
    def test_coverage_within_tolerance(self):
        """Test that coverage is within ±2% tolerance."""
        np.random.seed(42)
        n_samples = 10000  # Large sample for accurate coverage
        
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples),
            'Model1': np.random.randn(n_samples)
        })
        
        # Add perfect intervals
        cv_df['Model1-lo-80'] = cv_df['y'] - stats.norm.ppf(0.9)
        cv_df['Model1-hi-80'] = cv_df['y'] + stats.norm.ppf(0.9)
        cv_df['Model1-lo-90'] = cv_df['y'] - stats.norm.ppf(0.95)
        cv_df['Model1-hi-90'] = cv_df['y'] + stats.norm.ppf(0.95)
        cv_df['Model1-lo-95'] = cv_df['y'] - stats.norm.ppf(0.975)
        cv_df['Model1-hi-95'] = cv_df['y'] + stats.norm.ppf(0.975)
        
        coverage_df = compute_coverage(cv_df, ['Model1'], levels=[80, 90, 95])
        
        for _, row in coverage_df.iterrows():
            nominal = row['level'] / 100
            empirical = row['empirical_coverage']
            tolerance = 0.02  # ±2% as per requirements
            assert abs(empirical - nominal) <= tolerance, \
                f"Coverage {empirical:.3f} not within ±2% of nominal {nominal:.3f}"


class TestPITAnalysis:
    """Test suite for Probability Integral Transform analysis."""
    
    def test_compute_pit_uniform(self):
        """Test PIT computation for uniform distribution."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate uniform PIT values (perfect calibration)
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples)
        })
        
        # Create perfectly calibrated quantile predictions
        models = ['Model1']
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for q in quantiles:
            cv_df[f'Model1-q{int(q*100)}'] = stats.norm.ppf(q, cv_df['y'], 1)
            
        pit_values = compute_pit(cv_df, models[0], mode='quantile', quantiles=quantiles)
        
        assert len(pit_values) == n_samples, "PIT should have one value per sample"
        assert all(0 <= p <= 1 for p in pit_values), "PIT values should be in [0, 1]"
        
    def test_compute_pit_distributional(self):
        """Test PIT computation for distributional models."""
        np.random.seed(42)
        n_samples = 500
        
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples),
            'Model1': np.random.randn(n_samples)
        })
        
        # Add distribution parameters
        cv_df['Model1-scale'] = np.ones(n_samples) * 0.5
        cv_df['Model1-df'] = np.ones(n_samples) * 5.0
        
        pit_values = compute_pit(cv_df, 'Model1', mode='distribution', distribution='StudentT')
        
        assert len(pit_values) == n_samples, "PIT should have one value per sample"
        assert all(0 <= p <= 1 for p in pit_values), "PIT values should be in [0, 1]"
        
    def test_check_pit_uniformity(self):
        """Test PIT uniformity checking."""
        # Test uniform distribution (should pass)
        np.random.seed(42)
        uniform_pit = np.random.uniform(0, 1, 1000)
        is_uniform, ks_stat, p_value = check_pit_uniformity(uniform_pit)
        
        assert p_value > 0.05, "Uniform PIT should pass KS test"
        assert is_uniform, "Should detect uniform PIT"
        
        # Test non-uniform distribution (should fail)
        non_uniform_pit = np.random.beta(2, 5, 1000)  # Skewed distribution
        is_uniform, ks_stat, p_value = check_pit_uniformity(non_uniform_pit)
        
        assert p_value < 0.05, "Non-uniform PIT should fail KS test"
        assert not is_uniform, "Should detect non-uniform PIT"


class TestMetricsAggregation:
    """Test suite for metrics aggregation and ranking."""
    
    def test_aggregate_metrics_mean(self):
        """Test metrics aggregation with mean."""
        metrics_df = pd.DataFrame({
            'model': ['Model1', 'Model1', 'Model1', 'Model2', 'Model2', 'Model2'],
            'window': [0, 1, 2, 0, 1, 2],
            'sCRPS': [0.1, 0.12, 0.11, 0.15, 0.14, 0.16],
            'MAE': [0.5, 0.52, 0.51, 0.6, 0.58, 0.62]
        })
        
        agg_df = aggregate_metrics(metrics_df, method='mean')
        
        assert len(agg_df) == 2, "Should have one row per model"
        assert agg_df.loc[agg_df['model'] == 'Model1', 'sCRPS'].values[0] == pytest.approx(0.11, abs=0.001)
        assert agg_df.loc[agg_df['model'] == 'Model2', 'sCRPS'].values[0] == pytest.approx(0.15, abs=0.001)
        
    def test_aggregate_metrics_median(self):
        """Test metrics aggregation with median."""
        metrics_df = pd.DataFrame({
            'model': ['Model1'] * 5,
            'window': range(5),
            'sCRPS': [0.1, 0.11, 0.12, 0.13, 0.2],  # 0.2 is outlier
            'MAE': [0.5, 0.51, 0.52, 0.53, 1.0]
        })
        
        agg_mean = aggregate_metrics(metrics_df, method='mean')
        agg_median = aggregate_metrics(metrics_df, method='median')
        
        # Median should be less affected by outlier
        assert agg_median['sCRPS'].values[0] < agg_mean['sCRPS'].values[0]
        assert agg_median['sCRPS'].values[0] == pytest.approx(0.12, abs=0.001)
        
    def test_rank_models_by_scrps(self):
        """Test model ranking by sCRPS."""
        metrics_df = pd.DataFrame({
            'model': ['Model1', 'Model2', 'Model3'],
            'sCRPS': [0.15, 0.10, 0.12],
            'MAE': [0.6, 0.5, 0.55],
            'coverage_80': [0.79, 0.81, 0.80]
        })
        
        ranked_df = rank_models_by_scrps(metrics_df)
        
        assert len(ranked_df) == 3, "Should have all models"
        assert ranked_df.iloc[0]['model'] == 'Model2', "Model2 should rank first (lowest sCRPS)"
        assert ranked_df.iloc[1]['model'] == 'Model3', "Model3 should rank second"
        assert ranked_df.iloc[2]['model'] == 'Model1', "Model1 should rank third"
        assert list(ranked_df['rank']) == [1, 2, 3], "Ranks should be 1, 2, 3"


class TestCVRunner:
    """Test suite for CV runner functionality."""
    
    @pytest.fixture
    def mock_nf(self):
        """Create a mock NeuralForecast instance."""
        mock = Mock()
        mock.models = [Mock(__class__=Mock(__name__='NHITS'))]
        mock.cross_validation = Mock(return_value=pd.DataFrame({
            'unique_id': ['BTC'] * 10,
            'ds': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'y': np.random.randn(10),
            'NHITS': np.random.randn(10)
        }))
        return mock
        
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'y': np.random.randn(100)
        })
        
    @pytest.fixture
    def cv_config(self):
        """Create a sample CV configuration."""
        return {
            'n_windows': 6,
            'h': 4,
            'step_size': 4,
            'val_size': 16,
            'refit': True,
            'level': [80, 90],
            'verbose': False
        }
        
    def test_validate_cv_results(self, mock_nf):
        """Test CV results validation."""
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 10,
            'ds': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'y': np.random.randn(10),
            'NHITS': np.random.randn(10)
        })
        
        # Should not raise
        _validate_cv_results(cv_df, mock_nf.models, level=None)
        
        # Test with missing column
        cv_df_bad = cv_df.drop(columns=['NHITS'])
        with pytest.raises(ValueError, match="missing predictions"):
            _validate_cv_results(cv_df_bad, mock_nf.models, level=None)
            
    def test_compute_cv_metrics(self):
        """Test CV metrics computation."""
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=100, freq='15min')[::10].repeat(10),
            'y': np.random.randn(100),
            'Model1': np.random.randn(100),
            'Model1-lo-80': np.random.randn(100) - 1.28,
            'Model1-hi-80': np.random.randn(100) + 1.28
        })
        
        metrics_df = _compute_cv_metrics(cv_df, ['Model1'], levels=[80])
        
        assert 'model' in metrics_df.columns
        assert 'window' in metrics_df.columns
        assert 'sCRPS' in metrics_df.columns
        assert 'MAE' in metrics_df.columns
        assert 'coverage_80' in metrics_df.columns
        
        # Check that we have metrics for each window
        n_windows = cv_df['cutoff'].nunique()
        assert len(metrics_df) == n_windows
        
    def test_summarize_cv(self):
        """Test CV summarization."""
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * 100,
            'ds': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=100, freq='15min')[::10].repeat(10),
            'y': np.random.randn(100)
        })
        
        # Add two models
        for model in ['Model1', 'Model2']:
            cv_df[model] = cv_df['y'] + np.random.randn(100) * 0.1
            cv_df[f'{model}-lo-80'] = cv_df[model] - 1.28
            cv_df[f'{model}-hi-80'] = cv_df[model] + 1.28
            
        summary = summarize_cv(cv_df, horizon=4, levels=[80])
        
        assert 'metrics' in summary
        assert 'leaderboard' in summary
        assert 'best_model' in summary
        
        # Check leaderboard structure
        lb = summary['leaderboard']
        assert len(lb) == 2, "Should have 2 models"
        assert 'rank' in lb.columns
        assert 'model' in lb.columns
        assert 'sCRPS' in lb.columns
        
        # Best model should match rank 1
        assert summary['best_model'] == lb[lb['rank'] == 1]['model'].values[0]
        
    @patch('cv.runner.NeuralForecast')
    def test_run_cv_with_conformal(self, mock_nf_class, sample_df, cv_config):
        """Test CV execution with conformal prediction."""
        # Mock the NF instance
        mock_nf = Mock()
        mock_nf.models = [Mock(__class__=Mock(__name__='NHITS'))]
        
        # Mock CV results
        cv_result = pd.DataFrame({
            'unique_id': ['BTC'] * 10,
            'ds': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'cutoff': pd.date_range('2024-01-01', periods=10, freq='15min'),
            'y': np.random.randn(10),
            'NHITS': np.random.randn(10)
        })
        
        # Add prediction intervals
        for level in [80, 90]:
            cv_result[f'NHITS-lo-{level}'] = cv_result['NHITS'] - stats.norm.ppf((1 + level/100) / 2)
            cv_result[f'NHITS-hi-{level}'] = cv_result['NHITS'] + stats.norm.ppf((1 + level/100) / 2)
            
        mock_nf.cross_validation.return_value = cv_result
        
        # Run CV with conformal
        result = run_cv(mock_nf, sample_df, cv_config, use_conformal=True)
        
        assert isinstance(result, pd.DataFrame)
        assert 'NHITS' in result.columns
        assert 'NHITS-lo-80' in result.columns
        assert 'NHITS-hi-80' in result.columns


class TestIOFunctions:
    """Test suite for I/O operations."""
    
    def test_save_cv_artifacts(self):
        """Test saving CV artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cv_df = pd.DataFrame({'test': [1, 2, 3]})
            metrics_df = pd.DataFrame({'metric': ['sCRPS'], 'value': [0.5]})
            leaderboard_df = pd.DataFrame({'model': ['TestModel'], 'rank': [1]})
            
            paths = save_cv_artifacts(
                cv_df=cv_df,
                metrics_df=metrics_df,
                leaderboard_df=leaderboard_df,
                output_dir=tmpdir,
                horizon=4
            )
            
            assert 'cv_results' in paths
            assert 'metrics' in paths
            assert 'leaderboard' in paths
            
            # Check files exist
            assert Path(paths['cv_results']).exists()
            assert Path(paths['metrics']).exists()
            assert Path(paths['leaderboard']).exists()
            
    def test_load_cv_artifacts(self):
        """Test loading CV artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save artifacts first
            cv_df = pd.DataFrame({'test': [1, 2, 3]})
            metrics_df = pd.DataFrame({'metric': ['sCRPS'], 'value': [0.5]})
            leaderboard_df = pd.DataFrame({'model': ['TestModel'], 'rank': [1]})
            
            save_cv_artifacts(
                cv_df=cv_df,
                metrics_df=metrics_df,
                leaderboard_df=leaderboard_df,
                output_dir=tmpdir,
                horizon=4
            )
            
            # Load artifacts
            loaded = load_cv_artifacts(tmpdir, horizon=4)
            
            assert 'cv_results' in loaded
            assert 'metrics' in loaded
            assert 'leaderboard' in loaded
            
            # Check data integrity
            pd.testing.assert_frame_equal(loaded['metrics'], metrics_df)
            pd.testing.assert_frame_equal(loaded['leaderboard'], leaderboard_df)
            
    def test_load_missing_artifacts(self):
        """Test loading with missing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to load non-existent artifacts
            loaded = load_cv_artifacts(tmpdir, horizon=4)
            
            # Should return empty dict or handle gracefully
            assert loaded is not None


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            compute_coverage(empty_df, ['Model1'], levels=[80])
            
    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        df = pd.DataFrame({
            'unique_id': ['BTC'] * 10,
            'ds': pd.date_range('2024-01-01', periods=10, freq='15min')
            # Missing 'y' column
        })
        
        with pytest.raises((ValueError, KeyError)):
            compute_coverage(df, ['Model1'], levels=[80])
            
    def test_invalid_quantiles(self):
        """Test handling of invalid quantile values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        
        # Invalid quantiles (should be in [0, 1])
        with pytest.raises(ValueError):
            compute_scrps(y_true, y_pred, quantiles=np.array([0.1, 1.5]))
            
    def test_mismatched_array_sizes(self):
        """Test handling of mismatched array sizes."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different size
        
        with pytest.raises(ValueError):
            compute_mae(y_true, y_pred)
            
    def test_all_nan_handling(self):
        """Test handling of all-NaN arrays."""
        y_true = np.array([np.nan, np.nan, np.nan])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        mae = compute_mae(y_true, y_pred)
        assert np.isnan(mae) or mae == 0, "Should handle all-NaN gracefully"


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarks."""
    
    def test_metrics_computation_speed(self):
        """Test that metrics computation meets performance targets."""
        import time
        
        # Create large dataset
        n_samples = 10000
        y_true = np.random.randn(n_samples)
        y_pred = y_true + np.random.randn(n_samples) * 0.1
        
        # Test MAE speed
        start = time.time()
        mae = compute_mae(y_true, y_pred)
        mae_time = time.time() - start
        assert mae_time < 0.1, f"MAE computation too slow: {mae_time:.3f}s"
        
        # Test RMSE speed
        start = time.time()
        rmse = compute_rmse(y_true, y_pred)
        rmse_time = time.time() - start
        assert rmse_time < 0.1, f"RMSE computation too slow: {rmse_time:.3f}s"
        
        # Test sCRPS speed
        start = time.time()
        scrps = compute_scrps(y_true, y_pred)
        scrps_time = time.time() - start
        assert scrps_time < 0.5, f"sCRPS computation too slow: {scrps_time:.3f}s"
        
    def test_coverage_computation_speed(self):
        """Test that coverage computation is efficient."""
        import time
        
        # Create large CV results
        n_samples = 50000
        cv_df = pd.DataFrame({
            'unique_id': ['BTC'] * n_samples,
            'ds': pd.date_range('2024-01-01', periods=n_samples, freq='15min'),
            'y': np.random.randn(n_samples)
        })
        
        # Add model predictions and intervals
        for model in ['Model1', 'Model2', 'Model3']:
            cv_df[model] = cv_df['y'] + np.random.randn(n_samples) * 0.1
            for level in [80, 90, 95]:
                z = stats.norm.ppf((1 + level/100) / 2)
                cv_df[f'{model}-lo-{level}'] = cv_df[model] - z
                cv_df[f'{model}-hi-{level}'] = cv_df[model] + z
                
        start = time.time()
        coverage_df = compute_coverage(cv_df, ['Model1', 'Model2', 'Model3'], levels=[80, 90, 95])
        coverage_time = time.time() - start
        
        assert coverage_time < 1.0, f"Coverage computation too slow: {coverage_time:.3f}s"
        assert len(coverage_df) == 9, "Should have 3 models × 3 levels"


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])