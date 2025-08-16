"""
Calibration analysis module for uncertainty quantification.

This module provides functions for computing coverage, PIT (Probability Integral Transform),
and other calibration diagnostics for probabilistic forecasts.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_coverage(cv_df: pd.DataFrame,
                    models: List[str],
                    levels: List[int] = [80, 90, 95]) -> pd.DataFrame:
    """
    Compute empirical coverage at specified confidence levels.
    
    Following docs/forecasting_sf_plan.md lines 1661-1665 and 1717-1726,
    this function calculates the fraction of actuals that fall within
    predicted intervals and compares against nominal rates with ±2% tolerance.
    
    Args:
        cv_df: CV results with interval predictions
        models: List of model names to evaluate
        levels: Confidence levels to evaluate (default: [80, 90, 95])
        
    Returns:
        DataFrame with columns:
        - model: Model name
        - level: Confidence level
        - nominal_coverage: Expected coverage (e.g., 0.80 for 80%)
        - empirical_coverage: Actual coverage from CV
        - deviation: Difference from nominal
        - calibrated: Whether within ±2% tolerance
        - miscalibration_flag: String describing calibration issue if present
    """
    coverage_results = []
    
    for model_name in models:
        # Check if model predictions exist
        if model_name not in cv_df.columns:
            logger.warning(f"Model {model_name} not found in CV results")
            continue
        
        for level in levels:
            # Construct interval column names following NF convention
            lo_col = f"{model_name}-lo-{level}"
            hi_col = f"{model_name}-hi-{level}"
            
            # Check if interval columns exist
            if lo_col not in cv_df.columns or hi_col not in cv_df.columns:
                logger.warning(f"Interval columns not found for {model_name} at level {level}")
                continue
            
            # Compute coverage - calculate empirical hit rate
            y_true = cv_df['y'].values
            lo_pred = cv_df[lo_col].values
            hi_pred = cv_df[hi_col].values
            
            # Handle NaN values
            mask = ~(np.isnan(y_true) | np.isnan(lo_pred) | np.isnan(hi_pred))
            if not mask.any():
                logger.warning(f"No valid samples for {model_name} at level {level}")
                continue
            
            y_true_clean = y_true[mask]
            lo_pred_clean = lo_pred[mask]
            hi_pred_clean = hi_pred[mask]
            
            # Calculate empirical coverage (hit rate)
            within_interval = (y_true_clean >= lo_pred_clean) & (y_true_clean <= hi_pred_clean)
            empirical_coverage = float(within_interval.mean())
            
            # Nominal coverage
            nominal_coverage = level / 100.0
            
            # Deviation from nominal in percentage points
            deviation = empirical_coverage - nominal_coverage
            deviation_pp = deviation * 100  # Convert to percentage points
            
            # Check calibration (±2 percentage points tolerance as per spec)
            calibrated = abs(deviation) <= 0.02
            
            # Flag miscalibration with specific message
            if not calibrated:
                if deviation > 0.02:
                    miscalibration_flag = f"Over-coverage by {deviation_pp:.1f}pp"
                else:
                    miscalibration_flag = f"Under-coverage by {abs(deviation_pp):.1f}pp"
            else:
                miscalibration_flag = "Well-calibrated"
            
            coverage_results.append({
                'model': model_name,
                'level': level,
                'nominal_coverage': nominal_coverage,
                'empirical_coverage': empirical_coverage,
                'deviation': deviation,
                'deviation_pp': deviation_pp,
                'calibrated': calibrated,
                'miscalibration_flag': miscalibration_flag,
                'n_samples': len(y_true_clean)
            })
    
    coverage_df = pd.DataFrame(coverage_results)
    
    # Log calibration summary with detailed statistics
    if not coverage_df.empty:
        n_calibrated = coverage_df['calibrated'].sum()
        n_total = len(coverage_df)
        logger.info(f"Coverage analysis: {n_calibrated}/{n_total} model-level combinations calibrated")
        
        # Log specific miscalibration cases
        miscalibrated = coverage_df[~coverage_df['calibrated']]
        if not miscalibrated.empty:
            for _, row in miscalibrated.iterrows():
                logger.warning(f"{row['model']} at {row['level']}%: {row['miscalibration_flag']}")
    
    return coverage_df


def compute_pit(y_true: np.ndarray,
                predictions: Dict[str, np.ndarray],
                model_type: str) -> np.ndarray:
    """
    Compute Probability Integral Transform values.
    
    Following docs/forecasting_sf_plan.md lines 1743-1766 and design.md lines 135-149,
    this function computes PIT values for calibration assessment. For quantile models,
    we use a dense grid: quantiles = [i/100 for i in range(1, 100)].
    
    Args:
        y_true: Actual values
        predictions: Model predictions with keys depending on model_type:
            - For 'quantile': {'quantiles': array, 'values': array}
                             quantiles should be [i/100 for i in range(1, 100)]
            - For 'distributional': {'mean': array, 'std': array, 'df': float}
            - For 'conformal': Not supported (returns empty array)
        model_type: One of 'distributional', 'quantile', or 'conformal'
        
    Returns:
        PIT values for uniformity testing (should be uniform [0,1] if calibrated)
        
    Note:
        Conformal models don't provide insample PIs, so PIT is skipped.
        Per docs/forecasting_sf_plan.md lines 1632, 1642: "Conformal runs do not 
        provide insample PIs" - we rely on CV/test coverage checks instead.
    """
    if model_type == 'conformal':
        # Document why PIT is skipped for conformal models
        logger.info("PIT computation skipped for conformal model - NeuralForecast's conformal "
                   "prediction does not provide insample prediction intervals. "
                   "Calibration assessment relies on CV/test coverage statistics instead.")
        return np.array([])
    
    # Handle NaN values
    mask = ~np.isnan(y_true)
    if not mask.any():
        logger.warning("No valid y_true values for PIT computation")
        return np.array([])
    
    y_true_clean = y_true[mask]
    
    if model_type == 'quantile':
        # Quantile model: interpolate CDF at observed values
        if 'quantiles' not in predictions or 'values' not in predictions:
            logger.error("Quantile predictions must include 'quantiles' and 'values'")
            return np.array([])
        
        quantiles = predictions['quantiles']
        quantile_preds = predictions['values']
        
        # Ensure we're using dense quantile grid
        expected_quantiles = [i/100 for i in range(1, 100)]
        if len(quantiles) != 99:
            logger.warning(f"Expected 99 quantiles for dense grid, got {len(quantiles)}. "
                          f"PIT approximation may be less accurate.")
        
        # Apply same mask to predictions
        if len(quantile_preds.shape) == 2:
            quantile_preds_clean = quantile_preds[mask]
        else:
            quantile_preds_clean = quantile_preds
            
        pit_values = _compute_pit_quantile(y_true_clean, quantile_preds_clean, quantiles)
        
    elif model_type == 'distributional':
        # Distributional model: evaluate CDF at observed values
        if 'mean' not in predictions or 'std' not in predictions:
            logger.error("Distributional predictions must include 'mean' and 'std'")
            return np.array([])
        
        mean_pred = predictions['mean'][mask]
        std_pred = predictions['std'][mask]
        df = predictions.get('df', 10)  # Default degrees of freedom for StudentT
        
        pit_values = _compute_pit_distributional(y_true_clean, mean_pred, std_pred, df)
        
    else:
        logger.error(f"Unknown model type: {model_type}. Must be 'distributional', 'quantile', or 'conformal'")
        return np.array([])
    
    # Validate PIT values are in [0, 1]
    if len(pit_values) > 0:
        if np.any((pit_values < 0) | (pit_values > 1)):
            logger.warning("Some PIT values outside [0, 1] range - clipping")
            pit_values = np.clip(pit_values, 0, 1)
    
    return pit_values


def _compute_pit_quantile(y_true: np.ndarray,
                         quantile_preds: np.ndarray,
                         quantiles: np.ndarray) -> np.ndarray:
    """
    Compute PIT values for quantile predictions using efficient rank-based approximation.
    
    This implements the interpolated quantile rank method for PIT computation,
    which is efficient and suitable for dense quantile grids.
    
    Args:
        y_true: Actual values (n_samples,)
        quantile_preds: Quantile predictions (n_samples, n_quantiles)
        quantiles: Quantile levels (n_quantiles,) - should be [i/100 for i in range(1, 100)]
        
    Returns:
        PIT values using rank-based approximation
    """
    n_samples = len(y_true)
    
    # Ensure quantiles is a numpy array
    if not isinstance(quantiles, np.ndarray):
        quantiles = np.array(quantiles)
    
    # Handle 1D case (single sample)
    if quantile_preds.ndim == 1:
        quantile_preds = quantile_preds.reshape(1, -1)
    
    pit_values = np.zeros(n_samples)
    
    # Vectorized computation for efficiency
    for i in range(n_samples):
        y = y_true[i]
        q_preds = quantile_preds[i]
        
        # Handle edge cases and interpolate
        if np.isnan(y) or np.any(np.isnan(q_preds)):
            pit_values[i] = np.nan
        elif y <= q_preds[0]:
            # Below lowest quantile - linear extrapolation to 0
            pit_values[i] = quantiles[0] * max(0, y / (q_preds[0] + 1e-10))
        elif y >= q_preds[-1]:
            # Above highest quantile - linear extrapolation to 1
            remaining_prob = 1.0 - quantiles[-1]
            excess = (y - q_preds[-1]) / (np.abs(q_preds[-1]) + 1e-10)
            pit_values[i] = quantiles[-1] + remaining_prob * min(1.0, excess * 0.5)
        else:
            # Within quantile range - linear interpolation
            pit_values[i] = np.interp(y, q_preds, quantiles)
    
    return pit_values


def _compute_pit_distributional(y_true: np.ndarray,
                               mean_pred: np.ndarray,
                               std_pred: np.ndarray,
                               df: float = 10) -> np.ndarray:
    """
    Compute PIT values for distributional predictions (StudentT).
    
    For distributional models using StudentT distribution, we evaluate the CDF
    at the observed values to get PIT values.
    
    Args:
        y_true: Actual values
        mean_pred: Predicted means
        std_pred: Predicted standard deviations  
        df: Degrees of freedom for StudentT distribution (default: 10)
        
    Returns:
        PIT values from CDF evaluation
    """
    # Validate inputs
    if len(y_true) != len(mean_pred) or len(y_true) != len(std_pred):
        raise ValueError("Input arrays must have the same length")
    
    # Standardize observations to get z-scores
    # Avoid division by zero with small epsilon
    z_scores = (y_true - mean_pred) / (std_pred + 1e-10)
    
    # Compute CDF values using StudentT distribution
    # PIT values are the CDF evaluated at the actual observations
    pit_values = stats.t.cdf(z_scores, df=df)
    
    # Handle numerical issues
    pit_values = np.clip(pit_values, 0, 1)
    
    return pit_values


def plot_calibration_diagnostics(pit_values: np.ndarray,
                                coverage_df: pd.DataFrame,
                                output_dir: Path,
                                model_name: str,
                                horizon: Optional[int] = None) -> Dict[str, float]:
    """
    Generate calibration diagnostic plots with uniformity assessment.
    
    Creates PIT histogram with 20 bins and coverage reliability diagrams
    as specified in docs/forecasting_sf_plan.md lines 483-490 and 1664.
    
    Args:
        pit_values: PIT values for histogram (should be ~Uniform[0,1] if calibrated)
        coverage_df: Coverage analysis results
        output_dir: Directory for saving plots (e.g., reports/h{horizon}/)
        model_name: Name of the model for labeling
        horizon: Forecast horizon for directory structure (optional)
        
    Returns:
        Dictionary with diagnostic statistics:
        - ks_statistic: Kolmogorov-Smirnov test statistic
        - ks_pvalue: KS test p-value for uniformity
        - calibrated: Boolean indicating if model is well-calibrated
    """
    # Create output directory structure
    if horizon is not None:
        output_dir = Path(output_dir) / f"h{horizon}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize return statistics
    diagnostic_stats = {
        'ks_statistic': np.nan,
        'ks_pvalue': np.nan,
        'calibrated': False
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. PIT Histogram (20 bins as specified)
    ax1 = axes[0]
    if len(pit_values) > 0 and not np.all(np.isnan(pit_values)):
        # Clean PIT values
        pit_clean = pit_values[~np.isnan(pit_values)]
        
        # Create 20-bin histogram showing uniformity
        counts, bins, patches = ax1.hist(pit_clean, bins=20, density=True, 
                                         alpha=0.7, edgecolor='black', color='skyblue')
        
        # Add uniform reference line
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Uniform[0,1]')
        
        # Add confidence band for uniformity (approximate 95% CI)
        n_samples = len(pit_clean)
        expected_density = 1.0
        std_error = np.sqrt(expected_density * (1 - expected_density / 20) / n_samples)
        ax1.fill_between([0, 1], [expected_density - 2*std_error]*2, 
                         [expected_density + 2*std_error]*2,
                         alpha=0.2, color='red', label='95% CI')
        
        ax1.set_xlabel('PIT Value', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'PIT Histogram - {model_name}\n(20 bins for uniformity assessment)', fontsize=12)
        ax1.set_xlim([0, 1])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Perform KS test for uniformity
        ks_stat, ks_pval = stats.kstest(pit_clean, 'uniform')
        diagnostic_stats['ks_statistic'] = ks_stat
        diagnostic_stats['ks_pvalue'] = ks_pval
        
        # Add KS test result with interpretation
        calibration_text = "Well-calibrated" if ks_pval > 0.05 else "Miscalibrated"
        color = 'green' if ks_pval > 0.05 else 'red'
        ax1.text(0.05, 0.95, f'KS test p-value: {ks_pval:.4f}\nStatus: {calibration_text}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color=color, fontweight='bold')
        
        # Add sample size
        ax1.text(0.95, 0.05, f'n = {n_samples}',
                transform=ax1.transAxes, ha='right', fontsize=9)
        
    else:
        ax1.text(0.5, 0.5, 'No PIT values available\n(Conformal models skip PIT)',
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title(f'PIT Histogram - {model_name}')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 2])
    
    # 2. Coverage Reliability Diagram
    ax2 = axes[1]
    if not coverage_df.empty:
        model_coverage = coverage_df[coverage_df['model'] == model_name]
        if not model_coverage.empty:
            nominal = model_coverage['nominal_coverage'].values * 100
            empirical = model_coverage['empirical_coverage'].values * 100
            
            # Plot coverage points
            colors = ['green' if cal else 'red' for cal in model_coverage['calibrated'].values]
            ax2.scatter(nominal, empirical, s=150, alpha=0.8, c=colors, edgecolors='black', linewidth=1)
            
            # Perfect calibration line
            ax2.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect calibration')
            
            # ±2pp tolerance band
            ax2.fill_between([0, 100], [-2, 98], [2, 102], alpha=0.15, color='gray', label='±2pp tolerance')
            
            ax2.set_xlabel('Nominal Coverage (%)', fontsize=11)
            ax2.set_ylabel('Empirical Coverage (%)', fontsize=11)
            ax2.set_title(f'Coverage Reliability - {model_name}', fontsize=12)
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([75, 100])
            ax2.set_ylim([75, 100])
            
            # Annotate points with levels and deviations
            for i, (level, dev_pp) in enumerate(zip(model_coverage['level'].values,
                                                     model_coverage['deviation_pp'].values)):
                sign = '+' if dev_pp > 0 else ''
                ax2.annotate(f'{level}%\n({sign}{dev_pp:.1f}pp)',
                           (nominal[i], empirical[i]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, ha='left')
            
            # Overall calibration status
            n_calibrated = model_coverage['calibrated'].sum()
            n_total = len(model_coverage)
            diagnostic_stats['calibrated'] = (n_calibrated == n_total)
            
            status_text = f"Calibrated: {n_calibrated}/{n_total} levels"
            status_color = 'green' if diagnostic_stats['calibrated'] else 'orange'
            ax2.text(0.05, 0.95, status_text,
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    color=status_color, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No coverage data available',
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title(f'Coverage Reliability - {model_name}')
    else:
        ax2.text(0.5, 0.5, 'No coverage data available',
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title(f'Coverage Reliability - {model_name}')
    
    plt.suptitle(f'Calibration Diagnostics - {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"calibration_diagnostics_{model_name}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved calibration diagnostics to {plot_path}")
    
    return diagnostic_stats


def compute_pit_for_models(cv_df: pd.DataFrame,
                          models: List[str],
                          model_types: Dict[str, str],
                          use_dense_grid: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute PIT values for multiple models using dense quantile grids.
    
    Following design.md lines 135-149, this uses a dense quantile grid
    [i/100 for i in range(1, 100)] for accurate PIT computation.
    
    Args:
        cv_df: Cross-validation results with quantile/interval predictions
        models: List of model names to evaluate
        model_types: Dictionary mapping model names to types
                    ('distributional', 'quantile', 'conformal')
        use_dense_grid: Whether to use dense 99-quantile grid (default: True)
        
    Returns:
        Dictionary mapping model names to PIT value arrays
        
    Note:
        For quantile models, expects columns like: model-q1, model-q2, ..., model-q99
        For distributional models, expects: model_mean, model_std columns
        Conformal models return empty arrays (PIT not applicable insample)
    """
    pit_results = {}
    
    for model_name in models:
        if model_name not in cv_df.columns:
            logger.warning(f"Model {model_name} not found in CV results")
            pit_results[model_name] = np.array([])
            continue
        
        model_type = model_types.get(model_name, 'quantile')
        
        if model_type == 'conformal':
            # Skip conformal models with documentation
            logger.info(f"Skipping PIT for {model_name} (conformal) - not available insample")
            pit_results[model_name] = np.array([])
            continue
        
        # Get actual values
        y_true = cv_df['y'].values
        
        if model_type == 'quantile':
            # Use dense quantile grid for accurate PIT
            if use_dense_grid:
                # Dense grid: [i/100 for i in range(1, 100)]
                quantiles = [i/100 for i in range(1, 100)]
            else:
                # Fallback to available quantiles
                quantiles = []
            
            quantile_cols = []
            quantile_values = []
            
            # Check for quantile columns (model-q1 through model-q99)
            for q_level in range(1, 100):
                q_col = f"{model_name}-q{q_level}"
                if q_col in cv_df.columns:
                    quantile_cols.append(q_col)
                    quantile_values.append(q_level / 100.0)
            
            if len(quantile_cols) >= 50:  # Need reasonable density
                predictions = {
                    'quantiles': np.array(quantile_values),
                    'values': cv_df[quantile_cols].values
                }
                pit_values = compute_pit(y_true, predictions, model_type)
                logger.info(f"Computed PIT for {model_name} using {len(quantile_cols)} quantiles")
            else:
                # Try alternative naming convention (model-lo-X, model-hi-X)
                logger.warning(f"Insufficient quantile columns for {model_name} ({len(quantile_cols)} found). "
                              f"Need at least 50 for accurate PIT.")
                pit_values = np.array([])
                
        elif model_type == 'distributional':
            # Look for distribution parameters (StudentT)
            mean_col = f"{model_name}_mean"
            std_col = f"{model_name}_std"
            
            # Alternative naming conventions
            if mean_col not in cv_df.columns:
                mean_col = f"{model_name}"  # Point prediction as mean
            
            if std_col not in cv_df.columns:
                # Try to infer from interval widths if available
                lo_90 = f"{model_name}-lo-90"
                hi_90 = f"{model_name}-hi-90"
                if lo_90 in cv_df.columns and hi_90 in cv_df.columns:
                    # Approximate std from 90% interval (assuming StudentT with df=10)
                    # For StudentT(df=10), 90% interval ≈ ±1.812 std
                    interval_width = cv_df[hi_90] - cv_df[lo_90]
                    std_values = interval_width / (2 * 1.812)
                    
                    predictions = {
                        'mean': cv_df[mean_col].values,
                        'std': std_values.values,
                        'df': 10  # Default StudentT degrees of freedom
                    }
                    pit_values = compute_pit(y_true, predictions, model_type)
                    logger.info(f"Computed PIT for {model_name} using inferred distribution parameters")
                else:
                    logger.warning(f"Distribution parameters not found for {model_name}")
                    pit_values = np.array([])
            else:
                predictions = {
                    'mean': cv_df[mean_col].values,
                    'std': cv_df[std_col].values,
                    'df': 10  # Default StudentT degrees of freedom
                }
                pit_values = compute_pit(y_true, predictions, model_type)
                logger.info(f"Computed PIT for {model_name} using explicit distribution parameters")
        else:
            logger.warning(f"Unknown model type '{model_type}' for {model_name}")
            pit_values = np.array([])
        
        pit_results[model_name] = pit_values
    
    return pit_results


def generate_calibration_report(cv_df: pd.DataFrame,
                               models: List[str],
                               model_types: Dict[str, str],
                               output_dir: Path,
                               horizon: int,
                               levels: List[int] = [80, 90, 95]) -> pd.DataFrame:
    """
    Generate comprehensive calibration report for all models.
    
    This is the main entry point for calibration diagnostics, computing
    coverage, PIT, and generating all visualizations as per specifications.
    
    Args:
        cv_df: Cross-validation results with predictions and intervals
        models: List of model names to evaluate
        model_types: Dictionary mapping model names to types
        output_dir: Base directory for reports (typically 'reports/')
        horizon: Forecast horizon (4, 8, 16, or 32)
        levels: Confidence levels for coverage analysis
        
    Returns:
        Summary DataFrame with calibration metrics for all models
    """
    logger.info(f"Generating calibration report for horizon {horizon}")
    
    # Ensure output directory exists
    horizon_dir = Path(output_dir) / f"h{horizon}"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute coverage for all models
    logger.info("Computing coverage metrics...")
    coverage_df = compute_coverage(cv_df, models, levels)
    
    # Save coverage table
    coverage_path = horizon_dir / "coverage_metrics.csv"
    coverage_df.to_csv(coverage_path, index=False)
    logger.info(f"Saved coverage metrics to {coverage_path}")
    
    # 2. Compute PIT for all models
    logger.info("Computing PIT values...")
    pit_results = compute_pit_for_models(cv_df, models, model_types)
    
    # 3. Generate plots and collect statistics
    calibration_summary = []
    
    for model_name in models:
        model_type = model_types.get(model_name, 'quantile')
        pit_values = pit_results.get(model_name, np.array([]))
        
        # Generate diagnostic plots
        diagnostic_stats = plot_calibration_diagnostics(
            pit_values=pit_values,
            coverage_df=coverage_df,
            output_dir=output_dir,
            model_name=model_name,
            horizon=horizon
        )
        
        # Check interval crossing
        crossing_df = check_interval_crossing(cv_df, model_name, levels)
        has_crossing = crossing_df['has_crossing'].any() if not crossing_df.empty else False
        
        # Aggregate calibration status
        model_coverage = coverage_df[coverage_df['model'] == model_name]
        if not model_coverage.empty:
            coverage_80 = model_coverage[model_coverage['level'] == 80]['empirical_coverage'].values
            coverage_90 = model_coverage[model_coverage['level'] == 90]['empirical_coverage'].values
            coverage_95 = model_coverage[model_coverage['level'] == 95]['empirical_coverage'].values
            
            coverage_80_val = coverage_80[0] if len(coverage_80) > 0 else np.nan
            coverage_90_val = coverage_90[0] if len(coverage_90) > 0 else np.nan
            coverage_95_val = coverage_95[0] if len(coverage_95) > 0 else np.nan
            
            # Check calibration status
            all_calibrated = model_coverage['calibrated'].all()
        else:
            coverage_80_val = coverage_90_val = coverage_95_val = np.nan
            all_calibrated = False
        
        # Compile summary
        summary_row = {
            'model': model_name,
            'model_type': model_type,
            'horizon': horizon,
            'coverage_80': coverage_80_val,
            'coverage_90': coverage_90_val,
            'coverage_95': coverage_95_val,
            'coverage_calibrated': all_calibrated,
            'pit_ks_statistic': diagnostic_stats['ks_statistic'],
            'pit_ks_pvalue': diagnostic_stats['ks_pvalue'],
            'pit_uniform': diagnostic_stats['ks_pvalue'] > 0.05 if not np.isnan(diagnostic_stats['ks_pvalue']) else False,
            'has_interval_crossing': has_crossing,
            'overall_calibrated': all_calibrated and diagnostic_stats.get('calibrated', False) and not has_crossing
        }
        
        calibration_summary.append(summary_row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(calibration_summary)
    
    # Save summary
    summary_path = horizon_dir / "calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved calibration summary to {summary_path}")
    
    # Generate markdown report
    report_path = horizon_dir / "calibration_report.md"
    _write_calibration_report(summary_df, coverage_df, report_path, horizon)
    
    # Log overall status
    n_calibrated = summary_df['overall_calibrated'].sum()
    n_total = len(summary_df)
    logger.info(f"Calibration complete: {n_calibrated}/{n_total} models well-calibrated at horizon {horizon}")
    
    return summary_df


def _write_calibration_report(summary_df: pd.DataFrame,
                             coverage_df: pd.DataFrame,
                             output_path: Path,
                             horizon: int) -> None:
    """Write markdown calibration report."""
    with open(output_path, 'w') as f:
        f.write(f"# Calibration Report - Horizon {horizon}\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        n_calibrated = summary_df['overall_calibrated'].sum()
        n_total = len(summary_df)
        f.write(f"## Summary\n\n")
        f.write(f"- **Models evaluated**: {n_total}\n")
        f.write(f"- **Well-calibrated models**: {n_calibrated}/{n_total}\n")
        f.write(f"- **Coverage levels tested**: 80%, 90%, 95% (±2pp tolerance)\n\n")
        
        # Model-specific results
        f.write("## Model Results\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"### {row['model']}\n\n")
            f.write(f"- **Type**: {row['model_type']}\n")
            f.write(f"- **Coverage**:\n")
            f.write(f"  - 80%: {row['coverage_80']:.1%} {'✓' if abs(row['coverage_80'] - 0.80) <= 0.02 else '✗'}\n")
            f.write(f"  - 90%: {row['coverage_90']:.1%} {'✓' if abs(row['coverage_90'] - 0.90) <= 0.02 else '✗'}\n")
            f.write(f"  - 95%: {row['coverage_95']:.1%} {'✓' if abs(row['coverage_95'] - 0.95) <= 0.02 else '✗'}\n")
            
            if not np.isnan(row['pit_ks_pvalue']):
                f.write(f"- **PIT Uniformity**: KS p-value = {row['pit_ks_pvalue']:.4f} ")
                f.write(f"{'✓ Uniform' if row['pit_uniform'] else '✗ Non-uniform'}\n")
            else:
                f.write(f"- **PIT Uniformity**: N/A (conformal model)\n")
            
            f.write(f"- **Interval Crossing**: {'✗ Detected' if row['has_interval_crossing'] else '✓ None'}\n")
            f.write(f"- **Overall Status**: {'✓ Well-calibrated' if row['overall_calibrated'] else '✗ Miscalibrated'}\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        miscalibrated = summary_df[~summary_df['overall_calibrated']]
        if not miscalibrated.empty:
            f.write("Models requiring attention:\n\n")
            for _, row in miscalibrated.iterrows():
                f.write(f"- **{row['model']}**: ")
                issues = []
                if not row['coverage_calibrated']:
                    issues.append("coverage outside tolerance")
                if not row['pit_uniform'] and not np.isnan(row['pit_ks_pvalue']):
                    issues.append("non-uniform PIT")
                if row['has_interval_crossing']:
                    issues.append("interval crossing detected")
                f.write(", ".join(issues) + "\n")
            
            f.write("\nSuggested actions:\n")
            f.write("- For coverage issues: Consider switching from MQLoss to IQLoss\n")
            f.write("- For non-uniform PIT: Review model architecture or loss function\n")
            f.write("- For interval crossing: Use IQLoss instead of MQLoss\n")
        else:
            f.write("All models are well-calibrated. No immediate action required.\n")
    
    logger.info(f"Written calibration report to {output_path}")


def check_interval_crossing(cv_df: pd.DataFrame,
                           model_name: str,
                           levels: List[int] = [80, 90, 95]) -> pd.DataFrame:
    """
    Check for quantile crossing issues in interval predictions.
    
    Quantile crossing occurs when a narrower interval extends beyond
    a wider interval, violating monotonicity.
    
    Args:
        cv_df: Cross-validation results
        model_name: Model name to check
        levels: Confidence levels to check (should be sorted ascending)
        
    Returns:
        DataFrame with crossing detection results
    """
    crossing_results = []
    levels_sorted = sorted(levels)
    
    for i in range(len(levels_sorted) - 1):
        level_low = levels_sorted[i]
        level_high = levels_sorted[i + 1]
        
        # Get interval columns
        lo_low = f"{model_name}-lo-{level_low}"
        hi_low = f"{model_name}-hi-{level_low}"
        lo_high = f"{model_name}-lo-{level_high}"
        hi_high = f"{model_name}-hi-{level_high}"
        
        if all(col in cv_df.columns for col in [lo_low, hi_low, lo_high, hi_high]):
            # Check for crossing
            lower_crossing = (cv_df[lo_low] < cv_df[lo_high]).sum()
            upper_crossing = (cv_df[hi_low] > cv_df[hi_high]).sum()
            
            total_samples = len(cv_df)
            crossing_rate = (lower_crossing + upper_crossing) / (2 * total_samples)
            
            crossing_results.append({
                'model': model_name,
                'level_pair': f"{level_low}-{level_high}",
                'lower_crossings': lower_crossing,
                'upper_crossings': upper_crossing,
                'total_samples': total_samples,
                'crossing_rate': crossing_rate,
                'has_crossing': crossing_rate > 0
            })
    
    return pd.DataFrame(crossing_results)


def compute_coverage_by_volatility(cv_df: pd.DataFrame,
                                  models: List[str],
                                  levels: List[int] = [80, 90, 95],
                                  n_deciles: int = 10) -> pd.DataFrame:
    """
    Compute coverage segmented by volatility deciles for heteroscedasticity analysis.
    
    This function segments the data by volatility and computes coverage within
    each segment to identify regime-dependent calibration issues.
    
    Args:
        cv_df: Cross-validation results with predictions and intervals
        models: List of model names to evaluate
        levels: Confidence levels for coverage analysis
        n_deciles: Number of volatility segments (default: 10 for deciles)
        
    Returns:
        DataFrame with coverage by volatility segment for each model and level
    """
    # Compute rolling volatility (using 20-period window as proxy)
    window_size = min(20, len(cv_df) // 10)
    cv_df['volatility'] = cv_df['y'].rolling(window=window_size, min_periods=5).std()
    
    # Fill initial NaN values with expanding window std
    expanding_std = cv_df['y'].expanding(min_periods=2).std()
    cv_df['volatility'] = cv_df['volatility'].fillna(expanding_std)
    
    # Create volatility deciles
    cv_df['vol_decile'] = pd.qcut(cv_df['volatility'], q=n_deciles, labels=False, duplicates='drop')
    
    coverage_by_vol = []
    
    for model_name in models:
        if model_name not in cv_df.columns:
            continue
            
        for level in levels:
            lo_col = f"{model_name}-lo-{level}"
            hi_col = f"{model_name}-hi-{level}"
            
            if lo_col not in cv_df.columns or hi_col not in cv_df.columns:
                continue
            
            # Compute coverage by volatility decile
            for decile in range(n_deciles):
                decile_data = cv_df[cv_df['vol_decile'] == decile]
                if len(decile_data) < 10:  # Skip if too few samples
                    continue
                
                y_true = decile_data['y'].values
                lo_pred = decile_data[lo_col].values
                hi_pred = decile_data[hi_col].values
                
                # Handle NaN values
                mask = ~(np.isnan(y_true) | np.isnan(lo_pred) | np.isnan(hi_pred))
                if not mask.any():
                    continue
                
                y_clean = y_true[mask]
                lo_clean = lo_pred[mask]
                hi_clean = hi_pred[mask]
                
                # Calculate coverage
                within_interval = (y_clean >= lo_clean) & (y_clean <= hi_clean)
                empirical_coverage = float(within_interval.mean())
                
                nominal_coverage = level / 100.0
                deviation = empirical_coverage - nominal_coverage
                
                # Check if within tolerance
                calibrated = abs(deviation) <= 0.02
                
                # Get average volatility for this decile
                avg_volatility = decile_data['volatility'].mean()
                
                coverage_by_vol.append({
                    'model': model_name,
                    'level': level,
                    'vol_decile': decile + 1,  # 1-indexed for readability
                    'avg_volatility': avg_volatility,
                    'nominal_coverage': nominal_coverage,
                    'empirical_coverage': empirical_coverage,
                    'deviation': deviation,
                    'calibrated': calibrated,
                    'n_samples': len(y_clean)
                })
    
    coverage_vol_df = pd.DataFrame(coverage_by_vol)
    
    # Check for heteroscedasticity issues
    if not coverage_vol_df.empty:
        for model_name in models:
            model_data = coverage_vol_df[coverage_vol_df['model'] == model_name]
            if len(model_data) > 0:
                # Check if coverage varies significantly across volatility deciles
                coverage_std = model_data.groupby('level')['empirical_coverage'].std()
                for level, std in coverage_std.items():
                    if std > 0.05:  # More than 5pp variation
                        logger.warning(f"{model_name} at {level}% shows heteroscedastic calibration "
                                     f"(coverage std: {std:.3f} across volatility deciles)")
    
    return coverage_vol_df


def plot_coverage_by_volatility(coverage_vol_df: pd.DataFrame,
                               output_dir: Path,
                               model_name: str,
                               horizon: Optional[int] = None) -> None:
    """
    Plot coverage by volatility deciles for heteroscedasticity analysis.
    
    Args:
        coverage_vol_df: Coverage by volatility results
        output_dir: Directory for saving plots
        model_name: Model name for filtering
        horizon: Forecast horizon for directory structure
    """
    if horizon is not None:
        output_dir = Path(output_dir) / f"h{horizon}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = coverage_vol_df[coverage_vol_df['model'] == model_name]
    if model_data.empty:
        logger.warning(f"No volatility-segmented coverage data for {model_name}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot coverage by volatility for each level
    for level in [80, 90, 95]:
        level_data = model_data[model_data['level'] == level]
        if not level_data.empty:
            ax.plot(level_data['vol_decile'], 
                   level_data['empirical_coverage'] * 100,
                   marker='o', label=f'{level}% CI', linewidth=2)
            
            # Add nominal reference line
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
    
    # Add tolerance bands
    ax.fill_between(range(1, 11), [78]*10, [82]*10, alpha=0.1, color='gray')
    ax.fill_between(range(1, 11), [88]*10, [92]*10, alpha=0.1, color='gray')
    ax.fill_between(range(1, 11), [93]*10, [97]*10, alpha=0.1, color='gray')
    
    ax.set_xlabel('Volatility Decile', fontsize=11)
    ax.set_ylabel('Empirical Coverage (%)', fontsize=11)
    ax.set_title(f'Coverage by Volatility - {model_name}', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 10.5])
    ax.set_ylim([70, 100])
    
    # Add text for interpretation
    ax.text(0.02, 0.02, 'Flat lines indicate homoscedastic calibration',
           transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"coverage_by_vol_{model_name}.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved coverage by volatility plot to {plot_path}")