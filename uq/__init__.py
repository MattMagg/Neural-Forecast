"""
Uncertainty quantification module for probabilistic forecasting.

This module provides metrics computation, calibration analysis, and diagnostic tools.
"""

from .metrics import (
    compute_scrps,
    compute_mae,
    compute_rmse,
    compute_bias,
    compute_metrics_per_model,
    aggregate_metrics
)

from .calibration import (
    compute_coverage,
    compute_pit,
    plot_calibration_diagnostics,
    check_interval_crossing
)

__all__ = [
    # Metrics
    'compute_scrps',
    'compute_mae',
    'compute_rmse',
    'compute_bias',
    'compute_metrics_per_model',
    'aggregate_metrics',
    # Calibration
    'compute_coverage',
    'compute_pit',
    'plot_calibration_diagnostics',
    'check_interval_crossing'
]