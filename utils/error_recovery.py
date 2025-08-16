"""
Error recovery protocols for Neural-Forecast system.

This module provides centralized error handling and recovery mechanisms for
common failure scenarios in the forecasting pipeline. It implements the
Agent-Oriented Error Recovery Protocols from the design specification.

Author: Risk Mitigation Specialist
"""

import logging
import traceback
import warnings
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import pandas as pd
import torch
import gc
from pathlib import Path
import psutil
import GPUtil

logger = logging.getLogger(__name__)


class ErrorRecoveryManager:
    """Centralized error recovery and handling system."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        """
        Initialize error recovery manager.
        
        Args:
            enable_gpu_monitoring: Whether to monitor GPU memory usage
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.recovery_attempts = {}
        self.context7_queries = self._load_context7_queries()
        
    def _load_context7_queries(self) -> Dict[str, List[str]]:
        """Load suggested context7 queries for debugging."""
        return {
            'memory_exhaustion': [
                "neuralforecast memory optimization",
                "neuralforecast batch_size GPU",
                "neuralforecast inference_windows_batch_size",
                "neuralforecast gradient accumulation"
            ],
            'scrps_computation': [
                "neuralforecast sCRPS import path",
                "neuralforecast sCRPS numerical stability",
                "neuralforecast losses.pytorch sCRPS",
                "neuralforecast metric computation"
            ],
            'coverage_calibration': [
                "neuralforecast conformal prediction",
                "neuralforecast PredictionIntervals",
                "neuralforecast IQLoss quantile crossing",
                "neuralforecast level parameter"
            ],
            'pit_computation': [
                "probability integral transform quantile interpolation",
                "neuralforecast distributional calibration diagnostics",
                "scipy.stats.kstest uniformity",
                "neuralforecast PIT histogram"
            ],
            'cv_window_config': [
                "neuralforecast cross validation window parameters",
                "neuralforecast n_windows step_size",
                "neuralforecast val_size configuration",
                "neuralforecast minimum data requirements"
            ]
        }
    
    def handle_memory_exhaustion(self, 
                                  error: Exception,
                                  config: Dict[str, Any],
                                  attempt: int = 1) -> Dict[str, Any]:
        """
        Handle GPU memory exhaustion errors.
        
        Detection: OOM errors, kernel crashes, CV hanging
        Recovery: Progressive batch size reduction, n_windows reduction
        
        Args:
            error: The caught exception
            config: Current configuration dict
            attempt: Current recovery attempt number
            
        Returns:
            Updated configuration with reduced memory requirements
        """
        logger.warning(f"Memory exhaustion detected (attempt {attempt}): {str(error)}")
        
        # Log GPU status if available
        if self.enable_gpu_monitoring:
            self._log_gpu_status()
        
        # Progressive recovery strategies
        if attempt == 1:
            # First attempt: Reduce batch_size by 50%
            old_batch = config.get('batch_size', 512)
            new_batch = max(32, old_batch // 2)
            config['batch_size'] = new_batch
            logger.info(f"Reduced batch_size from {old_batch} to {new_batch}")
            
            # Add inference optimization
            config['inference_windows_batch_size'] = min(16, new_batch // 4)
            
        elif attempt == 2:
            # Second attempt: Further reduce batch_size and add gradient accumulation
            old_batch = config.get('batch_size', 256)
            new_batch = max(16, old_batch // 2)
            config['batch_size'] = new_batch
            config['gradient_steps'] = 4  # Simulate larger batch with accumulation
            logger.info(f"Reduced batch_size to {new_batch} with gradient accumulation")
            
        elif attempt == 3:
            # Third attempt: Reduce n_windows for CV
            old_windows = config.get('n_windows', 10)
            new_windows = max(2, old_windows - 4)
            config['n_windows'] = new_windows
            logger.info(f"Reduced n_windows from {old_windows} to {new_windows}")
            
            # Enable mixed precision if available
            if torch.cuda.is_available():
                config['use_amp'] = True
                logger.info("Enabled automatic mixed precision")
                
        else:
            # Final attempt: Switch to CPU or minimal config
            logger.error("Multiple GPU OOM failures. Switching to minimal configuration.")
            config['batch_size'] = 8
            config['n_windows'] = 2
            config['inference_windows_batch_size'] = 1
            
            # Suggest model pruning
            self._suggest_recovery_actions('memory_exhaustion')
            
        # Clear GPU cache
        self._clear_gpu_cache()
        
        return config
    
    def handle_scrps_failure(self,
                             error: Exception,
                             data: pd.DataFrame,
                             predictions: np.ndarray) -> Tuple[bool, str]:
        """
        Handle sCRPS computation failures.
        
        Detection: NaN values, import errors, metric crashes
        Recovery: Verify imports, check for NaN/Inf, fallback metrics
        
        Args:
            error: The caught exception
            data: Input data frame
            predictions: Model predictions
            
        Returns:
            Tuple of (recovery_successful, error_message)
        """
        logger.error(f"sCRPS computation failed: {str(error)}")
        
        # Check for NaN/Inf in data
        if np.any(np.isnan(data.values)) or np.any(np.isinf(data.values)):
            logger.error("NaN or Inf values detected in input data")
            return False, "Input data contains NaN or Inf values"
            
        # Check predictions
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            logger.error("NaN or Inf values detected in predictions")
            
            # Try to identify which model/quantile caused the issue
            nan_mask = np.isnan(predictions)
            inf_mask = np.isinf(predictions)
            
            if nan_mask.any():
                nan_locations = np.where(nan_mask)
                logger.error(f"NaN locations: {nan_locations}")
                
            if inf_mask.any():
                inf_locations = np.where(inf_mask)
                logger.error(f"Inf locations: {inf_locations}")
                
            return False, "Predictions contain NaN or Inf values"
        
        # Check for import issues
        try:
            from neuralforecast.losses.pytorch import sCRPS
            logger.info("sCRPS import successful")
        except ImportError as e:
            logger.error(f"sCRPS import failed: {e}")
            self._suggest_recovery_actions('scrps_computation')
            return False, "sCRPS import failed - check neuralforecast installation"
            
        # Suggest numerical stability improvements
        logger.info("Consider using StudentT distribution for heavy-tailed data")
        logger.info("Consider winsorizing extreme values during training")
        
        return True, "Recovery attempted - check logs for details"
    
    def handle_coverage_calibration(self,
                                     coverage: Dict[str, float],
                                     targets: Dict[str, float],
                                     predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Handle coverage calibration issues.
        
        Detection: Coverage outside ±2% tolerance, quantile crossing
        Recovery: Check intervals, add conformal, switch to IQLoss
        
        Args:
            coverage: Actual coverage percentages by level
            targets: Target coverage percentages
            predictions: Prediction dataframe with interval columns
            
        Returns:
            Recovery recommendations and actions
        """
        recommendations = {
            'issues_found': [],
            'actions_taken': [],
            'suggestions': []
        }
        
        # Check coverage tolerance
        for level, actual in coverage.items():
            target = targets.get(level, level)
            deviation = abs(actual - target)
            
            if deviation > 2.0:
                msg = f"Coverage {level}%: {actual:.1f}% (target: {target}%, deviation: {deviation:.1f}%)"
                logger.warning(msg)
                recommendations['issues_found'].append(msg)
                
                if actual < target - 2:
                    recommendations['suggestions'].append(
                        f"Under-coverage at {level}%: Consider conformal prediction or wider intervals"
                    )
                else:
                    recommendations['suggestions'].append(
                        f"Over-coverage at {level}%: Model may be too conservative"
                    )
        
        # Check for quantile crossing
        quantile_cols = [col for col in predictions.columns if 'lo-' in col or 'hi-' in col]
        
        if quantile_cols:
            # Sort quantile columns by level
            lo_cols = sorted([col for col in quantile_cols if 'lo-' in col])
            hi_cols = sorted([col for col in quantile_cols if 'hi-' in col])
            
            # Check monotonicity
            for i in range(len(lo_cols) - 1):
                # Lower quantiles should be decreasing
                violations = predictions[lo_cols[i]] < predictions[lo_cols[i+1]]
                if violations.any():
                    msg = f"Quantile crossing detected in lower bounds: {lo_cols[i]} < {lo_cols[i+1]}"
                    logger.error(msg)
                    recommendations['issues_found'].append(msg)
                    recommendations['suggestions'].append(
                        "Switch from MQLoss to IQLoss for monotonic quantiles"
                    )
                    
            for i in range(len(hi_cols) - 1):
                # Upper quantiles should be increasing  
                violations = predictions[hi_cols[i]] > predictions[hi_cols[i+1]]
                if violations.any():
                    msg = f"Quantile crossing detected in upper bounds: {hi_cols[i]} > {hi_cols[i+1]}"
                    logger.error(msg)
                    recommendations['issues_found'].append(msg)
                    recommendations['suggestions'].append(
                        "Consider IQLoss or post-processing quantile sorting"
                    )
        
        # Suggest context7 queries if issues found
        if recommendations['issues_found']:
            self._suggest_recovery_actions('coverage_calibration')
            
        return recommendations
    
    def handle_pit_errors(self,
                           pit_values: np.ndarray,
                           quantiles: np.ndarray) -> Dict[str, Any]:
        """
        Handle PIT computation errors.
        
        Detection: Non-uniform PIT, interpolation failures
        Recovery: Verify quantile grid, skip for conformal, KS test
        
        Args:
            pit_values: Computed PIT values
            quantiles: Quantile grid used
            
        Returns:
            PIT diagnostics and recovery recommendations
        """
        diagnostics = {
            'uniform': False,
            'ks_statistic': None,
            'ks_pvalue': None,
            'recommendations': []
        }
        
        # Check for NaN in PIT values
        if np.any(np.isnan(pit_values)):
            logger.error("NaN values in PIT computation")
            diagnostics['recommendations'].append(
                "Check quantile interpolation - ensure monotonic quantiles"
            )
            return diagnostics
            
        # Perform KS test for uniformity
        try:
            from scipy import stats
            ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
            diagnostics['ks_statistic'] = ks_stat
            diagnostics['ks_pvalue'] = ks_pval
            
            if ks_pval > 0.05:
                diagnostics['uniform'] = True
                logger.info(f"PIT appears uniform (KS test p-value: {ks_pval:.4f})")
            else:
                logger.warning(f"PIT not uniform (KS test p-value: {ks_pval:.4f})")
                
                # Analyze PIT histogram shape
                hist, _ = np.histogram(pit_values, bins=10)
                hist = hist / len(pit_values)
                
                if hist[0] > 0.15 and hist[-1] > 0.15:
                    diagnostics['recommendations'].append(
                        "U-shaped PIT: Model under-dispersed, consider heavier-tailed distribution"
                    )
                elif hist[4] > 0.15 and hist[5] > 0.15:
                    diagnostics['recommendations'].append(
                        "Inverse-U PIT: Model over-dispersed, consider tighter distribution"
                    )
                    
        except ImportError:
            logger.error("scipy not available for KS test")
            diagnostics['recommendations'].append("Install scipy for PIT uniformity testing")
            
        # Check quantile grid
        if len(quantiles) < 19:
            diagnostics['recommendations'].append(
                f"Quantile grid too coarse ({len(quantiles)} points), use 99-point grid"
            )
            
        # Suggest recovery actions
        if not diagnostics['uniform']:
            self._suggest_recovery_actions('pit_computation')
            
        return diagnostics
    
    def handle_cv_window_errors(self,
                                 data_length: int,
                                 config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle CV window configuration errors.
        
        Detection: Overlapping windows, insufficient data
        Recovery: Adjust parameters, validate relationships
        
        Args:
            data_length: Total number of time points
            config: CV configuration dict
            
        Returns:
            Tuple of (config_valid, updated_config)
        """
        n_windows = config.get('n_windows', 10)
        step_size = config.get('step_size', config.get('h', 12))
        val_size = config.get('val_size', 4 * config.get('h', 12))
        input_size = config.get('input_size', 512)
        h = config.get('h', 12)
        
        # Calculate minimum data requirements
        min_required = (n_windows - 1) * step_size + val_size + input_size
        
        logger.info(f"CV configuration check:")
        logger.info(f"  Data length: {data_length}")
        logger.info(f"  Minimum required: {min_required}")
        logger.info(f"  n_windows={n_windows}, step_size={step_size}, val_size={val_size}, input_size={input_size}")
        
        if data_length < min_required:
            logger.warning(f"Insufficient data: have {data_length}, need {min_required}")
            
            # Try to adjust n_windows
            max_windows = (data_length - val_size - input_size) // step_size + 1
            if max_windows >= 2:
                config['n_windows'] = max_windows
                logger.info(f"Adjusted n_windows from {n_windows} to {max_windows}")
                return True, config
            else:
                logger.error("Cannot fit even 2 CV windows with current configuration")
                
                # Try reducing input_size
                if input_size > 256:
                    config['input_size'] = 256
                    logger.info("Reduced input_size to 256")
                    return self.handle_cv_window_errors(data_length, config)
                    
                return False, config
                
        # Validate step_size = h relationship
        if step_size != h:
            logger.warning(f"step_size ({step_size}) != h ({h}), adjusting to maintain non-overlapping windows")
            config['step_size'] = h
            
        # Validate val_size = 4*h relationship
        expected_val = 4 * h
        if val_size != expected_val:
            logger.warning(f"val_size ({val_size}) != 4*h ({expected_val}), adjusting")
            config['val_size'] = expected_val
            
        return True, config
    
    def _log_gpu_status(self):
        """Log current GPU memory status."""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"GPU {i}: Allocated: {mem_alloc:.2f}GB, Reserved: {mem_reserved:.2f}GB")
                    
            # Try GPUtil for more detailed info
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                logger.info(f"GPU {gpu.id}: {gpu.name}, Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        except Exception as e:
            logger.debug(f"Could not log GPU status: {e}")
            
    def _clear_gpu_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared GPU cache")
            
    def _suggest_recovery_actions(self, error_type: str):
        """Suggest context7 queries for debugging."""
        queries = self.context7_queries.get(error_type, [])
        if queries:
            logger.info(f"Suggested context7 queries for debugging:")
            for query in queries:
                logger.info(f"  - {query}")
                

class NumericalStabilityChecker:
    """Check and ensure numerical stability in computations."""
    
    @staticmethod
    def check_array(arr: np.ndarray, 
                    name: str = "array",
                    fix: bool = True) -> Tuple[bool, np.ndarray]:
        """
        Check array for numerical issues.
        
        Args:
            arr: Array to check
            name: Name for logging
            fix: Whether to fix issues (clip/replace)
            
        Returns:
            Tuple of (is_clean, fixed_array)
        """
        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))
        
        if not has_nan and not has_inf:
            return True, arr
            
        logger.warning(f"{name} has numerical issues: NaN={has_nan}, Inf={has_inf}")
        
        if not fix:
            return False, arr
            
        # Fix issues
        fixed = arr.copy()
        
        # Replace NaN with 0 or median
        if has_nan:
            if arr.ndim == 1:
                median = np.nanmedian(arr)
                fixed = np.where(np.isnan(fixed), median if not np.isnan(median) else 0, fixed)
            else:
                for i in range(arr.shape[1]):
                    col = arr[:, i]
                    median = np.nanmedian(col)
                    fixed[:, i] = np.where(np.isnan(col), median if not np.isnan(median) else 0, col)
                    
        # Clip infinite values
        if has_inf:
            finite_vals = arr[np.isfinite(arr)]
            if len(finite_vals) > 0:
                vmin, vmax = np.percentile(finite_vals, [0.1, 99.9])
                fixed = np.clip(fixed, vmin, vmax)
            else:
                fixed = np.clip(fixed, -1e10, 1e10)
                
        logger.info(f"Fixed {name}: replaced NaN={has_nan}, clipped Inf={has_inf}")
        return False, fixed
        
    @staticmethod
    def ensure_monotonic_quantiles(quantiles: np.ndarray,
                                    epsilon: float = 1e-6) -> np.ndarray:
        """
        Ensure quantiles are monotonically increasing.
        
        Args:
            quantiles: Array of quantile predictions (shape: [n_samples, n_quantiles])
            epsilon: Small value to separate equal quantiles
            
        Returns:
            Fixed quantiles array
        """
        if quantiles.ndim == 1:
            quantiles = quantiles.reshape(1, -1)
            
        fixed = quantiles.copy()
        n_samples, n_quantiles = fixed.shape
        
        for i in range(n_samples):
            row = fixed[i]
            
            # Check if already monotonic
            if np.all(np.diff(row) >= 0):
                continue
                
            # Fix by sorting and adding small epsilon
            sorted_row = np.sort(row)
            for j in range(1, n_quantiles):
                if sorted_row[j] <= sorted_row[j-1]:
                    sorted_row[j] = sorted_row[j-1] + epsilon
                    
            fixed[i] = sorted_row
            
        return fixed


def with_error_recovery(recovery_manager: ErrorRecoveryManager,
                        error_type: str,
                        max_attempts: int = 3):
    """
    Decorator for functions that need error recovery.
    
    Args:
        recovery_manager: ErrorRecoveryManager instance
        error_type: Type of error to handle
        max_attempts: Maximum recovery attempts
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            last_error = None
            config = kwargs.get('config', {})
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if 'out of memory' in str(e).lower():
                        config = recovery_manager.handle_memory_exhaustion(e, config, attempt)
                        kwargs['config'] = config
                        last_error = e
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {e}")
                    last_error = e
                    
                    if attempt < max_attempts:
                        logger.info(f"Retrying... (attempt {attempt + 1}/{max_attempts})")
                    else:
                        raise
                        
            if last_error:
                raise last_error
                
        return wrapper
    return decorator