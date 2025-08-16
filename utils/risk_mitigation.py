"""
Risk mitigation for the four critical risk categories.

This module implements prevention and mitigation strategies for:
1. MTF Misalignment - multi-timeframe feature alignment
2. Data Leakage - future information bleeding into features  
3. Quantile Crossing - non-monotonic quantile predictions
4. GPU Memory - OOM errors and memory management

Author: Risk Mitigation Specialist
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class MTFAlignmentValidator:
    """Validate and fix multi-timeframe feature alignment."""
    
    @staticmethod
    def validate_resampling_params(df: pd.DataFrame,
                                    freq: str,
                                    label: str,
                                    closed: str) -> bool:
        """
        Validate that resampling uses correct parameters.
        
        MTF Rule: Always use label='right' and closed='right' for aggregation
        to ensure end-of-bar alignment and prevent future leakage.
        
        Args:
            df: DataFrame being resampled
            freq: Target frequency
            label: Label parameter (should be 'right')
            closed: Closed parameter (should be 'right')
            
        Returns:
            True if parameters are correct
        """
        if label != 'right':
            logger.error(f"MTF misalignment risk: label='{label}' should be 'right'")
            return False
            
        if closed != 'right':
            logger.error(f"MTF misalignment risk: closed='{closed}' should be 'right'")
            return False
            
        logger.info(f"MTF resampling validated: freq={freq}, label='right', closed='right'")
        return True
        
    @staticmethod
    def resample_with_alignment(df: pd.DataFrame,
                                 source_freq: str,
                                 target_freq: str,
                                 agg_func: str = 'last') -> pd.DataFrame:
        """
        Resample data with correct MTF alignment.
        
        Args:
            df: Source dataframe with datetime index
            source_freq: Source frequency (e.g., '15min')
            target_freq: Target frequency (e.g., '1h')
            agg_func: Aggregation function
            
        Returns:
            Resampled dataframe with proper alignment
        """
        # Always use right label and closed for end-of-bar alignment
        resampled = df.resample(
            target_freq,
            label='right',
            closed='right'
        ).agg(agg_func)
        
        logger.info(f"Resampled {source_freq} → {target_freq} with EOB alignment")
        return resampled
        
    @staticmethod
    def validate_mtf_features(df: pd.DataFrame,
                              base_freq: str = '15min',
                              mtf_freqs: List[str] = ['30min', '1h', '4h']) -> Dict[str, bool]:
        """
        Validate that MTF features are properly aligned.
        
        Args:
            df: DataFrame with MTF features
            base_freq: Base frequency
            mtf_freqs: List of MTF frequencies
            
        Returns:
            Validation results by frequency
        """
        results = {}
        
        for freq in mtf_freqs:
            # Check that MTF columns exist
            mtf_cols = [col for col in df.columns if f'_{freq}' in col]
            
            if not mtf_cols:
                results[freq] = False
                logger.warning(f"No {freq} MTF features found")
                continue
                
            # Check for proper shift (should have NaN at start)
            has_shift = df[mtf_cols].iloc[0].isna().any()
            
            if not has_shift:
                logger.error(f"MTF features for {freq} may not be shifted - leakage risk!")
                results[freq] = False
            else:
                results[freq] = True
                logger.info(f"MTF features for {freq} appear properly shifted")
                
        return results


class DataLeakageDetector:
    """Detect and prevent data leakage in features."""
    
    @staticmethod
    def check_correlation_leakage(features: pd.DataFrame,
                                   target: pd.Series,
                                   threshold: float = 0.1) -> Dict[str, Dict[str, float]]:
        """
        Detect leakage via correlation analysis.
        
        Rule: corr(feature_t, y_t) should be > corr(feature_t, y_{t+1})
        If next-period correlation exceeds current by >10%, likely leakage.
        
        Args:
            features: Feature dataframe
            target: Target series
            threshold: Threshold for flagging leakage (default 10%)
            
        Returns:
            Dictionary of suspicious features with correlation values
        """
        suspicious = {}
        
        # Shift target forward to get y_{t+1}
        target_next = target.shift(-1)
        
        for col in features.columns:
            if col == 'ds' or col == 'unique_id':
                continue
                
            # Calculate correlations
            corr_current = features[col].corr(target)
            corr_next = features[col].corr(target_next)
            
            # Check for leakage pattern
            if corr_next > corr_current * (1 + threshold):
                suspicious[col] = {
                    'corr_current': corr_current,
                    'corr_next': corr_next,
                    'ratio': corr_next / corr_current if corr_current != 0 else np.inf
                }
                logger.warning(
                    f"Potential leakage in {col}: "
                    f"corr(t)={corr_current:.3f}, corr(t+1)={corr_next:.3f}"
                )
                
        return suspicious
        
    @staticmethod
    def verify_historical_shift(df: pd.DataFrame,
                                 hist_cols: List[str]) -> Dict[str, bool]:
        """
        Verify all historical features have shift(1) applied.
        
        Args:
            df: Feature dataframe
            hist_cols: List of historical feature columns
            
        Returns:
            Dictionary of column shift validation results
        """
        results = {}
        
        for col in hist_cols:
            if col not in df.columns:
                results[col] = False
                logger.error(f"Historical column {col} not found")
                continue
                
            # Check for NaN in first row (indicates shift)
            if pd.isna(df[col].iloc[0]):
                results[col] = True
                logger.debug(f"Column {col} appears shifted (NaN at start)")
            else:
                results[col] = False
                logger.error(f"Column {col} may not be shifted - leakage risk!")
                
        # Summary
        n_valid = sum(results.values())
        n_total = len(results)
        
        if n_valid < n_total:
            logger.error(f"Shift validation failed: {n_valid}/{n_total} columns properly shifted")
        else:
            logger.info(f"All {n_total} historical columns properly shifted")
            
        return results
        
    @staticmethod
    def apply_safety_shift(df: pd.DataFrame,
                           hist_cols: List[str],
                           verify: bool = True) -> pd.DataFrame:
        """
        Apply shift(1) to historical columns with verification.
        
        Args:
            df: Input dataframe
            hist_cols: Historical columns to shift
            verify: Whether to verify after shifting
            
        Returns:
            DataFrame with shifted historical features
        """
        df_shifted = df.copy()
        
        for col in hist_cols:
            if col in df_shifted.columns:
                df_shifted[col] = df_shifted[col].shift(1)
                logger.debug(f"Applied shift(1) to {col}")
                
        if verify:
            # Run verification
            validator = DataLeakageDetector()
            results = validator.verify_historical_shift(df_shifted, hist_cols)
            
            if not all(results.values()):
                logger.error("Shift verification failed after applying shifts")
                
        return df_shifted


class QuantileCrossingFixer:
    """Fix and prevent quantile crossing issues."""
    
    @staticmethod
    def detect_crossing(predictions: pd.DataFrame,
                        quantile_cols: List[str]) -> Tuple[bool, List[str]]:
        """
        Detect quantile crossing in predictions.
        
        Rule: Quantiles must be monotonic (q10 < q50 < q90)
        
        Args:
            predictions: Prediction dataframe
            quantile_cols: Ordered list of quantile columns
            
        Returns:
            Tuple of (has_crossing, list_of_violations)
        """
        violations = []
        
        # Check each row for monotonicity
        for idx, row in predictions.iterrows():
            quantile_values = [row[col] for col in quantile_cols]
            
            # Check if monotonically increasing
            for i in range(len(quantile_values) - 1):
                if quantile_values[i] > quantile_values[i + 1]:
                    violations.append(
                        f"Row {idx}: {quantile_cols[i]}={quantile_values[i]:.3f} > "
                        f"{quantile_cols[i+1]}={quantile_values[i+1]:.3f}"
                    )
                    
        has_crossing = len(violations) > 0
        
        if has_crossing:
            logger.error(f"Quantile crossing detected in {len(violations)} cases")
            for v in violations[:5]:  # Show first 5
                logger.error(f"  {v}")
        else:
            logger.info("No quantile crossing detected")
            
        return has_crossing, violations
        
    @staticmethod
    def fix_crossing_simple(predictions: pd.DataFrame,
                            quantile_cols: List[str],
                            epsilon: float = 1e-6) -> pd.DataFrame:
        """
        Fix quantile crossing with averaging and separation.
        
        Args:
            predictions: Prediction dataframe
            quantile_cols: Ordered list of quantile columns
            epsilon: Small value to separate equal quantiles
            
        Returns:
            Fixed predictions dataframe
        """
        fixed = predictions.copy()
        n_fixed = 0
        
        for idx in fixed.index:
            values = [fixed.loc[idx, col] for col in quantile_cols]
            original = values.copy()
            
            # Sort to ensure monotonicity
            values_sorted = sorted(values)
            
            # Add epsilon separation if needed
            for i in range(1, len(values_sorted)):
                if values_sorted[i] <= values_sorted[i-1]:
                    values_sorted[i] = values_sorted[i-1] + epsilon
                    
            # Check if we needed to fix
            if values != values_sorted:
                n_fixed += 1
                for col, val in zip(quantile_cols, values_sorted):
                    fixed.loc[idx, col] = val
                    
        if n_fixed > 0:
            logger.info(f"Fixed quantile crossing in {n_fixed} rows")
        
        return fixed
        
    @staticmethod
    def recommend_loss_switch(n_violations: int,
                              n_total: int,
                              threshold: float = 0.1) -> bool:
        """
        Recommend switching from MQLoss to IQLoss if crossing is persistent.
        
        Args:
            n_violations: Number of crossing violations
            n_total: Total number of predictions
            threshold: Threshold for recommending switch (10%)
            
        Returns:
            True if should switch to IQLoss
        """
        violation_rate = n_violations / n_total if n_total > 0 else 0
        
        if violation_rate > threshold:
            logger.warning(
                f"High quantile crossing rate ({violation_rate:.1%}). "
                f"Recommend switching from MQLoss to IQLoss for monotonic quantiles."
            )
            return True
            
        return False


class GPUMemoryManager:
    """Manage GPU memory and handle OOM issues."""
    
    def __init__(self):
        """Initialize GPU memory manager."""
        self.initial_config = {}
        self.recovery_history = []
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory info in GB
        """
        info = {
            'available': False,
            'allocated_gb': 0,
            'reserved_gb': 0,
            'free_gb': 0
        }
        
        if not torch.cuda.is_available():
            return info
            
        try:
            info['available'] = True
            info['allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            info['reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            
            # Get total memory
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['total_gb'] = total
            info['free_gb'] = total - info['reserved_gb']
            
            logger.debug(
                f"GPU Memory: {info['allocated_gb']:.2f}GB allocated, "
                f"{info['free_gb']:.2f}GB free of {total:.2f}GB total"
            )
            
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            
        return info
        
    def suggest_batch_size(self,
                           model_size_mb: float,
                           current_batch: int,
                           safety_factor: float = 0.8) -> int:
        """
        Suggest optimal batch size based on available memory.
        
        Args:
            model_size_mb: Estimated model size in MB
            current_batch: Current batch size
            safety_factor: Safety margin (default 80% of available)
            
        Returns:
            Suggested batch size
        """
        mem_info = self.get_gpu_memory_info()
        
        if not mem_info['available']:
            return min(32, current_batch)  # Conservative CPU fallback
            
        # Estimate memory per batch item (rough heuristic)
        mem_per_item_mb = model_size_mb / 32  # Assume base batch of 32
        
        # Available memory with safety factor
        available_mb = mem_info['free_gb'] * 1024 * safety_factor
        
        # Calculate suggested batch size
        suggested = int(available_mb / mem_per_item_mb)
        suggested = max(8, min(suggested, current_batch))  # Clamp between 8 and current
        
        logger.info(
            f"GPU batch size suggestion: {suggested} "
            f"(based on {mem_info['free_gb']:.2f}GB free memory)"
        )
        
        return suggested
        
    def progressive_reduction(self,
                              config: Dict[str, Any],
                              stage: int = 1) -> Dict[str, Any]:
        """
        Progressive strategy for GPU memory reduction.
        
        Stages:
        1. Reduce batch_size by 50%
        2. Enable gradient accumulation
        3. Enable mixed precision
        4. Model pruning suggestions
        5. CPU offload
        
        Args:
            config: Current configuration
            stage: Recovery stage (1-5)
            
        Returns:
            Updated configuration
        """
        logger.info(f"GPU memory recovery stage {stage}")
        
        if stage == 1:
            # First: Reduce batch size
            old_batch = config.get('batch_size', 512)
            new_batch = max(16, old_batch // 2)
            config['batch_size'] = new_batch
            logger.info(f"Stage 1: Reduced batch_size {old_batch} → {new_batch}")
            
        elif stage == 2:
            # Second: Gradient accumulation
            config['gradient_steps'] = 4
            config['batch_size'] = max(8, config.get('batch_size', 128) // 2)
            logger.info("Stage 2: Enabled gradient accumulation (steps=4)")
            
        elif stage == 3:
            # Third: Mixed precision
            if torch.cuda.is_available():
                config['use_amp'] = True
                config['scaler_type'] = 'robust'  # More stable with AMP
                logger.info("Stage 3: Enabled automatic mixed precision (AMP)")
            else:
                logger.info("Stage 3: AMP not available, skipping")
                
        elif stage == 4:
            # Fourth: Model architecture suggestions
            logger.info("Stage 4: Consider model architecture changes:")
            logger.info("  - Reduce hidden_size or n_layers")
            logger.info("  - Reduce input_size (window length)")
            logger.info("  - Use smaller model variant (e.g., NHITS instead of PatchTST)")
            
        elif stage >= 5:
            # Final: CPU offload or minimal config
            config['device'] = 'cpu'
            config['batch_size'] = 8
            config['n_windows'] = 2
            logger.info("Stage 5: Switched to CPU with minimal configuration")
            
        # Clear cache after each stage
        self.clear_cache()
        
        # Record recovery attempt
        self.recovery_history.append({
            'stage': stage,
            'config': config.copy(),
            'memory_state': self.get_gpu_memory_info()
        })
        
        return config
        
    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared GPU cache")
            

class RiskMitigationOrchestrator:
    """Orchestrate all risk mitigation strategies."""
    
    def __init__(self):
        """Initialize orchestrator with all validators."""
        self.mtf_validator = MTFAlignmentValidator()
        self.leakage_detector = DataLeakageDetector()
        self.quantile_fixer = QuantileCrossingFixer()
        self.gpu_manager = GPUMemoryManager()
        
    def run_pre_training_checks(self,
                                 df: pd.DataFrame,
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all pre-training risk checks.
        
        Args:
            df: Training dataframe
            config: Model configuration
            
        Returns:
            Risk assessment report
        """
        report = {
            'mtf_alignment': {},
            'data_leakage': {},
            'gpu_memory': {},
            'recommendations': []
        }
        
        # Check MTF alignment
        if 'mtf_freqs' in config:
            mtf_results = self.mtf_validator.validate_mtf_features(
                df, 
                config.get('base_freq', '15min'),
                config['mtf_freqs']
            )
            report['mtf_alignment'] = mtf_results
            
            if not all(mtf_results.values()):
                report['recommendations'].append(
                    "Fix MTF alignment: use label='right', closed='right' in resampling"
                )
                
        # Check for data leakage
        if 'target_col' in config and 'hist_cols' in config:
            target = df[config['target_col']]
            features = df[config['hist_cols']]
            
            # Correlation check
            suspicious = self.leakage_detector.check_correlation_leakage(features, target)
            report['data_leakage']['suspicious_features'] = suspicious
            
            # Shift verification
            shift_results = self.leakage_detector.verify_historical_shift(df, config['hist_cols'])
            report['data_leakage']['shift_validation'] = shift_results
            
            if suspicious or not all(shift_results.values()):
                report['recommendations'].append(
                    "Apply shift(1) to all historical features to prevent leakage"
                )
                
        # Check GPU memory
        mem_info = self.gpu_manager.get_gpu_memory_info()
        report['gpu_memory'] = mem_info
        
        if mem_info['available'] and mem_info['free_gb'] < 2.0:
            report['recommendations'].append(
                f"Low GPU memory ({mem_info['free_gb']:.1f}GB free). Consider reducing batch_size."
            )
            
        return report
        
    def run_post_prediction_checks(self,
                                    predictions: pd.DataFrame,
                                    quantile_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run post-prediction risk checks.
        
        Args:
            predictions: Model predictions
            quantile_cols: List of quantile column names
            
        Returns:
            Risk assessment report
        """
        report = {
            'quantile_crossing': {},
            'recommendations': []
        }
        
        if quantile_cols:
            # Check for quantile crossing
            has_crossing, violations = self.quantile_fixer.detect_crossing(
                predictions, quantile_cols
            )
            
            report['quantile_crossing']['has_crossing'] = has_crossing
            report['quantile_crossing']['n_violations'] = len(violations)
            
            if has_crossing:
                # Fix crossing
                fixed = self.quantile_fixer.fix_crossing_simple(predictions, quantile_cols)
                report['quantile_crossing']['fixed'] = True
                
                # Check if should switch loss
                n_total = len(predictions)
                should_switch = self.quantile_fixer.recommend_loss_switch(
                    len(violations), n_total
                )
                
                if should_switch:
                    report['recommendations'].append(
                        "High quantile crossing rate. Switch from MQLoss to IQLoss."
                    )
                    
        return report