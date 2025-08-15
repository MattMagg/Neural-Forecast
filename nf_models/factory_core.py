"""
NeuralForecast Model Factory Core Module
Exported from notebook for production use.

This module provides the core factory functions for instantiating
NeuralForecast models according to the specifications in
docs/forecasting_sf_plan.md Section 4 (lines 1204-1564).
"""

from typing import List, Dict, Any, Optional, Union, Type
import inspect

from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss


# Default confidence levels for quantile losses
DEFAULT_LEVELS = [80, 90, 95]

# Model class mapping
MODEL_CLASSES = {
    "NHITS": NHITS,
    "NBEATSX": NBEATSx,
    "TIDE": TiDE,
    "PATCHTST": PatchTST
}


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class ModelInstantiationError(Exception):
    """Raised when model instantiation fails"""
    pass


def _loss_ctor(spec: Dict[str, Any]) -> Union[DistributionLoss, MQLoss, IQLoss]:
    """
    Create NF-native loss instances from specification.
    
    Per Section 4.1 (lines 1236-1285):
    - NHITS: DistributionLoss('StudentT', return_params=True)
    - NBEATSx: MQLoss(level=[10, 50, 90])
    - TiDE: IQLoss(level=[10, 50, 90])
    - PatchTST: DistributionLoss('StudentT', return_params=True)
    
    Args:
        spec: Loss specification with 'kind' and optional 'level'
              Examples:
              - {'kind': 'studentt'}
              - {'kind': 'mqloss', 'level': [80, 90]}
              - {'kind': 'iqloss', 'level': [80, 90]}
    
    Returns:
        Configured NF loss instance
        
    Raises:
        ConfigurationError: If loss specification is invalid
    """
    if 'kind' not in spec:
        raise ConfigurationError("Loss specification must include 'kind'")
    
    kind = spec['kind'].lower()
    
    if kind == 'studentt':
        # Per spec: For probabilistic models (NHITS, PatchTST)
        return DistributionLoss(distribution="StudentT", return_params=True)
    
    elif kind == 'mqloss':
        # Per spec: For NBEATSx - multi-quantile loss
        levels = spec.get('level', DEFAULT_LEVELS)
        if not isinstance(levels, list) or len(levels) == 0:
            raise ConfigurationError("MQLoss levels must be a non-empty list")
        if not all(0 < l < 100 for l in levels):
            raise ConfigurationError("All MQLoss levels must be between 0 and 100")
        return MQLoss(level=levels)
    
    elif kind == 'iqloss':
        # Per spec: For TiDE - implicit quantile loss
        levels = spec.get('level', DEFAULT_LEVELS)
        if not isinstance(levels, list) or len(levels) == 0:
            raise ConfigurationError("IQLoss levels must be a non-empty list")
        if not all(0 < l < 100 for l in levels):
            raise ConfigurationError("All IQLoss levels must be between 0 and 100")
        return IQLoss(level=levels)
    
    else:
        raise ConfigurationError(f"Unsupported loss kind: {kind}. Supported: studentt, mqloss, iqloss")


def _prune_kwargs(model_cls: Type, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unsupported constructor arguments using introspection.
    
    Args:
        model_cls: NeuralForecast model class (NHITS, NBEATSx, TiDE, PatchTST)
        params: Parameter dictionary from configuration
        
    Returns:
        Filtered parameter dictionary with only supported arguments
        
    Raises:
        ModelInstantiationError: If model doesn't support required exogenous lists
    """
    # Get model constructor signature
    sig = inspect.signature(model_cls.__init__)
    supported_params = set(sig.parameters.keys()) - {'self'}
    
    # Check for required exogenous support
    required_exog = {'hist_exog_list', 'futr_exog_list', 'stat_exog_list'}
    missing_exog = required_exog - supported_params
    
    if missing_exog:
        raise ModelInstantiationError(
            f"{model_cls.__name__} doesn't support required exogenous lists: {missing_exog}"
        )
    
    # Filter to only supported parameters
    pruned = {k: v for k, v in params.items() if k in supported_params}
    
    # Log pruned parameters for transparency
    removed = set(params.keys()) - set(pruned.keys())
    if removed:
        print(f"  ⚠️  {model_cls.__name__}: Removed unsupported params: {removed}")
    
    return pruned


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate experiment configuration structure.
    
    Args:
        cfg: Experiment configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Check required top-level keys
    required_keys = {'h', 'freq', 'models'}
    missing_keys = required_keys - set(cfg.keys())
    if missing_keys:
        raise ConfigurationError(f"Configuration missing required keys: {missing_keys}")
    
    # Validate horizon
    if not isinstance(cfg['h'], int) or cfg['h'] <= 0:
        raise ConfigurationError(f"Horizon 'h' must be positive integer, got: {cfg['h']}")
    
    # Validate frequency
    valid_freqs = ['15min', '15T', '30min', '30T', '1H', '4H']
    if cfg['freq'] not in valid_freqs:
        raise ConfigurationError(f"Frequency must be one of {valid_freqs}, got: {cfg['freq']}")
    
    # Validate models list
    if not isinstance(cfg['models'], list) or len(cfg['models']) == 0:
        raise ConfigurationError("'models' must be a non-empty list")
    
    # Validate each model entry
    for i, model_cfg in enumerate(cfg['models']):
        if 'alias' not in model_cfg:
            raise ConfigurationError(f"Model {i} missing required 'alias' field")
        if 'loss' not in model_cfg:
            raise ConfigurationError(f"Model {model_cfg['alias']} missing required 'loss' field")


def instantiate_models(
    exp_cfg: Dict[str, Any],
    exog_lists: Dict[str, List[str]],
    h: int
) -> List[Any]:
    """
    Instantiate NeuralForecast models from experiment configuration.
    
    Per Section 4.2 (lines 1431-1490), this is the main factory function
    that creates configured NeuralForecast model instances.
    
    Args:
        exp_cfg: Experiment configuration with 'models' list
        exog_lists: Dict with 'hist_cols', 'futr_cols', 'stat_cols' keys
        h: Forecast horizon
        
    Returns:
        List of instantiated NeuralForecast model objects
        
    Raises:
        ConfigurationError: If configuration is invalid
        ModelInstantiationError: If model instantiation fails
    
    Example:
        >>> exp_cfg = {
        ...     'h': 16,
        ...     'freq': '15min',
        ...     'models': [
        ...         {
        ...             'alias': 'NHITS_studentt',
        ...             'loss': {'kind': 'studentt'},
        ...             'n_blocks': [1, 1, 1]
        ...         }
        ...     ]
        ... }
        >>> exog_lists = {
        ...     'hist_cols': ['rsi', 'macd'],
        ...     'futr_cols': ['hour'],
        ...     'stat_cols': []
        ... }
        >>> models = instantiate_models(exp_cfg, exog_lists, 16)
    """
    # Validate configuration
    _validate_config(exp_cfg)
    
    # Extract configuration
    models_cfg = exp_cfg['models']
    scaler_overrides = exp_cfg.get('scaler_type', {})
    
    # Extract exogenous lists
    hist_exog_list = exog_lists.get('hist_cols', [])
    futr_exog_list = exog_lists.get('futr_cols', [])
    stat_exog_list = exog_lists.get('stat_cols', [])
    
    print(f"\nInstantiating models for h={h}:")
    print(f"  Historical features: {len(hist_exog_list)}")
    print(f"  Future features: {len(futr_exog_list)}")
    print(f"  Static features: {len(stat_exog_list)}")
    
    models = []
    
    for model_cfg in models_cfg:
        alias = model_cfg['alias']
        
        # Determine model class from alias
        model_type = alias.split('_')[0].upper()
        if model_type not in MODEL_CLASSES:
            raise ConfigurationError(f"Unknown model type in alias '{alias}': {model_type}")
        
        model_cls = MODEL_CLASSES[model_type]
        
        # Build base parameters (Section 4.2.A - Common parameters)
        params = {
            'h': h,
            'alias': alias,
            'hist_exog_list': hist_exog_list if hist_exog_list else None,
            'futr_exog_list': futr_exog_list if futr_exog_list else None,
            'stat_exog_list': stat_exog_list if stat_exog_list else None,
        }
        
        # Add common training parameters with production defaults
        params.update({
            'learning_rate': model_cfg.get('learning_rate', 1e-3),
            'batch_size': model_cfg.get('batch_size', 512),
            'max_steps': model_cfg.get('max_steps', 20000),
            'early_stop_patience_steps': model_cfg.get('early_stop_patience_steps', 400),
            'val_check_steps': model_cfg.get('val_check_steps', 100),
            'random_seed': model_cfg.get('random_seed', 1337),
        })
        
        # Set input_size (default 1024, except PatchTST uses 2048)
        if model_type == 'PATCHTST':
            params['input_size'] = model_cfg.get('input_size', 2048)
        else:
            params['input_size'] = model_cfg.get('input_size', 1024)
        
        # Set scaler_type (default 'robust', except PatchTST uses 'revin')
        if alias in scaler_overrides:
            params['scaler_type'] = scaler_overrides[alias]
        elif model_type == 'PATCHTST':
            params['scaler_type'] = model_cfg.get('scaler_type', 'revin')
        else:
            params['scaler_type'] = model_cfg.get('scaler_type', 'robust')
        
        # Add loss function
        params['loss'] = _loss_ctor(model_cfg['loss'])
        
        # Add model-specific parameters
        if model_type == 'NHITS':
            # NHITS-specific (Section 4.1.C, lines 1351-1370)
            params.update({
                'n_blocks': model_cfg.get('n_blocks', [1, 1, 1]),
                'n_pool_kernel_size': model_cfg.get('n_pool_kernel_size', [2, 2, 1]),
                'dropout_prob_theta': model_cfg.get('dropout_prob_theta', 0.1),
            })
        
        elif model_type == 'NBEATSX':
            # NBEATSx-specific (Section 4.1.C, lines 1371-1390)
            params.update({
                'stack_types': model_cfg.get('stack_types', ['trend', 'seasonality']),
                'n_blocks': model_cfg.get('n_blocks', [2, 2]),
                'mlp_units': model_cfg.get('mlp_units', [[512, 512], [512, 512]]),
                'dropout_prob_theta': model_cfg.get('dropout_prob_theta', 0.1),
            })
        
        elif model_type == 'TIDE':
            # TiDE-specific (Section 4.1.C, lines 1391-1410)
            params.update({
                'hidden_size': model_cfg.get('hidden_size', 256),
                'num_encoder_layers': model_cfg.get('num_encoder_layers', 2),
                'num_decoder_layers': model_cfg.get('num_decoder_layers', 2),
                'dropout': model_cfg.get('dropout', 0.1),
            })
        
        elif model_type == 'PATCHTST':
            # PatchTST-specific (Section 4.1.C, lines 1411-1430)
            params.update({
                'patch_len': model_cfg.get('patch_len', 16),
                'stride': model_cfg.get('stride', 8),
                'n_heads': model_cfg.get('n_heads', 8),
                'hidden_size': model_cfg.get('hidden_size', 256),
                'revin': model_cfg.get('revin', True),
            })
        
        # Copy any additional model-specific params from config
        for key, value in model_cfg.items():
            if key not in ['alias', 'loss', 'input_size', 'scaler_type']:
                if key not in params:
                    params[key] = value
        
        # Prune unsupported parameters
        params = _prune_kwargs(model_cls, params)
        
        # Instantiate model
        try:
            model = model_cls(**params)
            models.append(model)
            print(f"  ✅ {alias}: {model_cls.__name__} with {params['loss'].__class__.__name__}")
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to instantiate {alias} ({model_cls.__name__}): {str(e)}"
            )
    
    print(f"\n✅ Successfully instantiated {len(models)} models")
    return models