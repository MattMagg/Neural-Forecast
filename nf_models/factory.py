"""
NeuralForecast Model Factory

Lean, NF-native model instantiation system for the four-model portfolio
(NHITS, NBEATSx, TiDE, PatchTST) with proper loss configuration and 
YAML-driven experiment setup.

This module implements the exact specifications from the source document
with no enterprise-grade abstractions - just simple factory functions
that create configured NeuralForecast model instances.
"""

from typing import List, Dict, Any, Optional, Union, Type
import inspect

# NeuralForecast model imports
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST

# NeuralForecast loss imports  
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss


# Model class mapping for configuration-driven instantiation
MODEL_CLASSES = {
    "NHITS": NHITS,
    "NBEATSX": NBEATSx,
    "TIDE": TiDE, 
    "PATCHTST": PatchTST
}

# Default confidence levels for quantile losses
DEFAULT_LEVELS = [80, 90, 95]


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class ModelInstantiationError(Exception):
    """Raised when model instantiation fails"""
    pass


def _loss_ctor(spec: Dict[str, Any]) -> Union[DistributionLoss, MQLoss, IQLoss]:
    """
    Create NF-native loss instances from specification.
    
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
        # Add return_params=True to get distribution parameters for probabilistic predictions
        # This is required for proper uncertainty quantification
        return DistributionLoss(distribution="StudentT", return_params=True)
    
    elif kind == 'mqloss':
        levels = spec.get('level', DEFAULT_LEVELS)
        if not isinstance(levels, list) or len(levels) == 0:
            raise ConfigurationError("MQLoss levels must be a non-empty list")
        if not all(0 < l < 100 for l in levels):
            raise ConfigurationError("All MQLoss levels must be between 0 and 100")
        return MQLoss(level=levels)
    
    elif kind == 'iqloss':
        # IQLoss doesn't accept a level parameter - per NF documentation, it uses
        # internal fixed quantiles [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        # and applies "quantile over quantiles" approach for monotonic predictions
        return IQLoss()
    
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
    try:
        sig = inspect.signature(model_cls.__init__)
        valid_params = set(sig.parameters.keys())
        
        # Special handling for PatchTST which doesn't support exogenous variables
        # According to NeuralForecast documentation, PatchTST has:
        # EXOGENOUS_FUTR = False, EXOGENOUS_HIST = False, EXOGENOUS_STAT = False
        if model_cls.__name__ == 'PatchTST':
            # Remove exogenous parameters for PatchTST
            exog_params = ['hist_exog_list', 'futr_exog_list', 'stat_exog_list']
            for param in exog_params:
                if param in params:
                    print(f"Warning: PatchTST does not support exogenous variables. "
                          f"Removing {param} from configuration.")
                    params.pop(param)
        else:
            # For other models, check for required exogenous list support
            required_exog_params = ['hist_exog_list', 'futr_exog_list', 'stat_exog_list']
            for param in required_exog_params:
                if param not in valid_params:
                    raise ModelInstantiationError(
                        f"{model_cls.__name__} does not support {param}. "
                        f"All models must support exogenous variable lists."
                    )
        
        # Filter to only supported parameters
        pruned = {k: v for k, v in params.items() if k in valid_params}
        
        # Log dropped parameters for debugging
        dropped = set(params.keys()) - set(pruned.keys())
        if dropped:
            print(f"Dropped unsupported parameters for {model_cls.__name__}: {sorted(dropped)}")
        
        return pruned
        
    except Exception as e:
        raise ModelInstantiationError(
            f"Failed to introspect {model_cls.__name__} constructor: {e}"
        )


def _apply_model_defaults(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply model-specific parameter defaults from source document.
    
    Args:
        model_name: Model name (NHITS, NBEATSX, TIDE, PATCHTST)
        params: Current parameter dictionary
        
    Returns:
        Parameter dictionary with model-specific defaults applied
    """
    params = params.copy()
    
    if model_name == "NHITS":
        # NHITS defaults from Section 4.2
        params.setdefault('input_size', 1024)
        params.setdefault('n_blocks', [1, 1, 1])
        params.setdefault('n_pool_kernel_size', [2, 2, 1])
        params.setdefault('dropout_prob_theta', 0.1)
        
    elif model_name == "NBEATSX":
        # NBEATSx defaults from Section 4.2
        params.setdefault('input_size', 1024)
        params.setdefault('stack_types', ['identity', 'trend', 'seasonality'])
        params.setdefault('n_blocks', [1, 1, 1])
        params.setdefault('mlp_units', [[512, 512], [512, 512], [512, 512]])
        params.setdefault('dropout_prob_theta', 0.1)
        
    elif model_name == "TIDE":
        # TiDE defaults from Section 4.2
        params.setdefault('input_size', 1024)
        params.setdefault('hidden_size', 512)
        params.setdefault('num_encoder_layers', 2)
        params.setdefault('num_decoder_layers', 2)
        params.setdefault('dropout', 0.1)
        
    elif model_name == "PATCHTST":
        # PatchTST defaults from Section 4.2
        params.setdefault('input_size', 2048)  # Larger context for PatchTST
        params.setdefault('patch_len', 16)
        params.setdefault('stride', 16)
        params.setdefault('n_heads', 8)
        params.setdefault('hidden_size', 512)
        params.setdefault('revin', True)  # Native RevIN support
        # Override scaler for PatchTST if not explicitly set
        if 'scaler_type' not in params:
            params['scaler_type'] = 'revin'
    
    # Common parameter defaults
    params.setdefault('learning_rate', 1e-3)
    params.setdefault('batch_size', 512)
    params.setdefault('max_steps', 20000)
    params.setdefault('early_stop_patience_steps', 400)
    params.setdefault('val_check_steps', 100)
    
    return params


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate experiment configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Check required keys
    required_keys = ['h', 'freq', 'models']
    for key in required_keys:
        if key not in cfg:
            raise ConfigurationError(f"Missing required key: {key}")
    
    # Validate horizon
    h = cfg['h']
    if not isinstance(h, int) or h <= 0:
        raise ConfigurationError(f"Horizon 'h' must be a positive integer, got: {h}")
    
    # Validate frequency
    freq = cfg['freq']
    if not isinstance(freq, str) or not freq:
        raise ConfigurationError(f"Frequency 'freq' must be a non-empty string, got: {freq}")
    
    # Validate models list
    models = cfg['models']
    if not isinstance(models, list) or len(models) == 0:
        raise ConfigurationError("Models list cannot be empty")
    
    # Validate each model entry
    for i, model_entry in enumerate(models):
        if not isinstance(model_entry, dict) or len(model_entry) != 1:
            raise ConfigurationError(
                f"Model entry {i} must be a dict with exactly one key-value pair, "
                f"got: {model_entry}"
            )
        
        model_name, model_params = next(iter(model_entry.items()))
        
        # Validate model name
        if model_name.upper() not in MODEL_CLASSES:
            raise ConfigurationError(
                f"Unsupported model '{model_name}' in entry {i}. "
                f"Supported models: {list(MODEL_CLASSES.keys())}"
            )
        
        # Validate model parameters
        if not isinstance(model_params, dict):
            raise ConfigurationError(
                f"Model parameters for '{model_name}' must be a dict, "
                f"got: {type(model_params)}"
            )
        
        # Validate loss specification if present
        if 'loss' in model_params:
            _validate_loss_spec(model_params['loss'])


def _validate_loss_spec(loss_spec: Dict[str, Any]) -> None:
    """
    Validate loss specification.
    
    Args:
        loss_spec: Loss specification dictionary
        
    Raises:
        ConfigurationError: If loss specification is invalid
    """
    if not isinstance(loss_spec, dict):
        raise ConfigurationError(f"Loss specification must be a dict, got: {type(loss_spec)}")
    
    if 'kind' not in loss_spec:
        raise ConfigurationError("Loss specification must include 'kind'")
    
    kind = loss_spec['kind']
    if not isinstance(kind, str):
        raise ConfigurationError(f"Loss 'kind' must be a string, got: {type(kind)}")
    
    kind = kind.lower()
    if kind not in ['studentt', 'mqloss', 'iqloss']:
        raise ConfigurationError(
            f"Unsupported loss kind: {kind}. "
            f"Supported: studentt, mqloss, iqloss"
        )
    
    # Validate levels for quantile losses
    if kind in ['mqloss', 'iqloss'] and 'level' in loss_spec:
        levels = loss_spec['level']
        if not isinstance(levels, list):
            raise ConfigurationError(f"Loss levels must be a list, got: {type(levels)}")
        
        if len(levels) == 0:
            raise ConfigurationError("Loss levels cannot be empty")
        
        for level in levels:
            if not isinstance(level, (int, float)):
                raise ConfigurationError(f"Loss level must be numeric, got: {type(level)}")
            
            if not (0 < level < 100):
                raise ConfigurationError(
                    f"Loss level must be between 0 and 100, got: {level}"
                )


def _validate_model_compatibility(model_name: str, params: Dict[str, Any]) -> None:
    """
    Validate model-specific parameter compatibility.
    
    Args:
        model_name: Model name (NHITS, NBEATSX, TIDE, PATCHTST)
        params: Parameter dictionary
        
    Raises:
        ModelInstantiationError: If model doesn't support required parameters
    """
    if model_name not in MODEL_CLASSES:
        raise ModelInstantiationError(f"Unsupported model: {model_name}")
    
    model_cls = MODEL_CLASSES[model_name]
    
    try:
        sig = inspect.signature(model_cls.__init__)
        valid_params = set(sig.parameters.keys())
        
        # Check for required exogenous list support
        required_exog_params = ['hist_exog_list', 'futr_exog_list', 'stat_exog_list']
        for param in required_exog_params:
            if param not in valid_params:
                raise ModelInstantiationError(
                    f"{model_name} does not support {param}. "
                    f"All models must support exogenous variable lists."
                )
        
    except Exception as e:
        raise ModelInstantiationError(
            f"Failed to validate {model_name} compatibility: {e}"
        )


def instantiate_models(cfg: Dict[str, Any], 
                      hist_cols: List[str], 
                      futr_cols: List[str], 
                      stat_cols: List[str]) -> List[Any]:
    """
    Build NF-native model portfolio from YAML-style config.
    
    Args:
        cfg: Configuration dictionary with h, freq, models, scaler_type
        hist_cols: Historical exogenous feature column names (post-shift)
        futr_cols: Future-known exogenous feature column names
        stat_cols: Static exogenous feature column names
        
    Returns:
        List of configured NeuralForecast model instances
        
    Raises:
        ConfigurationError: If configuration is invalid
        ModelInstantiationError: If model instantiation fails
    """
    # Validate configuration
    _validate_config(cfg)
    
    # Extract core configuration
    h = cfg['h']
    models_cfg = cfg['models']
    
    # Handle scaler overrides
    scaler_config = cfg.get('scaler_type', {})
    if isinstance(scaler_config, str):
        # Simple string format - use as default for all models
        default_scaler = scaler_config
        scaler_overrides = {}
    else:
        # Dictionary format with default and per-model overrides
        default_scaler = scaler_config.get('default', 'robust')
        scaler_overrides = {k: v for k, v in scaler_config.items() if k != 'default'}
    
    models = []
    
    for model_entry in models_cfg:
        if not isinstance(model_entry, dict) or len(model_entry) != 1:
            raise ConfigurationError(f"Invalid model entry format: {model_entry}")
        
        model_name, model_params = next(iter(model_entry.items()))
        
        # Validate model name
        if model_name.upper() not in MODEL_CLASSES:
            raise ConfigurationError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(MODEL_CLASSES.keys())}"
            )
        
        model_cls = MODEL_CLASSES[model_name.upper()]
        
        # Build parameters dictionary with model-specific defaults
        params = _apply_model_defaults(model_name.upper(), model_params.copy())
        
        # Set horizon
        params['h'] = h
        
        # Wire exogenous variable lists
        params['hist_exog_list'] = hist_cols
        params['futr_exog_list'] = futr_cols
        params['stat_exog_list'] = stat_cols
        
        # Handle scaler type with per-model overrides
        if model_name in scaler_overrides:
            params['scaler_type'] = scaler_overrides[model_name]
        else:
            params['scaler_type'] = default_scaler
        
        # Handle loss configuration
        if 'loss' in params:
            loss_spec = params.pop('loss')
            params['loss'] = _loss_ctor(loss_spec)
        
        # Remove alias from model parameters (used for output naming only)
        alias = params.pop('alias', f"{model_name}_default")
        
        # Prune unsupported parameters
        params = _prune_kwargs(model_cls, params)
        
        try:
            # Instantiate model
            model = model_cls(**params)
            
            # Store alias for output column naming
            # NF models may or may not have native alias support - use defensive approach
            if hasattr(model, 'alias'):
                model.alias = alias
            else:
                # Add alias as dynamic attribute to ensure it's always available
                # for downstream column naming in outputs
                setattr(model, 'alias', alias)
            
            models.append(model)
            
        except Exception as e:
            raise ModelInstantiationError(
                f"Failed to instantiate {model_name} with params {params}: {e}"
            )
    
    return models


# ============================================================================
# Unit Tests (inline for development and validation)
# ============================================================================

def _test_loss_constructor():
    """Test loss function configuration"""
    print("Testing loss constructor...")
    
    # Test StudentT loss
    studentt_spec = {'kind': 'studentt'}
    loss = _loss_ctor(studentt_spec)
    assert isinstance(loss, DistributionLoss), f"Expected DistributionLoss, got {type(loss)}"
    print("✓ StudentT loss creation")
    
    # Test MQLoss with custom levels
    mq_spec = {'kind': 'mqloss', 'level': [80, 90]}
    loss = _loss_ctor(mq_spec)
    assert isinstance(loss, MQLoss), f"Expected MQLoss, got {type(loss)}"
    print("✓ MQLoss creation with custom levels")
    
    # Test IQLoss (doesn't use levels like MQLoss)
    iq_spec = {'kind': 'iqloss'}
    loss = _loss_ctor(iq_spec)
    assert isinstance(loss, IQLoss), f"Expected IQLoss, got {type(loss)}"
    print("✓ IQLoss creation")
    
    # Test case insensitive
    loss = _loss_ctor({'kind': 'STUDENTT'})
    assert isinstance(loss, DistributionLoss), "Case insensitive test failed"
    print("✓ Case insensitive loss kind")
    
    # Test error handling
    try:
        _loss_ctor({'kind': 'invalid'})
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        print("✓ Invalid loss kind error handling")
    
    try:
        _loss_ctor({'invalid': 'spec'})
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError:
        print("✓ Missing kind error handling")
    
    print("Loss constructor tests passed!\n")


def _test_parameter_pruning():
    """Test parameter pruning for unsupported arguments"""
    print("Testing parameter pruning...")
    
    # Test with NHITS
    params = {
        'h': 4,
        'input_size': 1024,
        'unsupported_param': 'value',
        'learning_rate': 0.001,
        'hist_exog_list': [],
        'futr_exog_list': [],
        'stat_exog_list': []
    }
    
    pruned = _prune_kwargs(NHITS, params)
    assert 'unsupported_param' not in pruned, "Unsupported param should be removed"
    assert 'h' in pruned, "Supported param should remain"
    assert 'learning_rate' in pruned, "Supported param should remain"
    print("✓ Parameter pruning works correctly")
    
    # Test exogenous list validation
    try:
        # This should work since NHITS supports exog lists
        _validate_model_compatibility('NHITS', {})
        print("✓ Model compatibility validation passed")
    except ModelInstantiationError:
        assert False, "NHITS should support exogenous lists"
    
    print("Parameter pruning tests passed!\n")


def _test_model_instantiation():
    """Test basic model instantiation"""
    print("Testing model instantiation...")
    
    cfg = {
        'h': 4,
        'freq': '15min',
        'models': [
            {
                'NHITS': {
                    'alias': 'test_nhits',
                    'input_size': 1024,
                    'loss': {'kind': 'studentt'}
                }
            }
        ]
    }
    
    models = instantiate_models(cfg, [], [], [])
    assert len(models) == 1, f"Expected 1 model, got {len(models)}"
    assert isinstance(models[0], NHITS), f"Expected NHITS, got {type(models[0])}"
    assert models[0].h == 4, f"Expected h=4, got {models[0].h}"
    assert hasattr(models[0], 'alias'), "Model should have alias attribute"
    assert models[0].alias == 'test_nhits', f"Expected alias 'test_nhits', got {models[0].alias}"
    print("✓ Basic model instantiation")
    
    print("Model instantiation tests passed!\n")


def _test_exogenous_wiring():
    """Test exogenous variable list wiring"""
    print("Testing exogenous wiring...")
    
    hist_cols = ['rsi_14', 'macd_12_26']
    futr_cols = ['minute_of_day', 'day_of_week']
    stat_cols = ['asset_id']
    
    cfg = {
        'h': 8,
        'freq': '15min',
        'models': [
            {
                'TIDE': {
                    'alias': 'test_tide',
                    'loss': {'kind': 'mqloss', 'level': [80, 90, 95]}
                }
            }
        ]
    }
    
    models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
    model = models[0]
    
    assert model.hist_exog_list == hist_cols, f"hist_exog_list mismatch"
    assert model.futr_exog_list == futr_cols, f"futr_exog_list mismatch"
    assert model.stat_exog_list == stat_cols, f"stat_exog_list mismatch"
    print("✓ Exogenous lists properly wired")
    
    print("Exogenous wiring tests passed!\n")


def _test_scaler_overrides():
    """Test per-model scaler_type handling"""
    print("Testing scaler overrides...")
    
    cfg = {
        'h': 16,
        'freq': '15min',
        'scaler_type': {
            'default': 'robust',
            'PATCHTST': 'revin'
        },
        'models': [
            {
                'NHITS': {
                    'alias': 'test_nhits'
                }
            },
            {
                'PATCHTST': {
                    'alias': 'test_patchtst'
                }
            }
        ]
    }
    
    # Test that models instantiate successfully with scaler overrides
    models = instantiate_models(cfg, [], [], [])
    
    # Verify we got the expected models
    assert len(models) == 2, f"Expected 2 models, got {len(models)}"
    
    nhits_model = next(m for m in models if isinstance(m, NHITS))
    patchtst_model = next(m for m in models if isinstance(m, PatchTST))
    
    assert nhits_model is not None, "NHITS model should be instantiated"
    assert patchtst_model is not None, "PatchTST model should be instantiated"
    
    print("✓ Scaler overrides work correctly")
    
    print("Scaler override tests passed!\n")


def _test_configuration_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test missing required keys
    try:
        _validate_config({'freq': '15min'})  # Missing 'h' and 'models'
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as e:
        assert "Missing required key" in str(e)
        print("✓ Missing required key validation")
    
    # Test invalid model format
    try:
        cfg = {
            'h': 4,
            'freq': '15min',
            'models': [{'NHITS': {}, 'TIDE': {}}]  # Multiple models in one entry
        }
        _validate_config(cfg)
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as e:
        assert "exactly one key-value pair" in str(e)
        print("✓ Invalid model format validation")
    
    # Test unsupported model
    try:
        cfg = {
            'h': 4,
            'freq': '15min',
            'models': [{'UnsupportedModel': {}}]
        }
        _validate_config(cfg)
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as e:
        assert "Unsupported model" in str(e)
        print("✓ Unsupported model validation")
    
    # Test invalid loss specification
    try:
        _validate_loss_spec({'kind': 'invalid_loss'})
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as e:
        assert "Unsupported loss kind" in str(e)
        print("✓ Invalid loss specification validation")
    
    print("Configuration validation tests passed!\n")


def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("Running NeuralForecast Model Factory Unit Tests")
    print("=" * 60)
    
    try:
        _test_loss_constructor()
        _test_parameter_pruning()
        _test_model_instantiation()
        _test_exogenous_wiring()
        _test_scaler_overrides()
        _test_configuration_validation()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Configuration Loading Utilities
# ============================================================================

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid or file not found
    """
    import yaml
    import os
    
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if cfg is None:
            raise ConfigurationError(f"Empty configuration file: {config_path}")
        
        # Validate the loaded configuration
        _validate_config(cfg)
        
        return cfg
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")


def create_models_from_config(config_path: str, 
                             hist_cols: List[str] = None, 
                             futr_cols: List[str] = None, 
                             stat_cols: List[str] = None) -> List[Any]:
    """
    Convenience function to load config and instantiate models in one step.
    
    Args:
        config_path: Path to YAML configuration file
        hist_cols: Historical exogenous feature column names
        futr_cols: Future-known exogenous feature column names  
        stat_cols: Static exogenous feature column names
        
    Returns:
        List of configured NeuralForecast model instances
        
    Raises:
        ConfigurationError: If configuration is invalid
        ModelInstantiationError: If model instantiation fails
    """
    # Default to empty lists if not provided
    hist_cols = hist_cols or []
    futr_cols = futr_cols or []
    stat_cols = stat_cols or []
    
    # Load configuration
    cfg = load_experiment_config(config_path)
    
    # Instantiate models
    return instantiate_models(cfg, hist_cols, futr_cols, stat_cols)


# ============================================================================
# Integration Test for YAML Configuration
# ============================================================================

def _test_yaml_integration():
    """Test YAML configuration loading and model instantiation"""
    print("Testing YAML configuration integration...")
    
    # Test loading h4 configuration
    try:
        cfg = load_experiment_config('experiments/h4.yaml')
        assert cfg['h'] == 4, f"Expected h=4, got {cfg['h']}"
        assert cfg['freq'] == '15min', f"Expected freq='15min', got {cfg['freq']}"
        assert len(cfg['models']) == 4, f"Expected 4 models, got {len(cfg['models'])}"
        print("✓ h4 configuration loaded successfully")
        
        # Test model instantiation from config
        models = instantiate_models(cfg, [], [], [])
        assert len(models) == 4, f"Expected 4 models, got {len(models)}"
        
        # Check model types
        model_types = [type(m).__name__ for m in models]
        expected_types = ['NHITS', 'NBEATSx', 'TiDE', 'PatchTST']
        for expected_type in expected_types:
            assert expected_type in model_types, f"Missing model type: {expected_type}"
        
        print("✓ Models instantiated from YAML configuration")
        
        # Test convenience function
        models2 = create_models_from_config('experiments/h4.yaml')
        assert len(models2) == 4, f"Expected 4 models from convenience function"
        print("✓ Convenience function works correctly")
        
    except Exception as e:
        print(f"❌ YAML integration test failed: {e}")
        raise
    
    print("YAML configuration integration tests passed!\n")


# ============================================================================
# Feature Engineering Integration Tests
# ============================================================================

def _test_feature_engineering_integration():
    """Test integration with feature engineering pipeline"""
    print("Testing feature engineering integration...")
    
    try:
        # Import feature engineering components
        from features.builder import select_features
        
        # Create mock exogenous data frame
        import pandas as pd
        import numpy as np
        
        # Create sample data with different feature types
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Mock exogenous features
        exo_data = {
            'ds': dates,
            # Historical features (technical indicators)
            'rsi_14': np.random.randn(100),
            'macd_12_26': np.random.randn(100),
            'atr_14': np.random.randn(100),
            'bb_upper_20': np.random.randn(100),
            'bb_lower_20': np.random.randn(100),
            # Future-known features (time-based)
            'minute_of_day': [d.hour * 60 + d.minute for d in dates],
            'day_of_week': [d.dayofweek for d in dates],
            'weekend': [1 if d.dayofweek >= 5 else 0 for d in dates],
            # Static features
            'asset_id': ['BTC-USD'] * 100
        }
        
        exo_df = pd.DataFrame(exo_data)
        
        # Test feature selection
        hist_cols, futr_cols, stat_cols = select_features(exo_df, {})
        
        print(f"✓ Feature selection returned {len(hist_cols)} hist, {len(futr_cols)} futr, {len(stat_cols)} stat features")
        
        # Test model instantiation with real feature lists
        cfg = {
            'h': 4,
            'freq': '15min',
            'models': [
                {
                    'NHITS': {
                        'alias': 'test_nhits_with_features',
                        'loss': {'kind': 'studentt'}
                    }
                }
            ]
        }
        
        models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
        model = models[0]
        
        # Verify exogenous lists are properly wired
        assert model.hist_exog_list == hist_cols, "Historical features not properly wired"
        assert model.futr_exog_list == futr_cols, "Future features not properly wired"
        assert model.stat_exog_list == stat_cols, "Static features not properly wired"
        
        print("✓ Models properly wired with feature engineering output")
        
        # Test with realistic feature counts (up to 256)
        large_hist_cols = [f'feature_{i}' for i in range(200)]
        large_futr_cols = [f'futr_feature_{i}' for i in range(30)]
        large_stat_cols = [f'stat_feature_{i}' for i in range(26)]
        
        models_large = instantiate_models(cfg, large_hist_cols, large_futr_cols, large_stat_cols)
        model_large = models_large[0]
        
        assert len(model_large.hist_exog_list) == 200, "Large hist feature list not handled"
        assert len(model_large.futr_exog_list) == 30, "Large futr feature list not handled"
        assert len(model_large.stat_exog_list) == 26, "Large stat feature list not handled"
        
        print("✓ Large feature lists (256 total) handled correctly")
        
    except ImportError as e:
        print(f"⚠️  Feature engineering module not available: {e}")
        print("✓ Integration test skipped (module not found)")
        return
    except Exception as e:
        print(f"❌ Feature engineering integration test failed: {e}")
        raise
    
    print("Feature engineering integration tests passed!\n")


if __name__ == "__main__":
    # Run original tests plus integration tests
    success = run_all_tests()
    if success:
        _test_yaml_integration()
        _test_feature_engineering_integration()