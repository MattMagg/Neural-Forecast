# Design Document

## Overview

The NeuralForecast Model Factory is a lean, NF-native model instantiation system that creates configured model instances for the four-model portfolio (NHITS, NBEATSx, TiDE, PatchTST) specified in the source document. The design follows the "keep it lean and explicit" principle with simple factory functions, no complex class hierarchies, and direct use of NeuralForecast's native APIs. The system integrates seamlessly with YAML-driven experiment configuration and the existing feature engineering pipeline to provide a complete model instantiation solution.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Experiment    │    │  Feature Lists   │    │  Model Factory  │
│  Configuration  │───▶│  (hist/futr/stat)│───▶│   Functions     │
│   (YAML)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  NeuralForecast │◀───│  Model Instance  │◀───│  Loss Functions │
│   Orchestrator  │    │     List         │    │  (StudentT/MQ)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Architecture

The system consists of three main architectural layers:

1. **Configuration Layer**: YAML-driven experiment configuration with model-specific parameters
2. **Factory Layer**: Core instantiation logic with loss configuration and parameter validation
3. **Integration Layer**: Seamless integration with NeuralForecast orchestrator and feature engineering

### Design Principles

- **NF-Native**: Use NeuralForecast models, losses, and APIs directly without abstraction
- **Lean Implementation**: Simple factory functions, no enterprise-grade class hierarchies
- **YAML-Driven**: Configuration-based model instantiation with parameter inheritance
- **Type Safety**: Proper parameter validation and constructor argument pruning
- **Integration-First**: Designed to work seamlessly with existing data processing and feature engineering

## Components and Interfaces

### Core Factory Module (`nf_models/factory.py`)

#### Primary Interface

```python
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
    """
```

#### Loss Configuration Interface

```python
def _loss_ctor(spec: Dict[str, Any]) -> Union[DistributionLoss, MQLoss, IQLoss]:
    """
    Create NF-native loss instances from specification.
    
    Args:
        spec: Loss specification with 'kind' and optional 'quantiles'
              Examples:
              - {'kind': 'studentt'}
              - {'kind': 'mqloss', 'level': [80, 90]}
              - {'kind': 'iqloss', 'level': [80, 90]}
    
    Returns:
        Configured NF loss instance
    """
```

#### Parameter Validation Interface

```python
def _prune_kwargs(model_cls: Type, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unsupported constructor arguments using introspection.
    
    Args:
        model_cls: NeuralForecast model class (NHITS, NBEATSx, TiDE, PatchTST)
        params: Parameter dictionary from configuration
        
    Returns:
        Filtered parameter dictionary with only supported arguments
    """
```

### Model Class Mapping

The factory maintains a simple mapping from configuration names to NF model classes:

```python
MODEL_CLASSES = {
    "NHITS": NHITS,
    "NBEATSX": NBEATSx, 
    "TIDE": TiDE,
    "PATCHTST": PatchTST
}
```

### Configuration Schema

#### YAML Configuration Structure

```yaml
# Global settings
seed: 1337
freq: "15min"
h: 16  # horizon steps

# Scaler configuration with per-model overrides
scaler_type:
  default: robust
  PatchTST: revin

# Cross-validation settings
n_windows: 6
step_size: 16
val_size: 64
refit: true

# Exogenous variable lists (populated at runtime)
hist_exog_list: []
futr_exog_list: [minute_of_day, day_of_week, weekend]
stat_exog_list: [asset_id]

# Model portfolio configuration
models:
  - NHITS:
      alias: NHITS_t1024_T
      input_size: 1024
      loss: {kind: studentt}
      learning_rate: 0.001
      batch_size: 512
      n_blocks: [1,1,1]
      early_stop_patience_steps: 400
      max_steps: 20000
      
  - PatchTST:
      alias: PatchTST_t2048_T
      input_size: 2048
      patch_len: 16
      stride: 16
      revin: true
      loss: {kind: studentt}
      learning_rate: 0.0005
      batch_size: 512
```

## Data Models

### Configuration Data Model

```python
@dataclass
class ModelConfig:
    """Individual model configuration"""
    name: str                    # Model class name (NHITS, NBEATSx, etc.)
    alias: str                   # Stable output column name
    input_size: int              # Historical context window size
    loss_spec: Dict[str, Any]    # Loss configuration
    learning_rate: float         # Training learning rate
    batch_size: int              # Training batch size
    max_steps: int               # Maximum training steps
    early_stop_patience_steps: int  # Early stopping patience
    model_specific_params: Dict[str, Any]  # Model-specific parameters

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    seed: int                    # Random seed for reproducibility
    freq: str                    # Time series frequency
    h: int                       # Forecast horizon
    scaler_type: Dict[str, str]  # Scaler configuration with overrides
    models: List[ModelConfig]    # Model portfolio configuration
    cv_params: Dict[str, Any]    # Cross-validation parameters
```

### Loss Specification Data Model

```python
@dataclass
class LossSpec:
    """Loss function specification"""
    kind: Literal["studentt", "mqloss", "iqloss"]
    level: Optional[List[int]] = None  # For quantile losses (confidence levels)
    
    # Default levels for symmetric coverage
    DEFAULT_LEVELS = [80, 90, 95]
```

### Model Parameter Data Model

Each model type has specific parameter requirements based on NeuralForecast signatures:

```python
# NHITS Parameters
@dataclass
class NHITSParams:
    n_blocks: List[int] = field(default_factory=lambda: [1,1,1])
    n_pool_kernel_size: List[int] = field(default_factory=lambda: [2,2,1])
    dropout_prob_theta: float = 0.1

# NBEATSx Parameters  
@dataclass
class NBEATSxParams:
    stack_types: List[str] = field(default_factory=lambda: ["identity", "trend", "seasonality"])
    n_blocks: List[int] = field(default_factory=lambda: [1,1,1])
    mlp_units: List[List[int]] = field(default_factory=lambda: [[512,512],[512,512],[512,512]])
    dropout_prob_theta: float = 0.1

# TiDE Parameters
@dataclass
class TiDEParams:
    hidden_size: int = 512
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout: float = 0.1

# PatchTST Parameters
@dataclass
class PatchTSTParams:
    patch_len: int = 16
    stride: int = 16
    n_heads: int = 8
    hidden_size: int = 512
    revin: bool = True
```

## Error Handling

### Configuration Validation

The system implements comprehensive validation for configuration errors:

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class ModelInstantiationError(Exception):
    """Raised when model instantiation fails"""
    pass

def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate experiment configuration"""
    required_keys = ['h', 'freq', 'models']
    for key in required_keys:
        if key not in cfg:
            raise ConfigurationError(f"Missing required key: {key}")
    
    if not isinstance(cfg['models'], list) or len(cfg['models']) == 0:
        raise ConfigurationError("Models list cannot be empty")
    
    for model_entry in cfg['models']:
        if not isinstance(model_entry, dict) or len(model_entry) != 1:
            raise ConfigurationError(f"Invalid model entry format: {model_entry}")
```

### Model Compatibility Validation

```python
def validate_model_compatibility(model_name: str, params: Dict[str, Any]) -> None:
    """Validate model-specific parameter compatibility"""
    if model_name not in MODEL_CLASSES:
        raise ModelInstantiationError(f"Unsupported model: {model_name}")
    
    model_cls = MODEL_CLASSES[model_name]
    sig = inspect.signature(model_cls.__init__)
    
    # Check for required exogenous list support
    required_exog_params = ['hist_exog_list', 'futr_exog_list', 'stat_exog_list']
    for param in required_exog_params:
        if param not in sig.parameters:
            raise ModelInstantiationError(f"{model_name} does not support {param}")
```

### Loss Configuration Validation

```python
def validate_loss_spec(loss_spec: Dict[str, Any]) -> None:
    """Validate loss specification"""
    if 'kind' not in loss_spec:
        raise ConfigurationError("Loss specification must include 'kind'")
    
    kind = loss_spec['kind'].lower()
    if kind not in ['studentt', 'mqloss', 'iqloss']:
        raise ConfigurationError(f"Unsupported loss kind: {kind}")
    
    if kind in ['mqloss', 'iqloss'] and 'level' in loss_spec:
        levels = loss_spec['level']
        if not isinstance(levels, list) or len(levels) == 0:
            raise ConfigurationError("Levels must be a non-empty list")
        
        if not all(0 < l < 100 for l in levels):
            raise ConfigurationError("All levels must be between 0 and 100")
```

## Testing Strategy

### Unit Testing Approach

The testing strategy focuses on the core factory functions and configuration validation:

#### Factory Function Tests

```python
def test_instantiate_models_basic():
    """Test basic model instantiation with minimal configuration"""
    cfg = {
        'h': 4,
        'freq': '15min',
        'models': [
            {'NHITS': {'alias': 'test_nhits', 'input_size': 1024, 'loss': {'kind': 'studentt'}}}
        ]
    }
    
    models = instantiate_models(cfg, [], [], [])
    assert len(models) == 1
    assert isinstance(models[0], NHITS)
    assert models[0].h == 4

def test_loss_configuration():
    """Test loss function configuration"""
    # Test StudentT loss
    studentt_spec = {'kind': 'studentt'}
    loss = _loss_ctor(studentt_spec)
    assert isinstance(loss, DistributionLoss)
    
    # Test MQLoss with custom levels
    mq_spec = {'kind': 'mqloss', 'level': [80, 90]}
    loss = _loss_ctor(mq_spec)
    assert isinstance(loss, MQLoss)

def test_parameter_pruning():
    """Test parameter pruning for unsupported arguments"""
    params = {
        'h': 4,
        'input_size': 1024,
        'unsupported_param': 'value',
        'learning_rate': 0.001
    }
    
    pruned = _prune_kwargs(NHITS, params)
    assert 'unsupported_param' not in pruned
    assert 'h' in pruned
    assert 'learning_rate' in pruned
```

#### Integration Tests

```python
def test_full_pipeline_integration():
    """Test integration with feature engineering pipeline"""
    # Mock feature lists from feature engineering
    hist_cols = ['rsi_14', 'macd_12_26', 'atr_14']
    futr_cols = ['minute_of_day', 'day_of_week']
    stat_cols = ['asset_id']
    
    cfg = load_experiment_config('experiments/h4.yaml')
    models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
    
    # Verify exogenous lists are properly wired
    for model in models:
        assert model.hist_exog_list == hist_cols
        assert model.futr_exog_list == futr_cols
        assert model.stat_exog_list == stat_cols

def test_neuralforecast_compatibility():
    """Test compatibility with NeuralForecast orchestrator"""
    cfg = load_experiment_config('experiments/h4.yaml')
    models = instantiate_models(cfg, [], [], [])
    
    # Should be able to create NeuralForecast instance
    nf = NeuralForecast(models=models, freq=cfg['freq'])
    assert nf.freq == '15min'
    assert len(nf.models) == len(models)
```

#### Configuration Validation Tests

```python
def test_config_validation():
    """Test configuration validation"""
    # Test missing required keys
    invalid_cfg = {'freq': '15min'}  # Missing 'h' and 'models'
    with pytest.raises(ConfigurationError):
        validate_config(invalid_cfg)
    
    # Test invalid model format
    invalid_cfg = {
        'h': 4,
        'freq': '15min',
        'models': [{'NHITS': {}, 'TiDE': {}}]  # Multiple models in one entry
    }
    with pytest.raises(ConfigurationError):
        validate_config(invalid_cfg)

def test_model_compatibility():
    """Test model compatibility validation"""
    # Test unsupported model
    with pytest.raises(ModelInstantiationError):
        validate_model_compatibility('UnsupportedModel', {})
    
    # Test supported model
    validate_model_compatibility('NHITS', {})  # Should not raise
```

### Performance Testing

```python
def test_instantiation_performance():
    """Test model instantiation performance"""
    cfg = load_experiment_config('experiments/h16.yaml')
    hist_cols = ['feature_' + str(i) for i in range(100)]  # 100 features
    
    start_time = time.time()
    models = instantiate_models(cfg, hist_cols, [], [])
    end_time = time.time()
    
    # Should instantiate quickly even with many features
    assert end_time - start_time < 1.0  # Less than 1 second
    assert len(models) > 0

def test_memory_usage():
    """Test memory usage of instantiated models"""
    cfg = load_experiment_config('experiments/h32.yaml')
    
    # Monitor memory before and after instantiation
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    models = instantiate_models(cfg, [], [], [])
    
    memory_after = process.memory_info().rss
    memory_increase = memory_after - memory_before
    
    # Memory increase should be reasonable (less than 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

### Error Handling Tests

```python
def test_error_handling():
    """Test comprehensive error handling"""
    # Test invalid loss specification
    cfg = {
        'h': 4,
        'freq': '15min',
        'models': [
            {'NHITS': {'loss': {'kind': 'invalid_loss'}}}
        ]
    }
    
    with pytest.raises(ConfigurationError):
        instantiate_models(cfg, [], [], [])
    
    # Test invalid levels
    cfg['models'][0]['NHITS']['loss'] = {
        'kind': 'mqloss',
        'level': [50, 150]  # Invalid level > 100
    }
    
    with pytest.raises(ConfigurationError):
        instantiate_models(cfg, [], [], [])
```

## Integration Points

### Feature Engineering Integration

The factory integrates with the feature engineering pipeline through the feature list interface:

```python
# In run_train.py
from features.builder import select_features
from nf_models.factory import instantiate_models

# Get feature lists from feature engineering
hist_cols, futr_cols, stat_cols = select_features(exo_df, policy={})

# Pass to model factory
models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
```

### NeuralForecast Integration

The factory produces model instances compatible with NeuralForecast orchestrator:

```python
# Direct integration with NeuralForecast
from neuralforecast import NeuralForecast

models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
nf = NeuralForecast(models=models, freq=cfg['freq'])

# Standard NF workflow
nf.fit(df=train_df, val_size=val_size)
forecasts = nf.predict(df=test_df)
cv_results = nf.cross_validation(df=df, n_windows=6, step_size=h)
```

### Configuration System Integration

The factory integrates with YAML-based experiment configuration:

```python
# Load experiment configuration
import yaml

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load and validate experiment configuration"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    validate_config(cfg)
    return cfg

# Usage in training scripts
cfg = load_experiment_config(f'experiments/h{horizon}.yaml')
models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
```

### Validation System Integration

The factory integrates with the existing validation system:

```python
# Integration with data validation
from utils.validate import assert_shifted

# After model instantiation, validate data
nf_df = merge_features_with_canonical_data(nf_base, exo_df)
assert_shifted(nf_df, hist_cols)  # Validate no leakage

# Proceed with model training
models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
nf = NeuralForecast(models=models, freq=cfg['freq'])
```

This design ensures seamless integration with all existing system components while maintaining the lean, NF-native approach specified in the requirements.