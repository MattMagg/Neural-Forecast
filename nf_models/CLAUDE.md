# NeuralForecast Model Factory Module

## Directory Overview
Handles instantiation and configuration of NeuralForecast models (NHITS, NBEATSx, TiDE, PatchTST).

## Files and Purpose

### ✅ **factory.py** (ACTIVELY USED)
- **Purpose**: Main model instantiation module
- **Key Function**:
  - `instantiate_models(cfg, verbose=True)` - Creates NF model instances
- **Features**:
  - Loss configuration (StudentT, MQLoss, IQLoss)
  - Parameter validation and pruning
  - Exogenous variable wiring
  - Model-specific parameter handling
- **Pipeline Role**: Called at line 378 in run_train.ipynb

### ❌ **factory_core.py** (NOT USED)
- **Purpose**: Appears to be duplicate/earlier version of factory.py
- **Status**: Contains similar code but not imported anywhere
- **Note**: Candidate for removal

### ✅ **__init__.py** (REQUIRED)
- **Purpose**: Makes nf_models/ a Python package
- **Status**: Empty but necessary for imports

### 📝 **01_model_factory_core.ipynb** (DEVELOPMENT TOOL)
- **Purpose**: Development notebook for model factory
- **Content**: Interactive model configuration, testing with synthetic data
- **Usage**: Development only, demonstrates factory usage

### 📝 **02_integration_tests.ipynb** (DEVELOPMENT TOOL)
- **Purpose**: Integration testing notebook
- **Usage**: Development testing, not part of pipeline

### 📝 **03_performance_profiling.ipynb** (DEVELOPMENT TOOL)
- **Purpose**: Performance and memory profiling
- **Usage**: Development analysis, not part of pipeline

### 📝 **04_usage_documentation.ipynb** (DEVELOPMENT TOOL)
- **Purpose**: Usage examples and documentation
- **Usage**: Reference material, not part of pipeline

## System Flow
```
run_train.ipynb
    ↓ line 63: import
from nf_models.factory import instantiate_models
    ↓ line 378: call
models = instantiate_models(cfg, verbose=True)
    ↓ creates
[NHITS, NBEATSx, TiDE, PatchTST] instances
    ↓ with
- Configured losses (StudentT/MQLoss/IQLoss)
- Exogenous variables (hist/futr/stat)
- Model-specific parameters
    ↓ passes to
NeuralForecast(models=models)
```

## Model Configurations
- **NHITS**: StudentT loss, input_size=1024
- **NBEATSx**: MQLoss, input_size=1024
- **TiDE**: IQLoss, input_size=1024
- **PatchTST**: StudentT loss, input_size=2048, revin scaler

## Key Insights
- Only factory.py is used in production
- factory_core.py is redundant and should be removed
- All notebooks are development/documentation tools
- Factory handles all NF-specific model configuration complexity