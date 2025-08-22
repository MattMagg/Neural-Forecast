# Cross-Validation Module

## Directory Overview
This module handles all cross-validation operations for the Neural-Forecast training pipeline.

## Files and Purpose

### ✅ **runner.py** (ACTIVELY USED)
- **Purpose**: Core cross-validation execution and orchestration
- **Key Functions**:
  - `run_cv()` - Executes NeuralForecast native cross-validation
  - `summarize_cv()` - Processes CV results and generates metrics
  - `save_cv_results()` - Persists CV outputs to disk
  - `generate_cv_summary_report()` - Creates markdown reports
- **Dependencies**: Uses utils/error_recovery.py and utils/risk_mitigation.py
- **Pipeline Role**: Called by run_train.ipynb at line 60, executes entire CV workflow

### ❌ **hpo.py** (NOT USED)
- **Purpose**: Hyperparameter optimization utilities
- **Status**: Available but not integrated into current pipeline
- **Note**: Could be used for future HPO implementation

### ✅ **__init__.py** (REQUIRED)
- **Purpose**: Makes cv/ a Python package
- **Status**: Empty but necessary for imports

### 📝 **README.md** (DOCUMENTATION)
- **Purpose**: Module documentation
- **Content**: Explains CV features, usage, and configuration

## System Flow
```
run_train.ipynb
    ↓ imports (line 60)
cv.runner functions
    ↓ calls (line 609)
run_cv() → cross-validation execution
    ↓
summarize_cv() → metrics computation
    ↓
save_cv_results() → data persistence
    ↓
generate_cv_summary_report() → reporting
```

## Key Insights
- Only runner.py is essential for training
- hpo.py exists for future HPO capabilities
- Module provides complete CV pipeline with metrics, saving, and reporting