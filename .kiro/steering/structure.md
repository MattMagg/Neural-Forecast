# Project Structure & Organization

## Directory Layout

### Core Modules
- **`data/`**: Canonical data frames and processed datasets
- **`features/`**: Feature engineering pipeline (registry + builders)
- **`nf_models/`**: Model factory and configurations  
- **`cv/`**: Cross-validation and hyperparameter optimization utilities
- **`uq/`**: Uncertainty quantification and diagnostics
- **`utils/`**: Validation, IO, and versioning utilities

### Experiment Organization
- **`experiments/`**: Per-horizon configurations and results
  - `h4.yaml`, `h8.yaml`, `h16.yaml`, `h32.yaml` (horizon configs)
  - `defaults.yaml` (shared parameters)
  - `h{horizon}/best/` (best model artifacts)
- **`reports/`**: Generated outputs and visualizations by horizon

### Entry Points
- **`run_train.py`**: Training and cross-validation orchestration (IMPLEMENTED)
  - Complete data assembly path: load → aggregate → regularize → canonical → validate
  - Command-line interface with data processing options
- **`run_predict.py`**: Inference and prediction pipeline
- **`settings.yaml`**: Global configuration parameters

## Key File Responsibilities

### Feature Engineering (`features/`)
- `registry.py`: Available indicators and feature definitions
- `builder.py`: Feature computation and alignment logic

### Model Management (`nf_models/`)
- `factory.py`: Model instantiation and configuration

### Cross-Validation (`cv/`)
- `runner.py`: CV execution and orchestration
- `hpo.py`: Hyperparameter optimization

### Uncertainty Quantification (`uq/`)
- `diag.py`: Metrics calculation and diagnostics
- `ensembles.py`: Model ensemble utilities

### Utilities (`utils/`)
- `validate.py`: Data validation and quality gates (IMPLEMENTED)
  - `assert_regular_grid()`, `assert_utc_eob()`, `assert_shifted()`, `assert_no_forward_fill_y()`
- `io.py`: Data loading and saving utilities (IMPLEMENTED)
  - `aggregate_1min_to_15min()`, `regularize_to_grid_utc()`, `make_nf_canonical()`, `drop_train_nans_and_winsorize()`
- `version.py`: Version tracking and management

## Naming Conventions

### Horizons
- Use `h{steps}` format: `h4`, `h8`, `h16`, `h32`
- Corresponds to 1h, 2h, 4h, 8h forecasting horizons

### Files & Artifacts
- Config files: `{horizon}.yaml`
- Model artifacts: `experiments/{horizon}/best/`
- Reports: `reports/{horizon}/`

### Code Style
- Snake_case for variables and functions
- PascalCase for classes
- ALL_CAPS for constants
- Descriptive names over brevity

## Quality Gates
All code must pass validation assertions (IMPLEMENTED in `utils/validate.py`):
- `assert_regular_grid(df, "15min")` - validates 15-minute time grid completeness
- `assert_utc_eob(df, "15min")` - validates UTC end-of-bar timestamp alignment
- `assert_shifted(df, hist_cols)` - detects data leakage using correlation analysis
- `assert_no_forward_fill_y(df)` - prevents target variable forward-filling

**Data Processing Pipeline**: Complete assembly path implemented in `run_train.py` processes 7M+ 1-minute bars into 477K+ 15-minute canonical format with 100% validation pass rate.