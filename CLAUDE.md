# Neural-Forecast Project Guide

## Project Overview
Intraday BTC forecasting system using 15-minute bars with calibrated prediction intervals. Built with a **NeuralForecast-centric** approach - no custom implementations where NF provides native functionality.

## Key Documents

### Planning & Design
- **Technical Specification**: See `docs/forecasting_sf_plan.md` for complete technical details
- **Implementation Workflow**: See `implementation_workflow.md` for 8-week project timeline
- **Versioning System**: See `docs/versioning_system.md` for version tracking

## Project Structure

### Core Directories
- **`data/`**: Canonical data frames and processed datasets
- **`features/`**: Feature engineering pipeline (registry + builders)
- **`nf_models/`**: Model factory and configurations
- **`cv/`**: Cross-validation and HPO utilities
- **`experiments/`**: Per-horizon configs and results
- **`uq/`**: Uncertainty quantification and diagnostics
- **`reports/`**: Generated outputs and visualizations
- **`utils/`**: Validation, IO, and versioning utilities

### Entry Points
- **`run_train.py`**: Training and cross-validation orchestration
- **`run_predict.py`**: Inference and prediction pipeline
- **`settings.yaml`**: Global configuration parameters

### Experiment Organization
Each horizon (h4, h8, h16, h32) has:
- Config file: `experiments/h{horizon}.yaml`
- Results directory: `experiments/h{horizon}/`
- Reports directory: `reports/h{horizon}/`

## Critical Constraints

### Data Requirements
- **Frequency**: 15-minute bars
- **Timezone**: UTC end-of-bar timestamps
- **Target**: Log returns
- **Grid**: Regular, no gaps

### Feature Engineering
- **Maximum features**: 256 after pruning
- **Leakage prevention**: All historical features must be shifted by 1 bar
- **Multi-timeframe**: 30min, 1h, 4h aligned to 15min base

### Model Portfolio
- **Models**: NHITS, NBEATSx, TiDE, PatchTST
- **Losses**: DistributionLoss("StudentT"), MQLoss, IQLoss
- **Horizons**: 4, 8, 16, 32 steps (1h, 2h, 4h, 8h)

### Validation Requirements
- **Primary metric**: sCRPS
- **Coverage targets**: 80±2%, 90±2%, 95±2%
- **Cross-validation**: n_windows=6→10, step_size=h, val_size=4h

## Development Guidelines

### NeuralForecast-Only Rules
- Use NF's native cross-validation - no custom backtesting
- Use NF's save/load - no custom serialization
- Use NF's scalers - no external normalization
- Use NF's conformal prediction when needed

### Import Patterns
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss
from neuralforecast.utils import PredictionIntervals  # Correct import
```

### Quality Gates
All code must pass these assertions:
- `assert_regular_grid(df, "15min")`
- `assert_utc_eob(df, "15min")`
- `assert_shifted(df, hist_cols)`
- `assert_no_forward_fill_y(df)`

## Quick Reference

### Check Project Status
- Current version: See `docs/versioning_system.md`
- Implementation phase: See `implementation_workflow.md`

### Find Specifications
- Data contracts: `docs/forecasting_sf_plan.md` → Section 2
- Feature details: `docs/forecasting_sf_plan.md` → Section 3
- Model configs: `docs/forecasting_sf_plan.md` → Section 4
- CV strategy: `docs/forecasting_sf_plan.md` → Section 5
- Acceptance criteria: `docs/forecasting_sf_plan.md` → Section 12

### Locate Code
- Validation functions: `utils/validate.py`
- Feature computation: `features/builder.py`
- Model instantiation: `nf_models/factory.py`
- CV execution: `cv/runner.py`
- Metrics calculation: `uq/diag.py`

## Implementation Status
**Current Phase**: Planning (v0.0.0)
**Next Steps**: See Phase 1 in `implementation_workflow.md`

---

*This file provides navigation and context. For specific details, refer to the referenced documents.*