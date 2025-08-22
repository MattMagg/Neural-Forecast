# Feature Engineering Module

## Directory Overview
Implements the complete feature engineering pipeline with technical indicators and multi-timeframe features.

## Files and Purpose

### ✅ **registry.py** (ACTIVELY USED)
- **Purpose**: Defines all technical indicators and their configurations
- **Key Exports**:
  - `REGISTRY` - List of IndicatorSpec objects defining all indicators
  - `MTF_TARGETS` - Multi-timeframe targets [(tf, [indicators])]
  - `IndicatorSpec` - Dataclass for indicator specifications
- **Content**: ~50+ indicator definitions with parameters
- **Pipeline Role**: Configuration source for all feature computation

### ✅ **builder.py** (ACTIVELY USED)
- **Purpose**: Implements feature computation and processing
- **Key Functions**:
  - `build_indicators()` - Computes base 15-minute indicators
  - `apply_mtf()` - Applies multi-timeframe features (30min, 1h, 4h)
  - `postprocess_shift_and_prune()` - Applies shift(1) for leakage prevention
  - `select_features()` - Selects final features with 256 hard cap
- **Internal Functions**:
  - `_compute_talib()` - TA-Lib indicator computation via vectorbt
  - `_compute_pandasta()` - pandas-ta indicator computation
  - `_compute_custom()` - Custom indicators (time-based features)
  - `_compute_mtf_one()` - Single timeframe MTF computation
- **Pipeline Role**: Core feature engineering implementation

### ✅ **__init__.py** (REQUIRED)
- **Purpose**: Makes features/ a Python package
- **Status**: Empty but necessary for imports

## System Flow
```
run_train.ipynb::integrate_features()
    ↓ line 222-223: imports
from features.registry import REGISTRY
from features.builder import functions
    ↓ line 227
build_indicators(df) → base 15min features
    ↓ line 230
apply_mtf(df) → multi-timeframe features
    ↓ merge
exo_raw = base + mtf features
    ↓ line 236
postprocess_shift_and_prune() → shift(1) applied
    ↓ line 239
select_features() → ≤256 features selected
    ↓
Returns: hist_cols, futr_cols, stat_cols
```

## Key Technical Details
- **Leakage Prevention**: All historical features shifted by 1 bar
- **Multi-Timeframe**: 30min, 1h, 4h features aligned to 15min base
- **Feature Cap**: Maximum 256 features after pruning
- **Libraries Used**: vectorbt (TA-Lib wrapper), pandas-ta, freqtrade/technical
- **Correlation Pruning**: Removes features with |ρ| > 0.95

## Key Insights
- Complete separation of configuration (registry) from implementation (builder)
- Strict leakage prevention through shift(1) enforcement
- Multi-library support for comprehensive indicator coverage
- Hard cap ensures model training efficiency