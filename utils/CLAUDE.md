# Utilities Module

## Directory Overview
Core utility functions for data I/O, validation, error handling, and risk mitigation.

## Files and Purpose

### ✅ **io.py** (ACTIVELY USED - DIRECT)
- **Purpose**: Data loading, processing, and persistence
- **Key Functions**:
  - `load_raw_1min_data()` - Load raw CSV data (line 145)
  - `aggregate_1min_to_15min()` - Aggregate to 15-min bars (line 150)
  - `regularize_to_grid_utc()` - Create regular UTC grid (line 155)
  - `make_nf_canonical()` - Convert to NF format with log returns (line 160)
  - `drop_train_nans_and_winsorize()` - Winsorize outliers (line 166)
  - `save_parquet()` - Save DataFrame to parquet (line 277)
  - `timestamped_path()` - Generate timestamped filenames (line 276)
- **Pipeline Role**: Core data pipeline implementation

### ✅ **validate.py** (ACTIVELY USED - DIRECT)
- **Purpose**: Data validation and quality gates
- **Key Functions**:
  - `assert_regular_grid()` - Verify regular time grid (line 178)
  - `assert_utc_eob()` - Verify UTC end-of-bar timestamps (line 182)
  - `assert_no_forward_fill_y()` - Check no forward filling (line 186)
  - `assert_shifted()` - Verify shift(1) for leakage prevention (line 246)
- **Pipeline Role**: Quality gates throughout data processing

### ✅ **error_recovery.py** (ACTIVELY USED - INDIRECT)
- **Purpose**: Error handling and recovery strategies
- **Usage**: Imported by cv/runner.py (line 32)
- **Features**: Retry logic, error classification, recovery strategies
- **Pipeline Role**: Provides robustness during CV execution

### ✅ **risk_mitigation.py** (ACTIVELY USED - INDIRECT)
- **Purpose**: GPU memory management and risk mitigation
- **Key Class**: `GPUMemoryManager`
- **Usage**: Imported by cv/runner.py (line 37)
- **Pipeline Role**: Prevents GPU OOM during training

### ❌ **version.py** (NOT USED)
- **Purpose**: Version tracking utilities
- **Status**: No references found in codebase
- **Note**: Available for future version management

### ✅ **__init__.py** (REQUIRED)
- **Purpose**: Makes utils/ a Python package
- **Status**: Required for imports

## System Flow

### Data Pipeline (io.py):
```
load_raw_1min_data()
    ↓
aggregate_1min_to_15min()
    ↓
regularize_to_grid_utc()
    ↓
make_nf_canonical()
    ↓
drop_train_nans_and_winsorize()
    ↓
save_parquet()
```

### Validation Gates (validate.py):
```
assert_regular_grid() → Check time grid
assert_utc_eob() → Check timestamps
assert_no_forward_fill_y() → Check data integrity
assert_shifted() → Check leakage prevention
```

### Error Handling Flow:
```
cv/runner.py imports
    ↓
error_recovery.py → Retry logic
risk_mitigation.py → GPU management
    ↓
Applied during CV execution
```

## Key Insights
- io.py and validate.py are directly critical for data pipeline
- error_recovery.py and risk_mitigation.py provide indirect robustness
- version.py exists but unused (could be removed)
- Clear separation between I/O, validation, and error handling concerns