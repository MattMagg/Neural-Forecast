---
inclusion: always
---

# Data Processing & Validation Guidelines

## Implementation Status
**COMPLETED**: Data processing and validation foundation is fully implemented and validated.
Reference `EXECUTION_STATUS.md` for complete implementation details and test results.

## Core Data Processing Pipeline

### Assembly Path (IMPLEMENTED)
The complete data assembly path is implemented in `run_train.py`:
```
load 1-min → aggregate_1min_to_15min → regularize_to_grid_utc → make_nf_canonical → validate
```

### Key Functions (IMPLEMENTED in utils/io.py)
- `aggregate_1min_to_15min()` - Aggregates 1-minute OHLCV to 15-minute bars with UTC EOB alignment
- `regularize_to_grid_utc()` - Creates complete UTC time grid with proper end-of-bar alignment
- `make_nf_canonical()` - Transforms to NeuralForecast canonical schema with log returns
- `drop_train_nans_and_winsorize()` - Applies winsorization while preserving evaluation data

### Validation Functions (IMPLEMENTED in utils/validate.py)
- `assert_regular_grid()` - Validates 15-minute time grid completeness and monotonicity
- `assert_utc_eob()` - Validates UTC end-of-bar timestamp alignment
- `assert_shifted()` - Detects data leakage using correlation analysis
- `assert_no_forward_fill_y()` - Prevents target variable forward-filling

## Data Quality Standards

### Mandatory Validation Gates
Every dataset MUST pass all validation gates before proceeding:
```python
assert_regular_grid(df, "15min")
assert_utc_eob(df, "15min") 
assert_shifted(df, hist_cols)  # When exogenous features present
assert_no_forward_fill_y(df)
```

### NeuralForecast Schema Compliance
All processed data must conform to NF canonical schema:
```python
["unique_id", "ds", "y", "open", "high", "low", "close", "volume"]
```
- `unique_id`: "BTC-USD" for all rows
- `ds`: UTC datetime64[ns] timestamps on 15-minute boundaries
- `y`: Log returns computed as `log(close_t) - log(close_t-1)`

### Timestamp Discipline
- **UTC Only**: All timestamps must be UTC timezone-aware
- **End-of-Bar**: Timestamps align to :00, :15, :30, :45 minute boundaries
- **Regular Grid**: No gaps or duplicates in the time series
- **No Forward-Fill**: Never forward-fill the target variable `y`

## Leakage Prevention

### Compute → Align → Shift(1) Rule
All historical exogenous features MUST follow this pattern:
1. **Compute**: Calculate the indicator/feature
2. **Align**: Align to 15-minute base frequency
3. **Shift(1)**: Shift by 1 period to prevent leakage

### Validation
Use `assert_shifted()` to detect leakage via correlation analysis:
- Compares correlation of feature with current vs future target
- Raises AssertionError if leakage detected
- Provides diagnostic information for debugging

## Data Processing Performance

### Validated Performance Metrics
- **Input**: 7,160,797 1-minute bars (13+ years of BTC data)
- **Output**: 477,464 15-minute canonical bars
- **Processing Time**: ~30 seconds for full dataset
- **Memory Efficiency**: Handles large datasets without issues
- **Validation**: 100% pass rate across all quality gates

### File I/O Standards
- **Save Format**: Parquet with pyarrow backend
- **Timestamped Paths**: Use `timestamped_path()` for versioning
- **Directory Structure**: Save to `data/processed/` with proper organization

## Usage Patterns

### Basic Data Processing
```python
from utils.io import load_raw_1min_data, aggregate_1min_to_15min, make_nf_canonical
from utils.validate import assert_regular_grid, assert_utc_eob, assert_no_forward_fill_y

# Load and process
df_1min = load_raw_1min_data("data/raw/btcusd_1-min_data.csv")
df_15min = aggregate_1min_to_15min(df_1min)
df_canonical = make_nf_canonical(df_15min)

# Validate
assert_regular_grid(df_canonical, "15min")
assert_utc_eob(df_canonical, "15min")
assert_no_forward_fill_y(df_canonical)
```

### Complete Pipeline
```bash
# Process full dataset with validation
python run_train.py --save-processed

# Process with custom data path
python run_train.py --raw-data path/to/data.csv --save-processed
```

## Error Handling

### Common Issues and Solutions
- **Timezone Errors**: Ensure all timestamps are UTC timezone-aware
- **Grid Irregularities**: Use `regularize_to_grid_utc()` to create complete grid
- **Leakage Detection**: Check feature computation and ensure proper shift(1)
- **Forward-Fill Violations**: Never forward-fill target variable, preserve NaN values

### Diagnostic Information
All validation functions provide detailed error messages with:
- Specific failure conditions
- Diagnostic values (correlations, timestamp ranges, etc.)
- Actionable guidance for fixing issues

## Integration Points

### Feature Engineering Integration
The data processing pipeline is designed to integrate with feature engineering:
- Canonical frame provides OHLCV data for technical indicators
- `assert_shifted()` validates all historical features
- Pipeline ready for exogenous feature merging

### Model Training Integration
Processed data is NeuralForecast-ready:
- Perfect schema compliance
- Proper temporal ordering
- Validated data quality
- Training/evaluation split preparation with winsorization

## Best Practices

### Development Workflow
1. Always validate data after each processing step
2. Use the complete assembly path in `run_train.py`
3. Test with small data samples before full processing
4. Monitor memory usage with large datasets
5. Save intermediate results for debugging

### Production Considerations
- Pipeline handles 7M+ records efficiently
- Validation gates catch data quality issues early
- Timestamped outputs enable data versioning
- Error messages provide actionable diagnostics
- Memory-efficient processing for production use