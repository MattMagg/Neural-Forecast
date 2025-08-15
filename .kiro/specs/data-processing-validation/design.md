# Design Document

## Overview

The Data Processing and Validation component serves as the foundational layer of the BTC forecasting system, ensuring data quality, preventing leakage, and creating canonical NeuralForecast-compatible data frames. This component implements a robust pipeline that transforms raw OHLCV data into validated, regularized time series data suitable for neural forecasting models.

The design follows a strict assembly path: raw data → regularization → canonical frame creation → validation → feature integration. Each step includes comprehensive validation to catch data quality issues early and prevent downstream failures.

## Architecture

### High-Level Data Flow

```mermaid
graph TD
    A[Raw 1-Min OHLCV Data<br/>data/raw/btcusd_1-min_data.csv] --> B[aggregate_1min_to_15min]
    B --> C[regularize_to_grid_utc]
    C --> D[make_nf_canonical]
    D --> E[drop_train_nans_and_winsorize]
    E --> F[Feature Integration]
    F --> G[Validation Gates]
    G --> H[NF-Ready DataFrame]
    
    I[Validation Utilities] --> G
    J[Error Handling] --> G
    
    subgraph "Validation Gates"
        G1[assert_regular_grid]
        G2[assert_utc_eob]
        G3[assert_shifted]
        G4[assert_no_forward_fill_y]
    end
```

### Component Architecture

The system is organized into two main utility modules:

1. **utils/validate.py** - Validation functions and assertion utilities
2. **utils/io.py** - Data processing, loading, and transformation functions

### Key Design Principles

1. **NeuralForecast Schema Compliance** - All data must conform to NF's long-format schema: `["unique_id", "ds", "y", <exog...>]`
2. **UTC End-of-Bar Discipline** - All timestamps use UTC timezone with end-of-bar semantics on 15-minute boundaries
3. **Leakage Prevention** - Strict temporal discipline with validation to prevent future information contamination
4. **Data Quality Gates** - Comprehensive validation at each processing step
5. **Graceful Error Handling** - Clear, actionable error messages with diagnostic information

## Components and Interfaces

### Data Processing Components (utils/io.py)

#### aggregate_1min_to_15min Function

**Purpose**: Aggregate 1-minute OHLCV data to 15-minute bars with proper UTC EOB alignment

**Interface**:
```python
def aggregate_1min_to_15min(df_1min: pd.DataFrame) -> pd.DataFrame
```

**Implementation Strategy**:
- Load 1-minute data from `data/raw/btcusd_1-min_data.csv`
- Convert timestamp column to UTC datetime64[ns] format
- Use pandas resample with label='right' and closed='right' for EOB semantics
- Apply aggregation rules:
  - Open: 'first' - first value in 15-minute window
  - High: 'max' - maximum value in 15-minute window
  - Low: 'min' - minimum value in 15-minute window
  - Close: 'last' - last value in 15-minute window
  - Volume: 'sum' - sum of volumes in 15-minute window
- Ensure output timestamps align to :00, :15, :30, :45 boundaries
- Handle gaps in 1-minute data by creating NaN entries in output

**Error Handling**:
- Raise FileNotFoundError if data file doesn't exist
- Raise ValueError if required columns (open, high, low, close, volume) are missing
- Handle timezone conversion errors with clear messages

**Implementation Example**:
```python
def aggregate_1min_to_15min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute OHLCV data to 15-minute bars.
    
    Args:
        df_1min: DataFrame with 1-minute OHLCV data
        
    Returns:
        DataFrame with 15-minute OHLCV bars, UTC EOB timestamps
    """
    # Ensure timestamp column exists
    if 'timestamp' not in df_1min.columns and 'ds' not in df_1min.columns:
        raise ValueError("No timestamp column found in 1-minute data")
    
    # Convert to UTC datetime
    ts_col = 'timestamp' if 'timestamp' in df_1min.columns else 'ds'
    df_1min['ds'] = pd.to_datetime(df_1min[ts_col], utc=True)
    df_1min = df_1min.set_index('ds')
    
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df_1min.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Resample to 15-minute bars with EOB alignment
    df_15min = df_1min.resample('15min', label='right', closed='right').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Reset index to get ds column back
    df_15min = df_15min.reset_index()
    
    # Ensure EOB timestamps (:00, :15, :30, :45)
    minutes = df_15min['ds'].dt.minute
    if not all(minutes.isin([0, 15, 30, 45])):
        raise ValueError("Aggregation produced non-EOB timestamps")
    
    return df_15min
```

#### regularize_to_grid_utc Function

**Purpose**: Create a complete UTC time grid with proper end-of-bar alignment

**Interface**:
```python
def regularize_to_grid_utc(df_ohlcv: pd.DataFrame, freq: str = "15min") -> pd.DataFrame
```

**Implementation Strategy**:
- Convert timestamps to UTC timezone with proper datetime64[ns] format
- Snap timestamps to 15-minute boundaries using `dt.floor(freq)`
- Create complete date range from min to max timestamp
- Merge original data with complete grid, filling gaps with NaN
- Preserve OHLCV columns: ["open", "high", "low", "close", "volume"]

**Error Handling**:
- Raise ValueError if no timestamp column found ("ds" or "timestamp")
- Handle timezone conversion gracefully with proper UTC localization

#### make_nf_canonical Function

**Purpose**: Transform OHLCV data into NeuralForecast canonical schema

**Interface**:
```python
def make_nf_canonical(df_ohlcv_15m: pd.DataFrame, unique_id: str = "BTC-USD") -> pd.DataFrame
```

**Implementation Strategy**:
- Call regularize_to_grid_utc to ensure proper time grid
- Compute log returns: `y = np.log(df["close"]).diff()`
- Create canonical schema: `["unique_id", "ds", "y", "open", "high", "low", "close", "volume"]`
- Set unique_id to "BTC-USD" for all rows
- Preserve original OHLCV data for feature engineering

**Error Handling**:
- Raise ValueError if "close" column missing
- Handle edge cases in log return computation (zero/negative prices)

#### drop_train_nans_and_winsorize Function

**Purpose**: Apply winsorization for training stability while preserving evaluation data

**Interface**:
```python
def drop_train_nans_and_winsorize(nf_df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame
```

**Implementation Strategy**:
- Compute quantiles on non-NaN y values only
- Create y_train column with clipped values using `df["y"].clip(lower=ql, upper=qu)`
- Preserve original y column for evaluation
- Return DataFrame with both y and y_train columns

**Data Integrity**:
- Never modify original y values for evaluation
- Only apply winsorization to training subset
- Maintain NaN values where appropriate

### Validation Components (utils/validate.py)

#### assert_regular_grid Function

**Purpose**: Validate 15-minute time grid completeness and monotonicity

**Interface**:
```python
def assert_regular_grid(df: pd.DataFrame, freq: str = "15min") -> None
```

**Implementation Strategy**:
- Extract DatetimeIndex from df["ds"]
- Assert monotonic increasing: `ds.is_monotonic_increasing`
- Create expected date range: `pd.date_range(ds.min(), ds.max(), freq=freq, tz="UTC")`
- Assert UTC timezone: `(ds.tz is not None) and (str(ds.tz) == "UTC")`
- Assert length match: `len(ds) == len(expected)`

**Error Conditions**:
- Non-monotonic timestamps
- Missing or extra bars vs regular grid
- Non-UTC timezone
- Irregular frequency

#### assert_utc_eob Function

**Purpose**: Validate UTC end-of-bar timestamp alignment

**Interface**:
```python
def assert_utc_eob(df: pd.DataFrame, freq: str = "15min") -> None
```

**Implementation Strategy**:
- Extract DatetimeIndex from df["ds"]
- Check minute alignment: `all(getattr(ts, "minute") % 15 == 0 for ts in ds)`
- Validate end-of-bar semantics on 15-minute boundaries

**Error Conditions**:
- Timestamps not aligned to 15-minute boundaries
- Non-end-of-bar timestamps detected

#### assert_shifted Function

**Purpose**: Detect potential data leakage using correlation analysis

**Interface**:
```python
def assert_shifted(df: pd.DataFrame, hist_cols: Sequence[str]) -> None
```

**Implementation Strategy**:
- Extract y values and create lead-1 series: `y_lead1 = np.roll(y, -1)`
- For each historical column, compute correlations:
  - Contemporaneous: `corr(x, y)`
  - Next-step: `corr(x, y_lead1)`
- Assert contemporaneous correlation < next-step correlation
- Use NaN-safe correlation computation

**Leakage Detection Logic**:
- If a feature has stronger correlation with current y than future y, it suggests proper shifting
- If correlation with future y is stronger, it indicates potential leakage

#### assert_no_forward_fill_y Function

**Purpose**: Prevent target variable forward-filling

**Interface**:
```python
def assert_no_forward_fill_y(df: pd.DataFrame) -> None
```

**Implementation Strategy**:
- Check rows where 'close' is NaN should have y as NaN
- Identify violations: `bad = df["close"].isna() & df["y"].notna()`
- Assert no violations exist

**Data Integrity Check**:
- Ensures missing price data corresponds to missing target data
- Prevents artificial target values from forward-filling

### Utility Functions (utils/io.py)

#### load_canonical_frame Function

**Purpose**: Load and validate stored canonical frames

**Interface**:
```python
def load_canonical_frame(path: str) -> pd.DataFrame
```

**Implementation Strategy**:
- Load parquet file using pandas
- Enforce canonical schema if needed
- Return validated DataFrame

#### save_parquet Function

**Purpose**: Save DataFrames with proper directory management

**Interface**:
```python
def save_parquet(df: pd.DataFrame, path: str) -> None
```

**Implementation Strategy**:
- Create parent directories: `Path(path).parent.mkdir(parents=True, exist_ok=True)`
- Save without index: `df.to_parquet(path, index=False)`

#### timestamped_path Function

**Purpose**: Generate timestamped file paths for versioning

**Interface**:
```python
def timestamped_path(base_dir: str, stem: str, ext: str = "parquet") -> str
```

**Implementation Strategy**:
- Generate UTC timestamp: `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")`
- Create path with timestamp: `{stem}_{ts}.{ext}`
- Ensure parent directory exists

## Data Models

### Input Data Schema

**Raw 1-Minute OHLCV Data** (from `data/raw/btcusd_1-min_data.csv`):
```python
{
    "timestamp": datetime64[ns],  # 1-minute frequency
    "open": float64,
    "high": float64,
    "low": float64,
    "close": float64,
    "volume": float64
}
```

**Aggregated 15-Minute OHLCV Data** (after `aggregate_1min_to_15min`):
```python
{
    "ds": datetime64[ns, UTC],  # 15-minute EOB timestamps
    "open": float64,            # First value in 15-min window
    "high": float64,            # Max value in 15-min window
    "low": float64,             # Min value in 15-min window
    "close": float64,           # Last value in 15-min window
    "volume": float64           # Sum of volumes in 15-min window
}
```

### Canonical NeuralForecast Schema

**NF Long Format**:
```python
{
    "unique_id": str,           # "BTC-USD"
    "ds": datetime64[ns, UTC],  # End-of-bar timestamps
    "y": float64,               # Log returns
    "open": float64,            # Preserved for features
    "high": float64,            # Preserved for features
    "low": float64,             # Preserved for features
    "close": float64,           # Preserved for features
    "volume": float64           # Preserved for features
}
```

### Processed Training Data

**Training-Ready Schema**:
```python
{
    "unique_id": str,
    "ds": datetime64[ns, UTC],
    "y": float64,               # Original for evaluation
    "y_train": float64,         # Winsorized for training
    "open": float64,
    "high": float64,
    "low": float64,
    "close": float64,
    "volume": float64,
    # ... additional exogenous features
}
```

## Error Handling

### Error Classification

1. **Data Quality Errors** (ValueError)
   - Missing required columns
   - Invalid data types
   - Corrupted timestamps

2. **Validation Errors** (AssertionError)
   - Grid irregularities
   - Timezone mismatches
   - Leakage detection
   - Forward-fill violations

3. **Processing Errors** (RuntimeError)
   - File I/O failures
   - Memory constraints
   - Computation failures

### Error Handling Strategy

#### Validation Functions
- Use AssertionError for validation failures
- Include specific diagnostic information in error messages
- Provide actionable guidance for fixing issues

#### Processing Functions
- Use ValueError for input validation issues
- Use RuntimeError for processing failures
- Include context about the failed operation

#### Example Error Messages
```python
# Grid validation failure
AssertionError: "missing or extra bars vs regular grid: expected 1440 bars, got 1438"

# Leakage detection
AssertionError: "Potential leakage in rsi_14: fix shift(1) - corr(x,y)=0.85 > corr(x,y_lead1)=0.23"

# Forward-fill detection
AssertionError: "Detected non-NaN y where close is NaN (forbidden forward-fill of target)"
```

## Testing Strategy

### Unit Testing Approach

1. **Validation Function Tests**
   - Test each assertion with valid and invalid data
   - Verify error messages are informative
   - Test edge cases (empty data, single row, etc.)

2. **Processing Function Tests**
   - Test data transformations with known inputs/outputs
   - Verify schema compliance
   - Test error handling paths

3. **Integration Tests**
   - Test complete assembly path
   - Verify end-to-end data flow
   - Test with realistic BTC data

### Test Data Strategy

1. **Synthetic Data**
   - Generate regular 15-minute grids
   - Create data with known gaps/irregularities
   - Simulate various error conditions

2. **Real Data Samples**
   - Use small BTC datasets for integration testing
   - Test with actual market data irregularities
   - Validate against known good outputs

### Performance Testing

1. **Memory Usage**
   - Test with large datasets (1M+ rows)
   - Monitor memory consumption during processing
   - Verify garbage collection effectiveness

2. **Processing Speed**
   - Benchmark key functions
   - Identify bottlenecks in the assembly path
   - Optimize critical sections

## Implementation Considerations

### Dependencies

**Required Libraries**:
- pandas >= 1.5.0 (DataFrame operations, datetime handling)
- numpy >= 1.21.0 (numerical computations, log returns)
- pathlib (file path management)
- datetime (timezone handling)
- typing (type hints)

### Memory Management

1. **Large Dataset Handling**
   - Process data in chunks if memory constrained
   - Use efficient pandas operations (vectorized)
   - Minimize data copying during transformations

2. **Garbage Collection**
   - Explicitly delete intermediate DataFrames
   - Use context managers for file operations
   - Monitor memory usage in long-running processes

### Performance Optimization

1. **Vectorized Operations**
   - Use pandas/numpy vectorized functions
   - Avoid Python loops for data processing
   - Leverage pandas' optimized datetime operations

2. **Efficient Data Types**
   - Use appropriate dtypes (float32 vs float64)
   - Optimize categorical data representation
   - Minimize string operations

### Configuration Management

1. **Parameterization**
   - Make frequency configurable ("15min" default)
   - Allow customizable winsorization quantiles
   - Support different unique_id formats

2. **Validation Thresholds**
   - Configurable correlation thresholds for leakage detection
   - Adjustable tolerance for timestamp alignment
   - Customizable error message verbosity

This design provides a robust, well-tested foundation for the BTC forecasting system's data processing needs, ensuring data quality and preventing common pitfalls in time series forecasting.