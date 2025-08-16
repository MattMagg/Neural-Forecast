# I/O Enhancements Summary (Tasks 14 & 15)

## Overview
Enhanced `utils/io.py` with comprehensive artifact management capabilities for cross-validation results and model persistence, implementing Tasks 14 and 15 from the CV metrics specification.

## Task 14: Results Persistence ✅

### Implementation Details
- **Location**: `utils/io.py` - `save_cv_artifacts()` and `load_cv_artifacts()` functions
- **Directory Structure**: 
  ```
  experiments/
    h4/
      cv_results_20240315T143022Z.parquet  # Timestamped CV predictions
      metrics.json                         # Detailed metrics per model/window
      leaderboard.csv                       # Ranked model comparison
    h8/
      ...
  ```

### Key Features
1. **Timestamped CV Results**: Saved as parquet with YYYYMMDDTHHMMSSZ format
2. **Metrics as JSON**: Changed from CSV to JSON for better structure (per spec)
3. **Leaderboard as CSV**: Human-readable ranking of models
4. **Atomic Writes**: All writes use temp file + rename for safety
5. **Error Handling**: Comprehensive error handling with logging

### Function Signatures
```python
save_cv_artifacts(
    cv_df: pd.DataFrame,
    metrics_dict: Union[Dict, pd.DataFrame],
    leaderboard_df: pd.DataFrame,
    horizon: int,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Dict[str, str]

load_cv_artifacts(
    horizon: int,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Dict[str, Any]
```

## Task 15: Model Saving ✅

### Implementation Details
- **Location**: `utils/io.py` - `save_nf_models()` and `load_nf_models()` functions
- **Directory Structure**:
  ```
  experiments/
    h4/
      models/
        NHITS_StudentT_20240315T143022Z.pkl
        NBEATSx_MQLoss_20240315T143022Z.pkl
        TiDE_IQLoss_20240315T143022Z.pkl
        nf_ensemble_20240315T143022Z.pkl
  ```

### Key Features
1. **NF Native Save**: Uses NeuralForecast's native save() method
2. **Metadata-Rich Filenames**: `{ModelClass}_{LossType}_{Timestamp}.pkl`
3. **Selective Saving**: Support for saving only best models
4. **Model Loading**: Load by name, timestamp, or latest
5. **Metadata Extraction**: Function to extract model metadata from files

### Function Signatures
```python
save_nf_models(
    nf_instance,
    horizon: int,
    model_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
    save_best_only: bool = False,
    best_metric: Optional[Dict[str, float]] = None
) -> Dict[str, str]

load_nf_models(
    horizon: int,
    model_name: Optional[str] = None,
    timestamp: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Any
```

## Additional Utility Functions

### Artifact Management
```python
# List all saved artifacts for a horizon
list_saved_artifacts(horizon: int, output_dir: Optional[str] = None) -> Dict[str, List[str]]

# Clean up old CV result files
cleanup_old_artifacts(horizon: int, keep_latest: int = 5, output_dir: Optional[str] = None) -> Dict[str, int]

# Extract metadata from model files
get_model_metadata(model_path: str) -> Dict[str, Any]
```

### Atomic Write Helpers
- `_atomic_write_parquet()`: Safe parquet writing
- `_atomic_write_json()`: Safe JSON writing
- `_atomic_write_csv()`: Safe CSV writing

All use temporary file + atomic rename pattern for safety.

## Error Handling

### Validation
- Horizon must be in [4, 8, 16, 32]
- Directory creation with permission checks
- File format validation on load

### Logging
- INFO level for successful operations
- WARNING for missing files
- ERROR for operation failures

### Recovery
- Graceful handling of missing artifacts
- Fallback to latest files when timestamp not found
- Continue with other models if one save fails

## Usage Examples

### Saving CV Artifacts
```python
from utils.io import save_cv_artifacts

# After CV execution
saved_paths = save_cv_artifacts(
    cv_df=cv_predictions,
    metrics_dict=metrics_summary,
    leaderboard_df=model_rankings,
    horizon=4
)
```

### Loading CV Artifacts
```python
from utils.io import load_cv_artifacts

# Load latest artifacts
artifacts = load_cv_artifacts(horizon=4)
cv_df = artifacts['cv_results']
metrics = artifacts['metrics']
leaderboard = artifacts['leaderboard']

# Load specific timestamp
artifacts = load_cv_artifacts(horizon=4, timestamp='20240315T143022Z')
```

### Saving Models
```python
from utils.io import save_nf_models

# Save all models
paths = save_nf_models(nf, horizon=4)

# Save only best models
best_models = {'NHITS': 0.012, 'TiDE': 0.015}  # sCRPS scores
paths = save_nf_models(
    nf, 
    horizon=4, 
    save_best_only=True, 
    best_metric=best_models
)
```

### Loading Models
```python
from utils.io import load_nf_models

# Load specific model
nhits_model = load_nf_models(horizon=4, model_name='NHITS')

# Load complete ensemble
nf_ensemble = load_nf_models(horizon=4)
```

## Testing
All functions have been tested with:
- ✅ Basic functionality
- ✅ Error handling for invalid inputs
- ✅ Directory creation
- ✅ Atomic write operations
- ✅ Loading with missing files

## Dependencies
- pandas
- numpy
- json
- tempfile (for atomic writes)
- shutil
- logging
- pathlib
- pickle (for model saving)
- glob (for file listing)

## Compliance with Specification
- ✅ Follows docs/forecasting_sf_plan.md lines 1799-1800
- ✅ Uses timestamp format YYYYMMDDTHHMMSSZ
- ✅ Creates directory structure automatically
- ✅ Implements atomic writes for safety
- ✅ Comprehensive error handling with logging
- ✅ Uses NF's native save() method for models
- ✅ Supports selective model saving

## Future Enhancements
1. Add compression options for large CV result files
2. Implement model versioning with git integration
3. Add cloud storage support (S3, GCS)
4. Create visualization functions for artifacts
5. Add artifact comparison utilities