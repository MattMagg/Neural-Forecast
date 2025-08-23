# BTC Forecasting System Training Guide

**Foundation Workflow (Specs 1-4) - v0.5.1.2**

This guide covers the complete training workflow for the implemented BTC forecasting system foundation, focusing on the 3-step sequential process: **setup.sh → kaggle_download_btc.py → run_train.ipynb**. This foundation provides basic forecasting/prediction capability with Specs 1-4 (Data Processing, Feature Engineering, Model Factory, Cross-Validation). Additional specs (5-14) will be implemented later.

## System Overview

**Implemented Components:**
- ✅ **Spec 1**: Data Processing & Validation (utils/io.py, utils/validate.py)
- ✅ **Spec 2**: Feature Engineering Pipeline (features/registry.py, features/builder.py)  
- ✅ **Spec 3**: NeuralForecast Model Factory (nf_models/factory.py)
- ✅ **Spec 4**: Cross-Validation & Metrics (cv/runner.py, uq/metrics.py)

**Thunder Compute Pre-installed:**
- ✅ **CUDA 12.9** - Pre-installed, no installation needed  
- ✅ **CUDNN 9.0** - Pre-installed  
- ✅ **Python 3.x** - Pre-installed with pip package manager  
- ✅ **PyTorch 2.7.1** - Pre-installed (setup script upgrades to 2.8.0)  
- ✅ **JupyterLab** - Pre-installed and ready to use

## Sequential Training Workflow

The foundation system follows a **3-step sequential process** for autonomous training execution:

### Step 1: Environment Setup
```bash
./setup.sh
```

### Step 2: Data Acquisition  
```bash
cd data/raw
python kaggle_download_btc.py
```

### Step 3: Training Execution
```bash
# Activate environment
source .venv/bin/activate

# Open JupyterLab and run run_train.ipynb
jupyter lab run_train.ipynb
```

**Expected Total Time:** 2-4 hours per horizon on A100XL (80GB VRAM)

## Step 1: Environment Setup (setup.sh)

The setup script automates the complete environment preparation for Thunder Compute A100XL instances.

### Command
```bash
./setup.sh
```

### What It Does
1. **System Dependencies**: Installs build tools, Node.js, Claude Code
2. **Python Environment**: Creates virtual environment with Python 3.13
3. **CUDA Integration**: Configures paths for pre-installed CUDA 12.9
4. **TA-Lib Installation**: Compiles TA-Lib C library for technical indicators
5. **Python Dependencies**: Installs packages in critical order:
   - numpy==2.3.2 (foundation)
   - pandas==2.3.1, pyarrow==20.0.0 (data handling)
   - torch==2.8.0 (upgraded from pre-installed 2.7.1)
   - neuralforecast==3.0.2 (core framework)
   - Feature engineering stack (vectorbt, TA-Lib, pandas-ta-openbb)

### Expected Output
```
[2025-01-XX XX:XX:XX] Starting Thunder Compute setup for BTC Forecasting System...
[2025-01-XX XX:XX:XX] Installing system packages...
[2025-01-XX XX:XX:XX] Installing TA-Lib C library...
[2025-01-XX XX:XX:XX] Creating virtual environment with pre-installed Python...
[2025-01-XX XX:XX:XX] Installing numpy==2.3.2...
[2025-01-XX XX:XX:XX] Installing neuralforecast==3.0.2...
[2025-01-XX XX:XX:XX] Testing project module imports...
utils.validate: OK
features.builder: OK
nf_models.factory: OK
cv.runner: OK
[2025-01-XX XX:XX:XX] Setup complete!
```

### Success Indicators
- ✅ All module imports successful
- ✅ CUDA available through PyTorch
- ✅ Virtual environment created at `.venv/`
- ✅ Setup report generated: `setup_report.txt`

### Time Estimate
**5-10 minutes** (first-time setup with compilation)

## Step 2: Data Acquisition (kaggle_download_btc.py)

Downloads the Bitcoin historical data from Kaggle using the kagglehub API.

### Prerequisites
- **kagglehub package**: Automatically installed by setup.sh or via `pip install kagglehub`
- **Kaggle API credentials**: Configure if needed (kagglehub handles authentication)

### Command
```bash
cd data/raw
python kaggle_download_btc.py
```

### What It Does
1. **Downloads Dataset**: Fetches 'mczielinski/bitcoin-historical-data' from Kaggle
2. **File Copy**: Copies `btcusd_1-min_data.csv` to `data/raw/`
3. **Data Integrity Verification**: 
   - Checks file size (>400MB expected)
   - Validates row count (>7M rows expected)
   - Verifies required columns exist
   - Confirms date range coverage

### Expected Output
```
Path to dataset files: /home/ubuntu/.cache/kagglehub/datasets/mczielinski/bitcoin-historical-data/versions/3
Copying to: /home/ubuntu/Neural-Forecast/data/raw/btcusd_1-min_data.csv
File copied successfully!
```

### Success Indicators
- ✅ File exists: `data/raw/btcusd_1-min_data.csv`
- ✅ File size: ~500MB (7M+ 1-minute bars)
- ✅ Date range: 2011-2024+ (13+ years of BTC data)

### Validation Check
```bash
# Verify file exists and check basic stats
ls -lh data/raw/btcusd_1-min_data.csv
head -5 data/raw/btcusd_1-min_data.csv
wc -l data/raw/btcusd_1-min_data.csv
```

### Time Estimate
**2-5 minutes** (depending on network speed)

## Step 3: Training Execution (run_train.ipynb)

The training notebook implements the complete data assembly and training pipeline with cell-by-cell execution guidance.

### Opening the Notebook
```bash
# Activate environment
source .venv/bin/activate

# Start JupyterLab
jupyter lab

# Navigate to and open: run_train.ipynb
```

### Cell-by-Cell Execution Guide

#### Cell 1-2: Imports and Configuration
- **Purpose**: Import all required modules and set configuration variables
- **Expected Output**: No errors, all imports successful
- **Key Variables**: 
  - `CONFIG = "experiments/h4.yaml"` (change for different horizons)
  - `RAW_DATA = "data/raw/btcusd_1-min_data.csv"`

#### Cell 3-4: Helper Functions
- **Purpose**: Define data processing and feature integration functions
- **Expected Output**: Function definitions loaded
- **Key Functions**: `load_and_process_data()`, `integrate_features()`

#### Cell 5: Main Execution - Data Processing
```python
# Load experiment configuration
cfg = load_experiment_config(CONFIG)

# Execute complete data assembly path
df_processed = load_and_process_data(RAW_DATA)
```
- **Expected Output**:
  ```
  🔄 Starting data assembly path...
  📥 Loading raw 1-minute data...
     Loaded 7,160,797 1-minute bars
  📊 Aggregating 1-minute to 15-minute bars...
     Aggregated to 477,464 15-minute bars
  🕐 Regularizing to UTC grid...
     Regularized grid: 477,464 bars
  🎯 Creating NF canonical format...
     Canonical format: 477,464 rows
  ✅ Running validation gates...
     ✓ Regular grid validation passed
     ✓ UTC EOB validation passed
     ✓ No forward-fill validation passed
  🎉 Data assembly path completed successfully!
  ```

#### Cell 6: Feature Engineering Integration
```python
# Integrate feature engineering pipeline
nf_df, hist_cols, futr_cols, stat_cols = integrate_features(df_processed)
```
- **Expected Output**:
  ```
  🔗 Integrating feature engineering pipeline...
     📊 Building base indicators...
     🕐 Computing multi-timeframe features...
     🔄 Merging feature sets...
     ⚡ Applying shift(1) and pruning...
     🎯 Selecting features with hard cap...
     ✅ Validating leakage prevention with assert_shifted...
     ✓ Leakage validation passed!
  
  📊 Feature Summary:
     Historical features: 45
     Future features: 0
     Static features: 3
     Total features: 48 (cap: 256)
  ```

#### Cell 7: Training and Cross-Validation
```python
# Run training and cross-validation
results = train_and_evaluate(
    nf_df=nf_df,
    hist_cols=hist_cols,
    futr_cols=futr_cols,
    stat_cols=stat_cols,
    cfg=cfg,
    output_dir=output_dir,
    use_conformal=USE_CONFORMAL,
    save_models=SAVE_MODELS
)
```
- **Expected Output**: Detailed progress through 6 steps:
  1. Model Instantiation
  2. Initial Model Fitting  
  3. Insample Predictions
  4. Cross-Validation (6 windows)
  5. Results Summarization
  6. Saving Artifacts

### A100XL-Specific Optimizations

The YAML configurations are pre-optimized for A100XL (80GB VRAM):

| Model | Batch Size | Input Size | Memory Usage | Notes |
|-------|------------|------------|--------------|-------|
| NHITS | 512 | 1024 | ~45GB | Optimal for A100XL |
| NBEATSx | 512 | 1024 | ~50GB | Stack-based architecture |
| TiDE | 512 | 1024 | ~40GB | Encoder-decoder efficient |
| PatchTST | 512 | 2048 | ~60GB | Larger input, RevIN scaler |

### Training Time Estimates (A100XL)

| Horizon | Description | CV Windows | Training Time | Memory Peak |
|---------|-------------|------------|---------------|-------------|
| h4 | 1 hour (4×15min) | 6 windows | 45-60 minutes | ~50GB |
| h8 | 2 hour (8×15min) | 6 windows | 45-60 minutes | ~50GB |
| h16 | 4 hour (16×15min) | 6 windows | 60-75 minutes | ~55GB |
| h32 | 8 hour (32×15min) | 6 windows | 60-75 minutes | ~60GB |

## Horizon Selection and Configuration

The system supports 4 forecasting horizons using existing YAML configurations:

### Available Horizons
- **h4** (`experiments/h4.yaml`): 1-hour forecasts (4 × 15min steps)
- **h8** (`experiments/h8.yaml`): 2-hour forecasts (8 × 15min steps)  
- **h16** (`experiments/h16.yaml`): 4-hour forecasts (16 × 15min steps)
- **h32** (`experiments/h32.yaml`): 8-hour forecasts (32 × 15min steps)

### Changing Horizons in Notebook

In the **Configuration** cell of `run_train.ipynb`, modify:

```python
# Change this line to select different horizon
CONFIG = "experiments/h4.yaml"   # For 1-hour forecasts
CONFIG = "experiments/h8.yaml"   # For 2-hour forecasts  
CONFIG = "experiments/h16.yaml"  # For 4-hour forecasts
CONFIG = "experiments/h32.yaml"  # For 8-hour forecasts
```

### Configuration Structure

Each YAML file contains:
```yaml
# Global settings
h: 4                    # Horizon steps
freq: "15min"          # Base frequency
seed: 1337             # Reproducibility

# Cross-validation settings  
n_windows: 6           # CV windows (6 for pilot, 10 for final)
step_size: 4           # Non-overlapping (= h, changes per horizon)
val_size: 16           # Should be 4*h (auto-corrected at runtime)
refit: true            # Refit each window

# Model portfolio (4 models with StudentT loss)
models:
  - NHITS: {...}       # Neural Hierarchical Interpolation
  - NBEATSX: {...}     # N-BEATS with exogenous variables
  - TIDE: {...}        # Time-series Dense Encoder
  - PATCHTST: {...}    # Patch Time Series Transformer
```

### Important Configuration Notes

**Val_size Auto-Correction:**
The system automatically corrects `val_size` to be `4*h` as per the specification:
- h=4: val_size should be 16 (1 hour forecast, 4 hours validation)
- h=8: val_size should be 32 (2 hour forecast, 8 hours validation)  
- h=16: val_size should be 64 (4 hour forecast, 16 hours validation)
- h=32: val_size should be 128 (8 hour forecast, 32 hours validation)

The YAML files have been updated with correct values. If misconfigured, `run_train.ipynb` will auto-correct at runtime (lines 271-276).

### Safe Parameters to Modify

**Adjustable without breaking system:**
- `batch_size`: Reduce if GPU memory issues (512 → 256 → 128)
- `learning_rate`: Adjust for convergence (0.001 → 0.0005)
- `max_steps`: Increase for longer training (20000 → 30000)
- `early_stop_patience_steps`: Adjust early stopping (400 → 600)

**Do NOT modify without understanding:**
- `h`, `step_size` (must maintain step_size=h for non-overlapping windows)
- `val_size` (should be 4*h per specification)
- `freq` (breaks data alignment)
- Model architecture parameters

## GPU Monitoring and Progress Indicators

### Real-Time GPU Monitoring
```bash
# Monitor GPU utilization during training
watch -n 1 nvidia-smi

# Detailed GPU stats with process info
nvidia-smi pmon -i 0 -s um

# Memory usage tracking
gpustat -i 1
```

### Training Progress Indicators

**In Jupyter Notebook Output:**
- ✅ **Data Processing**: Progress bars for 1min→15min aggregation
- ✅ **Feature Engineering**: Feature count and validation status
- ✅ **Model Fitting**: PyTorch Lightning progress bars per model
- ✅ **Cross-Validation**: Window-by-window progress (1/6, 2/6, etc.)
- ✅ **Results**: sCRPS scores and model rankings

**Expected Progress Pattern:**
```
STEP 1: Model Instantiation
Successfully instantiated 4 models

STEP 2: Initial Model Fitting  
Fitting 4 models with val_size=64...
✅ Model fitting completed in 180.5 seconds

STEP 4: Cross-Validation
Starting 6-window cross-validation...
Window 1/6: sCRPS=0.1234 (NHITS best)
Window 2/6: sCRPS=0.1198 (TiDE best)
...
✅ Cross-validation completed in 2847.3 seconds

STEP 5: Results Summarization
MODEL LEADERBOARD
Rank 1: TiDE_t1024_T - sCRPS: 0.1156
Rank 2: NHITS_t1024_T - sCRPS: 0.1189
```

### System Resource Monitoring
```bash
# CPU and memory usage
htop

# Disk space (artifacts can be large)
df -h

# Python process monitoring
ps aux | grep python | grep -v grep
```

## Artifact Locations and Results

### Training Artifacts

After successful training, artifacts are saved to `experiments/h{horizon}/`:

```
experiments/h4/
├── cv_results_20250822T120000Z.parquet  # Raw CV predictions
├── leaderboard.csv                      # Model rankings by sCRPS
├── metrics.json                         # Aggregated metrics
├── best/                               # Best model artifacts
│   ├── NHITS_20250822T120000Z.pkl
│   ├── NBEATSx_20250822T120000Z.pkl  
│   ├── TiDE_20250822T120000Z.pkl
│   └── PatchTST_20250822T120000Z.pkl
└── training_config.yaml                # Configuration used
```

### Interpreting Results

#### sCRPS Scores (Primary Metric)
- **Lower is better** (0.0 = perfect, higher = worse)
- **Typical range**: 0.08-0.15 for BTC 15-minute forecasts
- **Acceptance threshold**: < 0.12 for good performance

#### Coverage Metrics
- **Target**: 80%, 90%, 95% prediction intervals
- **Tolerance**: ±2% (e.g., 88-92% for 90% target)
- **Good coverage**: Within tolerance across all levels

#### Leaderboard Interpretation
```csv
rank,model,sCRPS_mean,sCRPS_std,coverage_80,coverage_90,coverage_95
1,TiDE_t1024_T,0.1156,0.0089,0.798,0.891,0.943
2,NHITS_t1024_T,0.1189,0.0094,0.801,0.896,0.948
3,NBEATSx_t1024_T,0.1203,0.0101,0.795,0.888,0.941
4,PatchTST_t2048_T,0.1247,0.0087,0.803,0.899,0.951
```

**Best Model**: TiDE with sCRPS=0.1156, excellent coverage

### Checking Results
```bash
# View leaderboard
cat experiments/h4/leaderboard.csv

# Check metrics summary
python -c "
import json
with open('experiments/h4/metrics.json') as f:
    metrics = json.load(f)
print(f'Best sCRPS: {metrics[\"best_scrps\"]:.4f}')
print(f'Coverage 90%: {metrics[\"coverage_90\"]:.3f}')
"

# Examine CV results
python -c "
import pandas as pd
df = pd.read_parquet('experiments/h4/cv_results_20250822T120000Z.parquet')
print(f'CV results shape: {df.shape}')
print(f'Models: {df.columns[df.columns.str.contains(\"_t\")].tolist()}')
"
```

## Checkpointing and Recovery

### Training Checkpoints (NEW)

To prevent loss of progress during long training runs, the system now supports checkpointing:

#### Manual Checkpointing
Add this to your notebook after each CV window:
```python
import torch
import pickle
from pathlib import Path

def save_checkpoint(cv_df, window_idx, horizon, checkpoint_dir="checkpoints"):
    """Save intermediate CV results and model states."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'cv_results': cv_df,
        'window_idx': window_idx,
        'horizon': horizon,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_h{horizon}_w{window_idx}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(horizon, checkpoint_dir="checkpoints"):
    """Load the most recent checkpoint for a horizon."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob(f"checkpoint_h{horizon}_*.pkl"))
    
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    with open(latest, 'rb') as f:
        checkpoint = pickle.load(f)
    
    logger.info(f"Loaded checkpoint: {latest}")
    return checkpoint
```

#### Resume from Checkpoint
```python
# Check for existing checkpoint before starting training
checkpoint = load_checkpoint(cfg['h'])
if checkpoint:
    logger.info(f"Resuming from window {checkpoint['window_idx']}")
    # Resume CV from saved window
    start_window = checkpoint['window_idx'] + 1
else:
    start_window = 0
```

## Troubleshooting and Error Recovery

### Common Error Scenarios

#### 1. CUDA Out of Memory (OOM)
**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# In notebook Configuration cell, reduce batch sizes:
# Modify the CONFIG file or create custom config
import yaml
with open('experiments/h4.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Reduce batch sizes for all models
for model_config in cfg['models']:
    for model_name, params in model_config.items():
        params['batch_size'] = 256  # Reduce from 512
        
# Save modified config
with open('experiments/h4_reduced.yaml', 'w') as f:
    yaml.dump(cfg, f)
    
# Update CONFIG variable
CONFIG = "experiments/h4_reduced.yaml"
```

#### 2. Import Errors
**Symptoms**: `ModuleNotFoundError` or import failures

**Recovery**:
```bash
# Reactivate environment
source .venv/bin/activate

# Test critical imports
python -c "
from utils.validate import assert_regular_grid
from features.builder import build_indicators  
from nf_models.factory import instantiate_models
from cv.runner import run_cv
print('All imports successful')
"

# If imports fail, reinstall
./setup.sh --clean
./setup.sh
```

#### 3. Data Validation Failures
**Symptoms**: `AssertionError` in validation gates

**Recovery**:
```bash
# Check raw data file
ls -lh data/raw/btcusd_1-min_data.csv

# Re-download if corrupted
cd data/raw
python kaggle_download_btc.py

# Clear processed data cache
rm -f data/processed/btc_canonical_*.parquet
```

#### 4. Training Interruption
**Symptoms**: Notebook kernel dies or training stops

**Recovery**:
```python
# In notebook, check for partial results
import os
exp_dir = "experiments/h4"
if os.path.exists(f"{exp_dir}/cv_partial_h4.parquet"):
    print("Partial results found - can resume analysis")
    
# Restart from data processing step
# The notebook is designed to be re-runnable
```

#### 5. GPU Driver Issues
**Symptoms**: `CUDA driver version is insufficient`

**Recovery**:
```bash
# Check CUDA status
nvidia-smi

# Verify PyTorch CUDA compatibility
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

# If issues persist, restart instance
sudo reboot
```

### Recovery Procedures

#### Complete Environment Reset
```bash
# Clean everything and start fresh
./setup.sh --clean
rm -rf data/processed/*
rm -rf experiments/h*/cv_results_*
./setup.sh
cd data/raw && python kaggle_download_btc.py
```

#### Partial Recovery (Keep Data)
```bash
# Keep processed data, reset environment only
./setup.sh --clean
./setup.sh
# Data and configs preserved
```

### Performance Troubleshooting

#### Slow Training
**Check GPU utilization**:
```bash
nvidia-smi
# Target: >80% GPU utilization
# If low: increase batch_size or check data loading
```

**Optimize memory**:
```bash
# Set memory optimization before starting notebook
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
```

#### Memory Leaks
```python
# In notebook, clear cache between runs
import torch
torch.cuda.empty_cache()

# Monitor memory usage
print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated')
print(f'GPU Memory: {torch.cuda.memory_reserved()/1e9:.1f}GB reserved')
```

## Data Integrity Verification

### Verify Downloaded Data
After running `kaggle_download_btc.py`, verify data integrity:

```python
import hashlib
import pandas as pd

def verify_btc_data(file_path="data/raw/btcusd_1-min_data.csv"):
    """Verify BTC data file integrity and basic statistics."""
    # Expected characteristics (adjust based on latest data)
    EXPECTED_MIN_ROWS = 7_000_000  # ~7M 1-minute bars
    EXPECTED_MIN_SIZE_MB = 400     # ~400-500MB file
    EXPECTED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    assert file_size_mb >= EXPECTED_MIN_SIZE_MB, f"File too small: {file_size_mb:.1f}MB"
    
    # Load and verify structure
    df = pd.read_csv(file_path, nrows=100000)  # Sample for quick check
    
    # Verify columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"
    
    # Count total rows
    total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
    assert total_rows >= EXPECTED_MIN_ROWS, f"Too few rows: {total_rows:,}"
    
    # Verify date range
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    
    print(f"✅ Data verification passed:")
    print(f"   File size: {file_size_mb:.1f}MB")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Date range: {date_range}")
    print(f"   Columns: {list(df.columns)}")
    
    return True

# Run verification after download
verify_btc_data()
```

## Local Pre-GPU Validation

Before deploying to expensive GPU resources, run comprehensive local validation to catch preventable issues.

### Validation Script
```bash
# Activate environment
source .venv/bin/activate

# Run complete local validation
python -c "
# Test 1: Import validation
print('Testing imports...')
try:
    from utils.validate import assert_regular_grid
    from features.builder import build_indicators
    from nf_models.factory import instantiate_models
    from cv.runner import run_cv
    print('✅ All imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)

# Test 2: Configuration validation
print('Testing configuration...')
try:
    import yaml
    with open('experiments/h4.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    required = ['h', 'n_windows', 'step_size', 'val_size', 'models']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f'Missing keys: {missing}')
    print('✅ Configuration valid')
except Exception as e:
    print(f'❌ Config failed: {e}')
    exit(1)

# Test 3: Data pipeline on small sample
print('Testing data pipeline...')
try:
    from utils.io import load_raw_1min_data, aggregate_1min_to_15min
    import pandas as pd
    
    # Load first 10k rows only
    df_sample = pd.read_csv('data/raw/btcusd_1-min_data.csv', nrows=10000)
    df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'], unit='s')
    df_15min = aggregate_1min_to_15min(df_sample)
    print(f'✅ Data pipeline: {len(df_sample)} → {len(df_15min)} bars')
except Exception as e:
    print(f'❌ Data pipeline failed: {e}')
    exit(1)

# Test 4: Model instantiation (CPU only)
print('Testing model instantiation...')
try:
    from nf_models.factory import instantiate_models
    cfg['hist_exog_list'] = []
    cfg['futr_exog_list'] = []  
    cfg['stat_exog_list'] = []
    models = instantiate_models(cfg, verbose=False)
    print(f'✅ Models instantiated: {len(models)} models')
except Exception as e:
    print(f'❌ Model instantiation failed: {e}')
    exit(1)

print('\\n🎉 Local validation passed! Ready for GPU deployment.')
"
```

### Expected Output
```
Testing imports...
✅ All imports successful
Testing configuration...
✅ Configuration valid
Testing data pipeline...
✅ Data pipeline: 10000 → 667 bars
Testing model instantiation...
✅ Models instantiated: 4 models

🎉 Local validation passed! Ready for GPU deployment.
```

### GPU vs Local Operations

**Local (CPU) Operations:**
- ✅ Data loading and processing
- ✅ Feature engineering computation
- ✅ Model instantiation and configuration
- ✅ YAML validation and parsing

**GPU-Only Operations:**
- ⚡ Model training (`.fit()`)
- ⚡ Cross-validation execution
- ⚡ Prediction generation

## Foundation System Summary

This training guide covers the **foundation workflow** implementing Specs 1-4:

### ✅ Implemented Capabilities
- **Complete Data Pipeline**: 1min → 15min → canonical → validated
- **Feature Engineering**: 13 indicators, MTF, shift(1), 256 cap
- **Model Factory**: 4 models (NHITS, NBEATSx, TiDE, PatchTST) with StudentT loss
- **Cross-Validation**: NF-native CV with sCRPS metrics and calibration diagnostics

### 🔄 Future Enhancements (Specs 5-14)
- **Hyperparameter Optimization**: Automated search and tuning
- **Model Selection & Ensembling**: Advanced selection and blending
- **Live Inference Pipeline**: Real-time prediction deployment
- **Monitoring & Retraining**: Drift detection and automated retraining
- **Risk Mitigation**: Advanced error handling and recovery
- **Quality Gates**: Automated acceptance testing

### Success Criteria
- ✅ **Data Quality**: All validation gates pass (regular grid, UTC EOB, no leakage)
- ✅ **Model Performance**: sCRPS < 0.12, coverage within ±2% tolerance
- ✅ **Training Stability**: Consistent results across CV windows
- ✅ **Artifact Generation**: Complete model persistence and metrics

The foundation provides robust basic forecasting capability. Additional specs will enhance automation, monitoring, and production readiness.

## Quick Reference Commands

```bash
# Complete workflow
./setup.sh
cd data/raw && python kaggle_download_btc.py
source .venv/bin/activate
jupyter lab  # Open run_train.ipynb

# Validation and monitoring
python -c "# local validation script above"
watch -n 1 nvidia-smi
ls -la experiments/h4/

# Troubleshooting
./setup.sh --clean && ./setup.sh  # Reset environment
rm -f data/processed/* && cd data/raw && python kaggle_download_btc.py  # Reset data
```

The system is designed for autonomous execution with comprehensive error handling and recovery procedures.
