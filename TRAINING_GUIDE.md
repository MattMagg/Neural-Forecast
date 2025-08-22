# Thunder Compute A100XL Training Guide

Complete training guide for BTC forecasting system on Thunder Compute A100XL (80GB VRAM).

## Pre-installed Software on Thunder Compute

✅ **CUDA 12.9** - Pre-installed, no installation needed  
✅ **CUDNN 9.0** - Pre-installed  
✅ **PyTorch 2.7.1** - Pre-installed (setup script upgrades to 2.8.0)  
✅ **JupyterLab** - Pre-installed and ready to use  
✅ **Scientific Python Libraries** - NumPy, Pandas, etc. pre-installed  
✅ **Docker** - Available (see Thunder Compute Docker guide)

## Quick Start

### 1. Activate Environment

```bash
cd Neural-Forecast
source .venv/bin/activate
```

### 2. Verify Setup

```bash
python validation_tests.py
```

### 3. Development Tools

Claude Code is available for AI-assisted development:

```bash
# Start Claude Code
claude-code

# Or use npm directly
npm list -g @anthropic-ai/claude-code
```

### 3. Start Training

```bash
# Train single horizon (1-hour forecast)
python run_train.py --horizon h4

# Train all horizons
python run_train.py --horizon h4
python run_train.py --horizon h8
python run_train.py --horizon h16
python run_train.py --horizon h32
```

## A100XL Optimizations

### Optimal Batch Sizes (80GB VRAM)

| Model | Optimal Batch Size | Fallback | Memory Usage |
|-------|-------------------|----------|--------------|
| NHITS | 512 | 256 | ~45GB |
| NBEATSx | 512 | 256 | ~50GB |
| TiDE | 512 | 256 | ~40GB |
| PatchTST | 256 | 128 | ~60GB |

### Memory Configuration

Set these environment variables for optimal performance:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0
```

### Training Commands with A100XL Settings

```bash
# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Train with optimal batch sizes
python run_train.py --horizon h4 --batch-size 512
python run_train.py --horizon h8 --batch-size 512
python run_train.py --horizon h16 --batch-size 512
python run_train.py --horizon h32 --batch-size 256  # Conservative for longer sequences
```

## Training Performance Estimates

### Expected Training Times (A100XL)

| Horizon | Models | CV Windows | Training Time | Memory Peak |
|---------|--------|------------|---------------|-------------|
| h4 (1h) | 4 models | 10 windows | 2-3 hours | ~50GB |
| h8 (2h) | 4 models | 10 windows | 2-3 hours | ~50GB |
| h16 (4h) | 4 models | 10 windows | 3-4 hours | ~55GB |
| h32 (8h) | 4 models | 10 windows | 3-4 hours | ~60GB |

### Full System Training

```bash
# Complete training pipeline (all horizons)
# Estimated total time: 10-14 hours
for horizon in h4 h8 h16 h32; do
    echo "Training horizon: $horizon"
    python run_train.py --horizon $horizon --batch-size 512
    echo "Completed: $horizon"
done
```

## Monitoring Commands

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi -l 1

# GPU utilization with process details
nvidia-smi pmon -i 0 -s um

# Memory usage tracking
gpustat -i 1
```

### Training Progress

```bash
# Monitor training logs
tail -f experiments/h4/training.log

# Check CV results
ls -la experiments/h4/cv_results_*.parquet

# View metrics
python -c "
import pandas as pd
df = pd.read_csv('experiments/h4/leaderboard.csv')
print(df.head())
"
```

### System Monitoring

```bash
# CPU and memory usage
htop

# Disk usage
df -h

# Process monitoring
ps aux | grep python
```

## Training Workflow

### 1. Data Processing

```bash
# Process raw data (if needed)
python run_train.py --save-processed --raw-data data/raw/btcusd_1-min_data.csv
```

### 2. Feature Engineering

```bash
# Test feature pipeline
python -c "
from features.builder import build_indicators
from utils.io import load_processed_data
df = load_processed_data('data/processed/btc_canonical_*.parquet')
features = build_indicators(df)
print(f'Generated {len(features.columns)} features')
"
```

### 3. Model Training

```bash
# Train with cross-validation
python run_train.py --horizon h4 --cv --n-windows 10

# Train final models
python run_train.py --horizon h4 --final-fit
```

### 4. Model Evaluation

```bash
# Generate predictions
python run_predict.py --horizon h4 --mode batch

# View results
python -c "
import pandas as pd
results = pd.read_parquet('experiments/h4/predictions_*.parquet')
print(results.describe())
"
```

## Configuration Files

### Horizon-Specific Configs

Each horizon has optimized settings in `experiments/h{horizon}.yaml`:

```yaml
# experiments/h4.yaml (example)
horizon: 4
batch_size: 512
learning_rate: 0.001
max_epochs: 100
patience: 10

models:
  NHITS:
    input_size: 168  # 1 week
    loss: "DistributionLoss"
  NBEATSx:
    input_size: 168
    loss: "MQLoss"
  TiDE:
    input_size: 168
    loss: "IQLoss"
  PatchTST:
    input_size: 168
    loss: "DistributionLoss"
```

### Global Settings

Modify `settings.yaml` for global parameters:

```yaml
# settings.yaml
data:
  frequency: "15min"
  target: "y"
  
training:
  seed: 42
  accelerator: "gpu"
  devices: 1
  
validation:
  n_windows: 10
  step_size: 4  # horizon-dependent
  val_size: 16  # 4 * horizon
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python run_train.py --horizon h4 --batch-size 256

# Or use gradient accumulation
python run_train.py --horizon h4 --batch-size 128 --accumulate-grad-batches 4
```

#### 2. Training Instability

```bash
# Reduce learning rate
python run_train.py --horizon h4 --learning-rate 0.0005

# Add gradient clipping
python run_train.py --horizon h4 --gradient-clip-val 1.0
```

#### 3. Slow Training

```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if memory allows
python run_train.py --horizon h4 --batch-size 1024

# Use mixed precision
python run_train.py --horizon h4 --precision 16
```

#### 4. Data Loading Issues

```bash
# Check data files
ls -la data/processed/

# Validate data
python validation_tests.py

# Regenerate processed data
python run_train.py --save-processed --force
```

### Error Recovery

#### Failed Training Run

```bash
# Check last checkpoint
ls -la experiments/h4/checkpoints/

# Resume from checkpoint
python run_train.py --horizon h4 --resume-from-checkpoint experiments/h4/checkpoints/last.ckpt
```

#### Corrupted Environment

```bash
# Clean and reinstall
./setup.sh --clean
./setup.sh
```

#### GPU Driver Issues

```bash
# Check driver status
nvidia-smi

# Restart if needed
sudo systemctl restart nvidia-persistenced
```

## Performance Optimization

### A100XL Specific Optimizations

```bash
# Enable Tensor Core usage
export NVIDIA_TF32_OVERRIDE=1

# Optimize CUDA kernels
export CUDA_LAUNCH_BLOCKING=0

# Memory pool optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,roundup_power2_divisions:16
```

### Training Speed Tips

1. **Use optimal batch sizes**: 512 for most models, 256 for PatchTST
2. **Enable mixed precision**: Add `--precision 16` to training commands
3. **Use multiple workers**: Set `--num-workers 4` for data loading
4. **Pin memory**: Add `--pin-memory` for faster GPU transfers

### Memory Management

```bash
# Monitor memory usage during training
python -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB')
print(f'Cached: {torch.cuda.memory_reserved()/1e9:.1f}GB')
"

# Clear cache between runs
python -c "import torch; torch.cuda.empty_cache()"
```

## Results and Artifacts

### Training Outputs

After training, check these directories:

```bash
experiments/h4/
├── cv_results_20250822T120000Z.parquet  # Cross-validation results
├── metrics.json                         # Aggregated metrics
├── leaderboard.csv                      # Model rankings
├── best/                               # Best model artifacts
│   ├── NHITS_20250822T120000Z.pkl
│   ├── NBEATSx_20250822T120000Z.pkl
│   ├── TiDE_20250822T120000Z.pkl
│   └── PatchTST_20250822T120000Z.pkl
└── training.log                        # Training logs
```

### Model Performance

```bash
# View leaderboard
cat experiments/h4/leaderboard.csv

# Check metrics
python -c "
import json
with open('experiments/h4/metrics.json') as f:
    metrics = json.load(f)
print(f'Best sCRPS: {metrics[\"best_scrps\"]:.4f}')
print(f'Coverage 90%: {metrics[\"coverage_90\"]:.3f}')
"
```

### Generate Reports

```bash
# Create acceptance report
python -c "
from uq.calibration import generate_acceptance_report
generate_acceptance_report('h4')
"

# View report
cat reports/h4/acceptance_report.md
```

## Next Steps

After successful training:

1. **Validate Results**: Check acceptance criteria in `reports/h{horizon}/`
2. **Deploy Models**: Use `run_predict.py` for inference
3. **Monitor Performance**: Set up monitoring for live deployment
4. **Retrain Schedule**: Plan regular retraining cadence

## Support

For issues or questions:

1. Check validation report: `validation_report.txt`
2. Review training logs: `experiments/h{horizon}/training.log`
3. Run diagnostics: `python validation_tests.py`
4. Check system status: `nvidia-smi` and `htop`

The system is designed to be robust and self-recovering. Most issues can be resolved by following the troubleshooting steps above.
