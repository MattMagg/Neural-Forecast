# 🚀 GPU Migration & Training Workflow
## Neural-Forecast A100 Deployment Plan

**Version**: 1.0  
**Created**: 2025-08-17  
**Estimated Total Time**: 4-6 hours active, 8-12 hours total with training  
**Required Resources**: A100 GPU (40GB+ VRAM), 100GB storage, 32GB+ system RAM

---

## 📋 Pre-Flight Checklist

### Local Machine Preparation
- [ ] Verify git status is clean: `git status`
- [ ] Commit any uncommitted changes
- [ ] Note current branch: `git branch --show-current`
- [ ] Record current commit hash: `git rev-parse HEAD`
- [ ] Verify data file exists: `ls -lh data/raw/btcusd_1-min_data.csv`
- [ ] Check SSH access to A100: `ssh a100-instance echo "Connected"`

### A100 Instance Information
```bash
# Record your instance details here:
A100_HOST=""  # e.g., ubuntu@123.45.67.89 or instance-name
A100_KEY=""   # e.g., ~/.ssh/a100-key.pem
A100_PATH="/home/ubuntu/Neural-Forecast"  # Target directory on A100
```

---

## 🏗️ Phase 1: Repository Preparation (10 minutes)

### 1.1 Create Git Worktree for Results Management
```bash
# On local machine - create separate worktree for GPU results
cd /Users/mac-main/Neural-Forecast

# Create a new branch for GPU results (keeps code separate from artifacts)
git branch gpu-results

# Create worktree in separate directory
git worktree add ../Neural-Forecast-Results gpu-results

# Verify worktree creation
git worktree list
```

### 1.2 Create Transfer Package
```bash
# Create a manifest of files to transfer
cat > transfer_manifest.txt << 'EOF'
# Code files
*.py
*.ipynb
*.md
*.yaml
*.json
*.txt

# Directories
cv/
uq/
utils/
features/
nf_models/
experiments/
tests/
docs/
.kiro/

# Data (essential only)
data/raw/btcusd_1-min_data.csv

# Exclude
!*.parquet
!*.pkl
!*.pt
!*.ckpt
!__pycache__/
!.git/
!*.pyc
EOF

# Create deployment scripts
mkdir -p deployment_scripts
```

### 1.3 Create Monitoring Script
```bash
cat > deployment_scripts/monitor_training.sh << 'EOF'
#!/bin/bash
# GPU Training Monitor Dashboard

while true; do
    clear
    echo "========================================="
    echo "   Neural-Forecast Training Monitor"
    echo "   $(date)"
    echo "========================================="
    
    # GPU Status
    echo -e "\n📊 GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu \
                --format=csv,noheader,nounits | column -t -s ','
    
    # Training Progress
    echo -e "\n📈 Training Progress:"
    for log in logs/*.log; do
        if [ -f "$log" ]; then
            echo "  $(basename $log):"
            tail -n 3 "$log" | sed 's/^/    /'
        fi
    done
    
    # Disk Usage
    echo -e "\n💾 Disk Usage:"
    df -h . | grep -v Filesystem
    
    # Recent Outputs
    echo -e "\n📁 Recent Artifacts:"
    find experiments/ -type f -mmin -30 -name "*.parquet" -o -name "*.csv" -o -name "*.json" | \
        head -5 | xargs -I {} ls -lh {} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    
    sleep 10
done
EOF

chmod +x deployment_scripts/monitor_training.sh
```

### 1.4 Create Training Orchestration Script
```bash
cat > deployment_scripts/run_all_training.sh << 'EOF'
#!/bin/bash
# Master training script for all horizons

set -e  # Exit on error

# Configuration
PYTHON_CMD="python"
BASE_DIR="/home/ubuntu/Neural-Forecast"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p $LOG_DIR
mkdir -p $BASE_DIR/experiments/{h4,h8,h16,h32}/{models,results}

# Function to run training for a horizon
run_horizon() {
    local horizon=$1
    echo "🚀 Starting training for horizon h=$horizon at $(date)"
    
    # Run training with proper logging
    $PYTHON_CMD $BASE_DIR/run_train.py \
        --config $BASE_DIR/experiments/h${horizon}.yaml \
        --raw-data $BASE_DIR/data/raw/btcusd_1-min_data.csv \
        --output-dir $BASE_DIR/data/processed \
        --save-models \
        2>&1 | tee $LOG_DIR/training_h${horizon}_${TIMESTAMP}.log
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Horizon h=$horizon completed successfully at $(date)"
    else
        echo "❌ Horizon h=$horizon failed with code $exit_code at $(date)"
        return $exit_code
    fi
}

# Main execution
echo "========================================="
echo "Neural-Forecast Full Training Pipeline"
echo "Started: $(date)"
echo "========================================="

# Option 1: Sequential execution (safer, easier to debug)
for h in 4 8 16 32; do
    run_horizon $h
done

# Option 2: Parallel execution (faster, uses more resources)
# Uncomment below and comment above for parallel execution
# for h in 4 8 16 32; do
#     run_horizon $h &
#     pids+=($!)
# done
# 
# # Wait for all background jobs
# for pid in ${pids[@]}; do
#     wait $pid
# done

echo "========================================="
echo "All training completed: $(date)"
echo "========================================="

# Generate summary report
$PYTHON_CMD -c "
import pandas as pd
import json
from pathlib import Path

results = {}
for h in [4, 8, 16, 32]:
    try:
        metrics_path = Path(f'experiments/h{h}/metrics.json')
        if metrics_path.exists():
            with open(metrics_path) as f:
                results[f'h{h}'] = json.load(f)
    except:
        results[f'h{h}'] = 'Failed or not found'

print('\n📊 Training Summary:')
for horizon, metrics in results.items():
    print(f'\n{horizon}:')
    if isinstance(metrics, list) and len(metrics) > 0:
        best = min(metrics, key=lambda x: x.get('sCRPS', float('inf')))
        print(f'  Best Model: {best.get(\"model\", \"Unknown\")}')
        print(f'  sCRPS: {best.get(\"sCRPS\", \"N/A\")}')
        print(f'  Coverage 90%: {best.get(\"coverage_90\", \"N/A\")}')
    else:
        print(f'  Status: {metrics}')
"
EOF

chmod +x deployment_scripts/run_all_training.sh
```

---

## 📤 Phase 2: Transfer to A100 (15-30 minutes)

### 2.1 Initial Connection Test
```bash
# Test connection and create directory structure
ssh ${A100_HOST} << 'EOF'
    echo "✅ Connected to A100 instance"
    mkdir -p ~/Neural-Forecast/{logs,results,temp}
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    python --version
    pip --version
EOF
```

### 2.2 Transfer Code and Data
```bash
# Use rsync for efficient transfer (handles interruptions gracefully)
# Exclude large artifacts and git history
rsync -avzP \
    --exclude='*.parquet' \
    --exclude='*.pkl' \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='experiments/h*/models/*' \
    --exclude='experiments/h*/results/*' \
    ./ ${A100_HOST}:${A100_PATH}/

# Verify transfer
ssh ${A100_HOST} "cd ${A100_PATH} && find . -type f -name '*.py' | wc -l"
```

### 2.3 Transfer Monitoring Scripts
```bash
# Copy deployment scripts
scp -r deployment_scripts/ ${A100_HOST}:${A100_PATH}/

# Make scripts executable on remote
ssh ${A100_HOST} "chmod +x ${A100_PATH}/deployment_scripts/*.sh"
```

---

## 🔧 Phase 3: Environment Setup on A100 (20 minutes)

### 3.1 Connect to A100 and Setup Python Environment
```bash
# SSH to A100
ssh ${A100_HOST}
cd ${A100_PATH}

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3.2 Install Core Dependencies
```bash
# Install in stages to handle dependencies properly
# Stage 1: Core scientific computing
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.11.1

# Stage 2: PyTorch with CUDA support (for A100)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Stage 3: NeuralForecast and dependencies
pip install neuralforecast==1.6.4

# Stage 4: Additional requirements
pip install \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    pyyaml==6.0.1 \
    joblib==1.3.1 \
    tqdm==4.65.0 \
    GPUtil==1.4.0

# Stage 5: Optional performance libraries
pip install \
    tensorboard==2.13.0 \
    wandb  # For experiment tracking (optional)

# Verify critical imports
python -c "
import torch
import neuralforecast
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'NeuralForecast version: {neuralforecast.__version__}')
print('✅ All critical imports successful')
"
```

### 3.3 Configure A100-Specific Optimizations
```bash
# Set environment variables for A100 optimization
cat >> ~/.bashrc << 'EOF'

# A100 Optimizations
export CUDA_VISIBLE_DEVICES=0  # Use first GPU (modify for multi-GPU)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging

# Mixed precision training (A100 has excellent FP16 support)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Better error messages
export TORCH_SHOW_CPP_STACKTRACES=1
EOF

source ~/.bashrc
```

### 3.4 Verify Data Integrity
```bash
# Check data file
ls -lh data/raw/btcusd_1-min_data.csv

# Quick data validation
python -c "
import pandas as pd
df = pd.read_csv('data/raw/btcusd_1-min_data.csv')
print(f'Data shape: {df.shape}')
print(f'Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}')
print(f'Columns: {list(df.columns)}')
print('✅ Data file validated')
"
```

---

## 🎯 Phase 4: Training Execution (2-4 hours)

### 4.1 Start Monitoring Dashboard (in separate terminal)
```bash
# Open new terminal/tmux/screen session
ssh ${A100_HOST}
cd ${A100_PATH}
tmux new-session -d -s monitor './deployment_scripts/monitor_training.sh'

# To view monitor:
# tmux attach -t monitor
# To detach: Ctrl+B, D
```

### 4.2 Run Test Training (Smoke Test)
```bash
# Quick test with limited data to verify setup
python run_train.py \
    --config experiments/h4.yaml \
    --raw-data data/raw/btcusd_1-min_data.csv \
    --output-dir data/processed \
    --max-periods 10000 \
    --save-models \
    2>&1 | tee logs/test_run.log

# Check for success
if [ $? -eq 0 ]; then
    echo "✅ Test run successful - proceeding to full training"
else
    echo "❌ Test run failed - check logs/test_run.log"
    exit 1
fi
```

### 4.3 Execute Full Training Pipeline
```bash
# Option A: Use orchestration script (recommended)
tmux new-session -d -s training './deployment_scripts/run_all_training.sh'

# Option B: Manual parallel execution
tmux new-session -d -s training
tmux send-keys -t training "source venv/bin/activate" Enter
tmux send-keys -t training "
for h in 4 8 16 32; do
    echo \"Starting horizon h=\$h\"
    python run_train.py \\
        --config experiments/h\${h}.yaml \\
        --raw-data data/raw/btcusd_1-min_data.csv \\
        --output-dir data/processed \\
        --save-models \\
        > logs/training_h\${h}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
wait
" Enter

# Monitor progress
tmux attach -t training
```

### 4.4 Training Monitoring Commands
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Training logs
tail -f logs/training_h*.log

# Check for errors
grep -i error logs/*.log

# Disk usage
watch -n 10 'du -sh experiments/h*/'

# Process status
ps aux | grep python | grep run_train
```

---

## 🔮 Phase 5: Prediction Generation (30 minutes)

### 5.1 Identify Best Models
```bash
# After training completes, identify best models
python -c "
import json
import pandas as pd
from pathlib import Path

for h in [4, 8, 16, 32]:
    print(f'\n=== Horizon h{h} ===')
    
    # Read leaderboard
    lb_path = Path(f'experiments/h{h}/leaderboard.csv')
    if lb_path.exists():
        df = pd.read_csv(lb_path)
        print(df.head(3).to_string())
        
        # Best model
        best = df.iloc[0]
        print(f'\nBest model: {best[\"model\"]} with sCRPS={best[\"sCRPS\"]:.4f}')
"
```

### 5.2 Generate Predictions
```bash
# Create prediction script
cat > generate_predictions.py << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path
from neuralforecast import NeuralForecast
from datetime import datetime, timedelta
import json

def generate_predictions(horizon):
    """Generate predictions for a specific horizon."""
    
    print(f"\n🔮 Generating predictions for h={horizon}")
    
    # Load the best model
    model_path = Path(f"experiments/h{horizon}/models/best_model")
    if not model_path.exists():
        print(f"  ❌ Model not found for h={horizon}")
        return None
    
    # Load model
    nf = NeuralForecast.load(model_path)
    
    # Load latest data for prediction
    df = pd.read_parquet(f"data/processed/canonical_15min.parquet")
    
    # Get last 1024 points for context
    df_tail = df.tail(1024 + horizon)
    
    # Generate predictions
    predictions = nf.predict(df_tail)
    
    # Save predictions
    output_path = f"experiments/h{horizon}/predictions_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    predictions.to_parquet(output_path)
    
    print(f"  ✅ Predictions saved to {output_path}")
    return predictions

# Generate for all horizons
for h in [4, 8, 16, 32]:
    try:
        generate_predictions(h)
    except Exception as e:
        print(f"  ❌ Error for h={h}: {e}")
EOF

python generate_predictions.py
```

---

## 📥 Phase 6: Results Synchronization (15 minutes)

### 6.1 Prepare Results for Transfer
```bash
# On A100 - Package results
cd ${A100_PATH}

# Create results archive
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz \
    experiments/h*/leaderboard.csv \
    experiments/h*/metrics.json \
    experiments/h*/cv_results*.parquet \
    experiments/h*/predictions*.parquet \
    experiments/h*/models/best_model* \
    logs/*.log

# Get file size
ls -lh results_*.tar.gz
```

### 6.2 Git-Based Results Sync (Recommended)
```bash
# On A100 - Commit results to gpu-results branch
cd ${A100_PATH}

# Initialize git if needed
git init
git remote add origin https://github.com/MattMagg/Neural-Forecast.git

# Switch to gpu-results branch
git fetch origin gpu-results
git checkout -b gpu-results origin/gpu-results

# Add results (use git-lfs for large files if configured)
git add experiments/h*/leaderboard.csv
git add experiments/h*/metrics.json
git add logs/*.log

# For large files, either use git-lfs or create references
find experiments -name "*.parquet" -size +50M | while read f; do
    echo "$(basename $f),$(stat -c%s $f),$(md5sum $f | cut -d' ' -f1)" >> large_files_manifest.csv
done
git add large_files_manifest.csv

# Commit and push
git commit -m "GPU training results - $(date +%Y%m%d_%H%M%S)

Training completed on A100 with following results:
- 4 horizons trained (h=4,8,16,32)
- Models: NHITS, NBEATSx, TiDE, PatchTST
- Primary metric: sCRPS
- See leaderboard.csv files for rankings"

git push origin gpu-results
```

### 6.3 Pull Results to Local Machine
```bash
# On local machine - pull results
cd /Users/mac-main/Neural-Forecast-Results

# Pull latest results
git pull

# View results
for h in 4 8 16 32; do
    echo -e "\n=== Horizon h$h ==="
    cat experiments/h${h}/leaderboard.csv 2>/dev/null | head -5 || echo "Not found"
done

# Copy best models to main repository if needed
cp -r experiments/h*/models/best_model* ../Neural-Forecast/experiments/
```

### 6.4 Alternative: Direct Transfer
```bash
# If git approach doesn't work, use direct transfer
# On local machine
scp ${A100_HOST}:${A100_PATH}/results_*.tar.gz ./

# Extract
tar -xzf results_*.tar.gz

# Move to results worktree
mv experiments/* /Users/mac-main/Neural-Forecast-Results/experiments/
```

---

## 📊 Phase 7: Results Analysis (Local)

### 7.1 Create Analysis Script
```bash
cat > analyze_gpu_results.py << 'EOF'
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_results():
    """Comprehensive analysis of GPU training results."""
    
    results_dir = Path("/Users/mac-main/Neural-Forecast-Results")
    
    # Collect all results
    summary = {}
    
    for h in [4, 8, 16, 32]:
        h_results = {}
        
        # Load leaderboard
        lb_path = results_dir / f"experiments/h{h}/leaderboard.csv"
        if lb_path.exists():
            df = pd.read_csv(lb_path)
            h_results['best_model'] = df.iloc[0]['model']
            h_results['best_scrps'] = df.iloc[0]['sCRPS']
            h_results['all_models'] = df.to_dict('records')
        
        # Load metrics
        metrics_path = results_dir / f"experiments/h{h}/metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                h_results['detailed_metrics'] = json.load(f)
        
        summary[f'h{h}'] = h_results
    
    # Print summary
    print("=" * 60)
    print("GPU TRAINING RESULTS SUMMARY")
    print("=" * 60)
    
    for horizon, data in summary.items():
        print(f"\n{horizon.upper()} ({int(horizon[1:])*15} minutes):")
        if 'best_model' in data:
            print(f"  Best Model: {data['best_model']}")
            print(f"  sCRPS: {data['best_scrps']:.4f}")
            
            # Show top 3
            print("  Top 3 Models:")
            for i, model in enumerate(data['all_models'][:3]):
                print(f"    {i+1}. {model['model']}: {model['sCRPS']:.4f}")
    
    return summary

if __name__ == "__main__":
    results = analyze_results()
    
    # Save summary
    with open('gpu_training_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Analysis complete. Summary saved to gpu_training_summary.json")
EOF

python analyze_gpu_results.py
```

### 7.2 Generate Visualizations
```bash
# Create comparison plots
python -c "
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance Across Horizons', fontsize=16)

for idx, h in enumerate([4, 8, 16, 32]):
    ax = axes[idx // 2, idx % 2]
    
    lb_path = Path(f'/Users/mac-main/Neural-Forecast-Results/experiments/h{h}/leaderboard.csv')
    if lb_path.exists():
        df = pd.read_csv(lb_path)
        df.plot(x='model', y='sCRPS', kind='bar', ax=ax, legend=False)
        ax.set_title(f'Horizon {h} ({h*15} min)')
        ax.set_ylabel('sCRPS (lower is better)')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print('✅ Visualization saved to model_comparison.png')
"
```

---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### Issue: CUDA Out of Memory
```bash
# Reduce batch size in experiments/h*.yaml
sed -i 's/batch_size: 32/batch_size: 16/g' experiments/h*.yaml

# Or use gradient accumulation
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256" >> ~/.bashrc
```

#### Issue: Training Hanging
```bash
# Check GPU utilization
nvidia-smi

# Check system resources
htop

# Kill hanging processes
pkill -f run_train.py
```

#### Issue: Slow Transfer Speed
```bash
# Use compression and parallel transfer
tar -czf - . | ssh ${A100_HOST} "cd ${A100_PATH} && tar -xzf -"

# Or use rsync with compression
rsync -avz --compress-level=9 --progress ./ ${A100_HOST}:${A100_PATH}/
```

#### Issue: Dependencies Conflict
```bash
# Create fresh environment
python -m venv venv_fresh
source venv_fresh/bin/activate
pip install --no-cache-dir -r requirements.txt
```

---

## ✅ Success Criteria

### Training Phase
- [ ] All 4 horizons complete without errors
- [ ] Each horizon has saved models in experiments/h*/models/
- [ ] Leaderboard.csv exists for each horizon
- [ ] Metrics.json shows sCRPS < 0.05 for at least one model
- [ ] Coverage metrics within ±3% of nominal (80/90/95%)

### Results Phase
- [ ] Results successfully synced to local machine
- [ ] Can load and inspect saved models locally
- [ ] Predictions generated for future periods
- [ ] Analysis shows clear model rankings

### Quality Metrics
- [ ] Training time < 4 hours for all horizons
- [ ] GPU utilization > 70% during training
- [ ] Memory usage < 32GB GPU RAM
- [ ] No data leakage detected in validation

---

## 📝 Post-Execution Checklist

### On A100
- [ ] Stop all running processes
- [ ] Save important logs
- [ ] Create backup of results
- [ ] Clean up temporary files
- [ ] Document any issues encountered

### On Local Machine
- [ ] Verify all results transferred
- [ ] Create summary report
- [ ] Update TASK_TRACKER.md
- [ ] Commit results to repository
- [ ] Plan next steps based on results

---

## 🚀 Next Steps After Successful Training

1. **Model Analysis**
   - Compare performance across horizons
   - Identify systematic patterns in errors
   - Check calibration quality

2. **Production Preparation**
   - Select best models for deployment
   - Create inference pipeline
   - Set up monitoring

3. **Optimization Opportunities**
   - Implement HPO if sCRPS > target
   - Add ensemble methods
   - Fine-tune on recent data

4. **Documentation**
   - Update model cards with performance metrics
   - Document training configuration
   - Create deployment guide

---

## 📞 Support Resources

- **NeuralForecast Docs**: https://nixtla.github.io/neuralforecast/
- **PyTorch A100 Guide**: https://pytorch.org/docs/stable/notes/cuda.html
- **Git Worktree Docs**: https://git-scm.com/docs/git-worktree

---

**End of Workflow Document**

*This workflow is designed to be executed step-by-step with verification at each phase. Total estimated time: 4-6 hours active work, 8-12 hours including training time.*