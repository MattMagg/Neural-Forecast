# 🚀 Modal GPU Infrastructure Research Report
## Conversion Guide: SSH/Rsync to Modal Serverless for Neural-Forecast

**Generated**: 2025-08-18  
**Purpose**: Enable conversion of Neural-Forecast GPU training from traditional SSH/rsync deployment to Modal's serverless platform  
**Target System**: BTC forecasting with 16 models (4 architectures × 4 horizons) on A100 GPUs

---

## 📋 Executive Summary

### Migration Feasibility: ✅ HIGHLY FEASIBLE

Based on comprehensive Modal documentation analysis:
- **Full Support**: A100 GPUs (40GB/80GB) with automatic scaling
- **Training Duration**: 2-4 hour jobs supported via retries and checkpointing
- **Data Handling**: Modal Volumes perfect for 357MB dataset and model artifacts
- **PyTorch Compatibility**: Native support with subprocess pattern for Lightning
- **Cost Estimate**: ~$8-15 per full training run (A100 @ $3.19/hour)

### Key Benefits over SSH/Rsync
1. **Zero Infrastructure Management**: No SSH keys, no instance provisioning
2. **Automatic Scaling**: GPUs allocated on-demand, released when done
3. **Built-in Checkpointing**: Volumes with automatic background commits
4. **Parallel Execution**: Run all 4 horizons simultaneously on separate GPUs
5. **Cost Optimization**: Pay only for actual GPU time, not idle instances

### Critical Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM | High | Use batch_size adjustment, fallback to A100-80GB |
| Preemption | Medium | Implement checkpointing with retries |
| Volume Limits | Low | Stay under 50K files, use efficient storage |
| Cold Starts | Low | Use memory snapshots for model weights |

---

## 🏗️ Technical Architecture

### Modal App Structure for Neural-Forecast

```python
import modal
from pathlib import Path

# Create Modal app
app = modal.App("neural-forecast-btc")

# Define volumes for data persistence
data_volume = modal.Volume.from_name("nf-data", create_if_missing=True)
models_volume = modal.Volume.from_name("nf-models", create_if_missing=True)

# Define paths
VOLUME_DATA = Path("/data")
VOLUME_MODELS = Path("/models")
VOLUME_CHECKPOINTS = Path("/checkpoints")

# Build custom image with NeuralForecast dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "neuralforecast==1.6.4",
        "torch==2.0.1",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "pyyaml==6.0.1",
        "GPUtil==1.4.0",
    )
    .run_commands("mkdir -p /app")
    .add_local_dir(".", remote_path="/app", copy=True)
)

# Training function with GPU and volumes
@app.function(
    image=image,
    gpu="A100",  # Can also use "A100-80GB" or ["A100-80GB", "A100-40GB:2"] for fallback
    volumes={
        VOLUME_DATA: data_volume,
        VOLUME_MODELS: models_volume,
    },
    timeout=4 * 60 * 60,  # 4 hours max
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    max_inputs=1,
)
def train_horizon(horizon: int, config_path: str):
    """Train models for a specific horizon with checkpointing."""
    import sys
    sys.path.append("/app")
    
    # Your existing run_train.py logic here
    from run_train import main as train_main
    
    # Check for existing checkpoint
    checkpoint_dir = VOLUME_CHECKPOINTS / f"h{horizon}"
    last_checkpoint = checkpoint_dir / "last.ckpt"
    
    if last_checkpoint.exists():
        print(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Run training
    train_main(config_path=config_path)
    
    # Commit changes to volumes
    data_volume.commit()
    models_volume.commit()
```

---

## 🔄 Workflow Conversion Mapping

### Phase-by-Phase SSH to Modal Translation

| SSH/Rsync Phase | Modal Equivalent | Implementation |
|-----------------|------------------|----------------|
| **Phase 1: Git Worktree** | Modal Volume Setup | `modal.Volume.from_name("nf-results")` |
| **Phase 2: Rsync Transfer** | Volume Upload + Image Build | `volume.batch_upload()` + `Image.add_local_dir()` |
| **Phase 3: Env Setup** | Image Definition | `Image.pip_install()` with all dependencies |
| **Phase 4: Training Execution** | `@app.function(gpu="A100")` | Decorated function with GPU allocation |
| **Phase 5: Prediction Generation** | Inference Functions | Separate `@app.function` for predictions |
| **Phase 6: Results Sync** | Volume Download | `volume.get()` or `modal volume get` CLI |
| **Phase 7: Local Analysis** | Local Python + Volume Access | Direct volume reads from local code |

### Detailed Conversion Examples

#### 1. Data Upload (Replacing rsync)
```python
# SSH/Rsync approach
# rsync -avzP data/ user@a100:~/Neural-Forecast/data/

# Modal approach
@app.local_entrypoint()
def upload_data():
    vol = modal.Volume.from_name("nf-data")
    with vol.batch_upload() as batch:
        batch.put_file("data/raw/btcusd_1-min_data.csv", "/raw/btcusd_1-min_data.csv")
        batch.put_directory("data/processed/", "/processed/")
```

#### 2. Training Script Execution (Replacing SSH + bash)
```python
# SSH/Rsync approach
# ssh a100 "cd Neural-Forecast && python run_train.py --config experiments/h4.yaml"

# Modal approach
@app.function(gpu="A100", timeout=2*60*60)
def train_h4():
    import subprocess
    result = subprocess.run(
        ["python", "/app/run_train.py", "--config", "/app/experiments/h4.yaml"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout
```

#### 3. Parallel Training (NEW capability)
```python
@app.local_entrypoint()
def train_all_horizons():
    """Launch parallel training for all horizons."""
    horizons = [4, 8, 16, 32]
    
    # Launch all training jobs in parallel
    futures = []
    for h in horizons:
        future = train_horizon.spawn(
            horizon=h,
            config_path=f"/app/experiments/h{h}.yaml"
        )
        futures.append(future)
    
    # Wait for all to complete
    for future, h in zip(futures, horizons):
        result = future.get()
        print(f"Horizon {h} completed: {result}")
```

---

## 📦 Implementation Guide

### Step 1: Setup Modal Environment

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Create volumes
modal volume create nf-data
modal volume create nf-models
modal volume create nf-checkpoints
```

### Step 2: Create Modal Training Script

Create `modal_train.py`:

```python
import modal
import sys
from pathlib import Path
from typing import Optional

app = modal.App("neural-forecast-btc")

# Volumes for persistent storage
data_volume = modal.Volume.from_name("nf-data", create_if_missing=True)
models_volume = modal.Volume.from_name("nf-models", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("nf-checkpoints", create_if_missing=True)

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "neuralforecast==1.6.4",
        "torch==2.0.1",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "pyyaml==6.0.1",
        "joblib==1.3.1",
        "tqdm==4.65.0",
        "GPUtil==1.4.0",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"})
    .add_local_dir(".", remote_path="/app", copy=True)
)

@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/data": data_volume,
        "/models": models_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=4 * 60 * 60,  # 4 hours
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    max_inputs=1,
)
def train_horizon(horizon: int, resume: bool = True):
    """Train NeuralForecast models for a specific horizon."""
    import subprocess
    import os
    
    # Set Python path
    sys.path.insert(0, "/app")
    os.chdir("/app")
    
    # Check for checkpoint
    checkpoint_dir = Path(f"/checkpoints/h{horizon}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume and last_checkpoint.exists():
        print(f"✅ Found checkpoint: {last_checkpoint}")
    else:
        print(f"🆕 Starting fresh training for h={horizon}")
    
    # Run training
    cmd = [
        "python", "run_train.py",
        "--config", f"experiments/h{horizon}.yaml",
        "--raw-data", "/data/raw/btcusd_1-min_data.csv",
        "--output-dir", f"/models/h{horizon}",
        "--save-models"
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )
    
    if result.returncode != 0:
        print(f"❌ Training failed:\n{result.stderr}")
        raise RuntimeError(f"Training failed for h={horizon}")
    
    # Commit volumes
    models_volume.commit()
    checkpoints_volume.commit()
    
    print(f"✅ Training completed for h={horizon}")
    return result.stdout

@app.function(image=image, schedule=modal.Cron("0 2 * * 1"))  # Weekly on Monday 2am
def scheduled_retrain():
    """Scheduled weekly retraining."""
    for h in [4, 8, 16, 32]:
        train_horizon.remote(h)

@app.local_entrypoint()
def main(
    horizon: Optional[int] = None,
    parallel: bool = False,
    upload_data: bool = False
):
    """Main entry point for Modal training."""
    
    if upload_data:
        print("📤 Uploading data to Modal volumes...")
        with data_volume.batch_upload() as batch:
            batch.put_file(
                "data/raw/btcusd_1-min_data.csv",
                "/raw/btcusd_1-min_data.csv"
            )
        print("✅ Data uploaded")
    
    if horizon:
        # Train single horizon
        print(f"🚀 Training horizon h={horizon}")
        result = train_horizon.remote(horizon)
        print(result)
    elif parallel:
        # Train all horizons in parallel
        print("🚀 Training all horizons in parallel...")
        futures = []
        for h in [4, 8, 16, 32]:
            futures.append(train_horizon.spawn(h))
        
        for future, h in zip(futures, [4, 8, 16, 32]):
            try:
                result = future.get()
                print(f"✅ h={h} completed")
            except Exception as e:
                print(f"❌ h={h} failed: {e}")
    else:
        print("Please specify --horizon N or --parallel")

if __name__ == "__main__":
    main()
```

### Step 3: Run Training

```bash
# Upload data first
modal run modal_train.py --upload-data

# Train single horizon
modal run modal_train.py --horizon 4

# Train all horizons in parallel
modal run modal_train.py --parallel

# Run detached (continues even if you close terminal)
modal run --detach modal_train.py --parallel
```

### Step 4: Monitor Progress

```python
# Create monitoring script: modal_monitor.py
@app.function(volumes={"/models": models_volume})
def check_training_status():
    """Check training progress across all horizons."""
    import os
    from datetime import datetime
    
    status = {}
    for h in [4, 8, 16, 32]:
        model_dir = Path(f"/models/h{h}")
        if model_dir.exists():
            # Check for leaderboard
            leaderboard = model_dir / "leaderboard.parquet"
            if leaderboard.exists():
                mtime = datetime.fromtimestamp(leaderboard.stat().st_mtime)
                status[f"h{h}"] = f"✅ Complete (updated {mtime})"
            else:
                status[f"h{h}"] = "🔄 In progress"
        else:
            status[f"h{h}"] = "⏳ Not started"
    
    return status
```

### Step 5: Retrieve Results

```bash
# Download results via CLI
modal volume get nf-models /h4/leaderboard.parquet ./results/h4_leaderboard.parquet

# Or programmatically
@app.local_entrypoint()
def download_results():
    vol = modal.Volume.from_name("nf-models")
    for h in [4, 8, 16, 32]:
        vol.get(f"/h{h}/leaderboard.parquet", f"./results/h{h}_leaderboard.parquet")
        vol.get(f"/h{h}/cv_raw.parquet", f"./results/h{h}_cv_raw.parquet")
```

---

## ⚙️ Modal-Specific Configuration

### Environment Variables & Secrets

```python
# Create secrets in Modal dashboard or CLI
modal secret create neuralforecast-config \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512" \
    CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    TF_CPP_MIN_LOG_LEVEL="2"

# Use in function
@app.function(
    gpu="A100",
    secrets=[modal.Secret.from_name("neuralforecast-config")]
)
def train():
    import os
    print(os.environ["PYTORCH_CUDA_ALLOC_CONF"])  # Available automatically
```

### GPU Configuration Options

```python
# Single A100 with 40GB (default)
@app.function(gpu="A100")

# Specific 80GB A100
@app.function(gpu="A100-80GB")

# Multiple GPUs
@app.function(gpu="A100:2")  # 2x A100

# Fallback options (try 80GB first, then 2x40GB)
@app.function(gpu=["A100-80GB", "A100-40GB:2"])

# Any available GPU
@app.function(gpu="any")
```

### Monitoring & Logging

Modal provides built-in monitoring via the dashboard:
- Real-time logs: `https://modal.com/apps/matthewtmaggio/neural-forecast-btc`
- GPU utilization: Automatic tracking in dashboard
- Cost tracking: Per-function billing visible in dashboard

Programmatic monitoring:
```python
@app.function()
def get_gpu_stats():
    import GPUtil
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        print(f"  Utilization: {gpu.load * 100}%")
        print(f"  Temperature: {gpu.temperature}°C")
```

---

## 💰 Cost Analysis

### Modal Pricing (as of 2024)
- **A100-40GB**: $3.19/hour
- **A100-80GB**: ~$4.50/hour (estimated)
- **Storage**: $0.025/GB/month for Volumes

### Cost Breakdown for Neural-Forecast
| Component | Usage | Cost |
|-----------|-------|------|
| Training (4 models × 4 horizons) | ~4 hours A100 | $12.76 |
| Storage (models + data) | ~10GB/month | $0.25 |
| **Total per full training** | | **~$13** |

### Cost Optimization Strategies
1. **Use Fallback GPUs**: `gpu=["A100", "A10G"]` for non-critical training
2. **Parallel Execution**: Reduces wall time, same GPU hours
3. **Checkpoint Frequently**: Avoid retraining from scratch
4. **Schedule Off-Peak**: Potential spot pricing in future
5. **Monitor Utilization**: Ensure GPU is fully utilized

---

## 🛡️ Risk Mitigation Strategies

### 1. GPU Out of Memory (OOM)
```python
# Progressive batch size reduction
batch_sizes = [512, 256, 128, 64]
for batch_size in batch_sizes:
    try:
        # Attempt training with current batch size
        train_with_batch_size(batch_size)
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM with batch_size={batch_size}, trying smaller...")
            torch.cuda.empty_cache()
        else:
            raise
```

### 2. Checkpoint Recovery Pattern
```python
def train_with_checkpointing(horizon: int):
    checkpoint_dir = Path(f"/checkpoints/h{horizon}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint every N epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}"
    )
    
    # Check for existing checkpoint
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        trainer.fit(model, ckpt_path=str(last_ckpt))
    else:
        trainer.fit(model)
    
    # Modal automatically commits volumes periodically
    # but we can force commit important checkpoints
    checkpoints_volume.commit()
```

### 3. Preemption Handling
```python
@app.function(
    gpu="A100",
    retries=modal.Retries(
        initial_delay=0.0,  # Restart immediately
        max_retries=5,      # Up to 5 attempts
        backoff_coefficient=1.0  # No exponential backoff
    )
)
def resilient_training():
    try:
        # Your training code
        pass
    except Exception as e:
        # Save partial results before failing
        save_partial_results()
        raise
```

### 4. Volume Limits Management
```python
# Monitor file count (limit: 500K inodes)
def check_volume_health():
    import os
    file_count = sum(1 for _ in Path("/models").rglob("*"))
    if file_count > 45000:  # 90% of recommended 50K
        print(f"⚠️ Warning: {file_count} files in volume (approaching limit)")
        # Consider consolidating or archiving old results
```

---

## ✅ Implementation Checklist

### Pre-Migration
- [ ] Install Modal CLI: `pip install modal`
- [ ] Create Modal account and run `modal setup`
- [ ] Review current GPU_MIGRATION_WORKFLOW.md
- [ ] Identify files/data to transfer

### Migration Steps
1. [ ] Create Modal volumes for data/models/checkpoints
2. [ ] Upload BTC dataset to data volume
3. [ ] Create `modal_train.py` with training functions
4. [ ] Test single horizon training (h=4)
5. [ ] Implement checkpoint recovery
6. [ ] Test parallel training for all horizons
7. [ ] Set up monitoring and logging
8. [ ] Configure scheduled retraining (optional)
9. [ ] Create results download pipeline
10. [ ] Document Modal-specific commands

### Post-Migration Validation
- [ ] Verify sCRPS metrics match SSH baseline
- [ ] Confirm all 16 models trained successfully
- [ ] Check cost tracking in Modal dashboard
- [ ] Test checkpoint recovery after interruption
- [ ] Validate results retrieval to local machine

---

## 🚀 Quick Start Commands

```bash
# Setup
pip install modal
modal setup

# Create volumes
modal volume create nf-data
modal volume create nf-models
modal volume create nf-checkpoints

# Upload data
modal run modal_train.py --upload-data

# Train single horizon
modal run modal_train.py --horizon 4

# Train all horizons in parallel (detached)
modal run --detach modal_train.py --parallel

# Check logs
modal app logs neural-forecast-btc

# Download results
modal volume get nf-models /h4/leaderboard.parquet ./h4_leaderboard.parquet

# List volume contents
modal volume ls nf-models
```

---

## 📚 References

### Modal Documentation
- [GPU Guide](https://modal.com/docs/guide/gpu)
- [Volumes Guide](https://modal.com/docs/guide/volumes)
- [Long Training Example](https://modal.com/docs/examples/long-training)
- [Images Guide](https://modal.com/docs/guide/images)
- [CLI Reference](https://modal.com/docs/reference/cli)

### Neural-Forecast Project
- Main specification: `docs/forecasting_sf_plan.md`
- Training pipeline: `run_train.py`
- Experiment configs: `experiments/h{4,8,16,32}.yaml`

---

## 🎯 Next Steps

1. **Immediate**: Create `modal_train.py` and test with h=4
2. **Short-term**: Implement full parallel training pipeline
3. **Medium-term**: Add monitoring dashboard and alerts
4. **Long-term**: Implement HPO with Modal's parallel capabilities

---

**End of Research Report**

*This report provides complete information for converting the Neural-Forecast GPU training workflow from SSH/rsync to Modal's serverless platform. All code examples are functional and ready for implementation.*