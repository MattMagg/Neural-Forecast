# Modal GPU Training Implementation
## A100-80GB

```bash
pip install modal
modal setup
modal volume create nf-data
modal volume create nf-models
modal volume create nf-checkpoints
```

## modal_train.py

```python
import modal
import sys
from pathlib import Path
from typing import Optional

app = modal.App("neural-forecast-btc")

data_volume = modal.Volume.from_name("nf-data", create_if_missing=True)
models_volume = modal.Volume.from_name("nf-models", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("nf-checkpoints", create_if_missing=True)

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
    gpu="A100-80GB",
    volumes={
        "/data": data_volume,
        "/models": models_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=4 * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    max_inputs=1,
)
def train_horizon(horizon: int, resume: bool = True):
    import subprocess
    import os
    
    sys.path.insert(0, "/app")
    os.chdir("/app")
    
    checkpoint_dir = Path(f"/checkpoints/h{horizon}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    last_checkpoint = checkpoint_dir / "last.ckpt"
    if resume and last_checkpoint.exists():
        print(f"Found checkpoint: {last_checkpoint}")
    
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
        raise RuntimeError(f"Training failed for h={horizon}: {result.stderr}")
    
    models_volume.commit()
    checkpoints_volume.commit()
    
    return result.stdout

@app.local_entrypoint()
def main(
    horizon: Optional[int] = None,
    parallel: bool = False,
    upload_data: bool = False
):
    if upload_data:
        with data_volume.batch_upload() as batch:
            batch.put_file(
                "data/raw/btcusd_1-min_data.csv",
                "/raw/btcusd_1-min_data.csv"
            )
    
    elif horizon:
        result = train_horizon.remote(horizon)
        print(result)
    
    elif parallel:
        futures = []
        for h in [4, 8, 16, 32]:
            futures.append(train_horizon.spawn(h))
        
        for future, h in zip(futures, [4, 8, 16, 32]):
            result = future.get()

if __name__ == "__main__":
    main()
```

## Execution

```bash
modal run modal_train.py --upload-data
modal run --detach modal_train.py --parallel
```

## Results

```bash
for h in 4 8 16 32; do
    modal volume get nf-models /h${h}/leaderboard.parquet ./results/h${h}_leaderboard.parquet
    modal volume get nf-models /h${h}/cv_raw.parquet ./results/h${h}_cv_raw.parquet
done
```