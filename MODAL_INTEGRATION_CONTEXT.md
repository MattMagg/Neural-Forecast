# Neural-Forecast BTC Project - Modal Integration Planning Context

## Purpose of This Document
You are being asked to review Modal's documentation and create a high-level implementation plan for integrating Modal GPU infrastructure into this existing Neural-Forecast project. You won't be writing code directly - just providing a strategic plan and architecture design.

## Project Overview
This is a complete, working Bitcoin price forecasting system that runs locally. It needs Modal GPU infrastructure to scale training and reduce costs. The core implementation is finished and tested - we just need to wrap it with Modal for cloud GPU execution.

## What This System Does
- Forecasts BTC prices at 15-minute intervals
- Produces predictions for 1, 2, 4, and 8 hours ahead (h=4,8,16,32 bars)
- Uses 4 deep learning models from NeuralForecast library
- Generates probability distributions, not just point forecasts
- Processes 7.16 million historical price bars

## Current Implementation Status
✅ **Complete and Working:**
- Data pipeline (1-min → 15-min aggregation)
- Feature engineering (256 technical indicators)
- Model training with cross-validation
- Prediction generation with confidence intervals
- All core logic thoroughly tested

❌ **Needs Modal Integration:**
- GPU acceleration for training (currently CPU-only)
- Parallel training across horizons
- Serverless inference endpoint
- Automated retraining schedule

## Technical Architecture

### Directory Structure That Modal Needs to Understand
```
Neural-Forecast/
├── data/
│   ├── raw/btcusd_1-min_data.csv (450MB historical data)
│   └── processed/experiments/h{4,8,16,32}/models/ (saved models)
├── features/ (feature computation code)
├── cv/ (cross-validation logic)
├── experiments/h{4,8,16,32}.yaml (configuration files)
├── run_train.py (main training script - uses argparse CLI)
├── run_predict.py (inference script)
├── run_train.ipynb (notebook version - IDENTICAL functionality to .py)
└── run_predict.ipynb (notebook version - IDENTICAL functionality to .py)
```

**Note on Scripts vs Notebooks:**
- Both .py and .ipynb versions exist with identical functionality
- Notebooks have configuration cells instead of argparse
- Notebooks are preferred for easier debugging and iteration
- Either can be used - choose based on Modal's optimal patterns

### Key Technical Details

**Training Entry Point:**
- Script: `run_train.py`
- Interface: Command-line arguments via argparse
- Example: `python run_train.py --config experiments/h16.yaml --output-dir data/processed --save-models`
- Duration: ~30-60 minutes per horizon on CPU
- Memory: ~16GB RAM, benefits from 40GB+ GPU memory

**Model Persistence Pattern:**
- NeuralForecast saves models as DIRECTORIES (not single files)
- Each model creates a folder with multiple internal files
- Loading requires the directory path, not a file path
- Size: ~500MB per horizon (includes all 4 models)

**Critical Dependencies:**
- TA-Lib: Requires C library installation (`apt-get install ta-lib`) BEFORE Python packages
- NeuralForecast: Version 3.0.2 (must stay consistent)
- PyTorch: GPU-enabled version needed

## Modal Integration Requirements

### Training Requirements
1. **GPU**: A100 80GB recommended for parallel model training
2. **Parallelization**: Train 4 horizons independently (can run simultaneously)
3. **Storage**: Need persistent volumes for:
   - Input data (450MB)
   - Trained models (~2GB total for all horizons)
   - Checkpoints for recovery
4. **Time**: Each horizon takes 30-60 minutes
5. **Memory Management**: Models need batch_size=256, fallback to 128 on OOM

### Inference Requirements
1. **GPU**: T4 sufficient (cheaper than A100)
2. **Latency**: Must respond in <1 second
3. **Model Loading**: Pre-load models and keep warm
4. **Scaling**: Auto-scale based on request volume
5. **API**: REST endpoint returning JSON predictions

### Data Flow Architecture
```
1. Training Flow:
   Raw CSV → Feature Engineering → Model Training → Save to Volume
   
2. Inference Flow:
   New Data → Load Model from Volume → Generate Predictions → Return JSON
```

## Specific Challenges to Address

### Challenge 1: TA-Lib Installation
- The TA-Lib Python package requires the C library installed first
- In Modal, this means using `apt_install(["ta-lib"])` before pip packages
- Many implementations fail because they only pip install

### Challenge 2: Model Save/Load Mismatch
- NeuralForecast saves models as directories, not pickle files
- The save path and load path must point to directories
- Example: `/models/h16/best_model/` contains multiple files

### Challenge 3: Argument Passing
- `run_train.py` uses argparse for CLI arguments
- Modal needs to either:
  - Use subprocess to call with CLI args
  - OR refactor to accept function parameters
  - Don't break the existing working code

### Challenge 4: Feature Computation
- Features take ~5 minutes to compute
- Options:
  - Pre-compute and store in volume
  - Compute on-demand in Modal
  - Hybrid: compute base features, derive others

## What the Modal Implementation Plan Should Cover

### Phase 1: Infrastructure Setup
- How to structure Modal app and images
- Volume configuration for data and models
- GPU allocation strategy
- Dependency installation order (especially TA-Lib)

### Phase 2: Training Pipeline
- How to wrap `run_train.py` for Modal execution
- Parallel execution strategy for 4 horizons
- Progress monitoring and logging
- Error recovery and checkpointing

### Phase 3: Inference Service
- Model loading and caching strategy
- API endpoint design
- Warm start optimization
- Auto-scaling configuration

### Phase 4: Operations
- Scheduled retraining (weekly recommended)
- Model versioning in volumes
- Monitoring and alerting
- Rollback procedures

## Success Criteria for Your Plan

Your Modal implementation plan should:
1. Preserve all existing functionality (don't break working code)
2. Enable parallel GPU training for 4 horizons
3. Provide <1 second inference latency
4. Support automated weekly retraining
5. Handle failures gracefully (OOM, preemption)

## Constraints and Non-Negotiables

1. **Must use existing code** - The system works; just wrap it for Modal
2. **Maintain NeuralForecast native patterns** - Don't reinvent their save/load
3. **Keep the same validation checks** - All data quality assertions must pass
4. **Preserve reproducibility** - Same seeds, same results

## Questions Your Plan Should Answer

1. **Data Management**: Should the 450MB CSV be uploaded once or fetched each time?
2. **Feature Engineering**: Pre-compute features or calculate on-demand?
3. **Model Versioning**: How to track model versions in Modal volumes?
4. **Coordination**: How to orchestrate 16 model training jobs (4 horizons × 4 models)?
5. **Monitoring**: What metrics to track for retraining triggers?
6. **Cost Optimization**: When to use A100 vs T4 vs CPU?

## Expected Deliverable

Please provide:
1. A high-level architecture diagram/description
2. Phase-by-phase implementation approach (optimal path for this specific repo)
3. Specific Modal patterns to use (Volumes, Apps, Functions, Classes)
4. Risk mitigation strategies for the challenges listed
5. Recommended Modal best practices for this use case
6. Clear recommendation: notebooks vs scripts for Modal (based on complexity trade-offs)

Remember: The goal is GPU acceleration and scalability without modifying the core forecasting logic. Think of Modal as infrastructure wrapped around the existing working system. Prioritize the simplest, most maintainable approach.

---

*Use this context along with Modal's documentation to design a clean, efficient GPU infrastructure for this forecasting system. Focus on simplicity and reliability over complex orchestration.*