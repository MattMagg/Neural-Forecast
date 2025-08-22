# Thunder Compute Setup Guide

Quick setup guide for BTC forecasting system on Thunder Compute A100XL instance.

## Pre-installed Software ✅

Thunder Compute instances come with these components pre-installed:

- **CUDA 12.9** - GPU compute platform
- **CUDNN 9.0** - Deep learning primitives
- **Python** - Python interpreter with pip package manager
- **PyTorch 2.7.1** - Machine learning framework (setup script upgrades to 2.8.0)
- **JupyterLab** - Interactive development environment
- **Docker** - Container platform (see Thunder Compute Docker guide)
- **Scientific Python Libraries** - NumPy, Pandas, etc.

## 1. Clone Repository

```bash
git clone -b v0.5.1.2 https://github.com/MattMagg/Neural-Forecast.git
cd Neural-Forecast
```

## 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This installs:

- System packages (build tools, development libraries)
- Node.js and Claude Code for AI-assisted development
- TA-Lib C library for technical analysis
- All Python dependencies in correct order (upgrades PyTorch to 2.8.0)
- Creates virtual environment with pre-installed Python and project structure

## 3. Verify Setup

```bash
# Check GPU access
nvidia-smi
source .venv/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 4. Start Training

Activate environment and start training:

```bash
source .venv/bin/activate

# Train single horizon (1-hour forecast)
python run_train.py --horizon h4

# Train all horizons
python run_train.py --horizon h4
python run_train.py --horizon h8  
python run_train.py --horizon h16
python run_train.py --horizon h32
```

Or use JupyterLab (pre-installed):

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 5. Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check training logs
tail -f experiments/h4/training.log
```

That's it! The system is ready for BTC forecasting model training with pre-installed CUDA 12.9, PyTorch, and JupyterLab.
