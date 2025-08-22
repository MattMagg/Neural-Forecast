# Thunder Compute Setup Guide

Quick setup guide for BTC forecasting system on Thunder Compute A100XL instance.

## 1. Install Python 3.13

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.13 python3.13-dev python3.13-pip
```

## 2. Clone Repository (Branch 0.5.1)

```bash
git clone -b 0.5.1 https://github.com/your-repo/Neural-Forecast.git
cd Neural-Forecast
```

## 3. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This installs:
- System packages (build tools, CUDA, NVIDIA drivers)
- TA-Lib C library
- All Python dependencies in correct order

## 4. Reboot (Required for NVIDIA drivers)

```bash
sudo reboot
```

## 5. Verify GPU Access

```bash
nvidia-smi
python3.13 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 6. Start Training

Navigate to notebooks and run training:

```bash
cd notebooks
python3.13 -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Or run training scripts directly:

```bash
# Train h4 model (1 hour horizon)
python3.13 run_train.py --horizon h4

# Train all horizons
python3.13 run_train.py --horizon h4
python3.13 run_train.py --horizon h8  
python3.13 run_train.py --horizon h16
python3.13 run_train.py --horizon h32
```

## 7. Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f logs/training.log
```

That's it! The system is ready for BTC forecasting model training.