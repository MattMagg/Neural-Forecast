#!/bin/bash
set -e

# Direct Setup Script for BTC Forecasting System
# No tests, no dry-run, just execution

# Configuration
PROJECT_DIR="/home/ubuntu/Neural-Forecast"
VENV_DIR=".venv"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Simple logging
log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check we're not root
if [[ $EUID -eq 0 ]]; then
   error "Don't run as root. Run as ubuntu user."
fi

log "Starting setup..."

# Install system packages
log "Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget software-properties-common \
    pkg-config libffi-dev libssl-dev python3-venv

# Install Node.js and Claude Code
log "Installing Node.js and Claude Code..."
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic-ai/claude-code

# Upgrade pip
log "Upgrading pip..."
python3 -m pip install --upgrade pip

# Setup CUDA paths (Thunder Compute has CUDA 12.9 pre-installed)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to bashrc if not already there
if ! grep -q "cuda" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Install TA-Lib C library
log "Installing TA-Lib..."
if ! sudo apt install -y libta-lib-dev; then
    # Compile from source if package fails
    cd /tmp
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr/local
    make && sudo make install
    echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf
    sudo ldconfig
fi

# Clone repository if needed
cd "$HOME"
if [ ! -d "Neural-Forecast" ]; then
    log "Cloning repository..."
    git clone -b instance-training-v0.5.2.X https://github.com/MattMagg/Neural-Forecast.git
fi

cd Neural-Forecast

# Switch to correct branch
git checkout instance-training-v0.5.2.X 2>/dev/null || true

# Create virtual environment
log "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies in critical order
log "Installing Python packages..."

# Core packages
pip install numpy==2.3.2
pip install pandas==2.3.1 pyarrow==20.0.0 kagglehub

# PyTorch (upgrade from pre-installed)
pip install --upgrade torch==2.8.0 pytorch-lightning==2.5.3

# NeuralForecast
pip install neuralforecast==3.0.2

# ML packages
pip install scikit-learn==1.7.1 statsmodels==0.14.5

# TA-Lib Python
pip install TA-Lib==0.6.5

# Feature engineering
pip install vectorbt==0.28.0
pip install pandas-ta-openbb==0.4.22
pip install technical==1.4.0
pip install freqtrade==2025.7

# Visualization
pip install matplotlib==3.10.5 seaborn==0.13.2 plotly==6.3.0

# Infrastructure
pip install modal==1.1.3
pip install GPUtil==1.4.0
pip install pyyaml==6.0.2
pip install joblib==1.5.1
pip install tqdm==4.67.1
pip install psutil==6.1.1

# Testing
pip install pytest==8.4.1 pytest-cov==6.0.0

# Create directory structure
log "Creating directories..."
mkdir -p data/raw data/processed
mkdir -p experiments/h4 experiments/h8 experiments/h16 experiments/h32
mkdir -p reports

# Quick GPU check
log "Checking GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

log "✅ Setup complete!"
log "Activate with: source $VENV_DIR/bin/activate"