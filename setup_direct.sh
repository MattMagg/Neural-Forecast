#!/bin/bash
set -e

echo "Starting setup..."

# Install system packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget software-properties-common \
    pkg-config libffi-dev libssl-dev python3-venv

# Install Node.js and Claude Code
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
npm install -g @anthropic-ai/claude-code

# Upgrade pip
python3 -m pip install --upgrade pip


# Install TA-Lib
sudo apt install -y libta-lib-dev || (
    cd /tmp
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr/local
    make && sudo make install
    echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf
    sudo ldconfig
)


# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install packages
pip install numpy==2.3.2
pip install pandas==2.3.1 pyarrow==20.0.0 kagglehub

pip install --upgrade torch==2.8.0 pytorch-lightning==2.5.3

pip install neuralforecast==3.0.2

pip install scikit-learn==1.7.1 statsmodels==0.14.5

pip install TA-Lib==0.6.5

pip install vectorbt==0.28.0
pip install pandas-ta-openbb==0.4.22
pip install technical==1.4.0
pip install freqtrade==2025.7

pip install matplotlib==3.10.5 seaborn==0.13.2 plotly==6.3.0

pip install modal==1.1.3
pip install GPUtil==1.4.0
pip install pyyaml==6.0.2
pip install joblib==1.5.1
pip install tqdm==4.67.1
pip install psutil==6.1.1

pip install pytest==8.4.1 pytest-cov==6.0.0

mkdir -p data/raw data/processed
mkdir -p experiments/h4 experiments/h8 experiments/h16 experiments/h32
mkdir -p reports

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; torch.cuda.is_available() and print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo "Done. Activate with: source .venv/bin/activate"