#!/bin/bash
set -e

# Thunder Compute Setup Script - Install Dependencies Only
echo "Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget software-properties-common pkg-config libffi-dev libssl-dev

# Install Python 3.13
echo "Installing Python 3.13..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.13 python3.13-dev python3.13-pip

# Install NVIDIA drivers and CUDA
echo "Installing NVIDIA drivers and CUDA..."
sudo apt install -y nvidia-driver-535
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install TA-Lib C library
echo "Installing TA-Lib..."
sudo apt install -y libta-lib-dev || {
    cd /tmp
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr/local
    make && sudo make install
    echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf
    sudo ldconfig
}

# Install Python dependencies in critical order
echo "Installing Python dependencies..."
python3.13 -m pip install --upgrade pip setuptools wheel

# Critical order from dependencies.yaml
python3.13 -m pip install numpy==2.3.2
python3.13 -m pip install pandas==2.3.1 pyarrow==20.0.0
python3.13 -m pip install torch==2.8.0 pytorch-lightning==2.5.3
python3.13 -m pip install neuralforecast==3.0.2
python3.13 -m pip install scikit-learn==1.7.1
python3.13 -m pip install TA-Lib==0.6.5

# Install remaining packages
python3.13 -m pip install -r requirements.txt

echo "Setup complete! Reboot recommended for NVIDIA drivers."