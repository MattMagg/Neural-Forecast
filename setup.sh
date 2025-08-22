#!/bin/bash
set -e

# Thunder Compute Setup Script for BTC Forecasting System v0.5.1.2
# Optimized for Thunder Compute instances with pre-installed CUDA 12.9, PyTorch 2.7.1, JupyterLab

# Configuration
PROJECT_DIR="/home/ubuntu/Neural-Forecast"
VENV_DIR=".venv"
LOG_FILE="setup.log"

# Pre-installed software versions on Thunder Compute
PREINSTALLED_CUDA="12.9"
PREINSTALLED_PYTORCH="2.7.1"
PREINSTALLED_CUDNN="9.0"
PREINSTALLED_PYTHON="pre-installed"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Dry run function - shows what would be executed
dry_run() {
    echo -e "${YELLOW}[DRY-RUN]${NC} Would execute: $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root. Run as ubuntu user with sudo access."
fi

# Parse command line arguments
TEST_ONLY=false
CLEAN=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--test-only] [--clean] [--dry-run]"
            echo ""
            echo "Note: CUDA 12.9, PyTorch 2.7.1, and JupyterLab are pre-installed on Thunder Compute"
            exit 1
            ;;
    esac
done

# Clean previous installation if requested
if [ "$CLEAN" = true ]; then
    log "Cleaning previous installation..."
    rm -rf "$VENV_DIR"
    rm -f "$LOG_FILE"
    log "Cleanup complete"
    exit 0
fi

# Test only mode
if [ "$TEST_ONLY" = true ]; then
    log "Running validation tests only..."
    if [ -f "validation_tests.py" ]; then
        python3.13 validation_tests.py
    else
        error "validation_tests.py not found. Run full setup first."
    fi
    exit 0
fi

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    log "DRY RUN MODE - Thunder Compute optimized setup..."
    echo ""
    echo -e "${GREEN}[PRE-INSTALLED]${NC} CUDA $PREINSTALLED_CUDA, PyTorch $PREINSTALLED_PYTORCH, CUDNN $PREINSTALLED_CUDNN, JupyterLab"
    echo ""
    dry_run "sudo apt update && sudo apt upgrade -y"
    dry_run "sudo apt install -y build-essential git curl wget software-properties-common pkg-config libffi-dev libssl-dev python3-venv"
    dry_run "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -"
    dry_run "sudo apt install -y nodejs"
    dry_run "npm install -g @anthropic-ai/claude-code"
    dry_run "python3 -m pip install --upgrade pip"
    dry_run "sudo apt install -y libta-lib-dev (or compile from source)"
    dry_run "git clone -b v0.5.1.2 https://github.com/MattMagg/Neural-Forecast.git"
    dry_run "python3 -m venv .venv"
    dry_run "source .venv/bin/activate"
    dry_run "pip install numpy (compatible with pre-installed PyTorch)"
    dry_run "pip install pandas pyarrow (compatible versions)"
    dry_run "pip install torch==2.8.0 pytorch-lightning==2.5.3 (upgrade from pre-installed 2.7.1)"
    dry_run "pip install neuralforecast==3.0.2"
    dry_run "pip install scikit-learn==1.7.1 statsmodels==0.14.5"
    dry_run "pip install TA-Lib==0.6.5"
    dry_run "pip install remaining packages from requirements.txt"
    dry_run "mkdir -p data/{raw,processed} experiments/{h4,h8,h16,h32} reports"
    dry_run "Test compatibility with pre-installed CUDA 12.9"
    
    echo ""
    log "DRY RUN completed. Use without --dry-run to execute."
    exit 0
fi

log "Starting Thunder Compute setup for BTC Forecasting System..."
log "Project directory: $PROJECT_DIR"
log "Pre-installed: CUDA $PREINSTALLED_CUDA, PyTorch $PREINSTALLED_PYTORCH, CUDNN $PREINSTALLED_CUDNN, Python $PREINSTALLED_PYTHON"

# Install system packages
log "Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget software-properties-common \
    pkg-config libffi-dev libssl-dev python3-venv

# Install Node.js and npm
log "Installing Node.js and npm..."
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# Install Claude Code globally
log "Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Upgrade pip for pre-installed Python
log "Upgrading pip for pre-installed Python..."
python3 -m pip install --upgrade pip

# Check pre-installed CUDA (Thunder Compute has CUDA 12.9 pre-installed)
log "Checking pre-installed CUDA..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    log "Found CUDA version: $cuda_version"
    
    # Add CUDA to PATH if not already there
    if ! grep -q "cuda" ~/.bashrc; then
        log "Adding CUDA to PATH..."
        # shellcheck disable=SC2016
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        # shellcheck disable=SC2016
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    
    # Source bashrc for current session
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    warn "CUDA not found. This is unexpected on Thunder Compute instances."
fi

# Install TA-Lib C library
log "Installing TA-Lib C library..."
if ! sudo apt install -y libta-lib-dev; then
    warn "Package installation failed, compiling from source..."
    cd /tmp
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr/local
    make && sudo make install
    echo '/usr/local/lib' | sudo tee -a /etc/ld.so.conf
    sudo ldconfig
fi

# Clone repository if not present
log "Setting up project repository..."
cd "$HOME"
if [ ! -d "Neural-Forecast" ]; then
    log "Cloning Neural-Forecast repository (v0.5.1.2)..."
    if ! git clone -b v0.5.1.2 https://github.com/MattMagg/Neural-Forecast.git; then
        error "Failed to clone repository. Please check the repository URL and branch."
    fi
fi

cd Neural-Forecast

# Verify we're on the correct branch
current_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
if [ "$current_branch" != "v0.5.1.2" ]; then
    log "Switching to v0.5.1.2 branch..."
    git checkout v0.5.1.2 || error "Failed to checkout v0.5.1.2 branch"
fi

# Create virtual environment
log "Creating virtual environment with pre-installed Python..."

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip and install build tools
log "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies in critical order from dependencies.yaml
log "Installing Python dependencies in critical order..."

# Step 1: numpy (must be first)
log "Installing numpy==2.3.2..."
pip install numpy==2.3.2

# Step 2: pandas and pyarrow
log "Installing pandas==2.3.1 and pyarrow==20.0.0..."
pip install pandas==2.3.1 pyarrow==20.0.0

# Step 3: PyTorch (upgrade from pre-installed 2.7.1 to 2.8.0 for compatibility)
log "Upgrading PyTorch from pre-installed 2.7.1 to 2.8.0 for CUDA 12.9 compatibility..."
pip install --upgrade torch==2.8.0 pytorch-lightning==2.5.3

# Step 4: NeuralForecast
log "Installing neuralforecast==3.0.2..."
pip install neuralforecast==3.0.2

# Step 5: scikit-learn and statsmodels
log "Installing scikit-learn==1.7.1 and statsmodels==0.14.5..."
pip install scikit-learn==1.7.1 statsmodels==0.14.5

# Step 6: TA-Lib Python package (after C library)
log "Installing TA-Lib==0.6.5..."
pip install TA-Lib==0.6.5

# Step 7: Feature engineering packages
log "Installing feature engineering packages..."
pip install vectorbt==0.28.0
pip install pandas-ta-openbb==0.4.22
pip install technical==1.4.0
pip install freqtrade==2025.7

# Step 8: Visualization packages
log "Installing visualization packages..."
pip install matplotlib==3.10.5 seaborn==0.13.2 plotly==6.3.0

# Step 9: Infrastructure and utilities
log "Installing infrastructure and utilities..."
pip install modal==1.1.3
pip install GPUtil==1.4.0
pip install pyyaml==6.0.2
pip install joblib==1.5.1
pip install tqdm==4.67.1
pip install psutil==6.1.1

# Step 10: Testing packages
log "Installing testing packages..."
pip install pytest==8.4.1 pytest-cov==6.0.0

# Create required directory structure
log "Creating directory structure..."
mkdir -p data/raw data/processed
mkdir -p experiments/h4 experiments/h8 experiments/h16 experiments/h32
mkdir -p reports

# Run basic validation
log "Running basic validation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import neuralforecast; print(f'NeuralForecast version: {neuralforecast.__version__}')"

# Test critical imports from the project
log "Testing project module imports..."
python -c "from utils.validate import assert_regular_grid; print('utils.validate: OK')" || warn "utils.validate import failed"
python -c "from features.builder import build_indicators; print('features.builder: OK')" || warn "features.builder import failed"
python -c "from nf_models.factory import instantiate_models; print('nf_models.factory: OK')" || warn "nf_models.factory import failed"
python -c "from cv.runner import run_cv; print('cv.runner: OK')" || warn "cv.runner import failed"

# Test GPU access with pre-installed CUDA
log "Testing GPU access with pre-installed CUDA $PREINSTALLED_CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"
else
    warn "CUDA not available through PyTorch. Check PyTorch-CUDA compatibility."
fi

# Test development tools
log "Testing development tools..."
if command -v npm &> /dev/null; then
    npm_version=$(npm --version)
    log "npm version: $npm_version"
else
    warn "npm not found"
fi

if command -v claude-code &> /dev/null; then
    log "claude-code installed successfully"
else
    warn "claude-code not found"
fi

# Generate setup report
log "Generating setup report..."
cat > setup_report.txt << EOF
Thunder Compute Setup Report
============================
Date: $(date)
Python Version: $(python --version)
Virtual Environment: $VENV_DIR
Project Directory: $(pwd)

Installed Packages:
$(pip list)

GPU Status:
$(python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA test failed")

Project Modules:
$(python -c "
try:
    from utils.validate import assert_regular_grid
    print('✓ utils.validate')
except: print('✗ utils.validate')

try:
    from features.builder import build_indicators
    print('✓ features.builder')
except: print('✗ features.builder')

try:
    from nf_models.factory import instantiate_models
    print('✓ nf_models.factory')
except: print('✗ nf_models.factory')

try:
    from cv.runner import run_cv
    print('✓ cv.runner')
except: print('✗ cv.runner')
")
EOF

log "Setup complete!"
log "Setup report saved to: setup_report.txt"
log "Log file saved to: $LOG_FILE"

log "Thunder Compute environment ready with:"
log "  - CUDA $PREINSTALLED_CUDA (pre-installed)"
log "  - PyTorch 2.8.0 (upgraded from pre-installed 2.7.1)"
log "  - Python $PYTHON_VERSION with BTC forecasting dependencies"
log "  - JupyterLab (pre-installed)"
log ""
log "Activate environment with: source $VENV_DIR/bin/activate"
log "Ready for BTC forecasting model training!"