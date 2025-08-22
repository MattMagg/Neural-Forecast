#!/bin/bash

# Thunder Compute Cleanup Script
# Cleans up failed installations and provides fresh start capability

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/ubuntu/Neural-Forecast"
VENV_DIR=".venv"

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Parse command line arguments
FULL_CLEANUP=false
CUDA_CLEANUP=false
PYTHON_CLEANUP=false
VENV_ONLY=false

show_help() {
    echo "Thunder Compute Cleanup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --full          Complete cleanup (CUDA, Python, packages, venv)"
    echo "  --cuda          Clean CUDA and NVIDIA drivers only"
    echo "  --python        Clean Python installations only"
    echo "  --venv-only     Clean virtual environment only (default)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Clean virtual environment only"
    echo "  $0 --venv-only        # Clean virtual environment only"
    echo "  $0 --cuda             # Clean CUDA/NVIDIA installations"
    echo "  $0 --full             # Complete system cleanup"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_CLEANUP=true
            shift
            ;;
        --cuda)
            CUDA_CLEANUP=true
            shift
            ;;
        --python)
            PYTHON_CLEANUP=true
            shift
            ;;
        --venv-only)
            VENV_ONLY=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Default to venv cleanup if no options specified
if [ "$FULL_CLEANUP" = false ] && [ "$CUDA_CLEANUP" = false ] && [ "$PYTHON_CLEANUP" = false ]; then
    VENV_ONLY=true
fi

log "Starting Thunder Compute cleanup..."

# Function to clean virtual environment
cleanup_venv() {
    log "Cleaning virtual environment..."
    
    if [ -d "$PROJECT_DIR" ]; then
        cd "$PROJECT_DIR"
        
        if [ -d "$VENV_DIR" ]; then
            log "Removing virtual environment: $VENV_DIR"
            rm -rf "$VENV_DIR"
        else
            info "Virtual environment not found"
        fi
        
        # Clean Python cache files
        log "Cleaning Python cache files..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        find . -name "*.pyo" -delete 2>/dev/null || true
        
        # Clean temporary files
        log "Cleaning temporary files..."
        rm -f setup.log setup_report.txt validation_report.txt
        rm -rf .pytest_cache
        rm -rf build/ dist/ *.egg-info/
        
    else
        warn "Project directory not found: $PROJECT_DIR"
    fi
}

# Function to clean Python installations
cleanup_python() {
    log "Cleaning Python installations..."
    
    # Remove Python 3.13 packages
    warn "This will remove Python 3.13 and all installed packages"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt remove --purge -y python3.13 python3.13-dev python3.13-venv python3.13-distutils
        sudo apt autoremove -y
        
        # Clean pip cache
        rm -rf ~/.cache/pip
        
        # Remove deadsnakes PPA (optional)
        read -p "Remove deadsnakes PPA? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo add-apt-repository --remove ppa:deadsnakes/ppa -y
        fi
    else
        info "Skipping Python cleanup"
    fi
}

# Function to clean Node.js and development tools
cleanup_nodejs() {
    log "Cleaning Node.js and development tools..."
    
    # Remove claude-code globally
    if command -v npm &> /dev/null; then
        npm uninstall -g @anthropic-ai/claude-code 2>/dev/null || true
    fi
    
    # Remove Node.js
    sudo apt remove --purge -y nodejs npm
    sudo apt autoremove -y
    
    # Clean npm cache
    rm -rf ~/.npm
    rm -rf ~/.cache/npm
}

# Function to clean CUDA installations
cleanup_cuda() {
    log "Cleaning CUDA and NVIDIA installations..."
    
    warn "This will remove NVIDIA drivers and CUDA toolkit"
    warn "You will need to reboot after this cleanup"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove CUDA packages
        sudo apt remove --purge -y cuda-toolkit-* cuda-runtime-* cuda-drivers nvidia-driver-*
        sudo apt remove --purge -y libnvidia-* nvidia-*
        
        # Remove CUDA directories
        sudo rm -rf /usr/local/cuda*
        sudo rm -rf /usr/lib/nvidia*
        
        # Clean package cache
        sudo apt autoremove -y
        sudo apt autoclean
        
        # Remove CUDA keyring
        sudo apt remove --purge -y cuda-keyring
        
        # Clean environment variables from bashrc
        if [ -f ~/.bashrc ]; then
            log "Cleaning CUDA paths from ~/.bashrc"
            sed -i '/cuda-12.2/d' ~/.bashrc
            sed -i '/CUDA/d' ~/.bashrc
        fi
        
        warn "REBOOT REQUIRED after CUDA cleanup"
    else
        info "Skipping CUDA cleanup"
    fi
}

# Function to clean TA-Lib
cleanup_talib() {
    log "Cleaning TA-Lib installations..."
    
    # Remove system TA-Lib
    sudo apt remove --purge -y libta-lib-dev libta-lib0
    
    # Remove compiled TA-Lib (if installed from source)
    if [ -d "/usr/local/lib" ] && [ -f "/usr/local/lib/libta_lib.so" ]; then
        log "Removing TA-Lib compiled from source..."
        sudo rm -f /usr/local/lib/libta_lib.*
        sudo rm -f /usr/local/include/ta_*.h
        sudo rm -rf /usr/local/include/ta-lib/
        sudo ldconfig
    fi
    
    # Clean temporary build files
    rm -rf /tmp/ta-lib*
}

# Function to clean downloaded packages
cleanup_downloads() {
    log "Cleaning downloaded packages..."
    
    # Clean apt cache
    sudo apt clean
    sudo apt autoremove -y
    
    # Clean temporary downloads
    rm -f /tmp/cuda-keyring_*.deb
    rm -rf /tmp/ta-lib*
    
    # Clean user cache
    rm -rf ~/.cache/pip
    rm -rf ~/.cache/matplotlib
}

# Function to reset system packages
reset_system_packages() {
    log "Resetting system packages..."
    
    # Update package lists
    sudo apt update
    
    # Fix any broken packages
    sudo apt --fix-broken install -y
    sudo dpkg --configure -a
    
    # Clean package cache
    sudo apt autoremove -y
    sudo apt autoclean
}

# Execute cleanup based on options
if [ "$VENV_ONLY" = true ]; then
    cleanup_venv
    log "Virtual environment cleanup complete"
fi

if [ "$PYTHON_CLEANUP" = true ] || [ "$FULL_CLEANUP" = true ]; then
    cleanup_venv
    cleanup_python
    log "Python cleanup complete"
fi

if [ "$CUDA_CLEANUP" = true ] || [ "$FULL_CLEANUP" = true ]; then
    cleanup_cuda
    log "CUDA cleanup complete"
fi

if [ "$FULL_CLEANUP" = true ]; then
    cleanup_venv
    cleanup_python
    cleanup_nodejs
    cleanup_cuda
    cleanup_talib
    cleanup_downloads
    reset_system_packages
    
    log "Full cleanup complete"
    warn "REBOOT RECOMMENDED after full cleanup"
    
    read -p "Reboot now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    fi
fi

log "Cleanup completed successfully!"

# Provide next steps
echo ""
info "Next steps:"
if [ "$VENV_ONLY" = true ]; then
    echo "  1. Run: ./setup.sh"
    echo "  2. The setup script will recreate the virtual environment"
elif [ "$PYTHON_CLEANUP" = true ]; then
    echo "  1. Run: ./setup.sh"
    echo "  2. Python 3.13 will be reinstalled with all packages"
elif [ "$CUDA_CLEANUP" = true ]; then
    echo "  1. Reboot the system: sudo reboot"
    echo "  2. Run: ./setup.sh"
    echo "  3. CUDA and drivers will be reinstalled"
elif [ "$FULL_CLEANUP" = true ]; then
    echo "  1. Reboot the system: sudo reboot"
    echo "  2. Run: ./setup.sh"
    echo "  3. Complete environment will be reinstalled"
fi

echo ""
log "Cleanup script finished"