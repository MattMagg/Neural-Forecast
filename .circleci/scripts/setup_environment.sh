#!/bin/bash

# CircleCI Environment Setup Script
# Implements Docker-based environment setup with Python 3.13.6
# Requirements: 4.1, 4.3, 4.4, 4.7

set -e  # Exit on any error for fail-fast behavior (Requirement 4.7)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler for fail-fast behavior (Requirement 4.7)
error_handler() {
    local line_number=$1
    log_error "Environment setup failed at line $line_number"
    log_error "Diagnostic information:"
    log_error "- Current directory: $(pwd)"
    log_error "- Available memory: $(free -h | grep '^Mem:' | awk '{print $7}' || echo 'N/A')"
    log_error "- Available disk space: $(df -h . | tail -1 | awk '{print $4}' || echo 'N/A')"
    log_error "- Python version: $(python --version 2>&1 || echo 'Python not available')"
    exit 1
}

trap 'error_handler $LINENO' ERR

main() {
    log_info "=== CircleCI Environment Setup ==="
    log_info "Implementing requirements 4.1, 4.3, 4.4, 4.7"
    
    # Requirement 4.1: Verify Docker executor environment
    verify_docker_environment
    
    # Requirement 4.3: Install system dependencies
    install_system_dependencies
    
    # Requirement 4.4: Setup Python 3.13.6 with virtual environment
    setup_python_environment
    
    # Install and validate Python dependencies
    install_python_dependencies
    
    # Comprehensive environment validation
    validate_environment
    
    log_success "Environment setup completed successfully"
    log_success "All requirements (4.1, 4.3, 4.4, 4.7) satisfied"
}

verify_docker_environment() {
    log_info "Verifying Docker executor environment (Requirement 4.1)"
    
    # Check if running in Docker
    if [ -f /.dockerenv ]; then
        log_success "Running in Docker container"
    else
        log_warning "Docker environment not detected (may be running locally)"
    fi
    
    # Verify we have standard Docker executor capabilities
    if command -v apt-get >/dev/null 2>&1; then
        log_success "Standard Docker executor with apt package manager available"
    else
        log_error "Standard Docker executor requirements not met"
        exit 1
    fi
    
    # Check resource availability
    local available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    local available_disk=$(df . | awk 'NR==2{print $4}')
    
    log_info "Resource status:"
    log_info "- Available memory: ${available_mem}MB"
    log_info "- Available disk space: $(df -h . | awk 'NR==2{print $4}')"
    
    if [ "$available_mem" -lt 500 ]; then
        log_warning "Low available memory: ${available_mem}MB"
    fi
    
    if [ "$available_disk" -lt 1000000 ]; then  # Less than 1GB in KB
        log_warning "Low disk space: $(df -h . | awk 'NR==2{print $4}')"
    fi
}

install_system_dependencies() {
    log_info "Installing system dependencies (Requirement 4.3)"
    
    # Update package lists
    log_info "Updating package lists..."
    sudo apt-get update -qq
    
    # Install build essentials and TA-Lib dependencies
    log_info "Installing build tools and TA-Lib dependencies..."
    sudo apt-get install -y \
        build-essential \
        wget \
        pkg-config \
        gcc \
        g++ \
        make \
        curl \
        git || handle_error "Failed to install system dependencies"
    
    # Verify build tools installation
    for tool in gcc g++ make pkg-config wget curl git; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "$tool installed successfully"
        else
            log_error "$tool installation failed"
            exit 1
        fi
    done
    
    # Install TA-Lib from source if not available
    install_talib_library
}

install_talib_library() {
    log_info "Installing TA-Lib C library..."
    
    # Check if TA-Lib is already available
    if ldconfig -p | grep -q ta_lib && pkg-config --exists ta-lib; then
        log_success "TA-Lib already available from cache"
        return 0
    fi
    
    log_info "Installing TA-Lib from source..."
    
    # Download and compile TA-Lib
    local talib_version="0.4.0"
    local talib_url="http://prdownloads.sourceforge.net/ta-lib/ta-lib-${talib_version}-src.tar.gz"
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    # Download with retry logic
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if wget -q "$talib_url"; then
            break
        else
            retry_count=$((retry_count + 1))
            log_warning "Download attempt $retry_count failed, retrying..."
            sleep 2
        fi
    done
    
    if [ $retry_count -eq $max_retries ]; then
        log_error "Failed to download TA-Lib after $max_retries attempts"
        exit 1
    fi
    
    # Extract and compile
    tar -xzf "ta-lib-${talib_version}-src.tar.gz"
    cd ta-lib/
    
    # Configure with proper prefix
    ./configure --prefix=/usr/local
    
    # Compile with parallel jobs
    make -j$(nproc)
    
    # Install
    sudo make install
    
    # Update library cache
    sudo ldconfig
    
    # Cleanup
    cd /
    rm -rf "$temp_dir"
    
    # Verify installation
    if ldconfig -p | grep -q ta_lib; then
        log_success "TA-Lib C library installed successfully"
    else
        log_error "TA-Lib installation verification failed"
        exit 1
    fi
}

setup_python_environment() {
    log_info "Setting up Python environment (Requirement 4.4)"
    
    # Verify Python version
    local python_version=$(python --version 2>&1)
    log_info "Detected Python version: $python_version"
    
    if ! echo "$python_version" | grep -q "Python 3.13"; then
        log_error "Python 3.13.x required but found: $python_version"
        log_error "Environment setup failed - Python version requirement not met"
        exit 1
    fi
    log_success "Python version requirement satisfied"
    
    # Create virtual environment if not cached
    if [ ! -d ".venv" ]; then
        log_info "Creating new virtual environment..."
        python -m venv .venv || {
            log_error "Failed to create virtual environment"
            log_error "Diagnostic: $(python -m venv --help | head -3)"
            exit 1
        }
        log_success "Virtual environment created"
    else
        log_info "Using cached virtual environment"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate || {
        log_error "Failed to activate virtual environment"
        log_error "Virtual environment may be corrupted"
        rm -rf .venv
        exit 1
    }
    
    # Verify virtual environment activation
    local venv_python=$(which python)
    if [[ "$venv_python" != *".venv"* ]]; then
        log_error "Virtual environment activation failed"
        log_error "Expected path containing '.venv', got: $venv_python"
        exit 1
    fi
    log_success "Virtual environment activated: $venv_python"
    
    # Upgrade pip, setuptools, and wheel
    log_info "Upgrading pip, setuptools, and wheel..."
    python -m pip install --upgrade pip setuptools wheel || {
        log_error "Failed to upgrade pip/setuptools/wheel"
        log_error "Diagnostic: $(pip --version 2>&1)"
        exit 1
    }
    log_success "Python package managers upgraded"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    source .venv/bin/activate
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found"
        exit 1
    fi
    
    # Install dependencies with comprehensive error handling
    if ! pip install --no-cache-dir -r requirements.txt; then
        log_warning "Initial dependency installation failed - attempting recovery"
        
        # Diagnostic information
        log_info "Diagnostic information:"
        log_info "- Pip version: $(pip --version)"
        log_info "- Available disk space: $(df -h . | tail -1)"
        log_info "- Memory usage: $(free -h | grep '^Mem:')"
        
        # Clear cache and retry
        log_info "Clearing cache and retrying..."
        rm -rf .venv ~/.cache/pip
        python -m venv .venv
        source .venv/bin/activate
        python -m pip install --upgrade pip setuptools wheel
        
        if ! pip install --no-cache-dir -r requirements.txt; then
            log_error "Dependency installation failed after retry"
            log_error "Requirements file contents:"
            head -10 requirements.txt
            exit 1
        fi
    fi
    log_success "Dependencies installed successfully"
}

validate_environment() {
    log_info "Performing comprehensive environment validation..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Verify critical imports with detailed diagnostics
    log_info "Verifying critical dependencies..."
    
    python << 'EOF'
import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'Not detected')}")
print()

# Critical package verification
packages = [
    ('talib', 'TA-Lib'),
    ('neuralforecast', 'NeuralForecast'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('pyarrow', 'PyArrow'),
    ('torch', 'PyTorch'),
    ('vectorbt', 'VectorBT'),
    ('sklearn', 'Scikit-learn')
]

failed = []
for pkg, name in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
    except ImportError as e:
        print(f'✗ {name}: Import failed - {e}')
        failed.append(name)
    except Exception as e:
        print(f'✗ {name}: Unexpected error - {e}')
        failed.append(name)

if failed:
    print(f'\n✗ CRITICAL ERROR: Failed to import {len(failed)} packages: {failed}')
    print('Environment setup validation failed')
    sys.exit(1)
else:
    print(f'\n✓ All {len(packages)} critical packages validated successfully')
EOF
    
    # Test critical functionality
    log_info "Testing critical functionality..."
    
    python << 'EOF'
import numpy as np
import talib
import pandas as pd

# Test TA-Lib functionality
test_data = np.random.random(100)
sma = talib.SMA(test_data, timeperiod=10)
assert not np.isnan(sma[-1]), 'TA-Lib SMA calculation failed'
print('✓ TA-Lib functionality verified')

# Test NeuralForecast import
from neuralforecast import NeuralForecast
print('✓ NeuralForecast core functionality verified')

# Test data processing
df = pd.DataFrame({'test': [1, 2, 3]})
assert len(df) == 3, 'Pandas functionality failed'
print('✓ Data processing libraries verified')

print('\n✅ Environment functionality validation completed')
EOF
    
    # Final system checks
    log_info "Final system validation..."
    
    # TA-Lib system library check
    if ldconfig -p | grep -q ta_lib; then
        log_success "TA-Lib C library available in system library path"
    else
        log_error "TA-Lib C library not found in system library path"
        exit 1
    fi
    
    # Virtual environment check
    if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
        log_success "Virtual environment properly activated: $VIRTUAL_ENV"
    else
        log_error "Virtual environment not properly activated"
        exit 1
    fi
    
    log_success "Environment validation completed successfully"
}

# Execute main function
main "$@"