# CircleCI Scripts

This directory contains scripts for the CircleCI environment setup job.

## Scripts

### setup_environment.sh
Main environment setup script that implements requirements 4.1, 4.3, 4.4, 4.7:
- **4.1**: Uses standard Docker executor (cimg/python:3.13.6)
- **4.3**: Installs system dependencies (TA-Lib C library, build tools)
- **4.4**: Sets up Python 3.13.6 with proper virtual environment
- **4.7**: Provides fail-fast behavior with diagnostic information

### validate_environment_setup.py
Comprehensive validation script that verifies:
- Docker executor environment
- System dependencies installation
- Python version and virtual environment
- Critical package imports
- Functionality testing

## Usage

These scripts are called from the CircleCI config.yml setup job:

```yaml
- run:
    name: "Environment Setup"
    command: ./.circleci/scripts/setup_environment.sh

- run:
    name: "Environment Validation"
    command: |
      source .venv/bin/activate
      python ./.circleci/scripts/validate_environment_setup.py
```

## Features

- **Comprehensive error handling**: Fail-fast behavior with detailed diagnostics
- **Caching integration**: Works with CircleCI's caching system
- **Resource monitoring**: Checks memory and disk space
- **Dependency validation**: Verifies all critical packages
- **Functionality testing**: Tests actual library functionality