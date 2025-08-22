# Experiments Configuration Directory

## Directory Overview
Contains YAML configuration files for each forecast horizon and utility notebooks for config management.

## Files and Purpose

### ✅ **h4.yaml** (ACTIVELY USED)
- **Purpose**: Configuration for h=4 (1 hour) horizon
- **Content**: Model definitions, loss functions, CV parameters
- **Usage**: Directly loaded by run_train.ipynb (default at line 90)

### ✅ **h8.yaml** (ACTIVELY USED)
- **Purpose**: Configuration for h=8 (2 hours) horizon
- **Content**: Model definitions, loss functions, CV parameters
- **Usage**: Loadable via load_experiment_config() function

### ✅ **h16.yaml** (ACTIVELY USED)
- **Purpose**: Configuration for h=16 (4 hours) horizon
- **Content**: Model definitions, loss functions, CV parameters
- **Usage**: Loadable via load_experiment_config() function

### ✅ **h32.yaml** (ACTIVELY USED)
- **Purpose**: Configuration for h=32 (8 hours) horizon
- **Content**: Model definitions, loss functions, CV parameters
- **Usage**: Loadable via load_experiment_config() function

### ❌ **defaults.yaml** (NOT USED AT RUNTIME)
- **Purpose**: Global default settings
- **Status**: Referenced in docs but NOT loaded by run_train.ipynb
- **Note**: Configs are self-contained, no inheritance implemented

### 📝 **experiment_configs.ipynb** (DEVELOPMENT TOOL)
- **Purpose**: Interactive config builder and validator
- **Features**: Widget-based config generation, validation functions
- **Usage**: Development only, not part of training pipeline

### 📝 **small_sample_generation.ipynb** (DEVELOPMENT TOOL)
- **Purpose**: Generate small sample datasets for testing
- **Usage**: Development only, not part of training pipeline

### 📁 **h4/** (RESULTS DIRECTORY)
- **Content**: CV results, metrics, leaderboards from h4 experiments
- **Files**: cv_results_*.parquet, leaderboard.csv, metrics.json

## System Flow
```
run_train.ipynb
    ↓ line 90: CONFIG = "experiments/h4.yaml"
load_experiment_config(CONFIG)
    ↓ validates required keys
    ↓ returns config dict
instantiate_models(cfg)
    ↓ uses model definitions
    ↓ creates NF models
training pipeline execution
```

## Key Insights
- Each horizon has independent, self-contained configuration
- defaults.yaml exists but isn't used (potential for DRY improvement)
- Notebooks are utility tools, not runtime components
- h4/ subdirectory shows results storage pattern