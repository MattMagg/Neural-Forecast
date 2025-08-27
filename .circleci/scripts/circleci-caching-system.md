# CircleCI Dependency Management and Caching System

## Overview

This document describes the comprehensive dependency management and caching system implemented for the CircleCI foundation setup. The system provides efficient caching for Python dependencies, system libraries, processed data, and model artifacts with robust fallback strategies and validation mechanisms.

## Implementation Summary

### Task 2 Requirements Fulfilled

✅ **3.1**: Python dependency caching with requirements.txt checksum  
✅ **3.2**: TA-Lib compilation caching for system libraries  
✅ **3.3**: Processed data caching for pipeline efficiency  
✅ **3.4**: Model artifacts persistence as build artifacts  
✅ **3.5**: Automatic cache key generation on changes  
✅ **3.6**: Graceful fallback when cache restoration fails  
✅ **3.7**: Correct cache invalidation when dependencies update  

## Cache Types and Strategies

### 1. Python Dependencies Cache

**Purpose**: Cache virtual environment and pip packages to avoid reinstallation  
**Cache Key**: `v2-python-deps-{{ checksum "requirements.txt" }}-{{ checksum "dependencies.yaml" }}-{{ arch }}`

**Fallback Strategy**:
1. Primary: Both requirements.txt and dependencies.yaml checksums + architecture
2. Fallback 1: Only requirements.txt checksum + architecture  
3. Fallback 2: Architecture-specific partial cache
4. Fallback 3: Any previous cache

**Cached Paths**:
- `~/.cache/pip` - Pip package cache
- `.venv` - Virtual environment
- `/home/circleci/.local/lib/python*/site-packages` - User site packages

### 2. TA-Lib Compilation Cache

**Purpose**: Cache compiled TA-Lib C library to avoid recompilation  
**Cache Key**: `v2-talib-{{ arch }}-{{ checksum "requirements.txt" }}-0.4.0`

**Fallback Strategy**:
1. Primary: Architecture + requirements checksum + TA-Lib version
2. Fallback 1: Architecture + TA-Lib version
3. Fallback 2: Architecture only

**Cached Paths**:
- `/usr/local/lib/libta_lib*` - TA-Lib shared libraries
- `/usr/local/include/ta_*.h` - TA-Lib header files
- `/usr/lib/libta_lib*` - Alternative library location

### 3. Processed Data Cache

**Purpose**: Cache processed datasets to avoid reprocessing raw data  
**Cache Key**: `v2-data-{{ checksum "data/raw/btcusd_1-min_data.csv" }}-{{ checksum "utils/io.py" }}`

**Fallback Strategy**:
1. Primary: Raw data checksum + processing code checksum
2. Fallback 1: Raw data checksum only
3. Fallback 2: Any processed data

**Cached Paths**:
- `data/processed` - All processed data files

### 4. Model Artifacts Cache

**Purpose**: Cache trained models and experiment results  
**Cache Key**: `v2-models-{{ .Branch }}-{{ .Revision }}`

**Fallback Strategy**:
1. Primary: Branch + revision specific
2. Fallback 1: Branch specific
3. Fallback 2: Any model artifacts

**Cached Paths**:
- `experiments/*/best` - Best model artifacts per horizon
- `*.pkl` - Pickle files
- `experiments/*/cv_results*.parquet` - Cross-validation results
- `experiments/*/metrics*.json` - Metrics files

### 5. System Packages Cache (Aggressive Mode Only)

**Purpose**: Cache apt packages to speed up system dependency installation  
**Cache Key**: `v2-system-{{ arch }}-{{ epoch }}`

**Cached Paths**:
- `/var/cache/apt/archives` - APT package cache

### 6. Test Results Cache (Aggressive Mode Only)

**Purpose**: Cache test results and coverage reports  
**Cache Key**: `v2-tests-{{ .Branch }}-{{ checksum "tests/**/*.py" }}`

**Cached Paths**:
- `test-results` - Test result files
- `coverage` - Coverage reports
- `.pytest_cache` - Pytest cache

## Cache Strategy Modes

### Disabled Mode
- **Parameter**: `cache_strategy: "disabled"`
- **Behavior**: No caching enabled, all dependencies installed fresh
- **Use Case**: Debugging cache issues, ensuring clean builds

### Conservative Mode  
- **Parameter**: `cache_strategy: "conservative"`
- **Behavior**: Core caches only (Python deps, TA-Lib, processed data, models)
- **Use Case**: Balanced performance with storage efficiency

### Aggressive Mode
- **Parameter**: `cache_strategy: "aggressive"`  
- **Behavior**: All caches enabled including system packages and test results
- **Use Case**: Maximum performance, fastest builds

## Cache Management Commands

### Restoration Commands
- `restore_python_dependencies` - Restore Python virtual environment and packages
- `restore_talib_cache` - Restore compiled TA-Lib libraries
- `restore_processed_data_cache` - Restore processed datasets
- `restore_model_artifacts_cache` - Restore model artifacts
- `restore_system_cache` - Restore system package cache (aggressive mode)
- `restore_test_cache` - Restore test results cache (aggressive mode)

### Saving Commands
- `save_python_dependencies` - Save Python environment to cache
- `save_talib_cache` - Save compiled TA-Lib to cache
- `save_processed_data_cache` - Save processed data to cache
- `save_model_artifacts_cache` - Save model artifacts to cache
- `save_system_cache` - Save system packages to cache
- `save_test_cache` - Save test results to cache

### Management Commands
- `cleanup_failed_caches` - Clean up corrupted or incomplete caches
- `validate_all_caches` - Comprehensive cache validation
- `generate_cache_report` - Generate cache performance report
- `monitor_cache_performance` - Monitor cache hit rates and metrics
- `test_caching_system` - Test caching system functionality

## Cache Validation and Error Handling

### Validation Mechanisms

1. **Virtual Environment Validation**
   - Check if `.venv` directory exists and is functional
   - Validate Python interpreter activation
   - Clean up corrupted environments automatically

2. **TA-Lib Validation**
   - Test TA-Lib import functionality
   - Check library files in system paths
   - Rebuild if import fails despite cached files

3. **Processed Data Validation**
   - Verify parquet files exist and are readable
   - Check data integrity with pandas
   - Remove corrupted cache automatically

4. **Model Artifacts Validation**
   - Check for expected model directory structure
   - Validate artifact file integrity

### Error Recovery

1. **Cache Corruption Detection**
   - Automatic detection of corrupted caches
   - Graceful fallback to fresh installation
   - Detailed error reporting

2. **Fallback Strategies**
   - Multi-level cache key fallbacks
   - Automatic retry with clean environment
   - Progressive degradation (aggressive → conservative → disabled)

3. **Cache Cleanup**
   - Remove incomplete installations
   - Clean up corrupted virtual environments
   - Reset failed TA-Lib compilations

## Performance Monitoring

### Cache Hit Rate Tracking
- Monitor cache restoration success/failure
- Calculate hit rates per cache type
- Generate performance recommendations

### Time Savings Analysis
- Estimated time savings per cache type:
  - Python dependencies: ~3-5 minutes
  - TA-Lib compilation: ~2-3 minutes  
  - Processed data: ~1-2 minutes
  - System packages: ~1 minute

### Storage Usage Monitoring
- Track cache sizes and growth
- Monitor total storage consumption
- Alert on storage limit approaches

## Integration with Setup Job

The caching system is fully integrated into the `setup` job workflow:

1. **Cache Restoration Phase**
   - Cleanup failed caches
   - Restore system cache (if aggressive)
   - Restore TA-Lib cache
   - Restore Python dependencies
   - Restore processed data cache

2. **Installation Phase**
   - Install system dependencies (with cache validation)
   - Setup Python environment (with cache validation)
   - Verify critical imports

3. **Cache Saving Phase**
   - Save system cache
   - Save TA-Lib cache  
   - Save Python dependencies

4. **Validation and Reporting Phase**
   - Validate environment
   - Validate all caches
   - Monitor cache performance
   - Generate cache report

## Usage Examples

### Basic Usage
The caching system is automatically enabled with the default `aggressive` strategy:

```yaml
parameters:
  cache_strategy: "aggressive"  # Default
```

### Conservative Caching
For storage-constrained environments:

```yaml
parameters:
  cache_strategy: "conservative"
```

### Disable Caching
For debugging or clean builds:

```yaml
parameters:
  cache_strategy: "disabled"
```

## Troubleshooting

### Common Issues

1. **Cache Corruption**
   - **Symptoms**: Import errors, activation failures
   - **Solution**: Automatic cleanup and rebuild
   - **Prevention**: Cache validation before use

2. **Storage Limits**
   - **Symptoms**: Cache save failures
   - **Solution**: Switch to conservative mode
   - **Prevention**: Monitor cache sizes

3. **Dependency Conflicts**
   - **Symptoms**: Package import errors
   - **Solution**: Clear Python cache, fresh install
   - **Prevention**: Proper cache key versioning

### Debug Commands

```bash
# Test caching system
circleci local execute --job setup

# Validate cache configuration  
python scripts/validate_caching_system.py

# Check cache status
# (Integrated into setup job output)
```

## Future Enhancements

1. **Cache Analytics**
   - Historical hit rate tracking
   - Performance trend analysis
   - Storage optimization recommendations

2. **Advanced Fallback**
   - Cross-branch cache sharing
   - Partial cache restoration
   - Smart cache invalidation

3. **Integration Extensions**
   - Model-specific caching strategies
   - Feature engineering cache optimization
   - GPU-specific cache handling

## Conclusion

The implemented caching system provides comprehensive dependency management with robust error handling, performance monitoring, and flexible configuration options. It significantly reduces build times while maintaining reliability through validation and fallback mechanisms.

The system fully satisfies all requirements (3.1-3.7) and provides a solid foundation for the CircleCI pipeline's efficiency and reliability.