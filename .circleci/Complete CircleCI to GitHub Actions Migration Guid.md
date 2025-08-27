# Complete CircleCI to GitHub Actions Migration Guide

Based on your comprehensive CircleCI setup with GPU support, multi-level caching, and validation scripts, here's an exhaustive migration strategy that addresses every aspect of your current implementation.

## Comprehensive Migration Checklist

- **Audit and assess current CircleCI implementation** - Use GitHub Actions Importer to analyze your 2200+ line configuration, validation scripts, and identify manual migration tasks
- **Configure automated migration toolchain** - Set up GitHub Actions Importer CLI with proper credentials and run audit/dry-run/migrate workflow
- **Implement comprehensive workflow structure** - Create `.github/workflows/ci.yml` with exact job mapping including environment setup, data processing, feature engineering, model factory, and cross-validation
- **Port advanced caching system** - Migrate your multi-level caching (Python, TA-Lib, processed data, model artifacts) using `actions/cache` with proper key strategies and fallbacks
- **Configure GPU compute infrastructure** - Set up self-hosted runners with CUDA 12 support or use GitHub's GPU runners for your model factory and cross-validation jobs
- **Implement matrix builds and job orchestration** - Use `strategy.matrix` for your 4-parameter system and `needs` for complex job dependencies
- **Migrate all validation scripts** - Port your 5 validation scripts with minimal modification while ensuring environment variable and secret management alignment
- **Comprehensive testing and validation** - Run complete workflow suite and verify outputs match CircleCI execution with proper artifact handling


## Phase 1: Migration Assessment and Planning

### Step 1: Install and Configure GitHub Actions Importer

```bash
# Install GitHub Actions Importer CLI extension
gh extension install github/gh-actions-importer

# Configure credentials
gh actions-importer configure
```

Your configuration will need:

- GitHub Personal Access Token with `workflow` and `repo` scopes
- CircleCI Personal API Token
- CircleCI organization name
- GitHub repository URL


### Step 2: Audit Current CircleCI Implementation

```bash
# Perform comprehensive audit of your CircleCI setup
gh actions-importer audit circle-ci --output-dir tmp/audit

# Generate forecast for resource planning
gh actions-importer forecast circle-ci --output-dir tmp/forecast_reports
```

The audit will identify :[^1]

- **Manual tasks**: Contexts, project environment variables, unknown orbs
- **Conversion success rate**: Automatic vs manual migration requirements
- **Resource requirements**: Compute usage patterns for GitHub Actions planning


### Step 3: Dry-Run Migration

```bash
# Convert your CircleCI config without creating PR
gh actions-importer dry-run circle-ci --output-dir tmp/dry-run --circle-ci-project your-project-name
```

This provides a preview of your converted workflows and identifies remaining manual tasks.[^1]

## Phase 2: Comprehensive Workflow Structure Implementation

### Complete GitHub Actions Workflow Template

Based on your CircleCI implementation, here's the comprehensive workflow structure:

```yaml
name: BTC Forecasting CI/CD Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:
    inputs:
      test_type:
        description: 'Test type to run'
        required: false
        default: 'foundation'
        type: choice
        options:
        - foundation
        - gpu
        - full

env:
  PYTHON_VERSION: "3.13"
  CACHE_VERSION: "v1"

jobs:
  # Environment Setup Job - Direct mapping from CircleCI
  environment-setup:
    runs-on: ubuntu-latest
    outputs:
      cache-key: ${{ steps.cache-key.outputs.value }}
      python-cache-key: ${{ steps.python-cache-key.outputs.value }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Generate cache keys
        id: cache-key
        run: |
          echo "value=env-${{ env.CACHE_VERSION }}-${{ runner.os }}-${{ hashFiles('requirements.txt', 'pyproject.toml') }}" >> $GITHUB_OUTPUT

      - name: Generate Python cache key
        id: python-cache-key
        run: |
          echo "value=python-${{ env.PYTHON_VERSION }}-${{ runner.os }}-${{ hashFiles('requirements.txt') }}" >> $GITHUB_OUTPUT

      - name: Cache Python dependencies
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/pip
            ~/.local/lib/python${{ env.PYTHON_VERSION }}/site-packages
          key: ${{ steps.python-cache-key.outputs.value }}
          restore-keys: |
            python-${{ env.PYTHON_VERSION }}-${{ runner.os }}-

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install system dependencies (TA-Lib)
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential wget
          cd /tmp
          wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
          tar -xzf ta-lib-0.4.0-src.tar.gz
          cd ta-lib/
          ./configure --prefix=/usr
          make
          sudo make install

      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Validate caching system
        run: python .github/scripts/validate_caching_system.py

  # Data Processing Validation Job
  data-processing-validation:
    needs: environment-setup
    runs-on: ubuntu-latest
    strategy:
      matrix:
        cache-strategy: [full, incremental]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Restore Python environment
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/pip
            ~/.local/lib/python${{ env.PYTHON_VERSION }}/site-packages
          key: ${{ needs.environment-setup.outputs.python-cache-key }}
          restore-keys: |
            python-${{ env.PYTHON_VERSION }}-${{ runner.os }}-

      - name: Cache processed data
        uses: actions/cache@v4
        with:
          path: data/processed
          key: processed-data-${{ matrix.cache-strategy }}-${{ hashFiles('data/raw/**/*') }}
          restore-keys: |
            processed-data-${{ matrix.cache-strategy }}-

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Run data processing validation
        run: python .github/scripts/validate_data_processing.py
        env:
          CACHE_STRATEGY: ${{ matrix.cache-strategy }}

      - name: Upload validation artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: data-validation-${{ matrix.cache-strategy }}
          path: |
            logs/
            reports/data_validation/

  # Feature Engineering Validation Job
  feature-engineering-validation:
    needs: [environment-setup, data-processing-validation]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Cache feature artifacts
        uses: actions/cache@v4
        with:
          path: |
            features/cache/
            artifacts/features/
          key: features-${{ matrix.python-version }}-${{ hashFiles('features/**/*') }}
          restore-keys: |
            features-${{ matrix.python-version }}-

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Download data validation artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: data-validation-*
          merge-multiple: true

      - name: Run feature engineering validation
        run: python .github/scripts/validate_feature_engineering.py
        env:
          PYTHON_VERSION: ${{ matrix.python-version }}

      - name: Upload feature validation results
        uses: actions/upload-artifact@v4
        with:
          name: feature-validation-${{ matrix.python-version }}
          path: |
            artifacts/features/
            reports/feature_validation/

  # Model Factory Validation Job (GPU Required)
  model-factory-validation:
    needs: [environment-setup, feature-engineering-validation]
    runs-on: [self-hosted, gpu, cuda-12]  # Self-hosted GPU runner
    strategy:
      matrix:
        model: [NHITS, NBEATSx, TiDE, PatchTST]
        loss: [DistributionLoss, MQLoss, IQLoss]
      fail-fast: false
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Verify GPU availability
        run: |
          nvidia-smi
          python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

      - name: Cache model artifacts
        uses: actions/cache@v4
        with:
          path: |
            models/cache/
            artifacts/models/
          key: models-${{ matrix.model }}-${{ matrix.loss }}-${{ hashFiles('models/**/*') }}
          restore-keys: |
            models-${{ matrix.model }}-${{ matrix.loss }}-
            models-${{ matrix.model }}-

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install GPU dependencies
        run: |
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
          pip install -r requirements-gpu.txt

      - name: Download feature validation artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: feature-validation-*
          merge-multiple: true

      - name: Run model factory validation
        run: python .github/scripts/validate_model_factory.py
        env:
          MODEL_TYPE: ${{ matrix.model }}
          LOSS_FUNCTION: ${{ matrix.loss }}
          CUDA_VISIBLE_DEVICES: "0"
        timeout-minutes: 30

      - name: Upload model artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: model-artifacts-${{ matrix.model }}-${{ matrix.loss }}
          path: |
            artifacts/models/
            reports/model_validation/

  # Cross-Validation Job (GPU Required)
  cross-validation:
    needs: [environment-setup, model-factory-validation]
    runs-on: [self-hosted, gpu, cuda-12]
    strategy:
      matrix:
        cv-fold: [1, 2, 3, 4, 5]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Cache cross-validation results
        uses: actions/cache@v4
        with:
          path: |
            cv/cache/
            artifacts/cv/
          key: cv-fold-${{ matrix.cv-fold }}-${{ hashFiles('cv/**/*') }}
          restore-keys: |
            cv-fold-${{ matrix.cv-fold }}-

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Download model artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: model-artifacts-*
          merge-multiple: true

      - name: Run cross-validation
        run: python .github/scripts/validate_cross_validation.py
        env:
          CV_FOLD: ${{ matrix.cv-fold }}
          CUDA_VISIBLE_DEVICES: "0"
        timeout-minutes: 60

      - name: Upload CV results
        uses: actions/upload-artifact@v4
        with:
          name: cv-results-fold-${{ matrix.cv-fold }}
          path: |
            artifacts/cv/
            reports/cv_validation/

  # Foundation Tests (CPU Only)
  foundation-tests:
    if: ${{ github.event.inputs.test_type == 'foundation' || github.event.inputs.test_type == 'full' || github.event_name != 'workflow_dispatch' }}
    needs: environment-setup
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-suite: [unit, integration, validation]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ env.PYTHON_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Run foundation tests
        run: |
          python -m pytest tests/${{ matrix.test-suite }}/ -v --tb=short
        env:
          TEST_SUITE: ${{ matrix.test-suite }}

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: foundation-test-results-${{ matrix.test-suite }}
          path: |
            test-results/
            coverage-reports/

  # Final validation and reporting
  validation-summary:
    needs: [
      data-processing-validation,
      feature-engineering-validation,
      model-factory-validation,
      cross-validation,
      foundation-tests
    ]
    runs-on: ubuntu-latest
    if: always()
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Download all artifacts
        uses: actions/download-artifact@v4

      - name: Generate validation summary
        run: python .github/scripts/generate_validation_summary.py

      - name: Upload final report
        uses: actions/upload-artifact@v4
        with:
          name: validation-summary-report
          path: |
            reports/final_summary/
            artifacts/combined/
```


## Phase 3: Advanced Caching Strategy Migration

### Multi-Level Caching Implementation

Your comprehensive caching system translates to GitHub Actions using these patterns :[^2][^3]

```yaml
# Python Dependencies Cache
- name: Cache Python dependencies
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/.local/lib/python${{ env.PYTHON_VERSION }}/site-packages
    key: python-${{ env.PYTHON_VERSION }}-${{ runner.os }}-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      python-${{ env.PYTHON_VERSION }}-${{ runner.os }}-

# TA-Lib System Dependencies Cache
- name: Cache TA-Lib installation
  uses: actions/cache@v4
  with:
    path: /usr/local/lib/ta-lib
    key: ta-lib-${{ runner.os }}-v1
    restore-keys: |
      ta-lib-${{ runner.os }}-

# Processed Data Cache with Fallback Strategy
- name: Cache processed data
  uses: actions/cache@v4
  with:
    path: |
      data/processed
      data/intermediate
    key: processed-data-${{ hashFiles('data/raw/**/*') }}-${{ hashFiles('src/data_processing/**/*') }}
    restore-keys: |
      processed-data-${{ hashFiles('data/raw/**/*') }}-
      processed-data-

# Model Artifacts Cache
- name: Cache model artifacts
  uses: actions/cache@v4
  with:
    path: |
      models/cache
      artifacts/models
    key: models-${{ matrix.model }}-${{ matrix.loss }}-${{ hashFiles('models/**/*') }}
    restore-keys: |
      models-${{ matrix.model }}-${{ matrix.loss }}-
      models-${{ matrix.model }}-
      models-

# Cross-Job Cache Sharing
- name: Cache cross-validation results
  uses: actions/cache@v4
  with:
    path: cv/cache
    key: cv-${{ needs.build.outputs.cache-key }}-${{ matrix.cv-fold }}
    restore-keys: |
      cv-${{ needs.build.outputs.cache-key }}-
```


### Cache Performance Optimization

Implement cache cleanup and monitoring :[^4]

```yaml
- name: Cache cleanup and monitoring
  if: always()
  run: |
    # Monitor cache hit rates
    echo "Cache statistics:"
    gh extension install actions/gh-actions-cache
    gh actions-cache list --sort created-at --limit 20
    
    # Cleanup old caches (keep last 5)
    CACHE_KEYS=$(gh actions-cache list --key "processed-data-" --sort created-at --order desc --limit 100 | cut -f 1 | tail -n +6)
    for key in $CACHE_KEYS; do
      gh actions-cache delete $key --confirm
    done
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```


## Phase 4: GPU Infrastructure Setup

### Self-Hosted GPU Runners

For your CUDA 12 and GPU requirements, set up self-hosted runners :[^5][^6]

```dockerfile
# Self-hosted GPU runner Dockerfile
FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNER_ALLOW_RUNASROOT=1
ENV AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache

# Install GitHub Actions runner
ARG GH_RUNNER_VERSION="2.317.0"
ARG TARGETPLATFORM

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    git \
    unzip \
    sudo \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib system dependencies
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install

# Set up GitHub Actions runner
WORKDIR /actions-runner
RUN curl -o actions-runner-linux-x64-${GH_RUNNER_VERSION}.tar.gz \
    -L https://github.com/actions/runner/releases/download/v${GH_RUNNER_VERSION}/actions-runner-linux-x64-${GH_RUNNER_VERSION}.tar.gz && \
    tar xzf ./actions-runner-linux-x64-${GH_RUNNER_VERSION}.tar.gz

# Runner configuration script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
```


### Runner Registration Script

```bash
#!/bin/bash
# entrypoint.sh for self-hosted GPU runner

# Configure runner with labels
./config.sh \
    --url https://github.com/${GITHUB_REPOSITORY} \
    --token ${RUNNER_TOKEN} \
    --labels "self-hosted,gpu,cuda-12,linux,x64" \
    --name "gpu-runner-$(hostname)" \
    --work _work \
    --replace

# Start runner
./run.sh
```


### Alternative: Third-Party GPU Runners

If you prefer managed GPU runners, consider RunsOn :[^7]

```yaml
jobs:
  model-factory-validation:
    runs-on: runs-on,runner=8cpu-32gb-gpu-t4,run-id=${{ github.run_id }}
    # 85% cheaper than GitHub's GPU runners with same T4 GPU
```


## Phase 5: Validation Scripts Migration

### Script Migration Strategy

Your existing validation scripts require minimal changes :[^8]

```yaml
# Port existing CircleCI validation scripts
- name: Copy validation scripts
  run: |
    mkdir -p .github/scripts
    cp .circleci/scripts/validate_*.py .github/scripts/
```


### Environment Variable Mapping

Map CircleCI environment variables to GitHub Actions equivalents :[^1]


| CircleCI Variable | GitHub Actions Equivalent |
| :-- | :-- |
| `CIRCLE_BRANCH` | `${{ github.ref_name }}` |
| `CIRCLE_SHA1` | `${{ github.sha }}` |
| `CIRCLE_JOB` | `${{ github.job }}` |
| `CIRCLE_WORKFLOW_ID` | `${{ github.run_number }}` |
| `CIRCLE_WORKING_DIRECTORY` | `${{ github.workspace }}` |

### Secrets Management

Migrate your CircleCI contexts and secrets :[^9][^10]

```yaml
# In your workflow
env:
  API_KEY: ${{ secrets.API_KEY }}
  DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
  MODEL_REGISTRY_TOKEN: ${{ secrets.MODEL_REGISTRY_TOKEN }}
```

Create repository secrets via GitHub CLI:

```bash
gh secret set API_KEY --body "your-api-key"
gh secret set DB_PASSWORD --body "your-db-password"
```


## Phase 6: Advanced Job Orchestration

### Complex Dependencies with Conditions

Implement your complex job dependencies using `needs` and conditional execution :[^11][^12]

```yaml
jobs:
  model-validation:
    needs: [data-processing, feature-engineering]
    if: |
      always() && 
      (needs.data-processing.result == 'success' || needs.data-processing.result == 'skipped') &&
      needs.feature-engineering.result == 'success'
    runs-on: [self-hosted, gpu]
    
  deployment:
    needs: [model-validation, cross-validation]
    if: |
      github.ref == 'refs/heads/main' &&
      needs.model-validation.result == 'success' &&
      needs.cross-validation.result == 'success'
    runs-on: ubuntu-latest
```


### Matrix Build Optimization

Optimize your 4-parameter system using dynamic matrices :[^13][^14]

```yaml
strategy:
  matrix:
    include:
      - test-type: foundation
        runner: ubuntu-latest
        python-version: "3.13"
        cache-strategy: full
      - test-type: gpu
        runner: [self-hosted, gpu]
        python-version: "3.13"
        cache-strategy: incremental
      - test-type: validation
        runner: ubuntu-latest
        python-version: ["3.11", "3.12", "3.13"]
        cache-strategy: [full, incremental]
  fail-fast: false
  max-parallel: 4
```


## Phase 7: Migration Execution

### Step-by-Step Migration Process

1. **Run automated migration**:

```bash
gh actions-importer migrate circle-ci \
  --target-url https://github.com/your-org/your-repo \
  --output-dir tmp/migrate \
  --circle-ci-project your-project-name
```

2. **Review generated PR**: GitHub Actions Importer creates a PR with:
    - Converted workflow files
    - Manual steps documentation
    - Required secrets list
    - Self-hosted runner requirements
3. **Manual post-migration tasks**:
    - Create repository secrets for contexts and environment variables
    - Set up self-hosted GPU runners
    - Configure organization-level secrets if needed
    - Update any unknown orbs with equivalent actions

### Validation and Testing

4. **Test workflow execution**:

```yaml
# Add workflow testing job
workflow-test:
  runs-on: ubuntu-latest
  steps:
    - name: Validate workflow syntax
      run: |
        yamllint .github/workflows/ci.yml
        actionlint .github/workflows/ci.yml
```

5. **Parallel execution during transition**:
    - Run both CircleCI and GitHub Actions in parallel
    - Compare artifacts and outputs
    - Gradually migrate confidence as validation passes

## Phase 8: Performance Optimization and Monitoring

### Workflow Performance Tuning

```yaml
# Optimize job concurrency
jobs:
  parallel-validation:
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
      max-parallel: 4
    steps:
      - name: Run validation shard ${{ matrix.shard }}
        run: python validate.py --shard ${{ matrix.shard }}/4
```


### Monitoring and Alerting

```yaml
# Add workflow monitoring
- name: Workflow monitoring
  if: failure()
  uses: actions/github-script@v7
  with:
    script: |
      github.rest.issues.create({
        owner: context.repo.owner,
        repo: context.repo.repo,
        title: 'CI/CD Pipeline Failure',
        body: `Workflow failed: ${context.workflow}\nRun: ${context.runNumber}\nCommit: ${context.sha}`
      })
```


## Migration Advantages and Expected Outcomes

### Performance Improvements

- **85% cost savings** on GPU compute using self-hosted or third-party runners vs CircleCI[^7]
- **Faster artifact sharing** between jobs using GitHub's native artifact system[^15][^16]
- **Improved caching efficiency** with GitHub Actions' multi-level cache strategy[^2]


### Operational Benefits

- **Native GitHub integration** eliminates webhook and permission management overhead
- **Unified secrets management** through GitHub's encrypted secrets system[^9]
- **Enhanced security** with OIDC authentication and environment protection rules[^17]


### Development Workflow Enhancements

- **Improved debugging** with enhanced logging and artifact inspection[^15]
- **Better matrix builds** with GitHub's flexible strategy options[^13]
- **Conditional execution** provides more granular workflow control[^12]


## Post-Migration Cleanup and Optimization

### CircleCI Deprecation Strategy

1. **Parallel execution phase**: Run both systems for 2-4 weeks
2. **Validation phase**: Compare outputs and performance metrics
3. **Gradual migration**: Move non-critical workflows first
4. **Full cutover**: Disable CircleCI workflows after validation
5. **Cleanup**: Remove `.circleci/` directory and update documentation

### Continuous Improvement

- **Monitor workflow performance** using GitHub's insights and analytics
- **Optimize cache hit rates** and reduce build times
- **Review and update** runner sizing based on actual usage patterns
- **Implement workflow templates** for consistency across projects

This comprehensive migration strategy ensures feature parity while leveraging GitHub Actions' native capabilities for improved performance, cost efficiency, and developer experience. The automated migration tools combined with detailed manual steps provide a complete path from your current CircleCI implementation to a fully optimized GitHub Actions workflow.
