# NeuralForecast BTC Forecasting Implementation Workflow

## Executive Summary

Systematic implementation workflow for an intraday BTC forecasting system using 15-minute bars with calibrated prediction intervals (80/90/95%). The system is **NeuralForecast-centric**, utilizing only native NF capabilities without custom implementations.

**Primary Metric**: sCRPS (Scaled Continuous Ranked Probability Score)  
**Horizons**: 4, 8, 16, 32 steps (1h, 2h, 4h, 8h)

---

## Core Principles & Constraints

### Non-Negotiables
- ✅ **NF-native only**: Use NeuralForecast for all modeling, cross-validation, and uncertainty quantification
- ✅ **Leakage hygiene**: All historic features computed then shifted by one bar (`shift(1)`)
- ✅ **UTC EOB**: All timestamps are UTC end-of-bar on regular 15-min grid
- ✅ **Target**: Log returns `y = log(close_t / close_{t-1})`
- ✅ **Feature cap**: Maximum 256 columns after pruning
- ✅ **Conformal import**: `from neuralforecast.utils import PredictionIntervals`

### Technical Stack
- **Core**: NeuralForecast 3.0.2+ (pinned version)
- **Indicators**: vectorbt, TA-Lib (primary), pandas-ta-openbb, freqtrade/technical (supplement)
- **Models**: NHITS, NBEATSx, TiDE, PatchTST
- **Losses**: DistributionLoss("StudentT"), MQLoss, IQLoss
- **Infrastructure**: Python 3.12+, GPU for training, PostgreSQL, Redis

---

## Phase 1: Foundation & Data Contracts (Week 1)

### Objectives
Establish repository structure, implement data contracts, and create validation utilities.

### Tasks

#### 1.1 Repository Setup
```bash
# Create directory structure
mkdir -p data features nf_models cv experiments uq reports utils
touch features/__init__.py nf_models/__init__.py cv/__init__.py uq/__init__.py utils/__init__.py
touch run_train.py run_predict.py settings.yaml
```

#### 1.2 Data Validation Utilities (`utils/validate.py`)
- [ ] Implement `assert_regular_grid(df, freq="15min")`
- [ ] Implement `assert_utc_eob(df, freq="15min")`
- [ ] Implement `assert_shifted(df, hist_cols)` for leakage detection
- [ ] Implement `assert_no_forward_fill_y(df)`

#### 1.3 IO Utilities (`utils/io.py`)
- [ ] Implement `load_canonical_frame(path)`
- [ ] Implement `save_parquet(df, path)`
- [ ] Implement `timestamped_path(base_dir, stem)`

#### 1.4 Settings Configuration (`settings.yaml`)
```yaml
freq: "15min"
horizons: [4, 8, 16, 32]
target: "log_return"
default_input_size:
  generic: 1024
  PatchTST: 2048
seed: 1337
winsor:
  lower_q: 0.001
  upper_q: 0.999
scaler_type:
  default: robust
  PatchTST: revin
```

### Deliverables
- Repository structure created
- Validation utilities tested and working
- Settings configuration in place

### Acceptance Criteria
- [ ] All validation functions pass unit tests
- [ ] Repository follows prescribed structure
- [ ] Settings loaded correctly

---

## Phase 2: Feature Engineering Pipeline (Week 2)

### Objectives
Implement indicator computation, MTF alignment, and feature selection with strict leakage prevention.

### Tasks

#### 2.1 Feature Registry (`features/registry.py`)
- [ ] Define indicator specifications (vectorbt/TA-Lib wrappers)
- [ ] Configure MTF specs (30min, 1h, 4h alignment to 15min)
- [ ] Mark features as `hist`/`futr`/`stat`

#### 2.2 Feature Builder (`features/builder.py`)
- [ ] Implement `build_indicators(df_ohlcv, registry)`
- [ ] Implement `apply_mtf(df_ohlcv, registry)` with EOB alignment
- [ ] **Critical**: Implement `postprocess_shift_and_prune(df_exog, shift=1)`
- [ ] Implement `select_features(df, policy)` with 256-column cap

#### 2.3 Feature Selection Criteria
- [ ] Availability filter (≥98% non-null after shift)
- [ ] Correlation pruning (|ρ| ≥ 0.95 clusters)
- [ ] Variance screening (remove near-constant features)

### Deliverables
- Feature pipeline with 100+ initial indicators
- Pruned feature set ≤256 columns
- Shift validation passing

### Acceptance Criteria
- [ ] `assert_shifted()` passes for all hist_exog
- [ ] Feature count ≤256 after pruning
- [ ] MTF features properly aligned to EOB

---

## Phase 3: Model Development & Configuration (Weeks 3-4)

### Objectives
Implement NF model factory with distributional losses and conformal prediction setup.

### Tasks

#### 3.1 Model Factory (`nf_models/factory.py`)
- [ ] Implement `instantiate_models(cfg, hist_cols, futr_cols, stat_cols)`
- [ ] Configure NHITS with DistributionLoss("StudentT")
- [ ] Configure NBEATSx with MQLoss
- [ ] Configure TiDE with IQLoss
- [ ] Configure PatchTST with appropriate losses

#### 3.2 Model Configurations
```python
# Common parameters
base_config = {
    "h": horizon,
    "input_size": 1024,  # 2048 for PatchTST
    "scaler_type": "robust",
    "learning_rate": 1e-3,
    "batch_size": 512,
    "max_steps": 20000,
    "early_stop_patience_steps": 400
}
```

#### 3.3 Conformal Prediction Setup
- [ ] Correct import: `from neuralforecast.utils import PredictionIntervals`
- [ ] Configure `PredictionIntervals(n_windows=6)`
- [ ] Pass levels via `fit(..., level=[80,90,95])`

### Deliverables
- Model factory with 4 model types
- Distributional losses configured
- Conformal prediction ready

### Acceptance Criteria
- [ ] All models instantiate without errors
- [ ] Loss functions properly configured
- [ ] Conformal import verified correct

---

## Phase 4: Training Pipeline & Cross-Validation (Week 5)

### Objectives
Implement training orchestration with NF-native cross-validation.

### Tasks

#### 4.1 CV Runner (`cv/runner.py`)
- [ ] Implement `run_cv(nf, df, n_windows, step_size, val_size, refit)`
- [ ] Use only `NeuralForecast.cross_validation()`
- [ ] Configure: `n_windows=6`, `step_size=h`, `val_size=4*h`, `refit=True`

#### 4.2 Training Script (`run_train.py`)
- [ ] Load canonical frame and build features
- [ ] Instantiate NF models from config
- [ ] Run cross-validation
- [ ] Compute metrics (sCRPS, MAE, RMSE)
- [ ] Generate PIT diagnostics via `predict_insample()`
- [ ] Save best model via `nf.save(path, save_dataset=True)`

#### 4.3 Experiment Configs (`experiments/h{4,8,16,32}.yaml`)
```yaml
h: 16
n_windows: 6  # 10 for final
step_size: 16
val_size: 64
refit: true
models:
  - NHITS:
      loss: "StudentT"
      input_size: 1024
  - NBEATSx:
      loss: "MQLoss"
      level: [10, 50, 90]
```

### Deliverables
- Training pipeline operational
- CV results for all horizons
- Saved models in `experiments/h{h}/best/`

### Acceptance Criteria
- [ ] CV completes without errors
- [ ] sCRPS metrics computed
- [ ] Models saved and loadable

---

## Phase 5: Model Selection & Calibration (Week 6)

### Objectives
Select best models by sCRPS, validate calibration, and create ensembles.

### Tasks

#### 5.1 Diagnostics (`uq/diag.py`)
- [ ] Implement `compute_coverage(preds, levels=[80,90,95])`
- [ ] Implement `plot_pit(insample_df)` for uniformity check
- [ ] Implement `coverage_by_vol_decile(preds, ref_vol)`

#### 5.2 Selection Protocol
- [ ] Rank models by mean sCRPS across CV windows
- [ ] Verify coverage within ±2pp at 80/90/95 levels
- [ ] Select top-2 models per horizon

#### 5.3 Ensemble Creation
- [ ] Implement `blend_equal(model1_preds, model2_preds)`
- [ ] Average quantiles level-wise (not distribution params)
- [ ] Validate ensemble coverage

### Deliverables
- Leaderboard with sCRPS rankings
- Coverage validation reports
- Selected models/ensembles per horizon

### Acceptance Criteria
- [ ] sCRPS documented for all models
- [ ] Coverage within ±2pp tolerance
- [ ] PIT approximately uniform

---

## Phase 6: Inference & Deployment (Week 7)

### Objectives
Implement production inference pipeline with monitoring.

### Tasks

#### 6.1 Inference Script (`run_predict.py`)
- [ ] Load saved model via `NeuralForecast.load(path)`
- [ ] Build tail features (last 1024/2048 bars)
- [ ] Generate forecasts via `predict(level=[80,90,95])`
- [ ] Apply conformal adjustment if needed
- [ ] Save predictions with timestamps

#### 6.2 Tail Builders (`utils/tail.py`)
- [ ] Implement `build_tail_features(df_latest, feature_registry)`
- [ ] Ensure proper shift(1) application
- [ ] Handle future exogenous (calendar features)

#### 6.3 Live Loop Components
```python
# Pseudo-code for 15-minute loop
while True:
    df_latest = fetch_latest_ohlcv()
    assert_regular_grid(df_latest)
    
    features = build_tail_features(df_latest)
    nf = NeuralForecast.load(model_path)
    preds = nf.predict(features, level=[80,90,95])
    
    save_predictions(preds)
    monitor_performance(preds)
    
    sleep_until_next_bar()
```

### Deliverables
- Inference pipeline operational
- Predictions generated for all horizons
- Monitoring metrics tracked

### Acceptance Criteria
- [ ] Inference latency <100ms
- [ ] Predictions saved with proper timestamps
- [ ] No leakage in live features

---

## Phase 7: Validation & Acceptance (Week 8)

### Objectives
Verify all acceptance criteria, document results, and prepare for production.

### Tasks

#### 7.1 Acceptance Testing
- [ ] Verify sCRPS improvement over baseline
- [ ] Confirm coverage targets met (80±2%, 90±2%, 95±2%)
- [ ] Validate no leakage via correlation tests
- [ ] Check inference performance (<100ms)

#### 7.2 Documentation
- [ ] Generate acceptance reports per horizon
- [ ] Document model configurations and hyperparameters
- [ ] Create operational runbook
- [ ] Prepare rollback procedures

#### 7.3 Production Readiness
- [ ] Container image built and tested
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set (sCRPS degradation >10%)
- [ ] Backup and recovery procedures documented

### Deliverables
- Acceptance reports showing PASS/FAIL per horizon
- Operational documentation
- Production deployment package

### Acceptance Criteria
- [ ] All quality gates passed
- [ ] Documentation complete
- [ ] Team sign-off obtained

---

## Risk Mitigation Strategies

### Technical Risks

#### 1. MTF Misalignment
- **Risk**: Incorrect temporal alignment of multi-timeframe features
- **Mitigation**: Strict EOB alignment, forward-fill then shift(1)
- **Validation**: Correlation analysis to detect leakage

#### 2. Quantile Crossing
- **Risk**: Invalid prediction intervals
- **Mitigation**: Switch from MQLoss to IQLoss if detected
- **Validation**: Check monotonicity of quantiles

#### 3. GPU OOM
- **Risk**: Out of memory during training
- **Mitigation**: Reduce batch_size, use gradient accumulation
- **Fallback**: CPU training with longer duration

#### 4. Data Gaps
- **Risk**: Missing bars in 15-minute grid
- **Mitigation**: Reindex to full grid, drop missing y rows
- **Validation**: `assert_regular_grid()` checks

### Operational Risks

#### 5. Model Drift
- **Risk**: Performance degradation in production
- **Mitigation**: Monthly retraining, drift detection metrics
- **Trigger**: Retrain if sCRPS degrades >15%

#### 6. Library Breaking Changes
- **Risk**: NeuralForecast API changes
- **Mitigation**: Pin exact version, comprehensive testing before upgrade
- **Documentation**: Maintain compatibility matrix

---

## Success Metrics & KPIs

### Primary Metrics
- **sCRPS**: <0.10 (target), measured on test set
- **Coverage**: 80±2%, 90±2%, 95±2% on all horizons
- **Inference Latency**: <100ms p95

### Secondary Metrics
- **MAE/RMSE**: Supporting metrics for point forecasts
- **PIT Uniformity**: p-value >0.05 for uniformity test
- **Feature Availability**: >98% after shift

### Operational Metrics
- **Uptime**: 99.9% availability
- **Retraining Frequency**: Monthly or triggered
- **Alert Response Time**: <5 minutes

---

## Parallel Work Streams

### Stream 1: Data & Features (Weeks 1-2)
- Data Engineer focus
- Can proceed independently
- Deliverable: Feature pipeline

### Stream 2: Modeling (Weeks 3-5)
- ML Engineer focus
- Depends on feature pipeline
- Deliverable: Trained models

### Stream 3: Infrastructure (Weeks 6-7)
- DevOps focus
- Can start early with deployment prep
- Deliverable: Production environment

### Integration Points
- Week 3: Feature pipeline → Model training
- Week 6: Trained models → Inference pipeline
- Week 7: All streams converge for validation

---

## Command Quick Reference

### Training
```bash
python run_train.py --exp experiments/h16.yaml --save
```

### Cross-Validation
```bash
python cv/runner.py --config experiments/h16.yaml --n_windows 10
```

### Inference
```bash
python run_predict.py --model experiments/h16/best/ --h 16
```

### Acceptance Testing
```bash
python reports/acceptance.py --hdir experiments/h16 --baseline_scrps 0.12
```

---

## Definition of Done

### Phase Completion Criteria
- [ ] All tasks completed and tested
- [ ] Code reviewed and merged
- [ ] Documentation updated
- [ ] Acceptance criteria verified

### Project Completion Criteria
- [ ] All horizons have trained models
- [ ] sCRPS targets achieved
- [ ] Coverage calibration verified
- [ ] Production deployment successful
- [ ] Monitoring operational
- [ ] Team trained on operations

---

## Appendix: Key Code Snippets

### Correct Conformal Import
```python
from neuralforecast.utils import PredictionIntervals  # ✅ Correct
# NOT from neuralforecast.auto import PredictionIntervals  # ❌ Wrong
```

### Feature Shift Application
```python
def postprocess_shift_and_prune(df_exog, shift=1):
    hist_cols = [c for c in df_exog.columns if c.startswith('hist_')]
    df_exog[hist_cols] = df_exog[hist_cols].shift(shift)
    return df_exog
```

### NF Model Save/Load
```python
# Save
nf.save('experiments/h16/best/', save_dataset=True, overwrite=True)

# Load
nf = NeuralForecast.load('experiments/h16/best/')
```

### Cross-Validation Call
```python
cv_results = nf.cross_validation(
    df=df_train,
    n_windows=6,
    step_size=h,
    val_size=4*h,
    refit=True,
    prediction_intervals=PredictionIntervals(n_windows=6),
    level=[80, 90, 95]
)
```

---

## Notes

This workflow emphasizes:
1. **NeuralForecast-native implementation** - No custom backtesting or scaling
2. **Systematic validation** - Evidence-based decision making
3. **Production readiness** - Monitoring and rollback procedures
4. **Incremental delivery** - Phases with clear acceptance criteria
5. **Risk mitigation** - Proactive identification and handling

The plan avoids over-engineering by using NF's built-in capabilities and focusing on essential components for a production forecasting system.