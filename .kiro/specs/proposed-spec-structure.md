# BTC Forecasting System - Proposed Spec Structure (CORRECTED LINE REFERENCES)

⚠️ **IMPORTANT**: This document contains corrected line references after validation against `docs/forecasting_sf_plan.md`. The original `docs/proposed-spec-structure.md` contained systematic errors in line numbering.

This document outlines the proposed breakdown of the BTC forecasting system into 14 manageable specifications for implementation. Each spec represents a distinct and manageable component that contributes to the overall system architecture.

## Overview

The BTC forecasting system is a complex probabilistic forecasting system using NeuralForecast for intraday Bitcoin predictions with calibrated prediction intervals. The system requires strict data discipline, feature engineering, cross-validation, uncertainty quantification, and production deployment capabilities.

### Global Guardrails (apply to all specs)
- Follow `docs/forecasting_sf_plan.md` Section 0 (Objectives & guardrails).
- NF-native only: use NF models, CV, scalers, conformal, and `nf.save()/NeuralForecast.load()`.
- UTC 15‑minute EOB grid; never forward‑fill `y`.
- Strict compute → align (EOB) → shift(1) for all historic exogs; enforce MTF alignment.
- Keep implementations lean and explicit; avoid over-engineering.

## Proposed Specification Breakdown

### **Core Infrastructure Specs**

#### **1. Data Processing and Validation Spec**
**Purpose:** Data contracts, regularization, and validation utilities  
**Core Components:**
- UTC timestamp handling and grid regularization
- Target computation (log returns) and data quality gates
- Canonical NF frame creation with schema validation
- Missing data handling and winsorization policies
- Data validation utilities (assert_regular_grid, assert_utc_eob, assert_shifted)

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 2) Data contracts & validation
  - 2.1 Canonical target frame
  - 2.2 Regularization & data hygiene
  - 2.3 Target and splits
  - 2.4 Canonical NF frame
  - 2.5 Regularization policy
  - 2.6 Deterministic seeding & data checks
  - 2.7 Assembly path
  - 2.8 Minimal integration diff
  - 2.9 Why this is correct
- **Supporting:** 
  - Section 0.2 Bar finalization & time ordering
  - Section 0.3 Leakage discipline
  - Section 1.5 Bootstrap stubs - utils/validate.py

**Key Deliverables:** utils/validate.py, utils/io.py, canonical frame creation functions

---

#### **2. Feature Engineering Pipeline Spec**
**Purpose:** Technical indicator registry and computation with leakage prevention  
**Core Components:**
- Technical indicator registry using vectorbt/TA-Lib
- Supplementary indicators via pandas-ta-openbb (Numba)
- MTF resampling/merge via freqtrade/technical utilities
- Multi-timeframe feature alignment and aggregation
- Feature selection with ≤256 cap and ≥98% availability filter
- Leakage prevention via strict shift(1) rule
- NF exogenous variable wiring (hist/futr/stat lists)
 - Crypto-specific data-driven features (CCXT/exchange APIs for funding, OI, basis; cryptofeed for order book imbalance) with compute → align (15m EOB) → shift(1)
 - Optional performance: RAPIDS cuDF pandas accelerator for resampling/joins (indicator kernels remain CPU)

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 3) Exogenous features
  - 3.1 Indicator registry
  - 3.2 Feature builder (compute → align → shift(1))
  - 3.3 Feature selection & hard cap
  - 3.4 End-to-end assembly
  - 3.5 NF wiring
  - 3.6 Hygiene & boundary cases
  - 3.7 Minimal tests
  - 3.8 Crypto-specific data-driven features
  - 3.9 Performance note: pandas-on-GPU accelerator (optional)

**Key Deliverables:** features/registry.py, features/builder.py, postprocess_shift_and_prune function

---

#### **3. NeuralForecast Model Factory Spec**
**Purpose:** Model instantiation system for NF models  
**Core Components:**
- Model instantiation for NHITS, NBEATSx, TiDE, PatchTST
- Loss function configuration (DistributionLoss, MQLoss, ISQF, IQLoss)
- Training parameter management (batch_size, learning_rate, etc.)
- Exogenous variable wiring (hist_exog_list, futr_exog_list, stat_exog_list)
- Scaler configuration (robust, revin for PatchTST)

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 4) NeuralForecast model portfolio & defaults
  - 4.0 Portfolio justification
  - 4.1 Common model parameters
  - 4.2 Model-specific configurations
- **Supporting:** Section 1.5 Bootstrap stubs - nf_models/factory.py

**Key Deliverables:** nf_models/factory.py, instantiate_models function, _make_loss function

---

### **Training and Evaluation Specs**

#### **4. Cross-Validation and Metrics Spec**
**Purpose:** NF-native cross-validation implementation and metrics computation  
**Core Components:**
- NF-native cross-validation with proper windowing
- sCRPS computation as primary metric
- CV windowing strategy per horizon (n_windows, step_size, val_size)
- Metrics aggregation and leaderboard generation
- Leakage prevention in CV setup

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 5) Cross-validation (NF-native)
  - 5.1 Windowing per horizon
  - 5.2 Metrics, outputs, and artifacts
    - 5.2.A What NF returns
    - 5.2.B Primary metric: sCRPS
    - 5.2.C Coverage & PIT diagnostics
    - 5.2.D Drop-in runner
    - 5.2.E PIT helper
    - 5.2.F Training flow integration
    - 5.2.G Leakage discipline
    - 5.2.H Conformal attachment
    - 5.2.I Save/Load
- **Supporting:** 
  - Section 0.4 Cross-validation semantics
  - Section 1.5 Bootstrap stubs - cv/runner.py

**Key Deliverables:** cv/runner.py, run_cv function, summarize_cv function

---

#### **5. Uncertainty Quantification Spec**
**Purpose:** Probabilistic evaluation and calibration  
**Core Components:**
- Probabilistic evaluation and calibration assessment
- PIT analysis and coverage diagnostics
- Conformal prediction integration via NF's PredictionIntervals
- Prediction interval generation at 80/90/95 levels
- Coverage validation by volatility decile

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 8) Uncertainty & calibration
  - 8.1 Quantile vs Distribution training
  - 8.2 Conformal prediction intervals
  - 8.3 Diagnostic checks
  - 8.4 Practical fixes
  - 8.5 What to persist
- **Supporting:** 
  - Section 0.5 Probabilistic forecasts
  - Section 1.5 Bootstrap stubs - uq/diag.py

**Key Deliverables:** uq/diag.py, compute_coverage, plot_pit, coverage_by_vol_decile functions

---

#### **6. Training Workflow Orchestration Spec**
**Purpose:** YAML-driven experiment configuration and training pipeline  
**Core Components:**
- YAML-driven experiment configuration system
- Training pipeline orchestration (run_train.py)
- Artifact management and model persistence
- Experiment tracking and results organization
- Insample diagnostics integration

**Scope boundary:** Offline orchestration only (training, CV, artifacts, optional batch predictions for evaluation). Live inference loop and runtime operations are covered by Spec 9.

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 9) Training & evaluation workflow
  - 9.1 YAML-driven experiment configs
  - 9.2 run_train.py implementation
  - 9.3 run_predict.py implementation
  - 9.4 Artifacts & file layout
  - 9.5 Safety measures
- **Supporting:** 
  - Section 1.6 Minimal settings.yaml
  - Section 1.7 Entry-point skeletons

**Key Deliverables:** run_train.py, experiments/*.yaml configs, artifact management system

---

### **Advanced Features Specs**

#### **7. Model Selection and Ensembling Spec**
**Purpose:** Model ranking and ensemble creation  
**Core Components:**
- Model ranking by mean sCRPS across CV windows
- Top-2 ensemble creation with equal-weight averaging
- Model promotion criteria (≥1% sCRPS improvement)
- Ensemble inference and blending utilities
- Performance tracking and selection protocols

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 7) Model selection & simple ensembling
  - 7.1 Selection protocol
  - 7.2 Simple ensembles
  - 7.3 Drop-in utilities
  - 7.4 Workflow per horizon
  - 7.5 Final fit & save
  - 7.6 Inference path
  - 7.7 Guardrails

**Key Deliverables:** Model selection logic, blend_equal function, ensemble utilities

---

#### **8. Hyperparameter Optimization Spec**
**Purpose:** Bounded search spaces and optimization strategy  
**Core Components:**
- Bounded, model-specific search spaces
- Pilot → promote → full CV workflow
- Promotion criteria (≥0.5% sCRPS improvement for advancement)
- Resource management and early stopping
- n_windows progression (6 → 10)

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 6) Hyperparameter strategy
  - 6.1 Search spaces
  - 6.2 Pilot → promote → full CV
  - 6.3 Concrete glue
  - 6.4 Promotion thresholds
  - 6.5 Practical guards
  - 6.6 Optional NF Auto*

**Key Deliverables:** cv/hpo.py, search space definitions, promotion logic

---

### **Production and Deployment Specs**

#### **9. Inference and Live Deployment Spec**
**Purpose:** Real-time prediction pipeline and live deployment  
**Core Components:**
- Real-time prediction pipeline (run_predict.py)
- Live loop implementation with 45+ second buffer
- Tail feature building and prediction generation
- One-shot and loop modes for inference
- Graceful degradation and error handling

**Scope boundary:** Runtime inference and live operations only. Uses persisted models/artifacts from Spec 6; does not include training orchestration or artifact generation.

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 10) Inference & live deployment
  - 10.1 15-minute sequence
  - 10.2 Tail builders
  - 10.3 run_predict.py modes
  - 10.4 Throughput & memory guards
  - 10.5 Live conformal & monitoring
  - 10.6 Minimal assertions
  - 10.7 Integration points

**Key Deliverables:** run_predict.py, live loop implementation, tail builders

---

#### **10. Monitoring and Maintenance Spec**
**Purpose:** Performance monitoring and drift detection  
**Core Components:**
- Performance monitoring and drift detection
- Retraining triggers and procedures
- Rollback mechanisms and version management
- Coverage drift tracking (±3pp deviation alerts)
- sCRPS degradation monitoring (>3% vs baseline)
- PSI thresholds for distribution shift (0.2 moderate, 0.3 major)

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 11) Maintenance & retraining
  - 11.1 Cadence & triggers
  - 11.2 Versioning & pinning
  - 11.3 Smoke tests
  - 11.4 Drift & monitoring
  - 11.5 Retrain procedure
  - 11.6 Rollback
  - 11.7 Optional siblings
  - 11.8 House rules

**Key Deliverables:** Monitoring system, drift detection, retraining procedures

---

### **Quality and Risk Management Specs**

#### **11. Risk Mitigation and Error Handling Spec**
**Purpose:** Comprehensive error handling and risk mitigation  
**Core Components:**
- Comprehensive error handling strategies
- Data quality safeguards and validation
- Training stability and GPU memory management
- MTF misalignment prevention
- Leakage detection and prevention
- Graceful degradation strategies

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 14) Risks & mitigations
  - 14.1 MTF misalignment
  - 14.2 Leakage from non-shifted exogs
  - 14.3 Target mishandling
  - 14.4 Quantile crossing
  - 14.5 GPU OOM
  - 14.6 Training instability
  - 14.7 Bad data
  - 14.8-14.15 Additional risks

**Key Deliverables:** Error handling framework, stability measures, risk mitigation utilities

---

#### **12. Quality Gates and Acceptance Testing Spec**
**Purpose:** Acceptance criteria and quality validation  
**Core Components:**
- Acceptance criteria and quality thresholds
- Automated testing and validation procedures
- Acceptance reporting and promotion gates
- Hard pass/fail gates per horizon
- Rollback criteria for live deployment

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 12) Acceptance criteria & quality gates
  - 12.1 Hard pass/fail gates
  - 12.2 Rollback criteria
  - 12.3 Acceptance report
  - 12.4 What to store
  - 12.5 Remediation guidance

**Key Deliverables:** Quality gates, acceptance tests, validation procedures

---

### **Integration and Configuration Specs**

#### **13. Configuration Management Spec**
**Purpose:** Settings management and project structure  
**Core Components:**
- Project structure and directory organization
- Settings management and YAML configuration
- Environment setup and dependency management
- Bootstrap utilities and project scaffolding
- Coding standards and conventions

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 1) Repository layout
  - 1.1 Directory scaffold
  - 1.2 File inventory
  - 1.3 Naming conventions
  - 1.4 Coding standards
  - 1.5 Bootstrap stubs
  - 1.6 Minimal settings.yaml
  - 1.7 Entry-point skeletons
- **Supporting:** Section 0) Objectives & guardrails

**Key Deliverables:** Project structure, settings.yaml, bootstrap stubs

---

#### **14. Integration Testing and Validation Spec**
**Purpose:** End-to-end pipeline testing and validation  
**Core Components:**
- End-to-end pipeline testing
- Integration validation and smoke tests
- Performance testing and benchmarking
- Implementation phases and checkpoints
- Definition of Done criteria

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 13) Implementation checklist
  - Phase 0-11 implementation phases
  - Parallelization guide
  - Command quick sheet
  - Definition of Done

**Key Deliverables:** Integration tests, smoke tests, implementation phases

---

## Implementation Strategy

### **Dependencies and Sequencing**

1. **Foundation Layer** (Specs 1, 13): Data processing and configuration management
2. **Core Features** (Specs 2, 3): Feature engineering and model factory
3. **Training Infrastructure** (Specs 4, 5, 6): CV, UQ, and training workflow
4. **Advanced Features** (Specs 7, 8): Model selection and HPO
5. **Production Layer** (Specs 9, 10): Inference and monitoring
6. **Quality Assurance** (Specs 11, 12, 14): Risk mitigation, quality gates, and testing

### **Key Principles**

- **NF-Native Approach**: Use NeuralForecast primitives exclusively
- **Data Discipline**: Strict leakage prevention via shift(1) rule
- **Probabilistic Focus**: Calibrated prediction intervals at 80/90/95 levels
- **Production Ready**: Live deployment with monitoring and rollback capabilities
- **Quality First**: Comprehensive testing and validation at every level

### **Success Criteria**

- Reproducible CV artifacts with sCRPS as primary metric
- Saved model winners with proper versioning
- 15-minute inference loop with calibrated prediction intervals
- Acceptance reports showing ACCEPT for ≥2 horizons
- Monitoring system with coverage within ±3pp of nominal levels

### **Mapping Note**

Line ranges in the core plan can drift as the document evolves. This spec anchors to section headers and subsection titles in `docs/forecasting_sf_plan.md`. When in doubt, prefer header anchors over line numbers.

This specification structure provides a systematic approach to implementing the complex BTC forecasting system while maintaining clear boundaries, dependencies, and quality standards throughout the development process.
