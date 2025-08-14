# BTC Forecasting System - Proposed Spec Structure (CORRECTED LINE REFERENCES)

⚠️ **IMPORTANT**: This document contains corrected line references after validation against `docs/forecasting_sf_plan.md`. The original `docs/proposed-spec-structure.md` contained systematic errors in line numbering.

This document outlines the proposed breakdown of the BTC forecasting system into 14 manageable specifications for implementation. Each spec represents a distinct and manageable component that contributes to the overall system architecture.

## Overview

The BTC forecasting system is a complex probabilistic forecasting system using NeuralForecast for intraday Bitcoin predictions with calibrated prediction intervals. The system requires strict data discipline, feature engineering, cross-validation, uncertainty quantification, and production deployment capabilities.

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
- **Primary:** Section 2) Data contracts & validation (lines 584-783)
  - 2.1 Canonical target frame (586-601)
  - 2.2 Regularization & data hygiene (602-620)
  - 2.3 Target and splits (621-626)
  - 2.4 Canonical NF frame (627-676)
  - 2.5 Regularization policy (677-713)
  - 2.6 Deterministic seeding & data checks (714-729)
  - 2.7 Assembly path (730-744)
  - 2.8 Minimal integration diff (745-773)
  - 2.9 Why this is correct (774-783)
- **Supporting:** 
  - Section 0.2 Bar finalization & time ordering (175-187)
  - Section 0.3 Leakage discipline (188-198)
  - Section 1.5 Bootstrap stubs - utils/validate.py (358-498)

**Key Deliverables:** utils/validate.py, utils/io.py, canonical frame creation functions

---

#### **2. Feature Engineering Pipeline Spec**
**Purpose:** Technical indicator registry and computation with leakage prevention  
**Core Components:**
- Technical indicator registry using vectorbt/TA-Lib
- Multi-timeframe feature alignment and aggregation
- Feature selection with ≤256 cap and ≥98% availability filter
- Leakage prevention via strict shift(1) rule
- NF exogenous variable wiring (hist/futr/stat lists)

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 3) Exogenous features (lines 784-1203)
  - 3.1 Indicator registry (824-907)
  - 3.2 Feature builder (908-1094)
  - 3.3 Feature selection & hard cap (1095-1146)
  - 3.4 End-to-end assembly (1147-1170)
  - 3.5 NF wiring (1171-1184)
  - 3.6 Hygiene & boundary cases (1185-1192)
  - 3.7 Minimal tests (1193-1203)

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
- **Primary:** Section 4) NeuralForecast model portfolio & defaults (lines 1204-1564)
  - 4.0 Portfolio justification (1216-1239)
  - 4.1 Common model parameters (1240-1330)
  - 4.2 Model-specific configurations (1331-1564)
- **Supporting:** Section 1.5 Bootstrap stubs - nf_models/factory.py (358-498)

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
- **Primary:** Section 5) Cross-validation (NF-native) (lines 1565-1807)
  - 5.1 Windowing per horizon (1587-1620)
  - 5.2 Metrics, outputs, and artifacts (1621-1807)
    - 5.2.A What NF returns (1625-1650)
    - 5.2.B Primary metric: sCRPS (1651-1680)
    - 5.2.C Coverage & PIT diagnostics (1681-1710)
    - 5.2.D Drop-in runner (1711-1740)
    - 5.2.E PIT helper (1741-1760)
    - 5.2.F Training flow integration (1761-1780)
    - 5.2.G Leakage discipline (1781-1790)
    - 5.2.H Conformal attachment (1791-1800)
    - 5.2.I Save/Load (1801-1807)
- **Supporting:** 
  - Section 0.4 Cross-validation semantics (199-216)
  - Section 1.5 Bootstrap stubs - cv/runner.py (358-498)

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
- **Primary:** Section 8) Uncertainty & calibration (lines 2291-2431)
  - 8.1 Quantile vs Distribution training (2293-2322)
  - 8.2 Conformal prediction intervals (2323-2362)
  - 8.3 Diagnostic checks (2363-2392)
  - 8.4 Practical fixes (2393-2412)
  - 8.5 What to persist (2413-2431)
- **Supporting:** 
  - Section 0.5 Probabilistic forecasts (217-246)
  - Section 1.5 Bootstrap stubs - uq/diag.py (358-498)

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

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 9) Training & evaluation workflow (lines 2432-2757)
  - 9.1 YAML-driven experiment configs (2434-2542)
  - 9.2 run_train.py implementation (2543-2642)
  - 9.3 run_predict.py implementation (2643-2712)
  - 9.4 Artifacts & file layout (2713-2742)
  - 9.5 Safety measures (2743-2757)
- **Supporting:** 
  - Section 1.6 Minimal settings.yaml (499-515)
  - Section 1.7 Entry-point skeletons (516-583)

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
- **Primary:** Section 7) Model selection & simple ensembling (lines 2142-2290)
  - 7.1 Selection protocol (2144-2172)
  - 7.2 Simple ensembles (2173-2202)
  - 7.3 Drop-in utilities (2203-2232)
  - 7.4 Workflow per horizon (2233-2252)
  - 7.5 Final fit & save (2253-2272)
  - 7.6 Inference path (2273-2282)
  - 7.7 Guardrails (2283-2290)

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
- **Primary:** Section 6) Hyperparameter strategy (lines 1808-2141)
  - 6.1 Search spaces (1810-1892)
  - 6.2 Pilot → promote → full CV (1893-1972)
  - 6.3 Concrete glue (1973-2012)
  - 6.4 Promotion thresholds (2013-2052)
  - 6.5 Practical guards (2053-2092)
  - 6.6 Optional NF Auto* (2093-2141)

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

**Reference Sections in `docs/forecasting_sf_plan.md`:**
- **Primary:** Section 10) Inference & live deployment (lines 2758-2955)
  - 10.1 15-minute sequence (2760-2792)
  - 10.2 Tail builders (2793-2822)
  - 10.3 run_predict.py modes (2823-2872)
  - 10.4 Throughput & memory guards (2873-2902)
  - 10.5 Live conformal & monitoring (2903-2922)
  - 10.6 Minimal assertions (2923-2942)
  - 10.7 Integration points (2943-2955)

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
- **Primary:** Section 11) Maintenance & retraining (lines 2956-3135)
  - 11.1 Cadence & triggers (2958-2992)
  - 11.2 Versioning & pinning (2993-3022)
  - 11.3 Smoke tests (3023-3052)
  - 11.4 Drift & monitoring (3053-3082)
  - 11.5 Retrain procedure (3083-3102)
  - 11.6 Rollback (3103-3122)
  - 11.7 Optional siblings (3123-3127)
  - 11.8 House rules (3128-3135)

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
- **Primary:** Section 14) Risks & mitigations (lines 3493-3661)
  - 14.1 MTF misalignment (3495-3512)
  - 14.2 Leakage from non-shifted exogs (3513-3532)
  - 14.3 Target mishandling (3533-3552)
  - 14.4 Quantile crossing (3553-3572)
  - 14.5 GPU OOM (3573-3592)
  - 14.6 Training instability (3593-3612)
  - 14.7 Bad data (3613-3632)
  - 14.8-14.15 Additional risks (3633-3661)

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
- **Primary:** Section 12) Acceptance criteria & quality gates (lines 3136-3323)
  - 12.1 Hard pass/fail gates (3138-3192)
  - 12.2 Rollback criteria (3193-3242)
  - 12.3 Acceptance report (3243-3292)
  - 12.4 What to store (3293-3323)
  - 12.5 Remediation guidance (3293-3323)

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
- **Primary:** Section 1) Repository layout (lines 291-583)
  - 1.1 Directory scaffold (293-310)
  - 1.2 File inventory (311-334)
  - 1.3 Naming conventions (335-349)
  - 1.4 Coding standards (350-357)
  - 1.5 Bootstrap stubs (358-498)
  - 1.6 Minimal settings.yaml (499-515)
  - 1.7 Entry-point skeletons (516-583)
- **Supporting:** Section 0) Objectives & guardrails (132-290)

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
- **Primary:** Section 13) Implementation checklist (lines 3324-3492)
  - Phase 0-11 implementation phases (3326-3441)
  - Parallelization guide (3442-3462)
  - Command quick sheet (3463-3481)
  - Definition of Done (3482-3492)

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

### **Document Validation Status**

✅ **VALIDATED**: All line references in this document have been verified against `docs/forecasting_sf_plan.md` (3661 lines total)  
⚠️ **ORIGINAL DOCUMENT**: `docs/proposed-spec-structure.md` contains systematic line reference errors and should not be used  
📋 **CORRECTION LOG**: See `docs/corrected-line-references.md` for detailed validation findings

This specification structure provides a systematic approach to implementing the complex BTC forecasting system while maintaining clear boundaries, dependencies, and quality standards throughout the development process.