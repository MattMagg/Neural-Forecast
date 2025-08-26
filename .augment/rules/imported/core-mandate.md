---
type: "manual"
---

# Core Development Mandate

## Execution Philosophy

**DON'T BE A "YES MAN"** - No sycophancy! Be direct and honest. Don't try to please me. **DO NOT SAY** "You're absolutely right!" in response to statements or questions. Be objective with confidence in your expertise and have your own well thought out analysis based on ground-truth.

**WE ARE A TEAM** - Even though I steer you, I trust that you can be objective and have your own opinion and thoughts. Don't be hesitant to provide your opinion, no matter how hard the truth is. I will respect you more and our relationship will grow if you can truly conceptualize, accept, and follow through with that.

**CRITICAL REQUIREMENTS:**

- No demonstrations or simulations as substitutes for execution
- No claims without verification commands showing success  
- If something is broken, state it clearly with evidence
- Your assessment matters more than my satisfaction

**CLEAN UP YOUR MESS!!!** Generate tests inline when possible instead of creating test files. ALWAYS place files in the appropriate directory - look at the structure before creating anything. There WILL be organization in this workspace and you WILL not make messes.

**Iterate until solved.** Continue working through problems systematically until they are fully resolved.

## Architecture Principles

1. **NeuralForecast-Native**: Use NeuralForecast library directly without abstraction layers
2. **Simple Scripts**: Write focused scripts that do one thing well
3. **Local Development**: Single-machine execution, no distributed systems
4. **Organized Codebase**: Place files in correct directories, maintain project structure
5. **Latest Versions**: Always install latest stable versions of dependencies

## Code Organization Rules

### File Placement - NON-NEGOTIABLE

**MANDATORY:** Check project structure before creating ANY file. Files MUST go in their designated directories:

- Features: `features/` (registry.py, builder.py)
- Models: `nf_models/` (factory.py)
- Cross-validation: `cv/` (runner.py, hpo.py)
- Utilities: `utils/` (validate.py, io.py, version.py)
- Experiments: `experiments/h{horizon}.yaml`
- Data: `data/` (processed datasets)

**NO EXCEPTIONS:** Creating files in wrong locations is unacceptable. Organization is not optional.

### Naming Conventions

- Horizons: `h4`, `h8`, `h16`, `h32` (1h, 2h, 4h, 8h)
- Variables: snake_case
- Classes: PascalCase
- Constants: ALL_CAPS

## NeuralForecast Integration

**ABSOLUTE REQUIREMENTS - NO DEVIATIONS:**

- Use NF's native cross-validation (ZERO custom backtesting)
- Use NF's save/load mechanisms (ZERO custom serialization)
- Use NF's scalers (ZERO external normalization)
- Import pattern: `from neuralforecast.utils import PredictionIntervals`

**VIOLATION = IMMEDIATE REJECTION:** Any custom implementation of existing NF functionality will be rejected without discussion.

## Data Quality Gates - MANDATORY VALIDATION

**EVERY DATASET MUST PASS - NO EXCEPTIONS:**

```python
assert_regular_grid(df, "15min")
assert_utc_eob(df, "15min")
assert_shifted(df, hist_cols)
assert_no_forward_fill_y(df)
```

**FAILURE = STOP EXECUTION:** If validation fails, fix the data. Don't proceed with broken datasets.

## Versioning System

### Project Versioning

- Format: `v{MAJOR}.{MINOR}.{PATCH}`
- Check VERSION file before changes
- Tag releases: `git tag -a v0.1.0 -m "Description"`

### Model Artifacts

- Format: `v{VERSION}-{MODEL_TYPE}-{TIMESTAMP}-{METRIC}`
- Directory: `experiments/{horizon}/best/`
- Include: model files, config, training history

### Data Versioning

- Format: `{SYMBOL}-{TIMEFRAME}-{START_DATE}-{END_DATE}-v{VERSION}`
- Track metadata in `data/processed/data_versions.json`

## Task Management

Always reference `.kiro/specs/btc-forecasting-system/tasks.md` for implementation progress and task tracking.

**Execution Status Tracking**: Reference `EXECUTION_STATUS.md` for completed implementation details and validation results. This document provides audit trails for completed specs and serves as a reference for understanding what has been built and validated.

## Focus Areas

**Prioritize:**

- Model accuracy through feature engineering
- Hyperparameter optimization
- Data quality and completeness
- Reproducible experiments with proper validation

**Avoid:**

- Premature optimization
- Complex architectures without proven need
- Custom implementations of existing NeuralForecast functionality

## Operational Efficiency

**Batch Operations:** When performing multiple related operations, execute them concurrently in a single message rather than sequentially. This applies to file operations, bash commands, and tool invocations.

**VERIFICATION REQUIRED:** All claims must be backed by verification commands showing success. No assumptions about system state. No "it should work" - PROVE IT WORKS.

## Data Processing Standards

**Foundation Complete**: Data processing and validation infrastructure is fully implemented. Reference `.kiro/steering/data-processing.md` for detailed guidelines and `EXECUTION_STATUS.md` for implementation audit trail.

**Key Principles Validated**:
- Complete assembly path processes 7M+ records with 100% validation pass rate
- All quality gates implemented and tested with real BTC data
- NeuralForecast canonical schema compliance verified
- Leakage detection working correctly with correlation analysis
- UTC timestamp discipline enforced throughout pipeline

**Performance Standards**: The implemented pipeline processes 13+ years of BTC data (7M+ 1-minute bars) in ~30 seconds with full validation, demonstrating production-ready performance.
