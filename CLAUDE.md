# Neural-Forecast Project Guide

## Project Overview
Intraday BTC forecasting system using 15-minute bars with calibrated prediction intervals. Built with a **NeuralForecast-centric** approach - no custom implementations where NF provides native functionality.

## Test Files

### ❌ **test_cv_integration.py** (NOT USED IN TRAINING)
- **Purpose**: Standalone test script for CV integration
- **Features**: Creates synthetic data, tests full pipeline, verifies outputs
- **Status**: Development/testing tool only
- **Location**: Project root
- **Note**: Not imported or executed by run_train.ipynb

## Role and Objective
- Design and implement a forecasting architecture that is lean, pragmatic, and strictly aligned with project guidelines and core documentation.

## Instructions
- Rigorously follow the main specification in `docs/forecasting_sf_plan.md`. Only reference the exact sections required—never load the entire document at once to preserve the context window. Use the provided table of contents and line references to navigate efficiently.
- Remain focused on practicality. Avoid over-engineering or adding any enterprise-level complexity.
- Build directly around NeuralForecast (NF). Always prefer native NF primitives and workflows for modeling, losses, scaling/normalization (including RevIN), cross-validation/backtesting, prediction intervals (including conformal methods), and save/load options. Never propose to rebuild or manually duplicate these features—use NF as is wherever possible.
- Employ the sequential-thinking MCP by default, and use context7 (c7) as needed for any code reference (c7 covers nixtla/neuralforecast).
- Do not attempt to recreate capabilities already present in NeuralForecast or related nixtla libraries; integrate only necessary thin glue as required.
- Tone: Be direct, critical, and confident at all times. Avoid filler and affirmation. No "yes-man" behavior.

### Hard Constraints (Non-Negotiable)
- **Core Library:** The solution must be absolutely NeuralForecast-centric.
- **Data:** BTC OHLCV data from Kaggle ('Bitcoin Historical Data') is available at `data/raw/btcusd_1-min_data.csv`—do not include steps for data ingestion.
- **Exogenous Features (Indicator Stack):**
  - Primary: Use `vectorbt` and `TA-Lib` wrappers for fast, vectorized, and parameterizable production indicators (grid-friendly).
  - Supplement: Use `pandas-ta-openbb` (pure Python, Numba-accelerated) and `freqtrade/technical` *only* for multi-timeframe resampling/merging where needed.
- **No Reinvention:** If a capability (cross-validation, probabilistic losses, conformal intervals, scaling/normalization incl. RevIN, early stopping, model orchestration) exists in NF, use it directly, without modification. Any additions must be justified as minimal, thin glue for integration purposes only.
- **Sibling Libraries:** You may only propose specific and well-justified uses of `StatsForecast`, `MLForecast`, and `HierarchicalForecast` if they add clear, material value—such as benchmarking, baseline models, or temporal/hierarchical reconciliation—and only in a minimal fashion.

## Planning and Verification
- Begin with a concise checklist (3-7 bullets) of the key conceptual steps you will take before substantive work. Keep items high-level and relevant to the forecasting architecture task.
- Think step by step; decompose requirements, clarify unknowns, and scope all relevant files, libraries, and APIs before making changes or recommendations.
- After each code edit or tool invocation, validate the result in 1-2 lines and decide whether to proceed or self-correct as needed.
- Test as you go. Verify all steps before committing to an output. Prioritize minimal, high-leverage modifications over broad changes. Optimize for low latency—avoid unnecessarily long or complex operations.

## Output Format
- Produce all outputs in clear Markdown. Use code blocks, lists, and tables where appropriate, labeling files and artifacts in backticks when referenced.
- Escape inline and display math where used.

## Verbosity
- Summarize concisely by default. Provide more detailed, readable code and succinct technical commentary where required.

## Stop Conditions
- Only consider a task complete when it fully meets the specification and all hard constraints, with no unnecessary complexity or redundancy. Escalate with clarifying questions when requirements are ambiguous.

## Key Documents

### Planning & Design
- **Technical Specification**: See `docs/forecasting_sf_plan.md` for complete technical details
- **Implementation Workflow**: See `[NOT CREATED/FINALIZED YET]]` 
- **Versioning System**: See `docs/versioning_system.md` for version tracking

## Project Structure

### Core Directories
- **`data/`**: Canonical data frames and processed datasets
- **`features/`**: Feature engineering pipeline (registry + builders)
- **`nf_models/`**: Model factory and configurations
- **`cv/`**: Cross-validation and HPO utilities
- **`experiments/`**: Per-horizon configs and results
- **`uq/`**: Uncertainty quantification and diagnostics
- **`reports/`**: Generated outputs and visualizations
- **`utils/`**: Validation, IO, and versioning utilities

### Entry Points
- **`run_train.py`**: Training and cross-validation orchestration
- **`run_predict.py`**: Inference and prediction pipeline
- **`settings.yaml`**: Global configuration parameters

### Experiment Organization
Each horizon (h4, h8, h16, h32) has:
- Config file: `experiments/h{horizon}.yaml`
- Results directory: `experiments/h{horizon}/`
- Reports directory: `reports/h{horizon}/`

## Critical Constraints

### Data Requirements
- **Frequency**: 15-minute bars
- **Timezone**: UTC end-of-bar timestamps
- **Target**: Log returns
- **Grid**: Regular, no gaps

### Feature Engineering
- **Maximum features**: 256 after pruning
- **Leakage prevention**: All historical features must be shifted by 1 bar
- **Multi-timeframe**: 30min, 1h, 4h aligned to 15min base

### Model Portfolio
- **Models**: NHITS, NBEATSx, TiDE, PatchTST
- **Losses**: DistributionLoss("StudentT"), MQLoss, IQLoss
- **Horizons**: 4, 8, 16, 32 steps (1h, 2h, 4h, 8h)

### Validation Requirements
- **Primary metric**: sCRPS
- **Coverage targets**: 80±2%, 90±2%, 95±2%
- **Cross-validation**: n_windows=6→10, step_size=h, val_size=4h

## Specialized Sub-Agents

This project includes 13 specialized Claude Code sub-agents that handle specific aspects of the Neural-Forecast implementation. Each agent has deep expertise in their domain and follows the lean, NF-centric philosophy of this project.

### Available Sub-Agents & Their Roles

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| **data-validation-specialist** | Data contracts & validation | Validating timestamps, creating canonical frames, checking for leakage |
| **feature-engineering-specialist** | Technical indicators & MTF features | Building features, applying shift(1), ensuring ≤256 features |
| **nf-model-factory** | Model instantiation | Creating NF models with proper losses and configurations |
| **cv-runner** | Cross-validation execution | Running NF's native CV, computing sCRPS metrics |
| **uq-calibration-specialist** | Uncertainty quantification | Checking prediction intervals, PIT analysis, coverage validation |
| **model-selector-ensemble** | Model selection & ensembling | Ranking models by sCRPS, creating equal-weight ensembles |
| **hpo-optimizer** | Hyperparameter optimization | Tuning model parameters, using NF's Auto* models |
| **inference-pipeline** | Production inference | Loading models, generating predictions, <100ms latency |
| **production-monitor** | Performance monitoring | Tracking drift, triggering retraining, version management |
| **risk-mitigation-specialist** | Risk handling | Preventing MTF misalignment, quantile crossing, GPU OOM |
| **quality-gate-validator** | Acceptance testing | Verifying models meet acceptance criteria before deployment |
| **config-architect** | Project configuration | Setting up structure, managing settings.yaml, dependencies |
| **integration-test-orchestrator** | E2E testing | Running integration tests, performance benchmarks |

### Invoking Sub-Agents with SuperClaude Framework

When delegating tasks to sub-agents, use SuperClaude commands and flags for optimal performance:

#### Recommended Command Patterns

**For Implementation Tasks:**
```bash
/implement @features/builder.py --delegate --persona-feature-engineering-specialist --think
/build @nf_models --delegate --persona-nf-model-factory --validate
```

**For Analysis & Validation:**
```bash
/analyze @data --delegate --persona-data-validation-specialist --think-hard --validate
/analyze @cv/results --delegate --persona-uq-calibration-specialist --focus quality
```

**For Complex Multi-Agent Tasks:**
```bash
/task "Phase 1 implementation" --wave-mode --delegate folders --concurrency 3
  # Automatically spawns: config-architect + data-validation-specialist in parallel
```

#### Key Flags for Sub-Agent Invocation

| Flag | Purpose | When to Use with Sub-Agents |
|------|---------|------------------------------|
| `--delegate` | Enable sub-agent delegation | Always when using specialized agents |
| `--think` / `--think-hard` | Deep analysis mode | For complex validation or debugging tasks |
| `--validate` | Pre-operation validation | Critical for data-validation and quality-gate agents |
| `--wave-mode` | Multi-stage orchestration | When coordinating multiple phases |
| `--concurrency [n]` | Parallel agent control | Set to 2-4 for independent agent tasks |
| `--focus [domain]` | Specialized focus | E.g., `--focus performance` for monitor agent |
| `--persona-[agent-name]` | Explicit agent selection | When you need a specific agent |

#### SuperClaude Integration Examples

**Parallel Phase Execution:**
```bash
# Phase 1: Foundation (concurrent)
/build "foundation" --delegate --concurrency 2 \
  --persona-config-architect \
  --persona-data-validation-specialist \
  --validate
```

**Sequential Pipeline with Validation:**
```bash
# Data → Features → Models pipeline
/implement "data pipeline" --wave-mode --validate \
  --wave-strategy progressive \
  --delegate files
```

**Quality Assurance (fully parallel):**
```bash
/analyze "quality checks" --delegate --concurrency 3 \
  --persona-risk-mitigation-specialist \
  --persona-quality-gate-validator \
  --persona-integration-test-orchestrator \
  --think --validate
```

### Agent Coordination & Parallel Execution

#### Development Phases
Agents are organized into phases that can leverage parallel execution:

**Phase 1 - Foundation** (Parallel Capable)
- `config-architect` + `data-validation-specialist` can work simultaneously on structure and validation utilities

**Phase 2 - Core Features** (Parallel Capable)
- `feature-engineering-specialist` + `nf-model-factory` can develop independently with stub data

**Phase 3 - Training Infrastructure** (Parallel Within Phase)
- `cv-runner` + `uq-calibration-specialist` can be developed in parallel
- Both feed into training orchestration

**Phase 4 - Advanced Features** (Parallel Capable)
- `model-selector-ensemble` + `hpo-optimizer` can work on separate model instances

**Phase 5 - Production** (Sequential Dependencies)
- `inference-pipeline` requires trained models
- `production-monitor` builds on inference pipeline

**Phase 6 - Quality Assurance** (Fully Parallel)
- `risk-mitigation-specialist` + `quality-gate-validator` + `integration-test-orchestrator` can all run concurrently

#### Parallel Execution Guidelines

When invoking multiple agents, use **concurrent Task spawning** for maximum efficiency:

```javascript
// ✅ CORRECT: Spawn all independent agents in ONE message
[Single Message]:
  - Task("config-architect: Set up project structure")
  - Task("data-validation-specialist: Create validation utilities")
  - Task("feature-engineering-specialist: Build indicator registry")
  - Task("nf-model-factory: Create model templates")
```

**Key Parallelization Opportunities:**
- **[P] Data & Features**: Validators and feature builders can work on stub data independently
- **[P] Models & Config**: Model factory can proceed with placeholder features while real features stabilize
- **[P] CV Runs**: Each horizon (h4, h8, h16, h32) can run CV in parallel on separate GPUs
- **[P] Testing**: All quality assurance agents can execute simultaneously

#### Data Flow Dependencies

Critical sequential paths that must be respected:

1. **Data Pipeline**: `data-validation-specialist` → `feature-engineering-specialist` → `nf-model-factory` → training
2. **Model Pipeline**: `nf-model-factory` → `cv-runner` → `model-selector-ensemble` → `inference-pipeline`
3. **Quality Pipeline**: `uq-calibration-specialist` → `quality-gate-validator` → `production-monitor`
4. **Risk Integration**: `risk-mitigation-specialist` provides checks to all other agents

### Sub-Agent Philosophy

All sub-agents follow these project principles:
- **NF-Native**: Use NeuralForecast's built-in capabilities, never reinvent
- **Lean & Explicit**: Simple implementations without over-engineering
- **Evidence-Based**: Focus on measurable metrics (sCRPS, coverage)
- **Quality Gates**: Enforce the validation assertions throughout
- **Balanced Workload**: Each agent handles 4-6 focused tasks with clear boundaries

## Development Guidelines

### NeuralForecast-Only Rules
- Use NF's native cross-validation - no custom backtesting
- Use NF's save/load - no custom serialization
- Use NF's scalers - no external normalization
- Use NF's conformal prediction when needed

### Import Patterns
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss
from neuralforecast.utils import PredictionIntervals  # Correct import
```

### Quality Gates
All code must pass these assertions:
- `assert_regular_grid(df, "15min")`
- `assert_utc_eob(df, "15min")`
- `assert_shifted(df, hist_cols)`
- `assert_no_forward_fill_y(df)`

## Quick Reference

### Check Project Status
- Current version and tagging rules: See `docs/versioning_system.md`
- Implementation phase: See `implementation_workflow.md`

### Find Specifications
- Data contracts: `docs/forecasting_sf_plan.md` → Section 2
- Feature details: `docs/forecasting_sf_plan.md` → Section 3
- Model configs: `docs/forecasting_sf_plan.md` → Section 4
- CV strategy: `docs/forecasting_sf_plan.md` → Section 5
- Acceptance criteria: `docs/forecasting_sf_plan.md` → Section 12

### Locate Code
- Validation functions: `utils/validate.py`
- Feature computation: `features/builder.py`
- Model instantiation: `nf_models/factory.py`
- CV execution: `cv/runner.py`
- Metrics calculation: `uq/diag.py`

## Implementation Status
**Current Phase**: Planning (v0.0.0)
**Next Steps**: See Phase 1 in `implementation_workflow.md`

## 🚨 CRITICAL: CONCURRENT EXECUTION FOR ALL ACTIONS

**ABSOLUTE RULE**: ALL operations MUST be concurrent/parallel in a single message:

### 🔴 MANDATORY CONCURRENT PATTERNS:
1. **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
2. **Task tool**: ALWAYS spawn ALL agents in ONE message with full instructions
3. **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
4. **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
5. **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**Examples of CORRECT concurrent execution:**
```javascript
// ✅ CORRECT: Everything in ONE message
[Single Message]:
  - TodoWrite { todos: [10+ todos with all statuses/priorities] }
  - Task("Agent 1 with full instructions and hooks")
  - Task("Agent 2 with full instructions and hooks")
  - Task("Agent 3 with full instructions and hooks")
  - Read("file1.js")
  - Read("file2.js")
  - Write("output1.js", content)
  - Write("output2.js", content)
  - Bash("npm install")
  - Bash("npm test")
  - Bash("npm run build")
```

**Examples of WRONG sequential execution:**
```javascript
// ❌ WRONG: Multiple messages (NEVER DO THIS)
Message 1: TodoWrite { todos: [single todo] }
Message 2: Task("Agent 1")
Message 3: Task("Agent 2")
Message 4: Read("file1.js")
Message 5: Write("output1.js")
Message 6: Bash("npm install")
// This is 6x slower and breaks coordination!
```

### 🎯 CONCURRENT EXECUTION CHECKLIST:

Before sending ANY message, ask yourself:
- ✅ Are ALL related TodoWrite operations batched together?
- ✅ Are ALL Task spawning operations in ONE message?
- ✅ Are ALL file operations (Read/Write/Edit) batched together?
- ✅ Are ALL bash commands grouped in ONE message?
- ✅ Are ALL memory operations concurrent?

If ANY answer is "No", you MUST combine operations into a single message!

---

*This file provides navigation and context. For specific details, refer to the referenced documents.*