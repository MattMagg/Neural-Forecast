# Notebook Conversion Workflow

## Objective
Convert Python scripts to Jupyter notebooks, preserving functionality while adding notebook-specific enhancements.

## Scope
- **Target Scripts**: 
  - run_train.py (main training pipeline)
  - run_predict.py (inference pipeline)
  - test_cv_integration.py (CV testing workflow)
  - test_feature_integration.py (feature testing workflow)
- **Excluded**: Scripts already covered by existing notebooks in nf_models/, examples/cv/

---

## Implementation Plan

### Phase 1: Script Conversion
**Execution**: Parallel (2 agents can work simultaneously)

**Agent 1**: Use existing project agent for training/testing notebooks
- Convert run_train.py to run_train.ipynb
- Convert test_cv_integration.py to test_cv_integration.ipynb
- Convert test_feature_integration.py to test_feature_integration.ipynb

**Agent 2**: Use existing project agent for inference notebook
- Convert run_predict.py to run_predict.ipynb

**Requirements**:
- Replace argparse with notebook variables
- Update all hardcoded paths to use notebook path variables
- Preserve all functions without modification
- Organize code into logical cells with markdown headers

---

### Phase 2: Validation
**Execution**: Sequential (after Phase 1 completes)

**Agent**: integration-test-orchestrator
- Verify all notebooks have correct cell structure
- Check that imports resolve without errors
- Validate path references are updated correctly
- Ensure notebook execution order is logical

---

## Path Variables to Define

All notebooks must define these at the top:
```python
# Path configuration
DATA_PATH = "data/raw/btcusd_1-min_data.csv"
PROCESSED_PATH = "data/processed"
EXPERIMENT_PATH = lambda h: f"experiments/h{h}"
REPORTS_PATH = "reports"
```

---

## Sub-Agent Deployment

### Phase 1: Parallel Execution
Two agents can work simultaneously:
- **Agent 1**: General agent for training/test notebooks
- **Agent 2**: General agent for prediction notebook

### Phase 2: Sequential Validation  
- **Agent**: integration-test-orchestrator (after Phase 1 completes)

---

When invoking sub-agents, you should construct detailed prompts that are rich with context, have clarity, and give specific, optimal directives. The sub-agents will be completely autonomous throughout their lifecycle, so ensure your prompts are comprehensive and unambiguous.

## SuperClaude Framework Integration

Integrate sub-agent prompts with the most optimal SuperClaude flags:

**For Implementation Tasks:**
```bash
--delegate --validate --think --safe-mode
--seq (for analysis) --c7 (for NeuralForecast patterns)
--focus [performance|quality|architecture]
```

**For Analysis & Validation:**
```bash
--think-hard --validate --systematic
--seq --c7 (for comprehensive understanding)
--scope [file|module|project]
```

**For Parallel Execution:**
```bash
--concurrency [2-4] (for independent agents)
--delegate folders (for directory-level work)
```

**For Complex Multi-Phase Tasks:**
```bash
--wave-mode --wave-strategy systematic
--validate (between phases)
```

## Project Context to Provide

Always state the higher-level purpose: Design and implement a forecasting architecture that is lean, pragmatic, efficient, and strictly aligned with project guidelines and core documentation.

Remind sub-agents to stick to core principles:
- Rigorously follow the main specification in `docs/forecasting_sf_plan.md`. Only reference exact sections required—never load the entire document to preserve context window.
- Remain focused on practicality. Avoid over-engineering or adding enterprise-level complexity.
- Build directly around NeuralForecast (NF). Always prefer native NF primitives and workflows.
- Never recreate capabilities already present in NeuralForecast:
  - Use `NeuralForecast.cross_validation` for CV
  - Use `PredictionIntervals` for uncertainty
  - Use `NeuralForecast.save/load` for persistence
  - Use native losses (DistributionLoss, MQLoss, IQLoss)
- Employ sequential-thinking MCP by default, use context7 (c7) for code reference.

## Planning and Verification Instructions

Sub-agents should:
- Begin with a concise checklist (3-7 bullets) of key conceptual steps
- Think step-by-step; decompose requirements, clarify unknowns, scope all relevant files
- After each code edit or tool invocation, validate the result in 1-2 lines
- Test as they go—verify all steps before committing to output
- Prioritize minimal, high-leverage modifications over broad changes
- Optimize for low latency—avoid unnecessarily long operations

## Quality Gates to Enforce

Ensure sub-agents validate:
- Data contracts: `assert_regular_grid()`, `assert_utc_eob()`, `assert_shifted()`, `assert_no_forward_fill_y()`
- Feature constraints: Maximum 256 features after pruning, all historical features shifted by 1
- Model targets: sCRPS metric, coverage 80±2%, 90±2%, 95±2%
- Code quality: Passes linting, type checking, ≥80% test coverage

## Execution Patterns

When spawning multiple agents, batch in ONE message:
```javascript
[Single Message]:
  - Task("agent-1: Complete task with full context")
  - Task("agent-2: Complete task with full context")
  - Task("agent-n: Complete task with full context")
```

## Resource Management Guidance

Inform sub-agents about resource zones:
- Green (0-60%): Full operations
- Yellow (60-75%): Enable `--uc`, optimize
- Orange (75-85%): Defer non-critical work
- Red (85%+): Essential operations only

## Critical Reminders

Sub-agents must:
- Never create custom implementations of NF native features
- Always validate data contracts before operations
- Maintain strict leakage prevention with shift(1)
- Reference only specific document sections, not entire files
- Test incrementally—don't accumulate untested code
- Focus on lean, efficient solutions without enterprise complexity

## Deliverables

1. **run_train.ipynb**: Training pipeline notebook with cell-based structure
2. **run_predict.ipynb**: Prediction pipeline notebook with cell-based structure
3. **test_cv_integration.ipynb**: CV testing notebook
4. **test_feature_integration.ipynb**: Feature testing notebook
5. **Updated TASK_TRACKER.md**: Document notebook creation

## Success Criteria

- All functions from original scripts preserved
- Cell structure follows logical organization
- Configuration converted from argparse to notebook variables
- Path references use notebook variables
- Notebooks syntactically correct (no execution required)

/sc:spawn notebook-conversion-workflow docs/notebook_conversion_workflow.md \
--parallel \
--delegate auto \
--validate \
--safe-mode \
--scope module \
--monitor