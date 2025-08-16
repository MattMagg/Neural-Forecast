/sc:spawn "Task: Implement the Cross-Validation and Metrics system for the Neural-Forecast project using parallel agent deployment
for optimal efficiency.

Your Role as Orchestrator

You are the main Claude Code agent responsible for understanding the complete specification, planning the implementation
strategy, and deploying specialized sub-agents with precise context. This is a complex multi-phase implementation
requiring careful coordination.

---
Step 1: Gather Complete Context (15 minutes)

Before deploying any agents, you must thoroughly understand the specification:

1. Read all three specification documents in order:
    - Start with `.kiro/specs/cross-validation-metrics/requirements.md`
    * Understand the 10 requirements and their acceptance criteria
    * Note the WHEN-THEN format for validation

    - Continue with `.kiro/specs/cross-validation-metrics/design.md`
    * Study the architecture diagrams and component interfaces
    * Pay attention to the Agent-Oriented Error Recovery Protocols section
    * Note the specific import paths and function signatures

    - Finish with `.kiro/specs/cross-validation-metrics/tasks.md`
    * Identify task dependencies and groupings
    * Map which tasks can be parallelized
    * Note the deliverables for each task

2. Cross-reference with the source document:
    - Read `docs/forecasting_sf_plan.md` Section 5 (lines 1583-1825)
    - Understand the NF-native philosophy - NO custom CV implementations
    - Note the specific windowing parameters: n_windows=6→10, step_size=h, val_size=4*h
    - Verify the sCRPS computation approach and coverage targets

3. Review available sub-agents:
    - List agents in `.claude/agents/` to understand available specialists
    - Identify which agents are best suited for each task group
    - Note any agents that might need specific configuration

---
Step 2: Create Implementation Plan (10 minutes)

Based on your understanding, create a detailed implementation plan:

1. Validate the specification is complete:
    - Confirm all line references to docs/forecasting_sf_plan.md are accurate
    - Verify no conflicts between requirements and design
    - Check that all tasks have clear deliverables

2. Identify parallelization opportunities from tasks.md:
    - Tasks 1-3 must be sequential (foundation)
    - Tasks 4,5,7,8 can run in parallel (metrics and calibration)
    - Tasks 17-20 can run in parallel (testing and documentation)

3. Map agents to task groups:
    - Which agent handles CV runner implementation?
    - Who implements metrics computation?
    - Who handles calibration diagnostics?
    - Who does integration and testing?

4. Prepare agent-specific context:
    - Extract relevant design interfaces for each agent
    - Identify specific line references each agent needs
    - Note error recovery protocols relevant to each task

---
Step 3: Phase 1 - Foundation Setup (Sequential Deployment)

Deploy the first agent with complete context:

Deploy config-architect agent with this context:

"You need to set up the foundation for the Cross-Validation and Metrics system.

Read the specification at `.kiro/specs/cross-validation-metrics/` focusing on Tasks 1-3 in tasks.md.

Your specific responsibilities:
1. Create cv/runner.py with these exact imports:
    - from neuralforecast import NeuralForecast
    - from neuralforecast.losses.pytorch import sCRPS
    - import pandas as pd, numpy as np
    - from typing import Dict, Any, List, Optional

2. Set up the module structure:
    - cv/__init__.py for module exports
    - uq/metrics.py for metrics computation
    - uq/calibration.py for coverage and PIT analysis
    - utils/io.py updates for CV artifact persistence

3. Implement the core run_cv function signature from design.md lines 55-69:
    def run_cv(nf: NeuralForecast, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame

Reference docs/forecasting_sf_plan.md lines 1620-1628 for the exact NF.cross_validation parameters.

Deliverables: Working module structure with proper imports and function stubs."

Wait for completion before proceeding to Phase 2.

---
Step 4: Phase 2 - Parallel Core Implementation

Deploy three agents simultaneously with specific contexts:

PARALLEL DEPLOYMENT - Execute all three in one message:

[Agent 1: nf-model-factory]
"Implement metrics computation for Cross-Validation (Tasks 4-5 from .kiro/specs/cross-validation-metrics/tasks.md).

Key requirements:
- Implement compute_scrps() using from neuralforecast.losses.pytorch import sCRPS
- Support both distributional (StudentT) and quantile (MQLoss/IQLoss) models
- Add MAE, RMSE, bias computation for supporting metrics
- Reference design.md lines 94-111 for exact function signatures
- Follow docs/forecasting_sf_plan.md lines 1655-1660 for sCRPS implementation

Deliverables: Complete metrics computation in uq/metrics.py"

[Agent 2: uq-calibration-specialist]  
"Implement coverage and PIT diagnostics (Tasks 7-8 from .kiro/specs/cross-validation-metrics/tasks.md).

Key requirements:
- Compute coverage at 80%, 90%, 95% with ±2pp tolerance validation
- Implement PIT with dense quantile grid: quantiles = [i/100 for i in range(1, 100)]
- Reference design.md lines 135-149 for compute_pit() signature
- Follow docs/forecasting_sf_plan.md lines 1661-1665 for coverage requirements
- Skip PIT for conformal models (document why per lines 1632, 1642)

Deliverables: Complete calibration diagnostics in uq/calibration.py"

[Agent 3: data-validation-specialist]
"Enhance CV validation (Task 3 from .kiro/specs/cross-validation-metrics/tasks.md).

Key requirements:
- Validate CV output DataFrame structure (unique_id, ds, cutoff, y)
- Implement assert_shifted() checks for leakage prevention
- Verify interval columns when level=[80, 90, 95] specified
- Reference docs/forecasting_sf_plan.md lines 1806-1812 for validation rules
- Add error handling for incomplete CV runs

Deliverables: Validation functions in cv/runner.py"

Monitor all three agents and ensure they complete their tasks.

---
Step 5: Phase 3 - Integration and Aggregation

Deploy agents based on Phase 2 completion:

Stage 3.1 - PARALLEL DEPLOYMENT:

[Agent 1: cv-runner or nf-model-factory]
"Implement aggregation and visualization (Tasks 6, 9, 13 from tasks.md):
- Aggregate metrics across CV windows (mean, std, min, max)
- Create calibration visualization suite (PIT histograms, coverage plots)
- Add conformal prediction support via NF's PredictionIntervals
- Reference design.md lines 273-318 for error recovery protocols"

[Agent 2: inference-pipeline]
"Implement results persistence (Task 14 from tasks.md):
- Save CV results to experiments/h{horizon}/cv_results.parquet
- Store metrics as JSON, leaderboard as CSV
- Use timestamp format YYYYMMDDTHHMMSSZ
- Reference docs/forecasting_sf_plan.md lines 1799-1800"

Stage 3.2 - SEQUENTIAL (after 3.1 completes):

[Agent: model-selector-ensemble]
"Implement model selection and ranking (Tasks 10-12, 15):
- Create summarize_cv() function orchestrating all metrics
- Rank models by mean sCRPS (ascending)
- Select best StudentT and best quantile model
- Save models using NF.save() to experiments/h{horizon}/models/
- Reference design.md lines 74-88 for summarize_cv signature"

---
Step 6: Phase 4 - Pipeline Integration

Deploy integration specialist:

[Agent: integration-test-orchestrator]
"Wire CV into the training pipeline (Task 16 from tasks.md):

Integration points:
- Modify run_train.py to call cv/runner.py functions
- Load configuration from experiments/h{horizon}.yaml
- Connect with feature pipeline output (canonical DataFrame)
- Add progress logging and partial failure recovery
- Reference docs/forecasting_sf_plan.md lines 1770-1801 for integration pattern

Test the integration with a small dataset to verify it works end-to-end."

---
Step 7: Phase 5 - Quality Assurance (Parallel)

Final parallel deployment for testing and documentation:

PARALLEL DEPLOYMENT - All four agents:

[Agent 1: integration-test-orchestrator]
"Create comprehensive tests (Tasks 17-18):
- Unit tests in tests/test_cv.py covering all metrics
- Integration tests for end-to-end pipeline
- Use pytest framework with >90% coverage target"

[Agent 2: quality-gate-validator]
"Validate acceptance criteria:
- Verify sCRPS computation matches NF implementation
- Check coverage is within ±2pp tolerance
- Confirm no custom CV loops exist
- Generate acceptance report"

[Agent 3: risk-mitigation-specialist]
"Implement error recovery from design.md lines 273-318:
- GPU OOM handling (batch_size reduction)
- Quantile crossing prevention (IQLoss fallback)
- Numerical stability checks
- Add specific context7 queries for debugging"

[Agent 4: spec-architect or appropriate documentation agent]
"Create documentation (Tasks 19-20):
- Comprehensive docstrings for all functions
- cv/README.md with usage examples
- Jupyter notebook demonstrating CV execution
- Troubleshooting guide for common issues"

---
Step 8: Final Validation

After all agents complete:

1. Run the complete CV pipeline with test data
2. Verify all deliverables from tasks.md are present
3. Check that all acceptance criteria from requirements.md are met
4. Ensure no violations of NF-native philosophy
5. Confirm line references to docs/forecasting_sf_plan.md are accurate

---
Critical Reminders for Main Agent

- Always read the full specification first - Don't skip Step 1
- Provide complete context to each agent - Include specific line references and function signatures
- Monitor parallel deployments - Ensure agents don't create conflicts
- Validate incrementally - Check outputs after each phase
- Use error recovery protocols - Reference design.md lines 273-318 if agents encounter issues
- Maintain NF-native approach - Reject any custom CV implementations

This workflow should take approximately 8-10 hours with parallel execution. Start with Step 1 immediately to understand
the full scope before deploying any agents." --sequential --delegate auto --think --seq --validate --concurrency 4