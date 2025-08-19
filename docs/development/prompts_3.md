Can you provide a specific example? For example, I want to add the following static flag that can be applied to any slash command. See below.

Slash command content:
```markdown
...ts for the objective of the prompt in provided below.

$ARGUMENTS
```

Flag:
--interative-response

```markdown
## Context & Scenario

You (the LLM) have **unlimited time and computational resources** to craft your response. The goal is to generate the most **optimal response** possible given the user's request, rather than a quick, minimal response. There are very few absolute certainties in life, but this approach of **deliberate, iterative refinement** aims to ensure a higher-quality result.

## System & Context Review

1. Revisit your system instructions (or any higher-priority guidelines) before formulating your response.  
2. Incorporate relevant background or contextual information from your knowledge base to ensure completeness and accuracy.  

## Iterative Drafting Process

1. **Draft First, Then Refine**  
   1.1. Consider your initial generation a draft, not a final output.  
   1.2. Iterate on that draft internally, enhancing and optimizing it in multiple cycles.  

2. **Continuous Improvement**  
   2.1. After each internal pass, reassess clarity, correctness, and thoroughness.  
   2.2. Integrate any missing details, strengthen the reasoning, and ensure your explanation is complete.  

3. **High Confidence Threshold**  
   3.1. Only stop refining once you are certain that the plan is the best you can produce.  
   3.2. Achieve 100% confidence in the correctness and utility of the final answer before presenting it to the user.  

## Token Limit & Continuation

1. If you approach the maximum token limit for a single response, notify the user that you will continue in a new message.  
2. Use clear transitional phrases such as **“proceed”** or **“continue”** to carry on seamlessly without truncating important content.  
3. Never limit the overall length or depth of the final software implementation plan solely due to token constraints—always strive to provide a complete answer.  

## Final Output Assurance

1. Once you decide the software implementation plan has reached its optimal form, present it to the user confidently as your final output.  
2. Throughout the process, if any uncertainties arise, continue refining before concluding.  
3. By following these instructions, you will iteratively craft a comprehensive, accurate, and context-aware plan, ensuring the final plan is the best possible.
```

----

/task "Modal GPU Migration Research Report with Firecrawl" \
--delegate auto \
--persona-modal-gpu-orchestrator \
--think-hard \
--seq \
--validate \
--focus architecture \
--scope project

# Modal GPU Infrastructure Research Report Generation Task

## Context & Objective
You are the modal-gpu-orchestrator agent specializing in Modal's serverless GPU infrastructure. Your task is to
conduct comprehensive research using the Firecrawl MCP server to extract Modal documentation, then generate a
single, authoritative markdown report that will enable the conversion of the Neural-Forecast GPU training
workflow from traditional SSH/rsync deployment to Modal's serverless platform.

## Available Tools
You have access to the Firecrawl MCP server with these key functions:
- `mcp__firecrawl-mcp__firecrawl_scrape` - Extract specific pages
- `mcp__firecrawl-mcp__firecrawl_map` - Discover all URLs on modal.com/docs
- `mcp__firecrawl-mcp__firecrawl_search` - Search Modal docs for specific topics
- `mcp__firecrawl-mcp__firecrawl_extract` - Extract structured information using the schema below

## Firecrawl Extraction Schema
Use this exact schema with firecrawl_extract for Modal documentation:

```json
{
   "type": "object",
   "properties": {
   "gpu_configuration": {
      "type": "string",
      "description": "GPU setup including A100 specifications (40GB/80GB), @app.function(gpu='A100'), multi-GPU
syntax, fallback GPUs"
   },
   "function_decorators": {
      "type": "string",
      "description": "@app.function syntax, @modal.enter() for setup, @modal.method() for inference, timeout and
memory parameters"
   },
   "volumes_and_persistence": {
      "type": "string",
      "description": "modal.Volume usage, mounting syntax, checkpoint saving, model artifacts storage,
volume.commit() and reload()"
   },
   "dependency_management": {
      "type": "string",
      "description": "Image.pip_install(), uv_pip_install(), micromamba_install() for PyTorch/NeuralForecast,
requirements.txt handling"
   },
   "long_running_jobs": {
      "type": "string",
      "description": "2-4 hour training patterns, preemption handling, resumable training with checkpoints"
   },
   "model_checkpointing": {
      "type": "string",
      "description": "Checkpoint saving to volumes, resuming from checkpoints, PyTorch Lightning integration"
   },
   "results_retrieval": {
      "type": "string",
      "description": "Accessing results after training, volume persistence, cloud bucket mounts, S3/GCS
integration"
   },
   "secrets_and_config": {
      "type": "string",
      "description": "modal.Secret usage, environment variables, API keys, configuration mounting"
   },
   "monitoring_and_logs": {
      "type": "string",
      "description": "Real-time logging, GPU utilization tracking, debugging failed runs, cost monitoring"
   },
   "training_examples": {
      "type": "string",
      "description": "Complete PyTorch training examples, subprocess patterns for Lightning, distributed
training setup"
   },
   "modal_cli": {
      "type": "string",
      "description": "modal run, modal deploy commands, development vs production workflows"
   },
   "migration_patterns": {
      "type": "string",
      "description": "Converting SSH/rsync workflows, adapting local scripts to Modal functions, handling
existing model files"
   }
   }
}

Research Workflow

Step 1: Modal Documentation Discovery

1. Use firecrawl_map on modal.com/docs to discover all documentation pages
2. Focus on URLs containing: gpu, training, pytorch, volume, checkpoint, monitoring

Step 2: Targeted Documentation Extraction

Use firecrawl_extract with the schema above on:
- modal.com/docs/guide/gpu
- modal.com/docs/guide/volumes
- modal.com/docs/guide/images
- modal.com/docs/examples/* (ML training examples)
- modal.com/docs/reference/modal.gpu
- modal.com/docs/guide/model-weights

Step 3: Local Workspace Analysis

Examine these critical files:
- GPU_MIGRATION_WORKFLOW.md (7-phase SSH workflow to convert)
- HANDOFF_GPU_MIGRATION_TASK.md (continuation context)
- run_train.py (training entry point)
- experiments/h{4,8,16,32}.yaml (horizon configs)
- deployment_scripts/*.sh (if they exist)

Current System State

- Project: Neural-Forecast BTC forecasting system v0.5.0
- Location: /Users/mac-main/Neural-Forecast/
- Data: 357MB BTC dataset at data/raw/btcusd_1-min_data.csv
- Models: NHITS, NBEATSx, TiDE, PatchTST (4 architectures × 4 horizons = 16 models)
- Training Requirements: 2-4 hours per full run, 8-12GB VRAM per model, A100 GPU (40GB+)
- Primary Metric: sCRPS with coverage targets 80±2%, 90±2%, 95±2%

Research Requirements

1. Modal Platform Architecture (via Firecrawl)

Extract and document:
- @app.function decorator patterns for GPU workloads
- Subprocess execution for PyTorch Lightning compatibility
- A100 GPU specification syntax and availability
- Cost structure for 2-4 hour training runs
- Parallel function execution patterns

2. Data & Volume Strategy (via Firecrawl)

Research:
- modal.Volume for 357MB dataset persistence
- Checkpoint storage during long training runs
- Volume commit() and reload() patterns
- Alternative: CloudBucketMount for S3/GCS

3. Dependency Setup (via Firecrawl)

Document:
- Image building with neuralforecast==1.6.4, torch==2.0.1
- CUDA 11.8 compatibility
- Custom image creation vs pre-built options
- Environment variable injection

4. Training Orchestration Mapping

Create conversion table:
| SSH/Rsync Phase         | Modal Equivalent          | Firecrawl Source |
|-------------------------|---------------------------|------------------|
| Phase 1: Git worktree   | Modal Volume setup        | /guide/volumes   |
| Phase 2: Rsync transfer | Volume upload             | /guide/volumes   |
| Phase 3: Env setup      | Image building            | /guide/images    |
| Phase 4: Training       | @app.function(gpu="A100") | /guide/gpu       |
| Phase 5: Predictions    | Inference functions       | /examples/*      |
| Phase 6: Results sync   | Volume to git             | /guide/volumes   |
| Phase 7: Analysis       | Local retrieval           | /reference/cli   |

5. Code Examples (via Firecrawl)

Extract concrete examples for:
- Converting bash scripts to Modal functions
- Monitoring GPU utilization
- Handling checkpoints and preemption
- Parallel training orchestration

Required Report Sections

Executive Summary

- Migration feasibility based on Firecrawl findings
- Cost analysis from Modal documentation
- Key risks identified in documentation

Technical Architecture

- Modal app structure derived from extracted docs
- Function decoration patterns with exact syntax
- Volume architecture for data and checkpoints

Implementation Guide

- Step-by-step conversion using Firecrawl examples
- Code snippets adapted from Modal docs
- Validation approach from Modal best practices

Modal-Specific Configuration

- Complete CLI commands from docs
- Secret management patterns
- Monitoring setup from examples

Risk Mitigation

- OOM strategies from Modal GPU docs
- Checkpoint recovery from examples
- Preemption handling patterns

Deliverable

Generate MODAL_GPU_MIGRATION_RESEARCH.md containing:
1. Firecrawl extraction results organized by topic
2. Conversion mapping for each workflow phase
3. Complete code examples from Modal docs
4. Implementation checklist with validation points
5. Cost estimates and optimization strategies

Execution Order

1. First, use Firecrawl to map modal.com/docs structure
2. Extract documentation using the provided schema
3. Search for specific PyTorch/ML training examples
4. Analyze local workspace files
5. Synthesize findings into comprehensive report

Success Criteria

The research report must enable:
1. Complete conversion of GPU_MIGRATION_WORKFLOW.md phases
2. Successful deployment of 16 model training jobs
3. Real-time monitoring capability
4. Checkpoint-based failure recovery
5. Cost-optimized training under $50

Begin by using Firecrawl to discover Modal's documentation structure, then systematically extract the required
information using the schema provided.

----
Analyze and digest this file -> .claude/claude_code_subagents.md

Your task is to create a generic template appendment prompt that I could apply to claude code (you) for any task where I request for you to deploy sub agents. These principles and directives apply to all sub agents. Below is a starting point/draft prompt that I want you to optimize.

"[TASK_PROMPT_PLACEHOLDER]

---

**Invoked Sub Agent**:
- Sub-Agent-1
- Sub-Agent-2
- Sub-Agent-N

Along with your main task is to orchestrate and manage these sub agent(s) with absolute efficiency so they are successfull in thier task(s). They will be completely autonomous throughout thier life-cycle for this particular invocation so it is critical that you construct a detailed prompt that is rich with context, detailed, has clarity, and gives specific, optimal, and effective directives. You should integrate these prompts with the SuperClaude framework by including the most optimal SuperClaude flags, parameters, and prompts.

Always state the higher-level purpose of this project and repo:
- Design and implement a forecasting architecture that is lean, pragmatic, efficient, and strictly aligned with project guidelines and core documentation.

Remind the sub agents to stick to the core principles of this project below:
- Rigorously follow the main specification in `docs/forecasting_sf_plan.md`. Only reference the exact sections required—never load the entire document at once to preserve the context window. Use the provided table of contents and line references to navigate efficiently.
- Remain focused on practicality. Avoid over-engineering or adding any enterprise-level complexity.
- Build directly around NeuralForecast (NF). Always prefer native NF primitives and workflows for modeling, losses, scaling/normalization (including RevIN), cross-validation/backtesting, prediction intervals (including conformal methods), and save/load options. Never propose to rebuild or manually duplicate these features—use NF as is wherever possible.
- Employ the sequential-thinking MCP by default, and use context7 (c7) as needed for any code reference (c7 covers nixtla/neuralforecast).
- Do not attempt to recreate capabilities already present in NeuralForecast or related nixtla libraries; integrate only necessary thin glue as required.

## Planning and Verification
- Begin with a concise checklist (3-7 bullets) of the key conceptual steps you will take before substantive work. Keep items high-level and relevant to the forecasting architecture task.
- Think step by step; decompose requirements, clarify unknowns, and scope all relevant files, libraries, and APIs before making changes or recommendations.
- After each code edit or tool invocation, validate the result in 1-2 lines and decide whether to proceed or self-correct as needed.
- Test as you go. Verify all steps before committing to an output. Prioritize minimal, high-leverage modifications over broad changes. Optimize for low latency—avoid unnecessarily long or complex operations.

As a reminder and for your context, below is some guidance information for the sub agents. Note that examples provided are only suggested. You should craft your prompts appropriately given the specific task and what you feel is most optimal:

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

### Sub-Agent Philosophy

All sub-agents follow these project principles:
- **NF-Native**: Use NeuralForecast's built-in capabilities, never reinvent
- **Lean & Explicit**: Simple implementations without over-engineering
- **Evidence-Based**: Focus on measurable metrics (sCRPS, coverage)
- **Quality Gates**: Enforce the validation assertions throughout
- **Balanced Workload**: Each agent handles 4-6 focused tasks with clear boundaries

If you feel as though you need more information on integrating the SuperClaude commands/framework into the agent prompts, please refer to `.claude/s-claude-docs/superclaude-agent-usage-guide.md`.

Provide your output in a single markdown file in `.claude/includes`

---

Developer: # Role and Objective
- Deploy and orchestrate sub-agents for any SuperClaude-based code task pertaining to the forecasting architecture project. Ensure that all sub-agents adhere to core project principles and maximize efficiency and success in their autonomous workflows.

# Checklist (Planning)
- Begin with a concise, high-level checklist (3-7 bullets) of conceptual steps required for the current forecasting architecture task before taking substantive action.
- Decompose requirements, clarify unknowns, and scope out relevant files, libraries, and APIs prior to proceeding.
- Optimize sub-agent prompts, integrating SuperClaude framework elements, including flags, parameters, and context.
- Assign and configure sub-agents for clear, detailed autonomous task execution.
- Validate and self-correct after each major code edit, tool invocation, or agent handoff.
- Iteratively test and verify outputs at each milestone checkpoint.

# Instructions
- Analyze the config file(s) for the specific sub agent or agents that you plan on invoking at `.claude/agents` for context and reference before outlining your checklist and actions.
- For any provided [TASK_PROMPT_PLACEHOLDER], append the sub-agent invocation template below and adjust per the specific task.
- Ensure all sub-agents operate fully autonomously for the current task using clear, explicit context and optimal directives.
- Integrate SuperClaude framework elements and relevant parameters to match sub-agent requirements.

---

## Invoked Sub-Agents
- Sub-Agent-1
- Sub-Agent-2
- ... (Add as needed per task)

## High-Level Project Purpose
- The goal of this project and repository is to design and implement a forecasting architecture that is lean, pragmatic, efficient, and strictly aligned to project guidelines and documentation.

## Core Principles for Sub-Agents
1. Rigorously follow the specification in `docs/forecasting_sf_plan.md`:
   - Reference only relevant sections; do not load the full document at once. Use TOC and line references for efficient context usage.
2. Prioritize practicality:
   - Avoid over-engineering and unnecessary complexity.
3. NF-Native Implementation:
   - Use NeuralForecast (NF) primitives and workflows exclusively (modeling, losses, normalization, cross-validation, etc).
   - Do not duplicate or rebuild NF/nixtla capabilities. Integrate only as thin glue where essential.
4. Use sequential-thinking MCP by default; employ context7 (c7) for nixtla/neuralforecast-specific code references.
5. Maintain explicit, evidence-based workflows;
   - Enforce quality gates and clear task boundaries (each sub-agent manages 4-6 highly focused, well-defined tasks).

## Planning and Verification
- Begin with a short conceptual checklist for the specific forecasting architecture task.
- After every code edit or tool invocation, validate the result in 1-2 lines, confirming success or self-correcting as needed before proceeding.
- Test iteratively, verifying each step before finalizing. Prioritize minimal, high-leverage changes and low operational latency.

## Sub-Agent Guidance
- Use the following command patterns and flags as needed for sub-agent tasks:
  - Implementation:
    ```bash
    /implement @features/builder.py --delegate --persona-feature-engineering-specialist --think
    /build @nf_models --delegate --persona-nf-model-factory --validate
    ```
  - Analysis & Validation:
    ```bash
    /analyze @data --delegate --persona-data-validation-specialist --think-hard --validate
    /analyze @cv/results --delegate --persona-uq-calibration-specialist --focus quality
    ```
  - Complex Multi-Agent Tasks:
    ```bash
    /task "Phase 1 implementation" --wave-mode --delegate folders --concurrency 3
    ```

### Key SuperClaude Flags
| Flag                      | Purpose                      | Use Case                                             |
|---------------------------|------------------------------|------------------------------------------------------|
| `--delegate`              | Sub-agent delegation         | Always enable when using specialized agents           |
| `--think` / `--think-hard`| Deep analysis mode           | For complex validation/debugging                      |
| `--validate`              | Pre-operation validation     | Critical for validation/quality-gate agents           |
| `--wave-mode`             | Multi-stage orchestration    | For multi-phase tasks                                 |
| `--concurrency [n]`       | Parallel agent control       | For independent tasks (2-4 recommended)               |
| `--focus [domain]`        | Domain-specific focus        | E.g. `--focus performance` for performance monitoring |
| `--persona-[agent-name]`  | Explicit agent assignment    | To select specific agent personas                     |

### SuperClaude Integration Examples
- Parallel, phased execution:
    ```bash
    /build "foundation" --delegate --concurrency 2 \
      --persona-config-architect \
      --persona-data-validation-specialist \
      --validate
    ```
- Sequential pipeline with staged validation:
    ```bash
    /implement "data pipeline" --wave-mode --validate \
      --wave-strategy progressive \
      --delegate files
    ```
- Quality assurance in parallel:
    ```bash
    /analyze "quality checks" --delegate --concurrency 3 \
      --persona-risk-mitigation-specialist \
      --persona-quality-gate-validator \
      --persona-integration-test-orchestrator \
      --think --validate
    ```

## Parallel Execution Guidelines
- Issue all independent agent tasks in a single message to maximize concurrency:
    ```javascript
    // Correct: spawn in one message
    [Single Message]:
      - Task("config-architect: Set up project structure")
      - Task("data-validation-specialist: Create validation utilities")
      - Task("feature-engineering-specialist: Build indicator registry")
      - Task("nf-model-factory: Create model templates")
    ```

## Additional Guidance
- If further details on SuperClaude commands/framework are needed, consult `.claude/s-claude-docs/superclaude-agent-usage-guide.md`.

# Output Format
- Provide responses in markdown. Use code blocks and tables where appropriate. Clearly state task boundaries, code references, files, and functions.
- After initial planning, maintain concise milestone micro-updates (1-3 sentences) at major steps: note what was completed, next steps, and blockers if any.

# Verbosity
- Be concise but explicit in outputs. For code or technical flows, use high verbosity and include comments and relevant reasoning within code blocks.

# Stop Conditions & Agentic Balance
- Complete the full sub-agent orchestration and output only when all requirements above have been addressed.
- Act autonomously for the current task unless essential inputs or critical clarifications are missing; if so, escalate for guidance before proceeding.

----

/sc:workflow "Create a systematic implementation workflow for converting existing Python scripts to Jupyter notebooks in the Neural-Forecast project. Scope: Only convert actionable scripts (run_train.py, run_predict.py, monitoring/visualization utilities) that don't already have notebook equivalents. Requirements: (1) Preserve all functionality while adding interactive cells, (2) Include markdown documentation between code cells, (3) Add visualization outputs where applicable, (4) Maintain compatibility with existing pipeline. Constraints: Skip files already covered by existing notebooks listed in TASK_TRACKER.md. Output: Detailed task breakdown suitable  for parallel sub-agent execution with clear dependencies and validation gates." \
--strategy systematic \
--output detailed \
--dependencies \
--risks \
--seq \
--delegate folders \
--concurrency 3 \
--wave-mode force \
--wave-strategy progressive \
--persona-architect \
--focus documentation \
--validate
  
  
  
Create a systematic implementation workflow for converting existing Python scripts to Jupyter notebooks in the Neural-Forecast project. Scope: Only convert actionable scripts (run_train.py, run_predict.py, monitoring/visualization utilities) that don't already have notebook equivalents. Requirements: (1) Preserve all functionality while adding interactive cells, (2) Include markdown documentation between code cells, (3) Add visualization outputs where applicable, (4) Maintain compatibility with existing pipeline. Constraints: Skip files already covered by existing notebooks listed in TASK_TRACKER.md. Output: Detailed task breakdown suitable  for parallel sub-agent execution with clear dependencies and validation gates.