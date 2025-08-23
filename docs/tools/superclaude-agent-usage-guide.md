# SuperClaude Agent Usage Guide (Repo-Focused)

> Purpose: Convert any user task or draft prompt into the optimal SuperClaude command + flags for this Neural-Forecast repository while activating the most efficient and optimal specialists for the objective of the prompt in provided below.

## Global Rules for Agents
- Prefer one precise command with targeted flags over generic chatter.
- Use auto-activation to your advantage; override only when necessary.
- Constrain scope (`--scope`, `--focus`) and tokens (`--uc`) for large contexts.
- Increase thinking depth (`--think*`) only when complexity demands it.
- Prioritize validation for model changes (`--validate`, `--preview`, `--with-tests`).
- Enable MCP servers only when they add material value (`--seq`, `--c7`).

## Core Flags (Reference)
- Thinking depth: `--think` (~4K), `--think-hard` (~10K), `--ultrathink` (~32K)
- Efficiency & safety: `--uc` (ultra-compress), `--safe-mode`, `--validate`, `--verbose`, `--answer-only`, `--preview`, `--fix`, `--safe`, `--aggressive`
- MCP servers: `--seq|--sequential`, `--c7|--context7`, `--morph|--morphllm`, `--serena`, `--all-mcp`, `--no-mcp`
- Behavioral modes: `--brainstorm`, `--introspect`, `--task-manage`, `--orchestrate`, `--token-efficient`
- Orchestration: `--delegate [files|folders|auto]`, `--wave-mode [auto|force|off]`, `--loop`, `--concurrency [1-15]`, `--iterations [1-10]`, `--parallel`, `--strategy [systematic|agile|enterprise]`
- Targeting: `--scope [file|module|project|system]`, `--focus [quality|performance|architecture|testing]`
- Command-specific: `--type`, `--format`, `--depth`, `--level`, `--with-tests`, `--documentation`, `--coverage`, `--watch`
- Repo-specific agents: `@agent-{config-architect|cv-runner|data-validation-specialist|feature-engineering-specialist|hpo-optimizer|inference-pipeline|integration-test-orchestrator|modal-gpu-orchestrator|model-selector-ensemble|nf-model-factory|nf-validation-expert|production-monitor|quality-gate-validator|risk-mitigation-specialist|spec-architect|spec-validation-auditor|uq-calibration-specialist}`
- Generic agents (when needed): `@agent-{python-expert|backend-architect|system-architect|performance-engineer|root-cause-analyst|refactoring-expert|technical-writer}`

## Behavioral Mode Auto-Activation
- **Brainstorming**: Vague requests, "maybe", "possibly", exploration keywords → collaborative discovery
- **Introspection**: Error recovery, "analyze reasoning", complex problems → transparent reasoning (🤔, 🎯, 💡)
- **Task Management**: >3 steps, >2 directories, complex dependencies → hierarchical coordination
- **Orchestration**: Multi-tool operations, performance constraints → optimal tool selection
- **Token Efficiency**: Context >75%, large operations, `--uc` flag → 30-50% compression

## MCP Server Auto-Activation
- **context7**: Import statements, NeuralForecast/pandas/numpy keywords, library documentation needs
- **sequential-thinking**: Complex debugging, `--think` flags, architectural analysis, multi-step reasoning
- **morphllm**: Multi-file edits, refactoring, pattern transformations, bulk code changes
- **serena**: Symbol operations, large codebases, session persistence, project memory

## Agent Auto-Activation Triggers (Repo-Specific)
| Domain | Keywords/Patterns | Activated Agents |
|--------|-------------------|------------------|
| **NeuralForecast** | "NF", "NHITS", "NBEATSx", "TiDE", "PatchTST" | @agent-nf-model-factory, @agent-nf-validation-expert |
| **Cross-Validation** | "CV", "cross-validation", "n_windows", "sCRPS" | @agent-cv-runner |
| **Features** | "indicator", "shift(1)", "MTF", "feature engineering" | @agent-feature-engineering-specialist |
| **Data Validation** | "timestamp", "leakage", "canonical", "grid" | @agent-data-validation-specialist |
| **HPO** | "hyperparameter", "tuning", "Auto*", "optimization" | @agent-hpo-optimizer |
| **UQ/Calibration** | "intervals", "coverage", "PIT", "quantiles" | @agent-uq-calibration-specialist |
| **Production** | "inference", "predict", "deploy", "monitor" | @agent-inference-pipeline, @agent-production-monitor |
| **GPU/Modal** | "GPU", "Modal", "A100", "serverless" | @agent-modal-gpu-orchestrator |
| **Quality Gates** | "acceptance", "criteria", "rollback" | @agent-quality-gate-validator |
| **Risk** | "OOM", "quantile crossing", "MTF misalignment" | @agent-risk-mitigation-specialist |
| **Config** | "settings.yaml", "structure", "dependencies" | @agent-config-architect |
| **Testing** | "integration", "E2E", "benchmark" | @agent-integration-test-orchestrator |
| **Ensemble** | "model selection", "ensemble", "equal-weight" | @agent-model-selector-ensemble |
| **Specs** | "design.md", "requirements.md", "tasks.md" | @agent-spec-architect, @agent-spec-validation-auditor |

Notes
- Auto-activation triggers: large context → `--uc`; complex/multi-step → `--think`, `--seq`; libraries/frameworks → `--c7`
- Precedence: safety flags > optimization; explicit flags > auto-detection; `--no-mcp` overrides all MCP
- Flag priority: `--ultrathink` > `--think-hard` > `--think`; `--safe-mode` > `--validate` > standard

---

# Command Reference (Agent-Centric)

Each command below includes: purpose/criteria, applicable flags, helpful combos, 3 example conversions, and notable edge cases.

## /sc:index — Command Discovery
- Purpose: Discover relevant commands for a task; filter by category or search.
- Use when: User is unsure of command; meta-navigation.
- Applicable flags: `--category <cat>`, `--search <term>`; optional `--answer-only`.
- Helpful combos: `--search <term>`, `--category analysis`.
- Examples:
  - User Prompt: "What should I use to diagnose cross-validation issues?"
    Optimized SuperClaude Prompt: /sc:index --search cross-validation
  - User Prompt: "List analysis commands"
    Optimized SuperClaude Prompt: /sc:index --category analysis --answer-only
  - User Prompt: "Find commands about documentation"
    Optimized SuperClaude Prompt: /sc:index --search documentation
- Edge cases: Pure navigation; do not overuse. Prefer direct command if intent is clear.

## /sc:load — Project Context Loading
- Purpose: Load and summarize project structure for orientation.
- Use when: New codebase; before major changes; onboarding.
- Applicable flags: `--deep`, `--focus <area>`, `--summary`; global: `--uc` for large repos.
- Helpful combos: `--deep --summary`, `--focus architecture`.
- Examples:
  - User Prompt: "Understand this repo quickly"
    Optimized SuperClaude Prompt: /sc:load --deep --summary
  - User Prompt: "Map NF modules involved in training"
    Optimized SuperClaude Prompt: /sc:load nf_models/ cv/ uq/ --focus architecture
  - User Prompt: "Summarize dependencies"
    Optimized SuperClaude Prompt: /sc:load . --focus dependencies --summary
- Edge cases: Can be slow on huge repos; prefer scoping and `--uc`.

## /sc:analyze — Code/Project Analysis
- Purpose: Quality, performance, or architecture analysis.
- Use when: Audits, understanding, targeted reviews.
- Applicable flags: `--focus <domain>`, `--depth [quick|deep]`, `--format [text|json|report]`; global: `--think*`, `--uc`, `--seq`, `--c7`, `--scope`, personas.
- Helpful combos: `--focus performance --think --seq`, `--focus architecture --c7`, `--scope module --uc`.
- Examples:
  - User Prompt: "Where can leakage occur in feature pipeline?"
    Optimized SuperClaude Prompt: /sc:analyze features/builder.py utils/ --focus architecture --think --seq @agent-data-validation-specialist @agent-risk-mitigation-specialist
  - User Prompt: "Spot performance bottlenecks in cross-validation runner"
    Optimized SuperClaude Prompt: /sc:analyze cv/runner.py --focus performance --depth deep --seq
  - User Prompt: "Give me a concise report for the whole repo"
    Optimized SuperClaude Prompt: /sc:analyze . --format report --uc --scope project
- Edge cases: Large repos need `--uc` and `--scope`; performance analysis benefits from personas.

## /sc:workflow — Implementation Planning
- Purpose: Turn PRDs/requirements into an actionable workflow.
- Use when: Starting a feature from spec; need sequenced plan.
- Applicable flags: `--strategy [systematic|mvp]`, `--output [roadmap|tasks|detailed]`, `--risks`, `--dependencies`, `--persona <role>`, MCP: `--c7`, `--sequential`.
- Helpful combos: `--strategy systematic --risks --dependencies --sequential`.
- Examples:
  - User Prompt: "Plan feature builder with compute→align→shift(1)"
    Optimized SuperClaude Prompt: /sc:workflow features/builder.py --strategy systematic --output detailed --sequential @agent-feature-engineering-specialist
  - User Prompt: "Create CV plan for h∈{4,8,16,32}"
    Optimized SuperClaude Prompt: /sc:workflow cv/ --strategy systematic --risks --dependencies --sequential @agent-cv-runner
  - User Prompt: "Design uncertainty calibration workflow"
    Optimized SuperClaude Prompt: /sc:workflow uq/ --strategy mvp --output roadmap @agent-uq-calibration-specialist
- Edge cases: Planning only; use /sc:implement for code.

## /sc:implement — Feature/Module/API Implementation
- Purpose: Implement features/services/modules with expert activation.
- Use when: Creating new code or expanding capability.
- Applicable flags: `--type [api|service|feature|module|pipeline]`, `--framework <name>`, `--safe`, `--iterative`, `--with-tests`, `--documentation`; global: personas, `--validate`.
- Helpful combos: `--type api --safe --with-tests`, `--type module --framework pandas --documentation`.
- Examples:
  - User Prompt: "Build login feature with tests"
    Optimized SuperClaude Prompt: /sc:implement "user authentication" --type feature --with-tests --safe
  - User Prompt: "Create feature engineering pipeline"
    Optimized SuperClaude Prompt: /sc:implement "feature builder" --type module --framework pandas --with-tests @agent-feature-engineering-specialist
  - User Prompt: "Add orders REST API"
    Optimized SuperClaude Prompt: /sc:implement "orders" --type api --iterative --validate
- Edge cases: Specify `--type`/`--framework` for precision; prefer `--safe` in production.

## /sc:build — Build/Bundle/Package
- Purpose: Build or package projects; prep for deployment.
- Use when: Running builds, debugging build failures, optimizing bundles.
- Applicable flags: `--type [dev|prod|test]`, `--clean`, `--optimize`, `--verbose`; global: `--uc` for logs.
- Helpful combos: `--type prod --optimize`, `--clean --verbose`.
- Examples:
  - User Prompt: "Create a production build"
    Optimized SuperClaude Prompt: /sc:build --type prod --optimize
  - User Prompt: "Rebuild from scratch with logs"
    Optimized SuperClaude Prompt: /sc:build --clean --verbose
  - User Prompt: "Build only models folder"
    Optimized SuperClaude Prompt: /sc:build nf_models/ @agent-nf-model-factory
- Edge cases: Custom build systems may require manual hints; ensure tools in PATH.

## /sc:design — System/API/Architecture Design
- Purpose: Produce designs/specs for systems, APIs, DBs, pipelines.
- Use when: Planning; formalizing interfaces; upfront architecture.
- Applicable flags: `--type [architecture|api|pipeline|database]`, `--format [diagram|spec|code]`, `--iterative`; global: `--c7`, personas.
- Helpful combos: `--type api --format spec`, `--type database --iterative`.
- Examples:
  - User Prompt: "Design user management API spec"
    Optimized SuperClaude Prompt: /sc:design user-management --type api --format spec
  - User Prompt: "Plan ecommerce DB schema"
    Optimized SuperClaude Prompt: /sc:design ecommerce --type database --format diagram
  - User Prompt: "Outline auth system architecture"
    Optimized SuperClaude Prompt: /sc:design auth-system --type architecture --format spec --c7
- Edge cases: Conceptual; not code-generation.

## /sc:troubleshoot — Systematic Debugging
- Purpose: Investigate and resolve issues methodically.
- Use when: Unknown failures; confusing errors; performance incidents.
- Applicable flags: `--logs <file>`, `--systematic`, `--focus [network|database|pipeline]`; global: `--think`, `--seq`, `--validate`.
- Helpful combos: `--systematic --think --seq`, `--logs server.log`.
- Examples:
  - User Prompt: "API returning 500s"
    Optimized SuperClaude Prompt: /sc:troubleshoot "API returning 500" --logs server.log --systematic
  - User Prompt: "Slow queries in DB layer"
    Optimized SuperClaude Prompt: /sc:troubleshoot "slow queries" --focus database --think
  - User Prompt: "Build keeps failing intermittently"
    Optimized SuperClaude Prompt: /sc:troubleshoot "intermittent build failure" --systematic --seq
- Edge cases: Provide specific logs/errors if available for better results.

## /sc:explain — Educational Explanations
- Purpose: Explain code/concepts at appropriate depth.
- Use when: Learning, onboarding, documentation-by-explanation.
- Applicable flags: `--beginner`, `--advanced`, `--code <file>`, `--examples`; global: `--answer-only`, `--verbose`, personas.
- Helpful combos: `--beginner --examples`, `--code <file> --advanced`.
- Examples:
  - User Prompt: "Explain async/await simply"
    Optimized SuperClaude Prompt: /sc:explain "async/await" --beginner --examples
  - User Prompt: "Explain this file at expert level"
    Optimized SuperClaude Prompt: /sc:explain --code src/utils.js --advanced
  - User Prompt: "Teach pandas groupby with examples"
    Optimized SuperClaude Prompt: /sc:explain "pandas groupby patterns" --examples --verbose
- Edge cases: Works best with concrete targets (file/topic).

## /sc:improve — Code Enhancement
- Purpose: Refactor, modernize, optimize, raise quality.
- Use when: Cleanup/refactors; perf or maintainability pushes.
- Applicable flags: `--type [quality|performance|maintainability|style]`, `--safe`, `--preview`; global: `--loop`, `--validate`.
- Helpful combos: `--type performance --safe`, `--preview` then apply.
- Examples:
  - User Prompt: "Safely improve API performance"
    Optimized SuperClaude Prompt: /sc:improve src/api/ --type performance --safe --validate
  - User Prompt: "Preview cleanup in legacy module"
    Optimized SuperClaude Prompt: /sc:improve utils/legacy.py --preview
  - User Prompt: "Style pass across project"
    Optimized SuperClaude Prompt: /sc:improve . --type style --safe
- Edge cases: Prefer `--preview` first; run on smaller scopes.

## /sc:cleanup — Technical Debt Reduction
- Purpose: Remove dead code, fix imports, reorganize files.
- Use when: Clutter, unused code, pre-refactor housekeeping.
- Applicable flags: `--dead-code`, `--imports`, `--files`, `--safe`; global: `--validate`.
- Helpful combos: `--dead-code --safe`, `--imports <path>`.
- Examples:
  - User Prompt: "Purge unused code in utils"
    Optimized SuperClaude Prompt: /sc:cleanup src/utils/ --dead-code --safe
  - User Prompt: "Fix messy imports in utils"
    Optimized SuperClaude Prompt: /sc:cleanup utils/ --imports
  - User Prompt: "Tidy file layout safely"
    Optimized SuperClaude Prompt: /sc:cleanup . --files --safe --validate
- Edge cases: Can be aggressive; always safe/validate on critical paths.

## /sc:test — Testing & QA
- Purpose: Run tests, check coverage, improve test quality.
- Use when: QA passes; regression checks; coverage work.
- Applicable flags: `--type [unit|integration|e2e|all]`, `--coverage`, `--watch`, `--fix`; MCP: `--play` for browser tests; personas.
- Helpful combos: `--type unit --coverage`, `--type e2e --play`.
- Examples:
  - User Prompt: "Run unit tests with coverage"
    Optimized SuperClaude Prompt: /sc:test --type unit --coverage
  - User Prompt: "Watch tests while editing features"
    Optimized SuperClaude Prompt: /sc:test --watch features/
  - User Prompt: "Fix failing cross-validation tests"
    Optimized SuperClaude Prompt: /sc:test --type integration --fix --coverage
- Edge cases: Requires configured test framework; review `--fix` changes.

## /sc:document — Documentation Generation
- Purpose: Generate API docs, READMEs, inline docs, guides.
- Use when: Missing docs; release prep; API reference.
- Applicable flags: `--type [inline|external|api|guide]`, `--style [brief|detailed]`, `--template`; personas.
- Helpful combos: `--type api --style detailed`, `--type guide --template <name>`.
- Examples:
  - User Prompt: "Create API docs for controllers"
    Optimized SuperClaude Prompt: /sc:document src/controllers/ --type api --style detailed
  - User Prompt: "Brief README for onboarding"
    Optimized SuperClaude Prompt: /sc:document README --style brief --type guide
  - User Prompt: "Inline docs for helpers"
    Optimized SuperClaude Prompt: /sc:document src/utils/helpers.js --type inline
- Edge cases: Quality depends on code structure and context.

## /sc:estimate — Estimation & Complexity
- Purpose: Time/effort/complexity estimation.
- Use when: Planning sprints; scoping features.
- Applicable flags: `--detailed`, `--complexity`, `--team-size <n>`.
- Helpful combos: `--detailed --complexity --team-size 3`.
- Examples:
  - User Prompt: "Estimate auth feature"
    Optimized SuperClaude Prompt: /sc:estimate "add user authentication" --detailed
  - User Prompt: "Complexity for dashboard"
    Optimized SuperClaude Prompt: /sc:estimate user-dashboard --complexity
  - User Prompt: "Team-aware migration estimate"
    Optimized SuperClaude Prompt: /sc:estimate "migrate to microservices" --detailed --team-size 3
- Edge cases: Estimates are rough; use as planning inputs only.

## /sc:task — Long-Running Feature Management
- Purpose: Create/track/breakdown multi-session tasks.
- Use when: Multi-day features, progress tracking, decomposition.
- Applicable flags: verbs `create|status|breakdown`, `--priority [high|medium|low]`.
- Helpful combos: `create --priority high`, `breakdown <feature>`.
- Examples:
  - User Prompt: "Track dashboard feature"
    Optimized SuperClaude Prompt: /sc:task create "implement user dashboard" --priority high
  - User Prompt: "Show current status"
    Optimized SuperClaude Prompt: /sc:task status
  - User Prompt: "Break down checkout flow"
    Optimized SuperClaude Prompt: /sc:task breakdown "e-commerce checkout flow"
- Edge cases: Persistence may be unreliable; treat as planning aid.

## /sc:spawn — Complex Workflow Orchestration
- Purpose: Coordinate multi-step/parallel operations, meta-orchestration for large projects.
- Use when: CI/CD, migrations, staged pipelines, large-scale projects.
- Applicable flags: `--parallel`, `--sequential`, `--monitor`, `--strategy [sequential|parallel|adaptive]`, `--depth [normal|deep]`; global: `--delegate` for sub-tasks.
- Helpful combos: `--parallel --monitor`, `--sequential` for migrations, `--strategy adaptive --depth deep`.
- Examples:
  - User Prompt: "Run deploy pipeline with monitoring"
    Optimized SuperClaude Prompt: /sc:spawn deploy-pipeline --monitor
  - User Prompt: "Parallel test and staging deploy"
    Optimized SuperClaude Prompt: /sc:spawn "test and deploy to staging" --parallel
  - User Prompt: "Sequence DB migration"
    Optimized SuperClaude Prompt: /sc:spawn database-migration --sequential
- Edge cases: Works best with well-defined workflows; expect iteration.

## /sc:git — Enhanced Git Operations
- Purpose: Smart commits, branching, merges with better messaging.
- Use when: Git workflows and hygiene.
- Applicable flags: `--smart-commit`, `--branch-strategy`, `--interactive`; personas.
- Helpful combos: `--smart-commit --branch-strategy`.
- Examples:
  - User Prompt: "Create a smart commit"
    Optimized SuperClaude Prompt: /sc:git --smart-commit "fix login bug"
  - User Prompt: "Create a feature branch"
    Optimized SuperClaude Prompt: /sc:git branch feature/user-dashboard --branch-strategy
  - User Prompt: "Interactive merge with review"
    Optimized SuperClaude Prompt: /sc:git merge develop --interactive
- Edge cases: Review generated messages; assumes standard workflows.

## /sc:save — Session Persistence
- Purpose: Save current session state to persistent memory via Serena MCP.
- Use when: Checkpointing work, preserving decisions, ending sessions.
- Applicable flags: `--consolidate` (merge memories), session description as argument.
- Helpful combos: Use with `/sc:reflect` before saving.
- Examples:
  - User Prompt: "Save my progress on the auth system"
    Optimized SuperClaude Prompt: /sc:save "authentication system phase 1 complete"
  - User Prompt: "Checkpoint before risky changes"
    Optimized SuperClaude Prompt: /sc:save "pre-refactor checkpoint"
  - User Prompt: "Store architectural decisions"
    Optimized SuperClaude Prompt: /sc:save "microservices boundaries defined"
- Edge cases: Requires Serena MCP; descriptive names improve retrieval.

## /sc:reflect — Progress Assessment
- Purpose: Analyze progress against goals and validate session completeness.
- Use when: Checking completion, assessing progress, session end.
- Applicable flags: `--type [task|session|completion]`, `--scope [project|session]`, `--analyze`, `--validate`, `--cleanup`, `--repair`.
- Helpful combos: `--type completion --validate`, `--scope project --analyze`.
- Examples:
  - User Prompt: "Am I done with this feature?"
    Optimized SuperClaude Prompt: /sc:reflect --type completion --validate
  - User Prompt: "Check project progress"
    Optimized SuperClaude Prompt: /sc:reflect --scope project --analyze
  - User Prompt: "Clean up old memories"
    Optimized SuperClaude Prompt: /sc:reflect --cleanup
- Edge cases: Works with Serena MCP memories; manual assessment without.

## /sc:select-tool — Tool Optimization
- Purpose: Analyze and optimize tool selection for operations.
- Use when: Performance optimization, tool selection questions.
- Applicable flags: `--analyze`, `--explain`.
- Helpful combos: `--analyze --explain` for detailed analysis.
- Examples:
  - User Prompt: "What's the best tool for this operation?"
    Optimized SuperClaude Prompt: /sc:select-tool "operation description" --explain
  - User Prompt: "Optimize tool usage"
    Optimized SuperClaude Prompt: /sc:select-tool --analyze
  - User Prompt: "Why use this tool?"
    Optimized SuperClaude Prompt: /sc:select-tool "tool name" --explain
- Edge cases: Advisory only; actual selection happens automatically.

## /sc:brainstorm — Requirements Discovery
- Purpose: Interactive requirements discovery through Socratic dialogue.
- Use when: Vague ideas, new projects, unclear requirements.
- Applicable flags: `--strategy [systematic|creative]`; activates brainstorming mode.
- Helpful combos: Often followed by `/sc:workflow` or `/sc:design`.
- Examples:
  - User Prompt: "I want to build something for productivity"
    Optimized SuperClaude Prompt: /sc:brainstorm "productivity app" --strategy creative
  - User Prompt: "Not sure what auth system to use"
    Optimized SuperClaude Prompt: /sc:brainstorm "authentication requirements"
  - User Prompt: "Explore payment options"
    Optimized SuperClaude Prompt: /sc:brainstorm "payment processing" --strategy systematic
- Edge cases: Shifts to discovery questions; may need multiple rounds.

## /sc:build vs /sc:implement — When to Choose Which
- Use /sc:implement for creating code (features/modules/services).
- Use /sc:build for compiling/packaging and build troubleshooting.

## /sc:document vs /sc:explain — When to Choose Which
- Use /sc:document for artifacts meant to ship (READMEs, API docs).
- Use /sc:explain for learning or clarifying concepts.

## /sc:design vs /sc:workflow — When to Choose Which
- Use /sc:design for system/API specs and architecture.
- Use /sc:workflow to turn specs/PRDs into stepwise implementation plans.

---

# Specialized Aids (Repo-Specific Agents)

- Repo agents auto-activate based on keywords; use `@agent-*` for explicit control
- High-yield repo agent combos:
  - NF validation: `@agent-nf-validation-expert @agent-nf-model-factory --c7`
  - Feature pipeline: `@agent-data-validation-specialist @agent-feature-engineering-specialist @agent-risk-mitigation-specialist`
  - Model training: `@agent-cv-runner @agent-uq-calibration-specialist @agent-hpo-optimizer`
  - Production: `@agent-modal-gpu-orchestrator @agent-inference-pipeline @agent-production-monitor`
  - Quality assurance: `@agent-quality-gate-validator @agent-integration-test-orchestrator`
- Precedence/conflicts: `--no-mcp` > MCP flags; safety > optimization; last persona wins.

## Experimental/Implicit Commands (seen in examples)
- `/sc:scan` (code audit): Prefer `/sc:analyze --focus quality --validate`.
- `/sc:review` (review/QA): Prefer `/sc:analyze --focus quality --persona-qa`.
- `/sc:deploy` (deployment): Prefer `/sc:spawn deploy-pipeline` or `/sc:build --type prod` + DevOps workflow.

---

# Quick Selection Matrix (Task → Command)
- Understand codebase → `/sc:load` → optionally `/sc:analyze`
- Find issues/perf/quality → `/sc:analyze` with `--focus`
- Vague requirements → `/sc:brainstorm` → `/sc:workflow`
- Plan from PRD → `/sc:workflow`
- Implement feature/API/module → `/sc:implement`
- Build/package/debug build → `/sc:build`
- System/API design → `/sc:design`
- Debug issues → `/sc:troubleshoot`
- Improve/refactor/cleanup → `/sc:improve` or `/sc:cleanup`
- Run tests/coverage → `/sc:test`
- Write docs → `/sc:document`
- Estimate complexity → `/sc:estimate`
- Long-running planning → `/sc:task`
- Orchestrate multi-step ops → `/sc:spawn`
- Git workflows → `/sc:git`
- Save progress → `/sc:save`
- Check completion → `/sc:reflect`
- Optimize tools → `/sc:select-tool`
- Find commands → `/sc:index`

---

# Advanced Workflow Patterns

## Complex Project Patterns
- **ML Pipeline Development**: `/sc:brainstorm` → `/sc:design` → `/sc:implement` → `/sc:test` → `/sc:document`
  - Flags: `--think --seq --c7 --with-tests --documentation`
  - Agents: @agent-system-architect + @agent-python-expert + @agent-backend-architect
  
- **Legacy Migration**: `/sc:load` → `/sc:analyze --focus architecture` → `/sc:workflow` → `/sc:improve`
  - Flags: `--ultrathink --seq --safe-mode --preview --validate`
  - Agents: @agent-refactoring-expert + @agent-system-architect
  
- **Performance Investigation**: `/sc:troubleshoot` → `/sc:analyze --focus performance` → `/sc:improve`
  - Flags: `--think-hard --seq --systematic --focus performance`
  - Agents: @agent-performance-engineer + @agent-root-cause-analyst

## Session Management Patterns
- **New Project**: `/sc:brainstorm` → `/sc:save "requirements"` → `/sc:workflow`
- **Resume Work**: `/sc:load project` → `/sc:reflect` → continue implementation
- **Checkpoint**: `/sc:reflect --validate` → `/sc:save "milestone description"`
- **Long Sessions**: Regular `/sc:save` every 30min or major milestone

## Multi-Agent Coordination (Repo-Specific)
- **Model Development**: @agent-nf-validation-expert + @agent-nf-model-factory + @agent-cv-runner + @agent-uq-calibration-specialist
- **Feature Pipeline**: @agent-data-validation-specialist + @agent-feature-engineering-specialist + @agent-risk-mitigation-specialist
- **Production Deploy**: @agent-modal-gpu-orchestrator + @agent-inference-pipeline + @agent-production-monitor + @agent-quality-gate-validator
- **HPO & Selection**: @agent-hpo-optimizer + @agent-cv-runner + @agent-model-selector-ensemble
- **Quality Assurance**: @agent-integration-test-orchestrator + @agent-quality-gate-validator + @agent-risk-mitigation-specialist
- **Specification**: @agent-spec-architect + @agent-spec-validation-auditor + @agent-nf-validation-expert

## Flag Combination Strategies
- **Model Validation**: `--validate --preview --think --with-tests`
- **Deep Analysis**: `--ultrathink --seq --c7`
- **Large Codebases**: `--uc --scope module --delegate auto --concurrency 5`
- **Quick Overview**: `--scope file --focus quality --answer-only`
- **Learning Mode**: `--introspect --verbose --examples`

## Common Task Optimizations

### For Neural-Forecast Repository
- **Feature Analysis**: `/sc:analyze features/ --focus architecture --think @agent-feature-engineering-specialist @agent-data-validation-specialist`
- **CV Implementation**: `/sc:implement cv/ --type module --seq --with-tests @agent-cv-runner`
- **Model Factory**: `/sc:design nf_models/ --type architecture @agent-nf-model-factory @agent-nf-validation-expert`
- **Leakage Check**: `/sc:analyze features/builder.py --focus architecture --think @agent-risk-mitigation-specialist`
- **GPU Deployment**: `/sc:implement "Modal training" @agent-modal-gpu-orchestrator`
- **Model Selection**: `/sc:analyze "model performance" @agent-model-selector-ensemble @agent-cv-runner`
- **Production Setup**: `/sc:implement "inference pipeline" @agent-inference-pipeline @agent-production-monitor`

### Context-Based Selection
- **High Complexity** (>5 components): Add `--ultrathink --seq --delegate`
- **Production Models**: Always add `--validate --with-tests`
- **Learning/Documentation**: Add `--verbose --examples`
- **Quick Fixes**: Use `--scope file --uc`
- **Exploration**: Start with `--brainstorm` mode

# Coverage Check
- Commands covered (21): index, load, analyze, workflow, implement, build, design, troubleshoot, explain, improve, cleanup, test, document, estimate, task, spawn, git, save, reflect, select-tool, brainstorm
- Behavioral modes (5): brainstorming, introspection, task-management, orchestration, token-efficiency
- MCP servers (4): context7, sequential-thinking, morphllm, serena
- Repo-specific agents (17): config-architect, cv-runner, data-validation-specialist, feature-engineering-specialist, hpo-optimizer, inference-pipeline, integration-test-orchestrator, modal-gpu-orchestrator, model-selector-ensemble, nf-model-factory, nf-validation-expert, production-monitor, quality-gate-validator, risk-mitigation-specialist, spec-architect, spec-validation-auditor, uq-calibration-specialist
- Generic agents (7): python-expert, backend-architect, system-architect, performance-engineer, root-cause-analyst, refactoring-expert, technical-writer
- Flags mapped: All thinking depth, efficiency/safety, MCP, orchestration, targeting, and command-specific flags with precedence rules
- Advanced patterns: Session management, multi-agent coordination, workflow combinations