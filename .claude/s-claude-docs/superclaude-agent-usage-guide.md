# SuperClaude Agent Usage Guide (Repo-Focused)

> Purpose: Convert any user task or draft prompt into the optimal SuperClaude command + flags for this Neural-Forecast repository, minimizing tokens while activating the right specialists.

## Global Rules for Agents
- Prefer one precise command with targeted flags over generic chatter.
- Use auto-activation to your advantage; override only when necessary.
- Constrain scope (`--scope`, `--focus`) and tokens (`--uc`) for large contexts.
- Increase thinking depth (`--think*`) only when complexity demands it.
- Prioritize safety for risky changes (`--safe-mode`, `--validate`, `--preview`).
- Enable MCP servers only when they add material value (`--seq`, `--c7`).

## Core Flags (Reference)
- Thinking depth: `--think` (~4K), `--think-hard` (~10K), `--ultrathink` (~32K)
- Efficiency & safety: `--uc` (ultra-compress), `--safe-mode`, `--validate`, `--verbose`, `--answer-only`
- MCP servers used here: `--seq|--sequential`, `--c7|--context7`
- Orchestration: `--delegate [files|folders|auto]`, `--wave-mode [auto|force|off]`, `--loop`, `--concurrency N`
- Targeting: `--scope [file|module|project|system]`, `--focus [quality|performance|architecture|testing]`
- Personas (relevant): `--persona-{architect|backend|performance|analyzer|qa|refactorer|devops|mentor|scribe}`

Notes
- Auto-activation triggers: large context → `--uc`; complex/multi-step → `--think`, `--seq`; libraries/frameworks → `--c7`.
- Precedence: safety flags take priority; last persona wins; thinking depth order: ultrathink > think-hard > think.

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
- Purpose: Quality, security, performance, or architecture analysis.
- Use when: Audits, understanding, targeted reviews.
- Applicable flags: `--focus <domain>`, `--depth [quick|deep]`, `--format [text|json|report]`; global: `--think*`, `--uc`, `--seq`, `--c7`, `--scope`, personas.
- Helpful combos: `--focus performance --think --seq`, `--focus architecture --c7`, `--scope module --uc`.
- Examples:
  - User Prompt: "Where can leakage occur in feature pipeline?"
    Optimized SuperClaude Prompt: /sc:analyze features/builder.py utils/ --focus architecture --think --seq --c7
  - User Prompt: "Spot performance bottlenecks in cross-validation runner"
    Optimized SuperClaude Prompt: /sc:analyze cv/runner.py --focus performance --depth deep --seq
  - User Prompt: "Give me a concise report for the whole repo"
    Optimized SuperClaude Prompt: /sc:analyze . --format report --uc --scope project
- Edge cases: Large repos need `--uc` and `--scope`; security and perf benefit from personas.

## /sc:workflow — Implementation Planning
- Purpose: Turn PRDs/requirements into an actionable workflow.
- Use when: Starting a feature from spec; need sequenced plan.
- Applicable flags: `--strategy [systematic|mvp]`, `--output [roadmap|tasks|detailed]`, `--risks`, `--dependencies`, `--persona <role>`, MCP: `--c7`, `--sequential`.
- Helpful combos: `--strategy systematic --risks --dependencies --sequential`.
- Examples:
  - User Prompt: "Plan feature builder with compute→align→shift(1)"
    Optimized SuperClaude Prompt: /sc:workflow features/builder.py --strategy systematic --output detailed --sequential
  - User Prompt: "Create CV plan for h∈{4,8,16,32}"
    Optimized SuperClaude Prompt: /sc:workflow cv/ --strategy systematic --risks --dependencies --sequential
  - User Prompt: "Design uncertainty calibration workflow"
    Optimized SuperClaude Prompt: /sc:workflow uq/ --strategy mvp --output roadmap --c7
- Edge cases: Planning only; use /sc:implement for code.

## /sc:implement — Feature/Component/API Implementation
- Purpose: Implement features/services/components with expert activation.
- Use when: Creating new code or expanding capability.
- Applicable flags: `--type [component|api|service|feature|module]`, `--framework <name>`, `--safe`, `--iterative`, `--with-tests`, `--documentation`; global: personas, `--validate`.
- Helpful combos: `--type api --safe --with-tests`, `--type component --framework react --documentation`.
- Examples:
  - User Prompt: "Build login feature with tests"
    Optimized SuperClaude Prompt: /sc:implement "user authentication" --type feature --with-tests --safe
  - User Prompt: "Create a React dashboard component"
    Optimized SuperClaude Prompt: /sc:implement "dashboard widget" --type component --framework react --documentation
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
  - User Prompt: "Build only components folder"
    Optimized SuperClaude Prompt: /sc:build src/components
- Edge cases: Custom build systems may require manual hints; ensure tools in PATH.

## /sc:design — System/API/Component Design
- Purpose: Produce designs/specs for systems, APIs, DBs, components.
- Use when: Planning; formalizing interfaces; upfront architecture.
- Applicable flags: `--type [architecture|api|component|database]`, `--format [diagram|spec|code]`, `--iterative`; global: `--c7`, personas.
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
- Applicable flags: `--logs <file>`, `--systematic`, `--focus [network|database|frontend]`; global: `--think`, `--seq`, `--validate`.
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
  - User Prompt: "Teach React context with examples"
    Optimized SuperClaude Prompt: /sc:explain "React context patterns" --examples --verbose
- Edge cases: Works best with concrete targets (file/topic).

## /sc:improve — Code Enhancement
- Purpose: Refactor, modernize, optimize, raise quality.
- Use when: Cleanup/refactors; perf or maintainability pushes.
- Applicable flags: `--type [quality|performance|maintainability|style]`, `--safe`, `--preview`; global: `--loop`, `--validate`.
- Helpful combos: `--type performance --safe`, `--preview` then apply.
- Examples:
  - User Prompt: "Safely improve API performance"
    Optimized SuperClaude Prompt: /sc:improve src/api/ --type performance --safe --validate
  - User Prompt: "Preview cleanup in legacy component"
    Optimized SuperClaude Prompt: /sc:improve src/components/Legacy.js --preview
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
  - User Prompt: "Fix messy imports in components"
    Optimized SuperClaude Prompt: /sc:cleanup src/components/ --imports
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
  - User Prompt: "Watch tests while editing src/components"
    Optimized SuperClaude Prompt: /sc:test --watch src/components/
  - User Prompt: "Try to auto-fix failing e2e tests"
    Optimized SuperClaude Prompt: /sc:test --type e2e --fix --play
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
- Purpose: Coordinate multi-step/parallel operations.
- Use when: CI/CD, migrations, staged pipelines.
- Applicable flags: `--parallel`, `--sequential`, `--monitor`; global: `--delegate` for sub-tasks.
- Helpful combos: `--parallel --monitor`, `--sequential` for migrations.
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

## /sc:build vs /sc:implement — When to Choose Which
- Use /sc:implement for creating code (features/components/services).
- Use /sc:build for compiling/packaging and build troubleshooting.

## /sc:document vs /sc:explain — When to Choose Which
- Use /sc:document for artifacts meant to ship (READMEs, API docs).
- Use /sc:explain for learning or clarifying concepts.

## /sc:design vs /sc:workflow — When to Choose Which
- Use /sc:design for system/API specs and architecture.
- Use /sc:workflow to turn specs/PRDs into stepwise implementation plans.

---

# Specialized Aids (from Flags/Personas Guides)

- Personas auto-activate; override with `--persona-*` only to change perspective.
- High-yield global combos:
  - Safe changes: `--safe-mode --validate --preview`
  - Deep analysis: `--think --seq --c7`
  - Large repos: `--delegate auto --uc --scope module`
  - Security work: `--persona-security --focus security --validate`
  - Performance work: `--persona-performance --focus performance --play`
- Precedence/conflicts: `--no-mcp` > MCP flags; safety > optimization; last persona wins.

## Experimental/Implicit Commands (seen in examples)
- `/sc:scan` (security audit): Prefer `/sc:analyze --focus security --persona-security --validate`.
- `/sc:review` (review/QA): Prefer `/sc:analyze --focus quality --persona-qa`.
- `/sc:deploy` (deployment): Prefer `/sc:spawn deploy-pipeline` or `/sc:build --type prod` + DevOps workflow.

---

# Quick Selection Matrix (Task → Command)
- Understand codebase → `/sc:load` → optionally `/sc:analyze`.
- Find issues/security/perf → `/sc:analyze` with `--focus`.
- Plan from PRD → `/sc:workflow`.
- Implement feature/API/component → `/sc:implement`.
- Build/package/debug build → `/sc:build`.
- System/API design → `/sc:design`.
- Debug issues → `/sc:troubleshoot`.
- Improve/refactor/cleanup → `/sc:improve` or `/sc:cleanup`.
- Run tests/coverage → `/sc:test`.
- Write docs → `/sc:document`.
- Estimate complexity → `/sc:estimate`.
- Long-running planning → `/sc:task`.
- Orchestrate multi-step ops → `/sc:spawn`.
- Git workflows → `/sc:git`.
- Find commands → `/sc:index`.

---

# Coverage Check
- Sources included: commands-guide.md, flags-guide.md, personas-guide.md.
- Commands covered (17): index, load, analyze, workflow, implement, build, design, troubleshoot, explain, improve, cleanup, test, document, estimate, task, spawn, git.
- Flags mapped: thinking depth, efficiency/safety, MCP servers, orchestration, focus/scope, personas; precedence and auto-activation rules captured.
- Examples: ≥3 per command with optimized formulations.