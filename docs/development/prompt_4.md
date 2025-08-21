/sc:workflow "Create a systematic implementation workflow for converting existing Python scripts to Jupyter notebooks in the Neural-Forecast project. Scope: Only convert actionable scripts (run_train.py, run_predict.py, monitoring/visualization utilities) that don't already have notebook equivalents. Requirements: (1) Preserve all functionality while adding interactive cells, (2) Include markdown documentation between code cells, (3) Add visualization outputs where applicable, (4) Maintain compatibility with existing pipeline. Constraints: Skip files already covered by existing notebooks listed in TASK_TRACKER.md.
Output: Detailed task breakdown suitable  for parallel sub-agent execution with clear dependencies and validation gates." \
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

---

/sc:analyze "research modal -> https://modal.com/docs/guide on a higher level, gathering context of all the features and different avenues of approach for using modal. This is because I want you you to provide a high-level suggestion of how you think you would approach using modal for the training runs and prediictions for the models based on the existing notebooks for the models. Provide your output inline in a brief sequence of specific steps (specific to modal) that is mostly high-level. For the research, first use firecrawl mcp then utilize context7 to validate. You output doesn't need to be detailed, only make sure you cover each modal topic and setup for the optimal path" --focus infrastructure --depth deep --think-hard --seq --c7 --delegate auto

---

Now compare your approach to the document here -> docs/infrastructure/GPU_MIGRATION_WORKFLOW.md

Is it the same? Is there one better than the other? Do the plans call for the same features? etc.

Setup the gpu instance through modal specifically for 

------

/superclaude-command-opt "Analyze the current project thoroughly and completley to gather information on the current state of the project. The entire project is based on the core document here -> docs/forecasting_sf_plan.md - the task tracker document acts as the changelog and versioning for this project. It is located here -> TASK_TRACKER.md - the last completed tasks was creating notebooks for the script files in order to conduct the training using the gpu instance through the Modal platform. Another minor change was making sure all that all the packages/libraries/dependencies referenced and used in this project are using the latest versions, and updating them if they are not.

Your task is primarily to analyze this project as a whole and to validate that EVERYTHING is good to go in terms of executing the first step, which would be data processing and then using the processed data to commence training on the a100 instance through the notebooks.

There should be no ambiguity or questions to be left unanswered. The current state of the project/repo should be solid as we have spent a lot of time slowly and meticulously building the foundation through sequential steps, all to avoid problems when it comes to execution, and you will validate this or document what needs work or needs to be done in order to make this happen.

Think hard about how you will execute this specific task and plan before execution.

Output a report in a single markdown file in docs/development."

---

/sc:analyze "Developer: Begin with a concise checklist (3-7 bullets) outlining your evaluation approach before performing a comprehensive analysis of the current project state for the data processing and training phases as specified in the project documentation and tracker. The project is structured around the core document at `docs/forecasting_sf_plan.md`, with `TASK_TRACKER.md` serving as both the changelog and the versioning record.

Recent tasks have included creating notebooks for training scripts and updating relevant packages, libraries, and dependencies for GPU compatibility via Modal.

Your objective is to assess whether the project is ready to initiate data processing and subsequent GPU-based training (on the A100 instance). Confirm all preparatory steps are in place, validate the robustness of the project's foundation, and explicitly document any gaps or incomplete areas requiring attention prior to execution.

After each discovery or review step, validate findings in 1-2 lines and revise your approach if necessary. If required documents or resources are missing, state these explicitly in the appropriate section.

Following your analysis, generate a readiness report summarizing key findings and actionable recommendations.

## Output Format
Generate a single Markdown file named `readiness_report.md` to be saved in `docs/development/`.

Structure the report using the following sections:

1. **Project Summary**
   - Concise overview highlighting project objectives and scope.
2. **Document & Resource Status**
   - Evaluation of the existence and completeness of key documents (e.g., `docs/forecasting_sf_plan.md`, `TASK_TRACKER.md`).
   - Clearly identify any missing or incomplete items.
3. **Readiness Checklist**
   - Use a table or checklist with checkboxes to review:
     - Code structure and organization
     - Data integrity and availability
     - Installation and versioning of packages, libraries, and dependencies
     - Hardware setup and compatibility (A100 instance, Modal platform)
     - Training notebook configuration
   - Assign a status to each item: Complete, Partial, or Incomplete.
4. **Issues and Recommendations**
   - List any detected errors, gaps, or risks, with succinct explanations.
   - Provide actionable recommendations for all partial or incomplete areas.
5. **Next Steps**
   - Define a clear set of necessary actions to achieve full project readiness, if applicable.

Use well-formatted tables, checklists, and headings for clarity and reviewability by both humans and automated systems." --focus architecture --depth deep --format report --ultrathink --seq --c7 --validate --scope project --wave-mode force --wave-strategy systematic --delegate auto --concurrency 4 --persona-analyzer --persona-qa --persona-architect

---

Use sequential-thinking for this task - Developer: Begin with a concise checklist (3-7 bullets) outlining your evaluation approach before performing a comprehensive analysis of the current project state for the data processing and training phases as specified in the project documentation and tracker. The project is structured around the core document at `docs/forecasting_sf_plan.md`, with `TASK_TRACKER.md` serving as both the changelog and the versioning record.

Recent tasks have included creating notebooks for training scripts and updating relevant packages, libraries, and dependencies for GPU compatibility via Modal.

Your objective is to assess whether the project is ready to initiate data processing and subsequent GPU-based training (on the A100 instance). Confirm all preparatory steps are in place, validate the robustness of the project's foundation, and explicitly document any gaps or incomplete areas requiring attention prior to execution.

After each discovery or review step, validate findings in 1-2 lines and revise your approach if necessary. If required documents or resources are missing, state these explicitly in the appropriate section.

Following your analysis, generate a readiness report summarizing key findings and actionable recommendations.

## Output Format
Generate a single Markdown file named `execution_readiness_report.md` to be saved in `docs/development/`.

Structure the report using the following sections:

1. **Project Summary**
   - Concise overview highlighting project objectives and scope.
2. **Document & Resource Status**
   - Evaluation of the existence and completeness of key documents (e.g., `docs/forecasting_sf_plan.md`, `TASK_TRACKER.md`).
   - Clearly identify any missing or incomplete items.
3. **Readiness Checklist**
   - Use a table or checklist with checkboxes to review:
     - Code structure and organization
     - Data integrity and availability
     - Installation and versioning of packages, libraries, and dependencies
     - Hardware setup and compatibility (A100 instance, Modal platform)
     - Training notebook configuration
   - Assign a status to each item: Complete, Partial, or Incomplete.
4. **Issues and Recommendations**
   - List any detected errors, gaps, or risks, with succinct explanations.
   - Provide actionable recommendations for all partial or incomplete areas.
5. **Next Steps**
   - Define a clear set of necessary actions to achieve full project readiness, if applicable.

Use well-formatted tables, checklists, and headings for clarity and reviewability by both humans and automated systems.

**DO NOT REGARD OR READ THE OTHER REPORT HERE -> `/Users/mac-main/Neural-Forecast/docs/development/readiness_report.md` - IT IS UNRELATED TO THIS!**


---

```

```


/sc:analyze "Independent, NF‑Native Validation of Modal A100 Integration and Execution Readin
ess

Context
You are validating a NeuralForecast‑centric BTC forecasting project that must tr
ain and infer on a 15‑minute UTC grid with calibrated probabilistic outputs (80/
90/95). The repo’s core specification lives at docs/forecasting_sf_plan.md, with
 TASK_TRACKER.md serving as the changelog/versioning record. Training and predic
tions will be executed from notebooks, and GPU runs are expected on Modal using 
an A100 instance.

Your goal is to independently assess whether the project is ready to initiate da
ta processing and GPU‑based training on Modal with an A100, and to identify any 
remaining operational gaps. Remain objective and avoid assumptions or bias. Do n
ot modify code; only analyze and report.

Instructions
- Use sequential thinking. After each major discovery step, validate your findin
g in 1–2 lines and adapt your next check as needed.
- Use Context7 to consult official Modal documentation as needed (focus on GPU r
esource selection, function/class configuration, image build, volumes, and memor
y snapshot lifecycle). Keep this usage minimal and targeted to confirm semantics
.
- Only reference the exact sections required from docs/forecasting_sf_plan.md; d
o not load the entire document. Follow its NF‑native guardrails strictly (cross_
validation, scalers/normalization, conformal intervals, save/load).
- Evaluate notebooks first-class (training and prediction) since execution will 
occur there. Do not rely solely on scripts.
- Maintain neutrality. Do not pre-suppose issues; verify them.

Scope of Review (high-level, non-prescriptive)
1) Repository readiness vs. spec
   - Structure matches spec; presence of required components (features, nf_model
s, cv, uq, utils, experiments, docs).
   - Existence/completeness of core documents and tracker.

2) Data and contracts
   - Availability of raw data and canonical frame pathing.
   - Presence and use of validators for regular grid, UTC EOB, no forward‑fill, 
and leakage checks.

3) Feature pipeline discipline
   - Indicator stack computed on a 15‑minute base, multi‑timeframe alignment ont
o the base grid, and shift(1) for historic exogenous features.
   - Feature selection and capping are enforced.

4) NF‑native modeling and CV
ilistic; NF cross_validation used with correct windowing semantics and PIs; opti
onal conformal via NF primitives.
   - Save/load uses NF’s native mechanisms; artifacts layout is stable.

5) Modal + A100 integration (objective checks)
   - GPU resource selection semantics are valid and current for A100.
   - Image definition and dependency installation are adequate for GPU training 
and the feature stack (including any required system‑level libraries).
   - Volumes and file paths align with the project’s data and artifact flows.
   - Concurrency controls and scheduling are configured using supported Modal AP
Is.
   - Memory snapshot lifecycle is correct with respect to GPU visibility and ini
tialization order.
   - Notebooks include (or can include) a neutral GPU preflight to assert visibi
lity without coupling to specific hardware assumptions.

6) Versioning and environment stability
   - Dependency versions are pinned or otherwise enforceable for reproducibility
.
   - Training configurations exist for required horizons; settings are centrally
 defined and non‑secret.

Deliverable
Produce a single Markdown report named execution_readiness_report.md with the fo
llowing sections:
1. Project Summary: concise overview of objectives and scope.
2. Document & Resource Status: existence/completeness of key docs and resources.
3. Readiness Checklist: table with checkboxes and status (Complete/Partial/Incom
plete) for:
   - Code structure and organization
   - Data integrity and availability
   - Packages/libraries/dependencies (including any system‑level needs)
   - Hardware setup and Modal compatibility (A100 instance)
   - Training notebook configuration
4. Issues and Recommendations: list any detected risks or gaps with succinct, ac
tionable guidance (keep generic to avoid bias).
5. Next Steps: minimal, ordered actions required for full readiness.

Constraints
- Stay NeuralForecast‑native; do not propose re‑implementing core NF capabilitie
s.
- Remain neutral and non‑prescriptive; avoid naming specific internal functions 
or lines unless necessary to explain a finding.
- If any required file or resource is missing, state this plainly.

Output
- Save the report to docs/development/execution_readiness_report.md (or return t
he full Markdown content ready to be saved if file IO is not permitted).
- Keep the tone direct and critical, with concise justifications and clearly lab
eled statuses." \
   --think-hard \
   --seq \
   --c7 \
   --validate \
   --scope project \
   --wave-mode auto \
   --wave-strategy systematic \
   --delegate folders \
   --concurrency 5

   ---

/sc:spawn "MODAL_GPU_INTEGRATION_WORKFLOW.md" \
   --sequential \
   --wave-mode force \
   --wave-strategy systematic \
   --validate \
   --safe-mode \
   --focus infrastructure \
   --delegate auto \
   --concurrency 4 \
   --seq \
   --c7 \
   --think-hard