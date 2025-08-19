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