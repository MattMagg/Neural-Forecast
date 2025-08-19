# Neural-Forecast GPU Migration Task - Session Handoff Prompt

**Copy and paste the text below into a new Claude Code session to continue this task:**

---

## CONTINUATION PROMPT FOR NEW SESSION

I need to continue the Neural-Forecast GPU migration and training workflow from a previous session. Please load the context and help me proceed with the next steps.

### 1. **Task Summary**
**Main Objective**: Migrate the Neural-Forecast BTC forecasting system to an A100 GPU instance for full-scale training, run cross-validation on all 4 horizons (h=4,8,16,32), and sync results back to local machine.

**Current Status**: 
- ✅ Core system implemented (v0.5.0) with 49/69 tasks complete
- ✅ Comprehensive GPU migration workflow created in `GPU_MIGRATION_WORKFLOW.md`
- ⏸️ READY TO EXECUTE: Awaiting execution of Phase 1 (Repository Preparation)
- 📍 Decision made to migrate to A100 NOW rather than complete remaining specs

### 2. **Critical References**
- **Project Version**: v0.5.0 (last commit: 9f7edb5)
- **Branch**: implementation-phase-1
- **Sequential Thinking Context**: 
  - Last thought #67 in Conclusion stage
  - Workflow generation completed with 5-thought synthesis
  - Tags: ["workflow", "gpu-migration", "training", "results-sync"]
- **Workflow Document**: `/Users/mac-main/Neural-Forecast/GPU_MIGRATION_WORKFLOW.md`
- **Key Specifications Completed**:
  - data-processing-validation (5/5 tasks)
  - feature-engineering-pipeline (8/8 tasks)
  - neuralforecast-model-factory (16/16 tasks)
  - cross-validation-metrics (20/20 tasks)

### 3. **Working Environment**
**Active Files**:
- `GPU_MIGRATION_WORKFLOW.md` - Master workflow document (7 phases)
- `run_train.py` - Training pipeline entry point
- `experiments/h{4,8,16,32}.yaml` - Horizon configurations
- `data/raw/btcusd_1-min_data.csv` - 357MB BTC data file

**Project Structure**:
```
/Users/mac-main/Neural-Forecast/
├── cv/                    # Cross-validation system (COMPLETE)
├── uq/                    # Uncertainty quantification (COMPLETE)
├── utils/                 # Utilities and validation (COMPLETE)
├── features/              # Feature engineering (COMPLETE)
├── nf_models/            # Model factory (COMPLETE)
├── experiments/          # Configs and results
├── data/
│   └── raw/
│       └── btcusd_1-min_data.csv (357MB)
├── deployment_scripts/   # TO BE CREATED
└── GPU_MIGRATION_WORKFLOW.md
```

### 4. **Progress Checkpoint**
**Completed**:
- ✅ Full CV and metrics system with NF-native implementation
- ✅ sCRPS computation, coverage diagnostics, PIT analysis
- ✅ Model factory supporting NHITS, NBEATSx, TiDE, PatchTST
- ✅ Feature engineering with 13 indicators, MTF, 256 cap
- ✅ Complete training pipeline integration
- ✅ GPU migration workflow document created

**Remaining for GPU Migration**:
- [ ] Phase 1: Create git worktree for results management
- [ ] Phase 2: Transfer code and data to A100 (~357MB)
- [ ] Phase 3: Set up Python environment on A100
- [ ] Phase 4: Run training on all 4 horizons (2-4 hours)
- [ ] Phase 5: Generate predictions
- [ ] Phase 6: Sync results back via git
- [ ] Phase 7: Analyze results locally

### 5. **Decisions & Approach**
**Key Decisions Made**:
1. **Move to A100 NOW** - 71% of system complete is sufficient for valuable training
2. **Use git worktree** for results management (not branches) - cleaner separation
3. **Parallel training** for all 4 horizons to maximize GPU utilization
4. **Skip remaining specs** (HPO, monitoring, etc.) until after baseline results
5. **Use rsync** for efficient file transfer, excluding artifacts
6. **Results in separate branch** (`gpu-results`) to avoid code conflicts

**Technical Approach**:
- NF-native only (no custom CV loops)
- Primary metric: sCRPS (lower is better)
- Coverage targets: 80±2%, 90±2%, 95±2%
- Training config: n_windows=6, step_size=h, val_size=4*h, refit=1

### 6. **Next Session Instructions**

**IMMEDIATE NEXT STEPS**:

1. **Verify workflow document exists**:
```bash
cat GPU_MIGRATION_WORKFLOW.md | head -50
```

2. **Start Phase 1 - Repository Preparation**:
```bash
# Create git worktree for results
git branch gpu-results
git worktree add ../Neural-Forecast-Results gpu-results

# Create deployment scripts directory
mkdir -p deployment_scripts
```

3. **Create monitoring scripts** (from workflow Phase 1.3):
```bash
# Copy monitoring script from GPU_MIGRATION_WORKFLOW.md section 1.3
# Copy orchestration script from section 1.4
```

4. **Set A100 connection variables**:
```bash
export A100_HOST="[YOUR_A100_HOST]"  # Fill this in
export A100_PATH="/home/ubuntu/Neural-Forecast"
```

5. **Execute Phase 2 - Transfer** (when ready):
```bash
# Test connection first
ssh ${A100_HOST} "nvidia-smi --query-gpu=name,memory.total --format=csv"

# Then run rsync transfer
rsync -avzP --exclude='*.parquet' --exclude='.git/' ./ ${A100_HOST}:${A100_PATH}/
```

**VALIDATION CHECKPOINTS**:
- Confirm 357MB data file exists: `ls -lh data/raw/btcusd_1-min_data.csv`
- Verify git worktree created: `git worktree list`
- Check A100 has CUDA: `ssh ${A100_HOST} "nvidia-smi"`

**IF RESUMING MID-WORKFLOW**:
Check which phase was last completed by looking for:
- `deployment_scripts/` directory → Phase 1 done
- Files on A100 → Phase 2 done
- `venv/` on A100 → Phase 3 done
- `logs/training_*.log` → Phase 4 in progress
- `experiments/h*/predictions*.parquet` → Phase 5 done

**CRITICAL CONTEXT**:
- Training will take 2-4 hours on A100 for all horizons
- Expect ~8-12GB VRAM usage per model
- Results will be ~500MB-1GB total
- Use tmux/screen for persistent sessions on A100

**For questions or to see the full workflow, read**: `GPU_MIGRATION_WORKFLOW.md`

---

END OF HANDOFF PROMPT