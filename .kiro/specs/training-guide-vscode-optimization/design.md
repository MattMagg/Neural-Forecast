# Design Document

## Overview

The Training Guide Optimization design creates a clear, practical training guide for the fully-implemented BTC forecasting system. The design leverages existing components (setup.sh, kaggle_download_btc.py, run_train.ipynb, YAML configs) to provide a sequential workflow with comprehensive local validation. The focus is on creating actionable documentation that minimizes GPU debugging time through thorough pre-deployment validation.

## Architecture

### Core Components

#### 1. Sequential Workflow Orchestrator
- **Step Sequencer**: Defines the exact order: setup → data → training → validation
- **Dependency Tracker**: Maps step dependencies and identifies parallelization opportunities
- **Progress Monitor**: Tracks completion status and provides time estimates
- **Recovery Manager**: Handles step failures and provides continuation procedures

#### 2. Local Validation Engine
- **Import Validator**: Tests all module imports before GPU deployment
- **Configuration Validator**: Validates YAML configs and parameter ranges
- **Data Pipeline Tester**: Runs data processing on small samples locally
- **Model Instantiation Tester**: Verifies models can be created without GPU

#### 3. Training Execution Guide
- **Notebook Cell Guide**: Cell-by-cell execution instructions for run_train.ipynb
- **Configuration Manager**: Horizon selection and parameter modification guidance
- **Progress Monitor**: GPU utilization and training progress tracking
- **Artifact Validator**: Verification of generated files and results

#### 4. Results Assessment Framework
- **Metrics Interpreter**: Explains sCRPS, coverage, and quality thresholds
- **Artifact Checker**: Validates expected output files and their contents
- **Quality Assessor**: Determines if models meet acceptance criteria
- **Report Generator**: Creates comprehensive training reports

## Components and Interfaces

### Sequential Workflow Orchestrator

```python
class SequentialWorkflowOrchestrator:
    """Orchestrates the complete training workflow."""
    
    def define_workflow_steps(self) -> List[WorkflowStep]:
        """Defines the sequential steps: setup → data → training → validation."""
        
    def validate_step_dependencies(self) -> DependencyMap:
        """Maps dependencies between workflow steps."""
        
    def estimate_step_duration(self, step: WorkflowStep) -> TimeEstimate:
        """Provides realistic time estimates for each step."""
        
    def handle_step_failure(self, step: WorkflowStep, error: Exception) -> RecoveryProcedure:
        """Provides recovery procedures for failed steps."""
```

### Local Validation Engine

```python
class LocalValidationEngine:
    """Comprehensive local validation before GPU deployment."""
    
    def validate_imports(self) -> ImportValidationResult:
        """Tests all module imports: utils.io, features.builder, nf_models.factory, cv.runner."""
        
    def validate_configurations(self) -> ConfigValidationResult:
        """Validates YAML configs can be loaded and parameters are valid."""
        
    def test_data_pipeline(self, sample_size: int = 10000) -> DataPipelineResult:
        """Tests data processing on small sample."""
        
    def test_model_instantiation(self) -> ModelInstantiationResult:
        """Verifies models can be created without GPU."""
        
    def generate_validation_report(self) -> LocalValidationReport:
        """Creates comprehensive pre-deployment validation report."""
```

### Training Execution Guide

```python
class TrainingExecutionGuide:
    """Guides execution of run_train.ipynb notebook."""
    
    def provide_cell_guidance(self, cell_index: int) -> CellExecutionGuide:
        """Provides guidance for each notebook cell."""
        
    def configure_horizon(self, horizon: str) -> ConfigurationGuide:
        """Explains how to set horizon (h4/h8/h16/h32) in configuration cells."""
        
    def monitor_training_progress(self) -> ProgressMonitoringGuide:
        """Documents GPU monitoring and training progress indicators."""
        
    def validate_artifacts(self, horizon: str) -> ArtifactValidationResult:
        """Validates expected output files in experiments/h{horizon}/."""
```

### Results Assessment Framework

```python
class ResultsAssessmentFramework:
    """Assesses training results and quality."""
    
    def interpret_metrics(self, metrics_file: str) -> MetricsInterpretation:
        """Explains sCRPS scores, coverage percentages, and thresholds."""
        
    def check_artifacts(self, experiment_dir: str) -> ArtifactCheckResult:
        """Validates expected files: cv_results.parquet, metrics.json, leaderboard.csv."""
        
    def assess_model_quality(self, results: Dict) -> QualityAssessment:
        """Determines if models meet acceptance criteria."""
        
    def generate_training_report(self, horizon: str) -> TrainingReport:
        """Creates comprehensive training completion report."""
```

## Data Models

### Workflow Step Structure

```python
@dataclass
class WorkflowStep:
    """Represents a single step in the training workflow."""
    name: str
    description: str
    command: str
    expected_duration: timedelta
    dependencies: List[str]
    validation_criteria: List[str]
    recovery_procedures: List[str]
```

### Local Validation Result

```python
@dataclass
class LocalValidationResult:
    """Results of local pre-GPU validation."""
    imports_valid: bool
    configs_valid: bool
    data_pipeline_valid: bool
    models_instantiable: bool
    issues_found: List[ValidationIssue]
    recommendations: List[str]
    gpu_ready: bool
```

### Training Configuration

```python
@dataclass
class TrainingConfiguration:
    """Training configuration for specific horizon."""
    horizon: str  # h4, h8, h16, h32
    config_file: str
    models: List[str]
    batch_sizes: Dict[str, int]
    expected_duration: timedelta
    memory_requirements: Dict[str, str]
```

## Error Handling

### Environment Setup Errors
- **Setup Script Failures**: Diagnostic commands and recovery procedures for setup.sh failures
- **Dependency Issues**: Resolution of Python, CUDA, and TA-Lib installation problems
- **Virtual Environment Problems**: Activation and package installation troubleshooting
- **GPU Access Issues**: CUDA driver and device accessibility validation

### Data Acquisition Errors
- **Kaggle Download Failures**: Network issues, authentication problems, and retry procedures
- **Data Validation Errors**: File corruption, incomplete downloads, and re-download procedures
- **Data Format Issues**: CSV parsing errors and data structure validation
- **Storage Problems**: Disk space issues and file permission problems

### Training Execution Errors
- **Import Errors**: Module path issues and environment activation problems
- **Configuration Errors**: YAML syntax errors and parameter validation failures
- **Memory Errors**: GPU OOM issues and batch size optimization
- **Training Failures**: Model convergence issues and checkpoint recovery

## Testing Strategy

### Local Validation Tests

#### 1. Import Validation
```python
def test_all_imports():
    """Tests all critical imports work correctly."""
    modules = [
        'utils.io', 'utils.validate',
        'features.builder', 'features.registry',
        'nf_models.factory', 'cv.runner',
        'uq.metrics', 'uq.calibration'
    ]
    for module in modules:
        assert_import_works(module)
```

#### 2. Configuration Validation
```python
def test_yaml_configs():
    """Validates all YAML configuration files."""
    configs = ['h4.yaml', 'h8.yaml', 'h16.yaml', 'h32.yaml']
    for config in configs:
        assert_yaml_valid(f'experiments/{config}')
        assert_parameters_valid(config)
```

#### 3. Data Pipeline Testing
```python
def test_data_pipeline_sample():
    """Tests data processing on small sample."""
    sample_data = load_sample_data(10000)
    processed = run_data_pipeline(sample_data)
    assert_data_quality(processed)
```

### Integration Testing

#### 1. End-to-End Workflow
- Complete workflow execution from setup to results
- Step dependency validation and error recovery
- Time estimation accuracy and performance optimization

#### 2. GPU Environment Simulation
- Local validation accuracy vs. GPU environment behavior
- Memory usage prediction and optimization
- Training time estimation and monitoring

## Implementation Phases

### Phase 1: Workflow Definition
1. **Step Sequencing**: Define exact order of operations
2. **Dependency Mapping**: Identify step dependencies and parallelization opportunities
3. **Time Estimation**: Provide realistic duration estimates for each step
4. **Recovery Procedures**: Define error handling and continuation procedures

### Phase 2: Local Validation Framework
1. **Import Testing**: Create comprehensive import validation
2. **Configuration Testing**: Implement YAML validation and parameter checking
3. **Data Pipeline Testing**: Create local data processing tests
4. **Model Testing**: Implement model instantiation validation

### Phase 3: Training Guide Creation
1. **Sequential Instructions**: Create step-by-step training procedures
2. **Cell-by-Cell Guidance**: Document run_train.ipynb execution
3. **Configuration Management**: Explain horizon selection and parameter modification
4. **Progress Monitoring**: Document GPU monitoring and progress tracking

### Phase 4: Results Assessment
1. **Metrics Interpretation**: Create guidance for understanding results
2. **Artifact Validation**: Implement output file checking
3. **Quality Assessment**: Define acceptance criteria and validation procedures
4. **Report Generation**: Create comprehensive training reports

### Phase 5: Documentation and Testing
1. **Guide Creation**: Write comprehensive training guide
2. **Example Validation**: Test all procedures and commands
3. **Integration Testing**: End-to-end workflow validation
4. **Quality Assurance**: Final validation and optimization

## Quality Assurance

### Validation Criteria
- All workflow steps execute successfully in sequence
- Local validation accurately predicts GPU environment behavior
- Training guide provides clear, actionable instructions
- Results assessment correctly identifies successful training
- Error recovery procedures handle common failure modes

### Success Metrics
- 100% local validation accuracy for preventable issues
- Complete workflow documentation with no gaps
- Realistic time estimates within ±20% accuracy
- Comprehensive error handling for all common failure modes
- Significant reduction in GPU debugging time

## Risk Mitigation

### Workflow Execution Risks
- **Step Dependencies**: Clear documentation of prerequisites and dependencies
- **Error Recovery**: Comprehensive procedures for handling step failures
- **Time Management**: Realistic estimates and progress monitoring
- **Resource Management**: GPU memory optimization and utilization monitoring

### Local Validation Risks
- **False Negatives**: Ensuring local tests catch all preventable issues
- **Environment Differences**: Accounting for Mac M4 Pro vs. A100XL differences
- **Configuration Drift**: Keeping local validation in sync with GPU requirements
- **Test Coverage**: Comprehensive coverage of all critical components

### Training Execution Risks
- **Memory Management**: A100XL-specific optimization and monitoring
- **Configuration Errors**: Validation of YAML configs and parameter ranges
- **Data Quality**: Comprehensive data validation and error handling
- **Model Performance**: Quality assessment and acceptance criteria

This design provides a practical framework for creating a comprehensive training guide that leverages existing components while providing thorough local validation to minimize GPU resource waste.