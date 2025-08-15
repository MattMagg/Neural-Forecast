# Requirements Document

## Introduction

The NeuralForecast Model Factory spec implements the exact model portfolio and configuration system specified in `docs/forecasting_sf_plan.md` Section 4. This component creates a lean, NF-native model instantiation system that supports the four-model portfolio (NHITS, NBEATSx, TiDE, PatchTST) with proper loss configuration, exogenous variable wiring, and YAML-driven experiment configuration. The implementation follows the "keep it lean and explicit" principle with no enterprise-grade abstractions, using NeuralForecast's native APIs directly for model creation, training, and persistence.

## Requirements

### Requirement 1: Implement Exact Model Portfolio from Source Document

**User Story:** As a model instantiation system, I want to implement the exact four-model portfolio specified in the source document (NHITS, NBEATSx, TiDE, PatchTST), so that I leverage proven architectures with complementary inductive biases for BTC intraday forecasting.

#### Acceptance Criteria

1. WHEN creating the model portfolio THEN the system SHALL support exactly four models: NHITS, NBEATSx, TiDE, PatchTST as specified in Section 4.0
2. WHEN justifying model selection THEN the system SHALL implement models that accept exogenous lists via hist_exog_list, futr_exog_list, stat_exog_list as documented in NF signatures
3. WHEN configuring NHITS THEN the system SHALL use it for multiscale patterns and long horizons via hierarchical interpolation with clean exog integration
4. WHEN configuring NBEATSx THEN the system SHALL use it for interpretable trend/seasonality blocks plus generic blocks with explicit exogenous projections
5. WHEN configuring TiDE THEN the system SHALL use it as a dense encoder/decoder that treats windows as tabular input with strong exogenous handling
6. WHEN configuring PatchTST THEN the system SHALL use it for long-range dependencies with patch-based Transformer architecture and native RevIN support
7. IF models beyond the specified four are added THEN the system SHALL fail validation to prevent over-engineering

### Requirement 2: Implement Exact Loss Configuration from Source Document

**User Story:** As a probabilistic forecasting system, I want to implement the exact loss functions specified in the source document (DistributionLoss StudentT, MQLoss, IQLoss), so that I support both distributional and quantile training approaches with proper sCRPS evaluation.

#### Acceptance Criteria

1. WHEN implementing loss functions THEN the system SHALL support exactly three loss types: DistributionLoss("StudentT"), MQLoss, IQLoss as specified in Section 4.1
2. WHEN using DistributionLoss THEN the system SHALL configure DistributionLoss(distribution="StudentT") for heavy-tailed intraday returns as specified
3. WHEN using MQLoss THEN the system SHALL configure MQLoss with symmetric confidence levels (e.g., [80, 90, 95]) for direct quantile estimation
4. WHEN using IQLoss THEN the system SHALL configure IQLoss as fallback for quantile crossing issues with same confidence levels
5. WHEN implementing _loss_ctor function THEN the system SHALL use the exact function signature: _loss_ctor(spec: Dict[str, Any]) with kind-based selection
6. WHEN selecting losses THEN the system SHALL follow the protocol: start with StudentT and parallel MQLoss, select by mean sCRPS across CV windows
7. IF custom loss implementations are added THEN the system SHALL fail validation to ensure NF-native approach

### Requirement 3: Implement Exact Common Parameters from Source Document

**User Story:** As a model configuration system, I want to implement the exact common parameters specified in the source document, so that I use proven defaults for input_size, scaler_type, learning_rate, batch_size, and training settings.

#### Acceptance Criteria

1. WHEN setting input_size THEN the system SHALL use 1024 for NHITS/NBEATSx/TiDE and 2048 for PatchTST as specified in Section 4.1
2. WHEN setting scaler_type THEN the system SHALL use "robust" as default and "revin" for PatchTST as specified in the global defaults table
3. WHEN setting learning_rate THEN the system SHALL use 1e-3 default with option to tune to 5e-4 if unstable as specified
4. WHEN setting batch_size THEN the system SHALL use 512 default with tuning range 256-1024 based on GPU memory as specified
5. WHEN setting max_steps THEN the system SHALL use 20,000 as cap with early_stop_patience_steps=400 as specified
6. WHEN setting validation THEN the system SHALL use val_check_steps=100 for periodic validation during training as specified
7. WHEN configuring exogenous lists THEN the system SHALL accept hist_exog_list, futr_exog_list, stat_exog_list from feature engineering pipeline
8. IF parameters deviate from the source specification THEN the system SHALL fail validation to ensure consistency

### Requirement 4: Implement Exact Model-Specific Configurations from Source Document

**User Story:** As a model-specific configuration system, I want to implement the exact model-specific parameters specified in the source document, so that I optimize each model architecture according to its documented strengths.

#### Acceptance Criteria

1. WHEN configuring NHITS THEN the system SHALL use n_blocks=[1,1,1] and n_pool_kernel_size=[2,2,1] with dropout_prob_theta=0.1 as specified in Section 4.2
2. WHEN configuring NBEATSx THEN the system SHALL use stack_types=['identity','trend','seasonality'] with n_blocks=[1,1,1] and mlp_units=[[512,512],[512,512],[512,512]] as specified
3. WHEN configuring TiDE THEN the system SHALL use hidden_size=512, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1 as specified
4. WHEN configuring PatchTST THEN the system SHALL use patch_len=16, stride=16, n_heads=8, hidden_size=512, revin=True as specified
5. WHEN handling PatchTST RevIN THEN the system SHALL use both revin=True model flag and scaler_type="revin" as documented
6. WHEN setting model aliases THEN the system SHALL use stable naming pattern like "NHITS_t1024_StudentT" for output column consistency
7. IF model-specific parameters deviate from source THEN the system SHALL fail validation to ensure proven configurations

### Requirement 5: Implement Exact Factory Function from Source Document

**User Story:** As a model instantiation interface, I want to implement the exact instantiate_models function specified in the source document, so that I create NF model instances with proper exogenous wiring and configuration.

#### Acceptance Criteria

1. WHEN implementing instantiate_models THEN the system SHALL use the exact function signature: instantiate_models(cfg: dict, hist_cols: List[str], futr_cols: List[str], stat_cols: List[str])
2. WHEN processing configuration THEN the system SHALL extract h, models_cfg, and scaler_overrides from cfg as specified in Section 4.2.A
3. WHEN wiring exogenous variables THEN the system SHALL set hist_exog_list=hist_cols, futr_exog_list=futr_cols, stat_exog_list=stat_cols for all models
4. WHEN handling scaler overrides THEN the system SHALL support per-model scaler_type overrides with fallback to default as specified
5. WHEN instantiating models THEN the system SHALL use exact class mapping: {"NHITS": NHITS, "NBEATSx": NBEATSx, "TiDE": TiDE, "PatchTST": PatchTST}
6. WHEN pruning kwargs THEN the system SHALL implement _prune_kwargs function to drop unsupported constructor arguments using inspect.signature
7. IF the factory function deviates from source implementation THEN the system SHALL fail validation to ensure NF compatibility

### Requirement 6: Implement Exact YAML Configuration Integration from Source Document

**User Story:** As a configuration management system, I want to implement the exact YAML-driven experiment configuration specified in the source document, so that I support horizon-specific model instantiation with proper parameter inheritance.

#### Acceptance Criteria

1. WHEN defining YAML structure THEN the system SHALL support the exact format from Section 4.2.B: seed, freq, h, scaler_type, n_windows, step_size, val_size, refit, models
2. WHEN configuring global scalers THEN the system SHALL support scaler_type with default and per-model overrides (e.g., PatchTST: revin)
3. WHEN defining model entries THEN the system SHALL use the exact format: - ModelName: {alias, input_size, loss, learning_rate, batch_size, model-specific params}
4. WHEN handling loss specifications THEN the system SHALL support loss: {kind: studentt|mqloss|iqloss, level: [...]} format
5. WHEN setting model aliases THEN the system SHALL use descriptive names like "NHITS_t1024_T" for stable column naming in outputs
6. WHEN configuring training parameters THEN the system SHALL include early_stop_patience_steps, max_steps, val_check_steps in YAML
7. IF YAML structure deviates from source specification THEN the system SHALL fail validation to ensure consistency

### Requirement 7: Implement Exact Integration Points from Source Document

**User Story:** As a system integration component, I want to implement the exact integration points specified in the source document, so that I work seamlessly with existing data processing and feature engineering components.

#### Acceptance Criteria

1. WHEN integrating with feature engineering THEN the system SHALL accept feature lists from features/builder.py select_features function
2. WHEN integrating with run_train.py THEN the system SHALL use the exact code pattern from Section 4.2.C: instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
3. WHEN creating NeuralForecast instance THEN the system SHALL use NeuralForecast(models=models, freq=cfg["freq"]) as specified
4. WHEN supporting horizon-specific instantiation THEN the system SHALL handle h ∈ {4,8,16,32} from experiment configuration
5. WHEN providing model aliases THEN the system SHALL ensure stable column naming for downstream evaluation and reporting
6. WHEN handling exogenous lists THEN the system SHALL validate that all models support the three exog list types as documented
7. IF integration points deviate from source specification THEN the system SHALL fail validation to ensure seamless operation

### Requirement 8: Implement Exact Sanity Checks from Source Document

**User Story:** As a validation system, I want to implement the exact sanity checks and gotchas specified in the source document, so that I catch configuration errors early and provide clear diagnostic information.

#### Acceptance Criteria

1. WHEN validating exog wire-up THEN the system SHALL verify that all models support hist_exog_list, futr_exog_list, stat_exog_list as documented in NF signatures
2. WHEN handling RevIN configuration THEN the system SHALL support both revin=True model flag and scaler_type="revin" with proper precedence
3. WHEN validating probabilistic outputs THEN the system SHALL ensure distributional models support predict(level=[80,90,95]) for interval generation
4. WHEN checking model compatibility THEN the system SHALL verify that all four models accept the required constructor arguments
5. WHEN handling memory constraints THEN the system SHALL provide guidance on batch_size tuning (256-1024) based on GPU memory
6. WHEN validating loss configuration THEN the system SHALL ensure loss specifications map correctly to NF loss classes using level parameter for MQLoss/IQLoss
7. IF sanity checks fail THEN the system SHALL provide clear diagnostic messages with specific configuration guidance

### Requirement 9: Implement File Organization and Dependencies from Source Document

**User Story:** As a maintainable codebase component, I want to implement the exact file organization and dependencies specified in the source document, so that I follow the established project structure and import patterns.

#### Acceptance Criteria

1. WHEN organizing files THEN the system SHALL place all model factory code in nf_models/factory.py as specified in Section 1.2
2. WHEN importing NF models THEN the system SHALL use: from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
3. WHEN importing NF losses THEN the system SHALL use: from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss
4. WHEN implementing utility functions THEN the system SHALL include _loss_ctor and _prune_kwargs as specified in Section 4.2.A
5. WHEN setting LOC budget THEN the system SHALL keep nf_models/factory.py under 200 lines as specified in the file inventory table
6. WHEN handling dependencies THEN the system SHALL use only NF-native imports without custom model implementations
7. IF file organization deviates from specification THEN the system SHALL fail validation to maintain project structure