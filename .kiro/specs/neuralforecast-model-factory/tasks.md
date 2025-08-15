# Implementation Plan

- [x] 1. Set up core factory module structure and imports
  - Create nf_models/factory.py with proper NeuralForecast imports
  - Import NHITS, NBEATSx, TiDE, PatchTST from neuralforecast.models
  - Import DistributionLoss, MQLoss, IQLoss from neuralforecast.losses.pytorch
  - Add typing imports for List, Dict, Any, Optional, Union
  - Add inspect module for constructor introspection
  - _Requirements: 1.1, 9.2, 9.3_

- [x] 2. Implement loss constructor function
  - Write _loss_ctor function with exact signature from source document
  - Support 'studentt', 'mqloss', 'iqloss' loss kinds with case-insensitive matching
  - Implement DistributionLoss("StudentT") for distributional training
  - Implement MQLoss and IQLoss with configurable confidence levels
  - Use DEFAULT_LEVELS = [80, 90, 95] as fallback for level parameter
  - Add proper error handling for unsupported loss kinds
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Implement parameter pruning utility function
  - Write _prune_kwargs function using inspect.signature for constructor introspection
  - Filter out unsupported constructor arguments for each model class
  - Handle model-specific parameters like 'revin' for PatchTST
  - Ensure all four models support hist_exog_list, futr_exog_list, stat_exog_list
  - Return filtered parameter dictionary with only supported arguments
  - _Requirements: 5.1, 5.2, 8.4_

- [x] 4. Implement core model instantiation function
  - Write instantiate_models function with exact signature from source document
  - Extract h, models_cfg, and scaler_overrides from configuration
  - Wire exogenous variable lists to all model instances
  - Implement model class mapping: NHITS, NBEATSx, TiDE, PatchTST
  - Handle scaler_type overrides with per-model precedence
  - Apply loss configuration using _loss_ctor function
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5. Add model-specific parameter handling
  - Implement NHITS configuration with n_blocks, n_pool_kernel_size, dropout_prob_theta
  - Implement NBEATSx configuration with stack_types, n_blocks, mlp_units, dropout_prob_theta
  - Implement TiDE configuration with hidden_size, num_encoder_layers, num_decoder_layers, dropout
  - Implement PatchTST configuration with patch_len, stride, n_heads, hidden_size, revin flag
  - Use _prune_kwargs to filter unsupported parameters for each model
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. Implement configuration validation functions
  - Write validate_config function to check required keys (h, freq, models)
  - Validate models list format and non-empty constraint
  - Write validate_loss_spec function for loss configuration validation
  - Check levels are between 0 and 100 for quantile losses
  - Write validate_model_compatibility function for exogenous list support
  - Add comprehensive error messages with specific diagnostic information
  - _Requirements: 8.1, 8.2, 8.3, 8.6_

- [x] 7. Add common parameter defaults and wiring
  - Implement common parameter defaults: input_size (1024 generic, 2048 PatchTST)
  - Set scaler_type defaults: "robust" general, "revin" for PatchTST
  - Configure training parameters: learning_rate=1e-3, batch_size=512, max_steps=20000
  - Set early_stop_patience_steps=400, val_check_steps=100
  - Wire h (horizon) parameter from configuration to all models
  - Ensure all models receive hist_exog_list, futr_exog_list, stat_exog_list
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7_

- [x] 8. Create comprehensive unit tests for factory functions
  - Write test_loss_constructor to verify DistributionLoss, MQLoss, IQLoss creation
  - Write test_parameter_pruning to verify unsupported argument filtering
  - Write test_model_instantiation to verify basic model creation
  - Write test_exogenous_wiring to verify hist/futr/stat lists are properly set
  - Write test_scaler_overrides to verify per-model scaler_type handling
  - Write test_configuration_validation to verify error handling
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 9. Implement YAML configuration integration
  - Create sample experiment configuration files for h4, h8, h16, h32
  - Implement exact YAML structure from source document Section 4.2.B
  - Support global scaler_type with per-model overrides
  - Configure model entries with alias, input_size, loss specification
  - Include model-specific parameters (n_blocks, patch_len, etc.)
  - Add cross-validation parameters (n_windows, step_size, val_size, refit)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 10. Add integration with feature engineering pipeline
  - Test integration with features/builder.py select_features function
  - Verify hist_cols, futr_cols, stat_cols are properly passed to models
  - Ensure feature lists are correctly wired to exogenous parameters
  - Test with realistic feature counts (up to 256 features)
  - Validate that all four models accept the exogenous lists
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 11. Implement NeuralForecast orchestrator integration
  - Test model instances work with NeuralForecast(models=models, freq=freq)
  - Verify models support standard NF workflow: fit, predict, cross_validation
  - Test model aliases provide stable column naming in outputs
  - Ensure probabilistic outputs work with predict(level=[80,90,95])
  - Validate model persistence with nf.save() and NeuralForecast.load()
  - _Requirements: 7.5, 7.6, 7.7_

- [ ] 12. Add comprehensive error handling and diagnostics
  - Implement ConfigurationError and ModelInstantiationError exception classes
  - Add detailed error messages for configuration validation failures
  - Provide specific guidance for common configuration mistakes
  - Handle edge cases like missing loss specifications or invalid confidence levels
  - Add memory and GPU constraint guidance for batch_size tuning
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 13. Create integration tests with run_train.py workflow
  - Test complete workflow: load config → instantiate models → create NeuralForecast
  - Verify integration with existing data processing and validation utilities
  - Test with realistic BTC forecasting configuration (h=16, multiple models)
  - Validate that models train successfully with sample data
  - Ensure cross-validation works with instantiated models
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [ ] 14. Add performance and memory optimization
  - Optimize model instantiation for large feature lists (256 features)
  - Ensure factory functions complete in under 1 second
  - Add memory usage monitoring for model instantiation
  - Implement lazy loading patterns if needed for large configurations
  - Test performance with all four models and full feature sets
  - _Requirements: 3.7, 8.5_

- [ ] 15. Create documentation and usage examples
  - Document factory function APIs with type hints and docstrings
  - Create usage examples for common configuration patterns
  - Document YAML configuration schema and parameter options
  - Add troubleshooting guide for common configuration errors
  - Document integration patterns with existing system components
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 16. Implement final validation and testing
  - Run complete test suite with all unit and integration tests
  - Validate against source document Section 4 requirements
  - Test with realistic BTC data and feature engineering pipeline
  - Verify all four models instantiate correctly with proper parameters
  - Ensure seamless integration with NeuralForecast orchestrator
  - Validate YAML-driven configuration works for all horizons
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1_