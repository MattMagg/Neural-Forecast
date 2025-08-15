Create Feature Engineering Pipeline Spec

## Objective

Create the **Feature Engineering Pipeline Spec** following the same rigorous approach, detail level, and format as the completed data-processing-validation spec. This is the next critical component in the BTC forecasting system development sequence.

## Context

The data processing and validation foundation has been successfully implemented and validated (see `EXECUTION_STATUS.md`). The system now has:
- ✅ Complete data assembly path processing 7M+ records with 100% validation pass rate
- ✅ NeuralForecast canonical schema compliance 
- ✅ UTC timestamp discipline and quality gates
- ✅ Leakage prevention infrastructure with `assert_shifted()` validation

The next logical step is implementing the feature engineering pipeline that will build technical indicators and exogenous features on top of this validated foundation.

## Specification Requirements

### **Spec Name**: `feature-engineering-pipeline`

### **Core Purpose**: 
Technical indicator registry and computation with strict leakage prevention, following the compute → align → shift(1) discipline for all historical exogenous features.

### **Key Components to Implement**:

1. **Technical Indicator Registry** (`features/registry.py`)
   - Primary indicators using vectorbt/TA-Lib for fast, vectorized computation
   - Supplementary indicators via pandas-ta-openbb (Numba-accelerated)
   - Multi-timeframe resampling/merge utilities via freqtrade/technical
   - Parameterizable indicator definitions with grid-friendly configurations

2. **Feature Builder** (`features/builder.py`)
   - Strict compute → align (15m EOB) → shift(1) implementation
   - Multi-timeframe feature alignment and aggregation
   - Feature selection with ≤256 cap and ≥98% availability filter
   - NF exogenous variable wiring (hist_exog_list, futr_exog_list, stat_exog_list)

3. **Crypto-Specific Features** (optional advanced features)
   - CCXT/exchange APIs for funding rates, open interest, basis
   - Cryptofeed integration for order book imbalance
   - All following compute → align → shift(1) discipline

4. **Integration Points**
   - Seamless integration with existing data processing pipeline
   - Validation using existing `assert_shifted()` function
   - NeuralForecast exogenous variable compatibility

### **Reference Documentation**:
Follow `docs/forecasting_sf_plan.md` Section 3 (Exogenous features):
- 3.1 Indicator registry
- 3.2 Feature builder (compute → align → shift(1))
- 3.3 Feature selection & hard cap
- 3.4 End-to-end assembly
- 3.5 NF wiring
- 3.6 Hygiene & boundary cases
- 3.7 Minimal tests
- 3.8 Crypto-specific data-driven features
- 3.9 Performance note: pandas-on-GPU accelerator (optional)

### **Critical Requirements**:

1. **Leakage Prevention**: Absolute adherence to compute → align → shift(1) rule
2. **Feature Cap**: Hard limit of ≤256 features after selection
3. **Availability Filter**: ≥98% data availability requirement
4. **MTF Alignment**: Proper multi-timeframe alignment to 15-minute base
5. **NF Integration**: Proper hist_exog_list, futr_exog_list, stat_exog_list wiring
6. **Validation**: All features must pass `assert_shifted()` validation

### **Quality Standards**:
- Same rigor as data-processing-validation spec
- Comprehensive requirements with EARS format acceptance criteria
- Detailed design document with architecture and implementation strategy
- Complete task breakdown with specific deliverables
- Full testing and validation of all components

### **Integration with Existing Foundation**:
- Build on validated data processing pipeline from `run_train.py`
- Use existing canonical NF frame format
- Leverage existing validation infrastructure
- Maintain same code organization and quality standards

## Expected Deliverables

Create the complete spec structure in `.kiro/specs/feature-engineering-pipeline/`:

1. **`requirements.md`** - Detailed requirements with user stories and EARS acceptance criteria
2. **`design.md`** - Comprehensive design document with architecture, components, and implementation strategy  
3. **`tasks.md`** - Complete task breakdown with specific implementation steps

## Success Criteria

The spec should enable implementation of:
- Technical indicator registry with vectorbt/TA-Lib integration
- Feature builder with strict leakage prevention
- Multi-timeframe alignment and aggregation
- Feature selection and pruning to ≤256 features
- NeuralForecast exogenous variable integration
- Complete validation and testing framework

The implementation should seamlessly integrate with the existing data processing foundation and maintain the same quality standards established in the first spec.

## Instructions

Follow the same systematic approach used for the data-processing-validation spec:
1. Create comprehensive requirements covering all aspects of feature engineering
2. Design a robust architecture that integrates with existing foundation
3. Break down implementation into manageable, testable tasks
4. Ensure all components support the NeuralForecast-native approach
5. Maintain strict leakage prevention and data quality standards

The goal is to create a spec that, when implemented, will provide a production-ready feature engineering pipeline that generates high-quality exogenous features for the BTC forecasting models while maintaining the same level of rigor and validation as the data processing foundation.

--------

The next task I will have you continue is to create a sub agent 


-------


You are a validation expert for nixtla's neuralforecast library. Your task is to validate document(s), script, notebook, or specific code thoroughly and systematically. You will use the mcp tool context7 to look up the up-to-date best practices and coding documentation for nixtla's neuralforecast library as your primary means of validation.

1.  Create a working document for this validation. You will update this document after validating each section by annotating and appending a brief validation report.
2.  Start by analyzing the document(s), script, notebook, or specific code.
3.  **Validation Process:**
    *   Use the mcp tool context7 to look up the up-to-date best practices and coding documentation for nixtla's neuralforecast library.
    *   Thoroughly validate the provided section content against these best practices.
    *   If context7 does not provide adequete context for a specific validation step and/or further validation is required, you will gather context directly from the neuralforecast repo at https://github.com/Nixtla/neuralforecast
    *   YOU MAY RESEARCH other libraries on context7 on a as needed basis if validation requires it.
4.  **Validation Report:**
    *   If everything looks good, annotate the working document with a brief validation report stating "Validation: OK."
    *   If you find something that needs revision in the document:
        *   State what is wrong.
        *   Provide a proposed solution.
        *   Annotate the working document with this information.
5.  **Further Investigation:**
    *   If further investigation is needed, or further research/information is needed in order to provide the most optimal solution, state specifically what information you need. Do not assume a solution if you need more information or cant find one reletivley quick, just state the case.
6.  **Important:** You are **NOT EDITING** the original document, only analyzing it and updating the working validation document you create throughout this session/conversation.