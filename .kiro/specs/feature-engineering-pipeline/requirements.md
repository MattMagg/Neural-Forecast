# Requirements Document

## Introduction

The Feature Engineering Pipeline spec implements the exact feature engineering approach specified in `docs/forecasting_sf_plan.md` Section 3. This component creates a lean, pragmatic indicator registry and computation system using vectorbt+TA-Lib (primary) and pandas-ta-openbb (supplement) with strict compute → align → shift(1) discipline. The implementation follows the "keep it small first" principle with a focused set of indicators, multi-timeframe processing via freqtrade/technical utilities, and hard feature cap of ≤256 to prevent overfitting.

## Requirements

### Requirement 1: Implement Exact Indicator Registry from Source Document

**User Story:** As a feature engineering system, I want to implement the exact IndicatorSpec dataclass and REGISTRY from the source document, so that I follow the proven "pragmatic starter set" without over-engineering.

#### Acceptance Criteria

1. WHEN creating features/registry.py THEN the system SHALL implement the exact IndicatorSpec dataclass with fields: name, lib, func, params, inputs, kind, tf, post
2. WHEN defining the REGISTRY THEN the system SHALL use the exact indicator specifications from the source document including RSI [7,14,28], ROC [4,8,16,32], STOCH, MACD, ATR [8,16,32], nvol [8,32,96], OBV, MFI, BBANDS with bandwidth post-processing, Donchian, and calendar features
3. WHEN categorizing features THEN the system SHALL use Kind = Literal["hist","futr","stat"] and TF = Literal["15min","30min","1h","4h"] exactly as specified
4. WHEN defining MTF targets THEN the system SHALL implement the exact MTF_TARGETS list: 30min [rsi,roc,atr,bbands], 1h [rsi,roc,atr,bbands,macd], 4h [rsi,atr,bbands]
5. WHEN using libraries THEN the system SHALL use vectorbt+TA-Lib as primary (with IndicatorFactory.from_talib()) and pandas-ta-openbb only as supplement
6. WHEN implementing calendar features THEN the system SHALL create minute_of_day, day_of_week, is_weekend as custom functions with kind="futr"
7. IF the registry deviates from the source specification THEN the system SHALL fail validation to prevent over-engineering

### Requirement 2: Implement Exact Computation Functions from Source Document

**User Story:** As a feature computation engine, I want to implement the exact _compute_talib, _compute_pandasta, and _compute_custom functions from the source document, so that I use the proven vectorbt and pandas-ta integration patterns.

#### Acceptance Criteria

1. WHEN implementing _compute_talib THEN the system SHALL use the exact implementation: vbt.IndicatorFactory.from_talib(spec.func) with parameter broadcasting via np.array(v) for cartesian grids
2. WHEN implementing _compute_pandasta THEN the system SHALL use the exact implementation: getattr(pta, spec.func) with itertools.product for parameter combinations
3. WHEN implementing _compute_custom THEN the system SHALL use the exact implementation: ds.hour*60 + ds.minute for minute_of_day, ds.dayofweek for day_of_week, (ds.dayofweek >= 5).astype(int) for is_weekend
4. WHEN implementing build_indicators THEN the system SHALL use the exact function signature and logic from the source document including post-processing for BBANDS bandwidth
5. WHEN handling vectorbt outputs THEN the system SHALL flatten MultiIndex columns using the exact pattern: f"{spec.name}{('_'+key if key!='real' else '')}_{suffix}"
6. WHEN processing BBANDS post-processing THEN the system SHALL implement the exact bandwidth calculation: (upper - lower) / middle and keep only bandwidth columns
7. IF any computation function deviates from the source implementation THEN the system SHALL fail validation to ensure consistency

### Requirement 3: Implement Exact MTF Processing from Source Document

**User Story:** As a multi-timeframe feature system, I want to implement the exact _compute_mtf_one and apply_mtf functions from the source document, so that I use the proven freqtrade/technical integration for reliable EOB alignment.

#### Acceptance Criteria

1. WHEN implementing _compute_mtf_one THEN the system SHALL use the exact implementation: resample_to_interval(base.set_index("ds"), tf) for upsampling and resampled_merge for downsampling
2. WHEN processing higher timeframes THEN the system SHALL use the exact column naming pattern: f"{c}_{tf}" for MTF indicators to avoid collisions
3. WHEN implementing apply_mtf THEN the system SHALL use the exact implementation: iterate through MTF_TARGETS and merge results on ds column
4. WHEN handling freqtrade utilities THEN the system SHALL import from technical.util: resample_to_interval, resampled_merge exactly as specified
5. WHEN merging MTF features THEN the system SHALL use the exact merge pattern: merged = resampled_merge(df_ohlcv_15m, hi_feats.set_index("ds"), tf)
6. WHEN filtering MTF outputs THEN the system SHALL exclude OHLCV columns: [c for c in merged.columns if c not in ["open","high","low","close","volume"]]
7. IF MTF processing deviates from the source implementation THEN the system SHALL fail validation to ensure EOB alignment correctness

### Requirement 4: Implement Exact Postprocess Function from Source Document

**User Story:** As a leakage prevention system, I want to implement the exact postprocess_shift_and_prune function from the source document, so that I apply the proven shift(1) discipline and pruning logic.

#### Acceptance Criteria

1. WHEN implementing postprocess_shift_and_prune THEN the system SHALL use the exact function signature: postprocess_shift_and_prune(exo_raw: pd.DataFrame, rules: Dict) -> pd.DataFrame
2. WHEN identifying feature types THEN the system SHALL use the exact logic: build futr_cols, hist_cols, stat_cols sets from REGISTRY and match by name prefixes
3. WHEN applying shift(1) THEN the system SHALL use the exact implementation: exo[c] = exo[c].shift(1) for all is_hist_col(c) columns
4. WHEN filtering by availability THEN the system SHALL use the exact threshold: exo[c].notna().mean() >= 0.98 for post-shift availability
5. WHEN removing constant features THEN the system SHALL use the exact logic: s.nunique() <= 3 for near-constant detection
6. WHEN preserving temporal discipline THEN the system SHALL leave futr and stat columns unchanged as specified in the source
7. IF the postprocess function deviates from the source implementation THEN the system SHALL fail validation to ensure leakage prevention

### Requirement 5: Implement Exact Feature Selection from Source Document

**User Story:** As a feature selection system, I want to implement the exact select_features function from the source document, so that I use the proven correlation pruning and hard cap logic.

#### Acceptance Criteria

1. WHEN implementing select_features THEN the system SHALL use the exact function signature: select_features(exo: pd.DataFrame, policy: Dict) -> Tuple[List[str], List[str], List[str]]
2. WHEN identifying feature types THEN the system SHALL use the exact logic: futr_roots and stat_roots from REGISTRY with name matching patterns
3. WHEN pruning correlations THEN the system SHALL use the exact implementation: Spearman correlation with |rho| >= 0.95 threshold and keep.remove(d) logic
4. WHEN enforcing hard cap THEN the system SHALL use the exact cap = 256 limit with futr/stat priority and hist ranking by variance
5. WHEN ranking historical features THEN the system SHALL use the exact implementation: exo[hist_cols].var().sort_values(ascending=False)
6. WHEN returning lists THEN the system SHALL return exactly (hist_cols, futr_cols, stat_cols) as specified in the source
7. IF the select_features function deviates from the source implementation THEN the system SHALL fail validation to ensure proper NF integration

### Requirement 6: Implement Exact End-to-End Assembly from Source Document

**User Story:** As a pipeline integration system, I want to implement the exact assembly sequence from the source document, so that I integrate seamlessly with run_train.py and run_predict.py.

#### Acceptance Criteria

1. WHEN implementing assembly THEN the system SHALL use the exact sequence: build_indicators → apply_mtf → merge → postprocess_shift_and_prune → select_features
2. WHEN integrating with run_train.py THEN the system SHALL use the exact code block provided in the source document starting with "From §2 we have: nf_base = make_nf_canonical"
3. WHEN merging features THEN the system SHALL use the exact merge pattern: base_feats.merge(mtf_feats, on="ds", how="left") then nf_base.merge(exo, on="ds", how="left")
4. WHEN validating leakage THEN the system SHALL use the exact validation: assert_shifted(nf_df, hist_cols) before any NF call
5. WHEN passing to models THEN the system SHALL use the exact instantiate_models call: instantiate_models(cfg, hist_cols=hist_cols, futr_cols=futr_cols, stat_cols=stat_cols)
6. WHEN implementing in run_predict.py THEN the system SHALL mirror the same 4 lines before NeuralForecast.load()
7. IF the assembly deviates from the source implementation THEN the system SHALL fail validation to ensure proper integration

### Requirement 7: Implement Minimal Testing from Source Document

**User Story:** As a quality assurance system, I want to implement the exact minimal tests specified in the source document, so that I validate correctness without over-engineering the testing framework.

#### Acceptance Criteria

1. WHEN implementing no-leak check THEN the system SHALL verify that hist feature at time t equals compute from data ≤ t-15m as specified in the source
2. WHEN implementing MTF alignment sanity THEN the system SHALL fabricate a tiny 1-day series and verify 1h moving average holds 08:00-09:00 value from 09:00 to 09:45
3. WHEN implementing cap enforcement THEN the system SHALL ensure total feature columns (excluding ds) ≤ 256 and fail fast otherwise
4. WHEN implementing stability test THEN the system SHALL verify repeated runs on same data yield identical exog matrices (deterministic)
5. WHEN running tests THEN the system SHALL keep tests fast and focused as specified: "fast, not fluffy"
6. WHEN validating shift(1) THEN the system SHALL confirm final shift(1) bumps MTF values by one 15m bar as specified
7. IF tests become complex or enterprise-grade THEN the system SHALL fail validation to maintain lean testing approach

### Requirement 8: Implement Hygiene and Boundary Cases from Source Document

**User Story:** As a robust feature system, I want to implement the exact hygiene and boundary case handling specified in the source document, so that I handle edge cases correctly without over-engineering.

#### Acceptance Criteria

1. WHEN handling shift(1) timing THEN the system SHALL ensure higher-TF bar closing at 10:00 is not used for 10:00 prediction but becomes available for 10:15
2. WHEN handling EOB correctness THEN the system SHALL rely on regularize_to_grid_utc() from Section 2 to snap to 15m EOB grid
3. WHEN handling NaN warmup THEN the system SHALL rely on availability filter (≥98%) plus NF masking to handle indicator warmup periods
4. WHEN handling multi-output indicators THEN the system SHALL keep only bandwidth for BBANDS as specified to avoid feature bloat
5. WHEN using parameter grids THEN the system SHALL use vectorbt broadcasting for arrays (e.g., RSI periods [7,14,28]) without Python loops
6. WHEN implementing boundary cases THEN the system SHALL follow the exact guidance from source document Section 3.6
7. IF hygiene handling becomes complex THEN the system SHALL fail validation to maintain lean implementation