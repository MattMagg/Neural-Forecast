# Implementation Plan

- [x] 1. Create features/registry.py with exact code from source document
  - Copy exact IndicatorSpec dataclass from docs/forecasting_sf_plan.md Section 3.1
  - Copy exact REGISTRY list with all 13 indicators as specified in source
  - Copy exact MTF_TARGETS list: 30min [rsi,roc,atr,bbands], 1h [rsi,roc,atr,bbands,macd], 4h [rsi,atr,bbands]
  - Import exact type literals: Kind = Literal["hist","futr","stat"], TF = Literal["15min","30min","1h","4h"]
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Create features/builder.py with exact computation functions from source document
  - [x] 2.1 Copy exact _compute_talib function from source
    - Use exact implementation: vbt.IndicatorFactory.from_talib(spec.func)
    - Use exact parameter broadcasting: params = {k: np.array(v) for k, v in spec.params.items()}
    - Use exact column naming: f"{spec.name}{('_'+key if key!='real' else '')}_{suffix}"
    - _Requirements: 2.1, 2.2, 2.7_

  - [x] 2.2 Copy exact _compute_pandasta function from source
    - Use exact implementation: getattr(pta, spec.func) with itertools.product
    - Use exact column naming pattern with parameter suffixes
    - Handle both Series and DataFrame outputs as specified
    - _Requirements: 2.1, 2.2, 2.7_

  - [x] 2.3 Copy exact _compute_custom function from source
    - Implement exact calendar features: ds.hour*60 + ds.minute, ds.dayofweek, (ds.dayofweek >= 5).astype(int)
    - Use exact function signature and logic from source document
    - _Requirements: 2.3, 2.7_

  - [x] 2.4 Copy exact build_indicators function from source
    - Use exact implementation including BBANDS bandwidth post-processing
    - Use exact post-processing logic: (upper - lower) / middle for bandwidth
    - Keep only bandwidth columns for BBANDS as specified
    - _Requirements: 2.4, 2.5, 2.7_

- [x] 3. Copy exact MTF functions from source document
  - [x] 3.1 Copy exact _compute_mtf_one function from source
    - Use exact implementation: resample_to_interval(base.set_index("ds"), tf)
    - Use exact column naming: f"{c}_{tf}" for MTF indicators
    - Use exact merge pattern: resampled_merge(df_ohlcv_15m, hi_feats.set_index("ds"), tf)
    - _Requirements: 3.1, 3.2, 3.7_

  - [x] 3.2 Copy exact apply_mtf function from source
    - Use exact implementation iterating through MTF_TARGETS
    - Use exact merge pattern on ds column with left joins
    - Return combined MTF feature DataFrame as specified
    - _Requirements: 3.1, 3.3, 3.7_

- [x] 4. Copy exact postprocess_shift_and_prune function from source document
  - Use exact function signature: postprocess_shift_and_prune(exo_raw: pd.DataFrame, rules: Dict) -> pd.DataFrame
  - Use exact feature type identification logic from REGISTRY
  - Use exact shift(1) implementation: exo[c] = exo[c].shift(1) for hist columns
  - Use exact availability filter: exo[c].notna().mean() >= 0.98
  - Use exact near-constant detection: s.nunique() <= 3
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Copy exact select_features function from source document
  - Use exact function signature: select_features(exo: pd.DataFrame, policy: Dict) -> Tuple[List[str], List[str], List[str]]
  - Use exact feature type identification by registry name prefixes
  - Use exact correlation pruning: Spearman |rho| >= 0.95 with keep.remove(d) logic
  - Use exact hard cap: cap = 256 with futr/stat priority and hist variance ranking
  - Use exact variance ranking: exo[hist_cols].var().sort_values(ascending=False)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 6. Implement exact integration code from source document
  - Copy exact assembly sequence from source Section 3.4 for run_train.py
  - Use exact merge pattern: base_feats.merge(mtf_feats, on="ds", how="left")
  - Use exact validation: assert_shifted(nf_df, hist_cols) before NF calls
  - Mirror same 4 lines in run_predict.py before NeuralForecast.load()
  - Use exact instantiate_models call with hist_cols, futr_cols, stat_cols
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 7. Implement minimal testing from source document Section 3.7
  - [ ] 7.1 Implement no-leak check
    - Verify hist feature at time t equals compute from data ≤ t-15m (not t)
    - Use targeted check for random timestamp as specified in source
    - _Requirements: 7.1_

  - [ ] 7.2 Implement MTF alignment sanity test
    - Fabricate tiny 1-day series and compute 1h moving average via MTF pipeline
    - Confirm 09:00 to 09:45 holds 08:00-09:00 value, updates at 10:00
    - Confirm final shift(1) bumps it one 15m bar
    - _Requirements: 7.2_

  - [ ] 7.3 Implement cap enforcement test
    - Ensure total feature columns (excluding ds) ≤ 256
    - Fail fast if exceeded as specified
    - _Requirements: 7.3_

  - [ ] 7.4 Implement stability test
    - Verify repeated runs on same data yield identical exog matrices (deterministic)
    - _Requirements: 7.4_

- [ ] 8. Implement hygiene and boundary cases from source document Section 3.6
  - Ensure higher-TF bar closing at 10:00 not used for 10:00 prediction (available for 10:15)
  - Rely on regularize_to_grid_utc() from Section 2 for EOB grid snapping
  - Use availability filter (≥98%) plus NF masking for NaN warmup handling
  - Keep only bandwidth for BBANDS to avoid feature bloat
  - Use vectorbt broadcasting for parameter arrays, no Python loops
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_