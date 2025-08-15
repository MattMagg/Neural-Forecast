# Design Document

## Overview

The Feature Engineering Pipeline implements the exact approach from `docs/forecasting_sf_plan.md` Section 3. This is a lean, pragmatic implementation using two drop-in files: `features/registry.py` (declarative indicator specs) and `features/builder.py` (computation functions). The design follows the "keep it small first" principle with vectorbt+TA-Lib primary, pandas-ta-openbb supplement, freqtrade/technical for MTF, and strict compute → align → shift(1) discipline.

The implementation consists of exactly three stages: 1) Compute base-TF indicators (15m), 2) Apply MTF resampling & merge, 3) Postprocess with shift(1) and pruning. No enterprise architecture - just the essential functions needed for NF integration.

## Architecture

### Simple Three-Stage Flow

```
Canonical NF DataFrame → build_indicators() → apply_mtf() → postprocess_shift_and_prune() → select_features() → NF Lists
```

**Library Stack (as specified in source):**
- **vectorbt + TA-Lib**: Primary (fast, vectorized, parameter broadcasting)
- **pandas-ta-openbb**: Supplement only (Numba-accelerated, when TA-Lib lacks variant)
- **freqtrade.technical**: MTF utilities (resample_to_interval, resampled_merge)
- **Custom functions**: Calendar features only

### Two Drop-In Files

1. **features/registry.py** - IndicatorSpec dataclass + REGISTRY list + MTF_TARGETS (exact from source)
2. **features/builder.py** - Four functions: build_indicators, apply_mtf, postprocess_shift_and_prune, select_features (exact from source)

### Core Principles (from source document)

1. **Keep it small first** - Pragmatic starter set, expand only if sCRPS drops materially
2. **vectorbt broadcasts** - Use parameter arrays for cartesian combos, no Python loops
3. **freqtrade/technical** - Reliable resample+merge, avoid ad-hoc pandas pitfalls
4. **Strict shift(1)** - All hist exogs shifted, futr/stat unchanged
5. **Hard cap 256** - Prevent memory bloat and overfitting

## Implementation Details

### features/registry.py (Exact from Source)

**IndicatorSpec dataclass** with fields: name, lib, func, params, inputs, kind, tf, post

**REGISTRY list** with exact indicators from source:
- Momentum: RSI [7,14,28], ROC [4,8,16,32], STOCH [14,3,3], MACD [8,12]/[21,26]/[9]
- Volatility: ATR [8,16,32], nvol [8,32,96] 
- Volume: OBV, MFI [14]
- Structure: BBANDS (bandwidth post-processing), Donchian [20]
- Calendar: minute_of_day, day_of_week, is_weekend (kind="futr")

**MTF_TARGETS** exactly as specified:
- 30min: [rsi,roc,atr,bbands]
- 1h: [rsi,roc,atr,bbands,macd] 
- 4h: [rsi,atr,bbands]

### features/builder.py (Exact from Source)

**Four functions exactly as provided in source document:**

1. **_compute_talib()** - Uses vbt.IndicatorFactory.from_talib() with parameter broadcasting
2. **_compute_pandasta()** - Uses getattr(pta, spec.func) with itertools.product  
3. **_compute_custom()** - Calendar features: ds.hour*60+ds.minute, ds.dayofweek, (ds.dayofweek>=5)
4. **build_indicators()** - Main function with post-processing for BBANDS bandwidth

**MTF Functions:**
5. **_compute_mtf_one()** - Uses resample_to_interval + resampled_merge from freqtrade
6. **apply_mtf()** - Iterates MTF_TARGETS and merges results

**Post-processing:**
7. **postprocess_shift_and_prune()** - shift(1) for hist, availability ≥98%, near_const ≤3
8. **select_features()** - Correlation pruning |rho|≥0.95, hard cap 256, variance ranking

### Integration Pattern (Exact from Source)

**In run_train.py:**
```python
# From §2 we have: nf_base = make_nf_canonical(raw, unique_id="BTC-USD")
from features.registry import REGISTRY
from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features

base_feats = build_indicators(nf_base.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}))
mtf_feats  = apply_mtf(nf_base[["ds","open","high","low","close","volume"]])
exo_raw    = base_feats.merge(mtf_feats, on="ds", how="left")

exo        = postprocess_shift_and_prune(exo_raw, rules={})
hist_cols, futr_cols, stat_cols = select_features(exo, policy={})

nf_df = nf_base.merge(exo, on="ds", how="left")
# Validate leakage discipline before any NF call (Section 2.3)
from utils.validate import assert_shifted
assert_shifted(nf_df, hist_cols)
```

**In run_predict.py:** Mirror the same 4 lines before NeuralForecast.load()

## Minimal Testing (from Source Section 3.7)

**Four essential tests only:**

1. **No-leak check**: Verify hist feature at time t equals compute from data ≤ t-15m
2. **MTF alignment sanity**: 1-day series, 1h MA holds 08:00-09:00 value from 09:00-09:45, updates at 10:00
3. **Cap enforcement**: Total features ≤ 256, fail fast otherwise  
4. **Stability**: Repeated runs yield identical exog matrices (deterministic)

**Keep tests "fast, not fluffy" as specified in source.**

## Dependencies

**Required (exact from source):**
- vectorbt (TA-Lib wrappers, parameter broadcasting)
- pandas-ta-openbb (Numba-accelerated supplement)  
- freqtrade[technical] (resample_to_interval, resampled_merge)
- Standard: pandas, numpy, itertools

**Import pattern:**
```python
import vectorbt as vbt
import pandas_ta as pta  
from technical.util import resample_to_interval, resampled_merge
```