# features/builder.py
from __future__ import annotations
import itertools, numpy as np, pandas as pd
try:
    import pandas_ta_openbb as pta  # NumPy 2 compatible fork
except ImportError:
    import pandas_ta as pta  # Fallback to original if openbb not available
import vectorbt as vbt
from typing import Dict, List, Tuple
from .registry import REGISTRY, MTF_TARGETS, IndicatorSpec
from technical.util import resample_to_interval, resampled_merge  # freqtrade technical
# ^ provides reliable resample+merge utilities for higher TFs → base TF forward-fill. 

# ---------- 3.2.1 Base-TF compute (15m) ----------
def _compute_talib(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    # vectorbt wraps TA-Lib w/ IndicatorFactory, broadcasting params combos efficiently. 
    fac = vbt.IndicatorFactory.from_talib(spec.func)
    # Build param grid dict -> vectorbt accepts arrays to create cartesian runs. 
    params = {k: np.array(v) for k, v in spec.params.items()} if spec.params else {}
    inputs = [df[i] for i in spec.inputs]
    out = fac.run(*inputs, **params)
    
    # Get output names from the indicator - vectorbt indicators have direct attributes
    # not _results. Each indicator type has specific output attributes:
    # RSI -> .real, BBANDS -> .upperband/.middleband/.lowerband, etc.
    if hasattr(out, 'output_names'):
        output_names = out.output_names
    else:
        # Default to 'real' for single-output indicators like RSI
        output_names = ['real']
    
    # Collect data from each output attribute
    data = {}
    for name in output_names:
        if hasattr(out, name):
            attr_value = getattr(out, name)
            # Ensure it's a DataFrame
            if not isinstance(attr_value, pd.DataFrame):
                attr_value = pd.DataFrame(attr_value, index=df.index)
            data[name] = attr_value
    
    frames = []
    for key, dfi in data.items():
        cols = []
        # vectorbt encodes parameter combinations in MultiIndex or as simple column values
        if isinstance(dfi.columns, pd.MultiIndex):
            # Multi-param combinations with MultiIndex
            for tup in dfi.columns:
                suffix = "_".join(f"{k[:1]}{v}" for k, v in zip(spec.params.keys(), tup))
                cols.append(f"{spec.name}{('_'+key if key!='real' else '')}_{suffix}")
        else:
            # Simple columns (e.g., [14, 21] for RSI periods)
            for col in dfi.columns:
                if spec.params:
                    # Create suffix from param names and column value
                    # For RSI with timeperiod=[14,21], columns are [14, 21]
                    suffix = "_".join(f"{k[:1]}{col}" for k in spec.params.keys())
                    cols.append(f"{spec.name}{('_'+key if key!='real' else '')}_{suffix}")
                else:
                    # No params, just use the base name
                    cols.append(f"{spec.name}{('_'+key if key!='real' else '')}")
        
        dfi = dfi.copy()
        dfi.columns = cols
        frames.append(dfi)
    
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)

def _compute_pandasta(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    func = getattr(pta, spec.func)
    all_params = spec.params or {}
    keys, grids = zip(*all_params.items()) if all_params else ([], [])
    frames = []
    for combo in itertools.product(*grids) if grids else [()]:
        kw = dict(zip(keys, combo)) if grids else {}
        ser_or_df = func(df[spec.inputs[0]] if len(spec.inputs)==1 else df[list(spec.inputs)], **kw)
        dfi = ser_or_df if isinstance(ser_or_df, pd.DataFrame) else ser_or_df.to_frame()
        suffix = "_".join(f"{k[:1]}{v}" for k,v in kw.items()) if kw else None
        dfi.columns = [f"{spec.name}_{c}" if suffix is None else f"{spec.name}_{c}_{suffix}" for c in dfi.columns]
        frames.append(dfi)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)

def _compute_custom(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    ds = pd.DatetimeIndex(df["ds"])
    if spec.func == "minute_of_day":
        return pd.DataFrame({"minute_of_day": ds.hour*60 + ds.minute}, index=df.index)
    if spec.func == "day_of_week":
        return pd.DataFrame({"day_of_week": ds.dayofweek}, index=df.index)
    if spec.func == "is_weekend":
        return pd.DataFrame({"is_weekend": (ds.dayofweek >= 5).astype(int)}, index=df.index)
    raise ValueError(f"Unknown custom func: {spec.func}")

def build_indicators(df_ohlcv_15m: pd.DataFrame, registry=REGISTRY) -> pd.DataFrame:
    """Compute base-15m indicators from registry; raw (no shift), EOB-aligned."""
    base = df_ohlcv_15m.set_index("ds").sort_index()
    frames = []
    for spec in registry:
        if spec.tf != "15min":  # base compute only 15m here
            continue
        if spec.kind == "stat":  # handled elsewhere as constants
            continue
        if spec.lib == "talib":
            dfi = _compute_talib(base, spec)
        elif spec.lib == "pandas_ta":
            dfi = _compute_pandasta(base, spec)
        elif spec.lib == "custom":
            dfi = _compute_custom(base.reset_index()[["ds"]].assign(dummy=0), spec).set_index(base.index)
        else:
            raise ValueError(spec.lib)
        # Optional post-processing
        if spec.name == "bbands" and "bandwidth" in (spec.post or ""):
            # percent bandwidth: (upper - lower) / middle
            cols = [c for c in dfi.columns if "upper" in c or "lower" in c or "middle" in c]
            # expecting columns contain 'upper','lower','middle' stems
            uppers = [c for c in cols if "upper" in c]; lowers = [c for c in cols if "lower" in c]; mids = [c for c in cols if "middle" in c]
            for u,l,m in zip(uppers, lowers, mids):
                dfi[f"{spec.name}_bw{u.split('_',1)[1]}"] = (dfi[u] - dfi[l]).div(dfi[m].replace(0,np.nan))
            dfi = dfi[[c for c in dfi.columns if "_bw" in c]]  # keep only bandwidths
        frames.append(dfi)
    return pd.concat(frames, axis=1).reset_index() if frames else df_ohlcv_15m[["ds"]].copy()

# ---------- 3.2.2 Higher-TF compute & merge ----------
def _compute_mtf_one(df_ohlcv_15m: pd.DataFrame, tf: str, names: List[str]) -> pd.DataFrame:
    """Resample OHLCV to tf, compute subset of indicators, then merge down to 15m."""
    base = df_ohlcv_15m.copy()
    # 1) Up-sample: resample_to_interval builds a higher-TF OHLCV DataFrame (EOB aligned).
    df_hi = resample_to_interval(base.set_index("ds"), tf)  # returns columns with suffixes like 'close'
    df_hi = df_hi.reset_index().rename(columns={"date":"ds"}) if "date" in df_hi.columns else df_hi.reset_index()
    # 2) Compute indicators on higher TF
    frames = [df_hi[["ds"]].copy()]
    for spec in REGISTRY:
        if spec.name not in names: 
            continue
        sp = IndicatorSpec(**{**spec.__dict__, "tf": tf})  # reuse spec on hi-TF
        if sp.lib == "talib":
            dfi = _compute_talib(df_hi.set_index("ds"), sp)
        elif sp.lib == "pandas_ta":
            dfi = _compute_pandasta(df_hi.set_index("ds"), sp)
        else:
            continue
        # Prefix with TF to avoid collisions, e.g., rsi_1h_p14
        dfi.columns = [f"{c}_{tf}" for c in dfi.columns]
        frames.append(dfi.reset_index())
    hi_feats = frames[0]
    for fr in frames[1:]:
        hi_feats = hi_feats.merge(fr, on="ds", how="left")
    # 3) Merge down to 15m with forward-fill inside each hi-TF bar
    #    resampled_merge aligns EOB and fills until next hi-TF close. 
    merged = resampled_merge(df_ohlcv_15m, hi_feats.set_index("ds"), tf).reset_index().rename(columns={"index":"ds"})
    return merged[[c for c in merged.columns if c not in ["open","high","low","close","volume"]]]

def apply_mtf(df_ohlcv_15m: pd.DataFrame, registry=REGISTRY) -> pd.DataFrame:
    out = df_ohlcv_15m[["ds"]].copy()
    for tf, names in MTF_TARGETS:
        add = _compute_mtf_one(df_ohlcv_15m, tf, names)
        out = out.merge(add, on="ds", how="left")
    return out

# ---------- 3.2.3 Post-process: shift + prune ----------
def postprocess_shift_and_prune(exo_raw: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """Shift all hist features by 1; leave futr/stat untouched; basic NA and variance guards."""
    exo = exo_raw.copy()
    exo = exo.sort_values("ds").reset_index(drop=True)
    ds = exo["ds"]

    # Identify columns by type from registry
    futr_cols = set()
    hist_cols = set()
    stat_cols = set()
    for spec in REGISTRY:
        if spec.kind == "futr":
            futr_cols.add(spec.name)
        elif spec.kind == "stat":
            stat_cols.add(spec.name)
        else:
            hist_cols.add(spec.name)

    # shift(1) every hist column (by prefix match)
    def is_hist_col(c): 
        return any(c.startswith(n) for n in hist_cols)
    def is_futr_col(c): 
        return any(c == n or c.startswith(f"{n}_") for n in futr_cols)

    cols = [c for c in exo.columns if c != "ds"]
    for c in cols:
        if is_hist_col(c):
            exo[c] = exo[c].shift(1)  # STRICT NO-LEAKAGE RULE

    # Simple availability/variance guards (keep pruning policy lean)
    # 1) Drop columns with <98% availability post-shift
    mask_ok = {c: exo[c].notna().mean() >= 0.98 for c in cols}
    exo = exo[["ds"] + [c for c in cols if mask_ok.get(c, True)]]

    # 2) Drop near-constant columns
    def near_const(s: pd.Series) -> bool:
        s = s.dropna()
        return s.nunique() <= 3
    keep = ["ds"] + [c for c in exo.columns if c=="ds" or not near_const(exo[c])]
    exo = exo[keep]

    return exo

def select_features(exo: pd.DataFrame, policy: Dict) -> Tuple[List[str], List[str], List[str]]:
    """Return (hist_exog_list, futr_exog_list, stat_exog_list) respecting caps and pruning."""
    cols = [c for c in exo.columns if c != "ds"]
    # Identify types by registry name prefixes
    futr_roots = [s.name for s in REGISTRY if s.kind=="futr"]
    stat_roots = [s.name for s in REGISTRY if s.kind=="stat"]

    futr_cols = [c for c in cols if any(c==r or c.startswith(f"{r}_") for r in futr_roots)]
    stat_cols = [c for c in cols if any(c==r or c.startswith(f"{r}_") for r in stat_roots)]
    hist_cols = [c for c in cols if c not in futr_cols and c not in stat_cols]

    # Correlation prune among hist features (Spearman |rho|>0.95 keep 1)
    H = exo[hist_cols].copy()
    corr = H.corr(method="spearman").abs()
    keep = set(hist_cols)
    for i in corr.columns:
        if i not in keep: continue
        dupes = [j for j in corr.index if j!=i and corr.loc[i,j] >= 0.95]
        for d in dupes:
            if d in keep:
                keep.remove(d)
    hist_cols = [c for c in hist_cols if c in keep]

    # Hard cap total count ≤ 256 (favor futr/stat first, then hist by variance)
    cap = 256
    if len(futr_cols) + len(stat_cols) >= cap:
        futr_cols = futr_cols[:cap]  # in practice futr/stat are few
        stat_cols = []
        hist_cols = []
    else:
        remain = cap - (len(futr_cols) + len(stat_cols))
        # Rank hist by variance (proxy for information)
        vari = exo[hist_cols].var().sort_values(ascending=False).index.tolist()
        hist_cols = [c for c in vari if c in hist_cols][:remain]

    return hist_cols, futr_cols, stat_cols