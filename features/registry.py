# features/registry.py
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any

Kind = Literal["hist","futr","stat"]
TF = Literal["15min","30min","1h","4h"]

@dataclass
class IndicatorSpec:
    name: str                        # column stem, e.g. "rsi"
    lib: Literal["talib","pandas_ta","custom"]
    func: str                        # e.g. "RSI" for TA-Lib, "rsi" for pandas_ta
    params: Dict[str, List[Any]]     # cartesian grid; vectorbt will broadcast when lib=talib
    inputs: Tuple[str, ...]          # source columns, e.g. ("close",) or ("high","low","close")
    kind: Kind                       # "hist" or "futr" or "stat"
    tf: TF = "15min"                 # base timeframe unless specified
    post: Optional[str] = None       # optional post-processing: e.g., "bandwidth", "zscore"

# Pragmatic starter set (tight, not bloated).
REGISTRY: List[IndicatorSpec] = [
    # --- Momentum / Trend ---
    IndicatorSpec(name="rsi", lib="talib", func="RSI",
                  params={"timeperiod":[7,14,28]}, inputs=("close",), kind="hist"),
    IndicatorSpec(name="roc", lib="talib", func="ROC",
                  params={"timeperiod":[4,8,16,32]}, inputs=("close",), kind="hist"),
    IndicatorSpec(name="stoch_k", lib="talib", func="STOCH",
                  params={"fastk_period":[14], "slowk_period":[3], "slowd_period":[3]},
                  inputs=("high","low","close"), kind="hist"),
    IndicatorSpec(name="macd", lib="talib", func="MACD",
                  params={"fastperiod":[8,12], "slowperiod":[21,26], "signalperiod":[9]},
                  inputs=("close",), kind="hist"),

    # --- Volatility ---
    IndicatorSpec(name="atr", lib="talib", func="ATR",
                  params={"timeperiod":[8,16,32]}, inputs=("high","low","close"), kind="hist"),
    IndicatorSpec(name="nvol", lib="pandas_ta", func="stdev",
                  params={"length":[8,32,96]}, inputs=("close",), kind="hist"),

    # --- Volume / Flow ---
    IndicatorSpec(name="obv", lib="talib", func="OBV",
                  params={}, inputs=("close","volume"), kind="hist"),
    IndicatorSpec(name="mfi", lib="talib", func="MFI",
                  params={"timeperiod":[14]}, inputs=("high","low","close","volume"), kind="hist"),

    # --- Structure ---
    IndicatorSpec(name="bbands", lib="talib", func="BBANDS",
                  params={"timeperiod":[20], "nbdevup":[2], "nbdevdn":[2]},
                  inputs=("close",), kind="hist", post="bandwidth"),
    IndicatorSpec(name="donchian", lib="pandas_ta", func="donchian",
                  params={"lower_length":[20], "upper_length":[20]}, inputs=("high","low"), kind="hist"),

    # --- Calendar (future-known -> futr_exog) ---
    IndicatorSpec(name="minute_of_day", lib="custom", func="minute_of_day",
                  params={}, inputs=("ds",), kind="futr"),
    IndicatorSpec(name="day_of_week", lib="custom", func="day_of_week",
                  params={}, inputs=("ds",), kind="futr"),
    IndicatorSpec(name="is_weekend", lib="custom", func="is_weekend",
                  params={}, inputs=("ds",), kind="futr"),
]

# MTF policy: compute these also on higher TFs and merge down (EOB-aligned) later.
MTF_TARGETS: List[Tuple[TF, List[str]]] = [
    ("30min", ["rsi","roc","atr","bbands"]),
    ("1h",    ["rsi","roc","atr","bbands","macd"]),
    ("4h",    ["rsi","atr","bbands"]),
]