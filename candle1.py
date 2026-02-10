from dataclasses import dataclass
from typing import Optional
import uuid
import os
import glob
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

@dataclass
class Config:
    """Configuration class to hold settings for data processing and plotting."""
    zip_files: list = None
    extract_dir: str = 'extracted_data'
    data_dir: str = 'extracted_data/parquet_out/symbol=NIFTY 50'
    date_start: str = "2024-04-15"
    date_end: str = "2024-04-19"
    lookback_1m: int = 5
    lookback_5m: int = 5
    lookback_15m: int = 5
    lookback_4h: int = 3
    fvg_min_range: float = 10.0
    fvg_body_ratio: float = 0.6
    fvg_min_confidence: float = 0.50
    fvg_interaction_penalty: float = 0.10
    ob_min_confidence: float = 0.50
    ob_interaction_penalty: float = 0.10
    plot_inverse_fvgs: bool = False
    plot_15m: bool = False
    plot_1m_5m: bool = False
    plot_1m: bool = False
    plot_5m: bool = True


class DataManager:
    """Handles data extraction, loading, and preprocessing."""

    def __init__(self, config: Config):
        self.config = config

    def extract_data(self):
        """Extract zip files to the specified directory if not already extracted."""
        extract_path = Path(self.config.extract_dir)
        if not self.config.zip_files:
            print("[WARN] No zip files configured; skipping extraction.")
            return
        if not extract_path.exists():
            extract_path.mkdir(parents=True)
            for zip_path in self.config.zip_files:
                if not Path(zip_path).exists():
                    print(f"[WARN] Zip file not found: {zip_path}")
                    continue
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.config.extract_dir)

    def load_parquet_data(self):
        """Load all parquet files from the data directory into a single DataFrame."""
        if not Path(self.config.data_dir).exists():
            print(f"[WARN] Data directory not found: {self.config.data_dir}")
            return pd.DataFrame()
        parquet_files = glob.glob(f'{self.config.data_dir}/date=2024-04-*/data.parquet', recursive=True)
        dataframes = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                date_str = Path(file).parent.name.replace('date=', '')
                df['date'] = pd.to_datetime(date_str)
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
            if "time" not in df.columns:
                print("[ERROR] Column 'time' not found in parquet data.")
                return pd.DataFrame()
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            return df
        return pd.DataFrame()

    def preprocess_ticks(self, df):
        """Preprocess tick data: ensure datetime, sort, filter market hours, keep required columns."""
        df = df.copy()
        if df.empty:
            return df
        if "time" not in df.columns or "ltp" not in df.columns:
            print("[ERROR] Required columns 'time' and/or 'ltp' are missing.")
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        market_open = pd.to_datetime("09:15:00").time()
        market_close = pd.to_datetime("15:30:00").time()
        df = df[
            (df["time"].dt.time >= market_open) &
            (df["time"].dt.time <= market_close)
        ]
        df = df[["time", "ltp"]]
        date_start = pd.to_datetime(self.config.date_start)
        date_end = pd.to_datetime(self.config.date_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        df = df[(df["time"] >= date_start) & (df["time"] <= date_end)]
        return df

class TimeframeBuilder:
    """Builds OHLC timeframes from tick data."""

    def __init__(self, config: Config):
        self.config = config

    def resample_ohlc(self, df, timeframe):
        """Resample tick data into OHLC candles for the given timeframe."""
        if df.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        df_idx = df.set_index("time").sort_index()
        resampled = df_idx['ltp'].resample(timeframe).agg(['first', 'max', 'min', 'last']).dropna()
        resampled.columns = ['open', 'high', 'low', 'close']
        return resampled.reset_index()

    def build_timeframes(self, df_ticks):
        """Build multiple timeframes from preprocessed tick data."""
        tf_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "4h": "4h"
        }
        if df_ticks.empty:
            return {name: pd.DataFrame(columns=["time", "open", "high", "low", "close"]) for name in tf_map}
        tf_data = {}
        for name, rule in tf_map.items():
            tf_data[name] = self.resample_ohlc(df_ticks, rule)
        return tf_data

class StructureAnalyzer:
    """Analyzes market structure: swings, hierarchy, BOS/CHoCH."""

    def __init__(self, config: Config):
        self.config = config

    def detect_swings(self, df, lookback):
        """Detect swing highs and lows in the DataFrame."""
        df = df.copy()
        df["swing_high"] = False
        df["swing_low"] = False
        last_swing_type = None

        if df.empty:
            return df

        lookback = max(1, int(lookback))
        window = 2 * lookback + 1
        roll_high = df["high"].rolling(window=window, center=True).max()
        roll_low = df["low"].rolling(window=window, center=True).min()

        for i in range(lookback, len(df) - lookback):
            is_high = df["high"].iloc[i] == roll_high.iloc[i]
            is_low = df["low"].iloc[i] == roll_low.iloc[i]

            if is_high and is_low:
                if last_swing_type == "HIGH":
                    is_high = False
                elif last_swing_type == "LOW":
                    is_low = False
                else:
                    if (df["high"].iloc[i] - df["low"].iloc[i]) > 0:
                        is_high = True
                        is_low = False

            if is_high and last_swing_type != "HIGH":
                df.at[df.index[i], "swing_high"] = True
                last_swing_type = "HIGH"
            elif is_low and last_swing_type != "LOW":
                df.at[df.index[i], "swing_low"] = True
                last_swing_type = "LOW"

        return df

    def compute_swing_trend(self, df):
        """Compute swing-based trend using HH/HL and LH/LL logic."""
        df = df.copy()
        df["swing_trend"] = None
        df["trend_high"] = np.nan
        df["trend_low"] = np.nan
        df["trend_high_idx"] = np.nan
        df["trend_low_idx"] = np.nan

        prev_high = prev_low = None
        last_high = last_low = None
        prev_high_idx = prev_low_idx = None
        last_high_idx = last_low_idx = None

        trend = None
        trend_high = trend_low = None
        trend_high_idx = trend_low_idx = None

        for i in range(len(df)):
            if df.iloc[i]["swing_high"]:
                prev_high, prev_high_idx = last_high, last_high_idx
                last_high, last_high_idx = df.iloc[i]["high"], i
            if df.iloc[i]["swing_low"]:
                prev_low, prev_low_idx = last_low, last_low_idx
                last_low, last_low_idx = df.iloc[i]["low"], i

            if (
                prev_high is not None
                and prev_low is not None
                and last_high is not None
                and last_low is not None
            ):
                if last_high > prev_high and last_low > prev_low:
                    if trend != "UPTREND":
                        trend_high = last_high
                        trend_low = last_low
                        trend_high_idx = last_high_idx
                        trend_low_idx = last_low_idx
                    trend = "UPTREND"
                elif last_high < prev_high and last_low < prev_low:
                    if trend != "DOWNTREND":
                        trend_high = last_high
                        trend_low = last_low
                        trend_high_idx = last_high_idx
                        trend_low_idx = last_low_idx
                    trend = "DOWNTREND"

            if trend == "UPTREND":
                if df.iloc[i]["swing_low"] and last_low is not None:
                    if trend_low is None or last_low > trend_low:
                        trend_low = last_low
                        trend_low_idx = last_low_idx
                if df.iloc[i]["swing_high"] and last_high is not None:
                    if trend_high is None or last_high > trend_high:
                        trend_high = last_high
                        trend_high_idx = last_high_idx
            elif trend == "DOWNTREND":
                if df.iloc[i]["swing_high"] and last_high is not None:
                    if trend_high is None or last_high < trend_high:
                        trend_high = last_high
                        trend_high_idx = last_high_idx
                if df.iloc[i]["swing_low"] and last_low is not None:
                    if trend_low is None or last_low < trend_low:
                        trend_low = last_low
                        trend_low_idx = last_low_idx

            df.at[df.index[i], "swing_trend"] = trend
            df.at[df.index[i], "trend_high"] = trend_high
            df.at[df.index[i], "trend_low"] = trend_low
            df.at[df.index[i], "trend_high_idx"] = trend_high_idx
            df.at[df.index[i], "trend_low_idx"] = trend_low_idx

        return df

    def classify_swing_hierarchy(self, df):
        """Classify swings into external and internal based on trend."""
        df = df.copy()
        df["swing_label"] = None
        external_high = external_low = None
        external_high_idx = external_low_idx = None
        current_trend = None

        swing_points = []
        for i in range(len(df)):
            if df.iloc[i]["swing_high"]:
                swing_points.append(("HIGH", i))
            elif df.iloc[i]["swing_low"]:
                swing_points.append(("LOW", i))

            if len(swing_points) >= 2:
                first, second = swing_points[0], swing_points[1]
                if first[0] == "LOW" and second[0] == "HIGH":
                    external_low = df.iloc[first[1]]["low"]
                    external_high = df.iloc[second[1]]["high"]
                    external_low_idx, external_high_idx = first[1], second[1]
                    current_trend = "BULLISH"
                elif first[0] == "HIGH" and second[0] == "LOW":
                    external_high = df.iloc[first[1]]["high"]
                    external_low = df.iloc[second[1]]["low"]
                    external_high_idx, external_low_idx = first[1], second[1]
                    current_trend = "BEARISH"

                if current_trend:
                    df.iloc[external_high_idx, df.columns.get_loc("swing_label")] = "EXTERNAL_HIGH"
                    df.iloc[external_low_idx, df.columns.get_loc("swing_label")] = "EXTERNAL_LOW"
                    break

        if current_trend is None:
            return df

        for i in range(max(external_high_idx or 0, external_low_idx or 0) + 1, len(df)):
            if df.iloc[i]["swing_high"]:
                price = df.iloc[i]["high"]
                if current_trend == "BULLISH" and price > external_high:
                    df.iloc[i, df.columns.get_loc("swing_label")] = "EXTERNAL_HIGH"
                    external_high = price
                else:
                    df.iloc[i, df.columns.get_loc("swing_label")] = "INTERNAL_HIGH"
            elif df.iloc[i]["swing_low"]:
                price = df.iloc[i]["low"]
                if current_trend == "BEARISH" and price < external_low:
                    df.iloc[i, df.columns.get_loc("swing_label")] = "EXTERNAL_LOW"
                    external_low = price
                else:
                    df.iloc[i, df.columns.get_loc("swing_label")] = "INTERNAL_LOW"

        return df
    def detect_bos_choch(self, df, initial_trend, disp_threshold=0.6):
        df = df.copy()
        if df.empty:
            return df

        # Outputs
        df["BOS_bullish"] = False
        df["BOS_bearish"] = False
        df["CHOCH_bullish"] = False
        df["CHOCH_bearish"] = False
        df["MSS"] = False
        df["structure_level"] = np.nan
        df["structure_type"] = None
        trend = initial_trend if initial_trend in ("BULLISH", "BEARISH") else "BULLISH"
        df["effective_trend"] = trend
        df["structure_state"] = None
        # -----------------------------
        # Structure anchors (SWING-BASED ONLY)
        # -----------------------------
        external_high = None
        external_low = None

        internal_high = None
        internal_low = None

        last_hl = None   # last Higher Low in bullish trend
        last_lh = None   # last Lower High in bearish trend

        # -----------------------------
        # State
        # -----------------------------
        phase = "NORMAL"                       # "NORMAL" / "CHOCH_SEEN"
        bos_allowed = True                     # One BOS per leg

        for i in range(len(df)):

            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]
            open_ = df.iloc[i]["open"]
            close = df.iloc[i]["close"]

            # -----------------------------
            # Acceptance (displacement)
            # -----------------------------
            rng = high - low
            if rng == 0:
                df.iloc[i, df.columns.get_loc("effective_trend")] = trend
                df.iloc[i, df.columns.get_loc("structure_state")] = phase
                continue

            accepted = abs(close - open_) / rng >= disp_threshold

            # -----------------------------
            # Update structure anchors from swings
            # -----------------------------
            if df.iloc[i]["swing_high"]:
                if external_high is None or high > external_high:
                    external_high = high
                    bos_allowed = True
                else:
                    if trend == "BEARISH":
                        last_lh = high
                    else:
                        internal_high = high

            elif df.iloc[i]["swing_low"]:
                if external_low is None or low < external_low:
                    external_low = low
                    bos_allowed = True
                else:
                    if trend == "BULLISH":
                        last_hl = low
                    else:
                        internal_low = low

            # Require valid external structure
            if (trend == "BULLISH" and external_high is None) or (
                trend == "BEARISH" and external_low is None
            ):
                continue

            # =============================
            # STATE: NORMAL
            # =============================
            if phase == "NORMAL":

                # ---- BOS (continuation)
                if (
                    bos_allowed
                    and trend == "BULLISH"
                    and close > external_high
                    and accepted
                ):
                    df.iloc[i, df.columns.get_loc("BOS_bullish")] = True
                    df.iloc[i, df.columns.get_loc("structure_level")] = external_high
                    df.iloc[i, df.columns.get_loc("structure_type")] = "BOS"
                    bos_allowed = False

                elif (
                    bos_allowed
                    and trend == "BEARISH"
                    and close < external_low
                    and accepted
                ):
                    df.iloc[i, df.columns.get_loc("BOS_bearish")] = True
                    df.iloc[i, df.columns.get_loc("structure_level")] = external_low
                    df.iloc[i, df.columns.get_loc("structure_type")] = "BOS"
                    bos_allowed = False

                # ---- CHoCH (internal structure violation)
                elif (
                    trend == "BULLISH"
                    and last_hl is not None
                    and close < last_hl
                ):
                    df.iloc[i, df.columns.get_loc("CHOCH_bearish")] = True
                    df.iloc[i, df.columns.get_loc("structure_level")] = last_hl
                    df.iloc[i, df.columns.get_loc("structure_type")] = "CHOCH"
                    phase = "CHOCH_SEEN"

                elif (
                    trend == "BEARISH"
                    and last_lh is not None
                    and close > last_lh
                ):
                    df.iloc[i, df.columns.get_loc("CHOCH_bullish")] = True
                    df.iloc[i, df.columns.get_loc("structure_level")] = last_lh
                    df.iloc[i, df.columns.get_loc("structure_type")] = "CHOCH"
                    phase = "CHOCH_SEEN"

            # =============================
            # STATE: CHOCH_SEEN
            # =============================
            elif phase == "CHOCH_SEEN":

                # ---- MSS (external confirmation)
                if (
                    trend == "BULLISH"
                    and close < external_low
                    and accepted
                ):
                    df.iloc[i, df.columns.get_loc("MSS")] = True
                    df.iloc[i, df.columns.get_loc("structure_level")] = external_low
                    df.iloc[i, df.columns.get_loc("structure_type")] = "MSS"
                    trend = "BEARISH"
                    phase = "NORMAL"
                    bos_allowed = True

                    # reset bullish references
                    last_hl = None
                    external_high = None

                elif (
                    trend == "BEARISH"
                    and close > external_high
                    and accepted
                ):
                    df.iloc[i, df.columns.get_loc("MSS")] = True
                    df.iloc[i, df.columns.get_loc("structure_level")] = external_high
                    df.iloc[i, df.columns.get_loc("structure_type")] = "MSS"
                    trend = "BULLISH"
                    phase = "NORMAL"
                    bos_allowed = True

                    # reset bearish references
                    last_lh = None
                    external_low = None

            # -----------------------------
            # Persist state
            # -----------------------------
            df.iloc[i, df.columns.get_loc("effective_trend")] = trend
            df.iloc[i, df.columns.get_loc("structure_state")] = phase

        return df
      
    def compute_htf_trend(self, df_4h):
        """Compute higher timeframe trend from 4H data."""
        df_4h = self.detect_swings(df_4h, self.config.lookback_4h)
        swing_highs = df_4h[df_4h["swing_high"]]["high"]
        swing_lows = df_4h[df_4h["swing_low"]]["low"]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        if swing_highs.iloc[-1] > swing_highs.iloc[-2]:
            return "BULLISH"
        if swing_lows.iloc[-1] < swing_lows.iloc[-2]:
            return "BEARISH"
        return "RANGE"

@dataclass
class FairValueGap:
    id: str
    direction: str            # "bullish" | "bearish"
    timeframe: str
    created_at: int           # displacement candle index
    upper: float
    lower: float
    market_state: str         # TREND | REPRICING
    displacement_strength: float
    confidence: float
    ref_high: Optional[float] = None
    ref_low: Optional[float] = None
    ref_high_idx: Optional[int] = None
    ref_low_idx: Optional[int] = None
    mid_level: Optional[float] = None
    mitigated_at: Optional[int] = None
    repriced_at: Optional[int] = None
    display_end: Optional[int] = None
    inverse: bool = False
    active: bool = True
    retired: bool = False
    invalidated_reason: Optional[str] = None
    closed_reason: Optional[str] = None
    age: int = 0
    taps: int = 0
    interactions: int = 0

@dataclass
class OrderBlock:
    id: str
    created_at: int
    timeframe: str
    direction: str            # "bullish" | "bearish"
    lower: float
    upper: float
    market_state: str
    displacement_index: int
    confidence: float
    interactions: int = 0
    active: bool = True
    retired: bool = False
    closed_reason: Optional[str] = None
    invalidated_reason: Optional[str] = None
    repriced_at: Optional[int] = None
    mitigated_at: Optional[int] = None
    display_end: Optional[int] = None
class FairValueGapDetector:
    @staticmethod
    def is_valid_displacement(candle, threshold=0.6):
        body = abs(candle["close"] - candle["open"])
        rng = candle["high"] - candle["low"]
        return rng > 0 and (body / rng) >= threshold

    @staticmethod
    def detect_fvg(
        df,
        market_state_series,
        htf_trend,
        timeframe,
        min_range=0.0,
        displacement_threshold=0.6,
    ):
        fvgs = []

        if df is None or len(df) < 3:
            return fvgs
        if market_state_series is None:
            print("[WARN] market_state missing; no FVGs generated.")
            return fvgs
        if htf_trend not in ("BULLISH", "BEARISH"):
            print("[WARN] HTF trend unknown; no FVGs generated.")
            return fvgs

        has_struct = any(col in df.columns for col in ("MSS", "BOS_bullish", "BOS_bearish"))
        if not has_struct:
            print("[WARN] Structure columns missing; no FVGs generated.")
            return fvgs

        def _get_value(series, idx, default=None):
            if series is None:
                return default
            if hasattr(series, "iloc"):
                if len(series) == 0:
                    return default
                if idx < len(series):
                    return series.iloc[idx]
                return series.iloc[-1]
            try:
                if idx < len(series):
                    return series[idx]
                return series[-1]
            except Exception:
                return default

        last_swing_high = None
        last_swing_low = None

        for i in range(1, len(df) - 1):
            prev_idx = i - 1
            if "swing_high" in df.columns and bool(df.iloc[prev_idx].get("swing_high", False)):
                last_swing_high = df.iloc[prev_idx]["high"]
            if "swing_low" in df.columns and bool(df.iloc[prev_idx].get("swing_low", False)):
                last_swing_low = df.iloc[prev_idx]["low"]

            candle = df.iloc[i]
            rng = candle["high"] - candle["low"]
            if rng <= 0:
                continue
            body = abs(candle["close"] - candle["open"])
            displacement = body / rng
            if displacement < displacement_threshold:
                continue

            prev_high = df.iloc[i - 1]["high"]
            prev_low = df.iloc[i - 1]["low"]
            next_low = df.iloc[i + 1]["low"]
            next_high = df.iloc[i + 1]["high"]
            body_low = min(candle["open"], candle["close"])
            body_high = max(candle["open"], candle["close"])

            market_state = _get_value(market_state_series, i)
            if market_state not in ("TREND", "REPRICING", "REVERSAL"):
                continue

            mss = bool(df.iloc[i]["MSS"]) if "MSS" in df.columns else False
            bos_bull = bool(df.iloc[i]["BOS_bullish"]) if "BOS_bullish" in df.columns else False
            bos_bear = bool(df.iloc[i]["BOS_bearish"]) if "BOS_bearish" in df.columns else False

            bullish_impulse = candle["close"] > candle["open"]
            bearish_impulse = candle["close"] < candle["open"]
            if not bullish_impulse and not bearish_impulse:
                continue

            structure_break = False
            if bullish_impulse and last_swing_high is not None and candle["close"] > last_swing_high:
                structure_break = True
            if bearish_impulse and last_swing_low is not None and candle["close"] < last_swing_low:
                structure_break = True

            if market_state != "REVERSAL":
                if bullish_impulse and htf_trend != "BULLISH":
                    continue
                if bearish_impulse and htf_trend != "BEARISH":
                    continue

            if bullish_impulse:
                if not (bos_bull or mss or structure_break):
                    continue
                if next_low <= prev_high:
                    continue
                if not (body_low <= prev_high <= body_high and body_low <= next_low <= body_high):
                    continue
                lower = prev_high
                upper = next_low
                direction = "bullish"
            else:
                if not (bos_bear or mss or structure_break):
                    continue
                if next_high >= prev_low:
                    continue
                if not (body_low <= next_high <= body_high and body_low <= prev_low <= body_high):
                    continue
                upper = prev_low
                lower = next_high
                direction = "bearish"

            fvg_range = abs(upper - lower)
            if fvg_range < min_range:
                continue

            s_trend = 1.0 if (
                (direction == "bullish" and htf_trend == "BULLISH") or
                (direction == "bearish" and htf_trend == "BEARISH")
            ) else (0.6 if market_state == "REVERSAL" else 0.0)
            s_liq = 0.5
            if "liquidity_score" in df.columns:
                s_liq = _get_value(df["liquidity_score"], i, default=0.5)
            try:
                s_liq = float(s_liq)
            except (TypeError, ValueError):
                s_liq = 0.5
            s_liq = max(0.0, min(1.0, s_liq))

            denom = max(1e-9, 1.0 - displacement_threshold)
            s_disp = max(0.0, min(1.0, (displacement - displacement_threshold) / denom))
            s_struct = 1.0
            if direction == "bullish":
                s_acc = (candle["close"] - candle["low"]) / rng
            else:
                s_acc = (candle["high"] - candle["close"]) / rng
            s_acc = max(0.0, min(1.0, s_acc))

            confidence = (
                0.25 * s_trend
                + 0.20 * s_liq
                + 0.20 * s_disp
                + 0.20 * s_struct
                + 0.15 * s_acc
            )

            fvgs.append(
                FairValueGap(
                    id=uuid.uuid4().hex,
                    direction=direction,
                    timeframe=timeframe,
                    created_at=i,
                    upper=upper,
                    lower=lower,
                    market_state=market_state,
                    displacement_strength=displacement,
                    confidence=confidence,
                )
            )

        return fvgs

    @staticmethod
    def annotate_mitigations(
        fvgs,
        df,
        market_state_series=None,
        min_confidence=0.50,
        interaction_penalty=0.10,
    ):
        if df is None or df.empty:
            return fvgs
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        last_idx = len(df) - 1
        min_confidence = max(0.0, float(min_confidence))
        interaction_penalty = max(0.0, float(interaction_penalty))

        def _get_value(series, idx, default=None):
            if series is None:
                return default
            if hasattr(series, "iloc"):
                if len(series) == 0:
                    return default
                if idx < len(series):
                    return series.iloc[idx]
                return series.iloc[-1]
            try:
                if idx < len(series):
                    return series[idx]
                return series[-1]
            except Exception:
                return default

        for fvg in fvgs:
            fvg.mitigated_at = None
            fvg.repriced_at = None
            fvg.display_end = None
            fvg.retired = False
            fvg.closed_reason = None
            fvg.invalidated_reason = None
            fvg.interactions = 0
            fvg.taps = 0
            fvg.active = True

            if fvg.confidence < min_confidence:
                fvg.active = False
                fvg.invalidated_reason = "CONFIDENCE"
                continue

            start = (fvg.created_at or 0) + 1
            if start > last_idx:
                fvg.display_end = last_idx
                continue

            upper = max(fvg.upper, fvg.lower)
            lower = min(fvg.upper, fvg.lower)
            end_at = last_idx

            for j in range(start, len(df)):
                state = _get_value(market_state_series, j)

                if (
                    state in ("TREND", "REVERSAL")
                    and fvg.market_state in ("TREND", "REVERSAL")
                    and state != fvg.market_state
                ):
                    fvg.active = False
                    fvg.invalidated_reason = "STATE_FLIP"
                    end_at = j
                    break

                close = closes[j]
                if fvg.direction == "bullish" and close <= lower:
                    fvg.active = False
                    fvg.retired = True
                    fvg.closed_reason = "FULL_MITIGATION"
                    fvg.mitigated_at = j
                    end_at = j
                    break
                if fvg.direction == "bearish" and close >= upper:
                    fvg.active = False
                    fvg.retired = True
                    fvg.closed_reason = "FULL_MITIGATION"
                    fvg.mitigated_at = j
                    end_at = j
                    break

                if state == "REPRICING":
                    if fvg.direction == "bullish":
                        touched = lows[j] <= upper
                    else:
                        touched = highs[j] >= lower
                    if touched:
                        if fvg.repriced_at is None:
                            fvg.repriced_at = j
                        fvg.interactions += 1
                        fvg.taps = fvg.interactions
                        fvg.confidence -= interaction_penalty
                        if fvg.confidence < min_confidence:
                            fvg.active = False
                            fvg.invalidated_reason = "CONFIDENCE"
                            end_at = j
                            break

            fvg.display_end = end_at
        return fvgs

    @staticmethod
    def detect_inverse_fvgs(df, fvgs, min_range=0.0):
        """
        Detect inverse FVGs: when price closes through the opposite side of an FVG,
        it becomes an inverse gap in the opposite direction.
        """
        if df is None or df.empty or not fvgs:
            return []

        inverse_fvgs = []

        for fvg in fvgs:
            upper = max(fvg.upper, fvg.lower)
            lower = min(fvg.upper, fvg.lower)
            if abs(upper - lower) < min_range:
                continue
            start = (fvg.created_at or 0) + 1
            if start >= len(df):
                continue

            for j in range(start, len(df)):
                close = df.iloc[j]["close"]
                if fvg.direction == "bullish" and close < lower:
                    inverse_fvgs.append(
                        FairValueGap(
                            id=uuid.uuid4().hex,
                            direction="bearish",
                            timeframe=fvg.timeframe,
                            created_at=j,
                            upper=upper,
                            lower=lower,
                            market_state="INVERSE",
                            displacement_strength=fvg.displacement_strength,
                            confidence=fvg.confidence,
                            inverse=True,
                        )
                    )
                    break
                if fvg.direction == "bearish" and close > upper:
                    inverse_fvgs.append(
                        FairValueGap(
                            id=uuid.uuid4().hex,
                            direction="bullish",
                            timeframe=fvg.timeframe,
                            created_at=j,
                            upper=upper,
                            lower=lower,
                            market_state="INVERSE",
                            displacement_strength=fvg.displacement_strength,
                            confidence=fvg.confidence,
                            inverse=True,
                        )
                    )
                    break

        return inverse_fvgs

    @staticmethod

    def update_fvg_lifecycle(
        fvgs,
        candle,
        market_state,
        min_confidence=0.50,
        interaction_penalty=0.10,
    ):
        for fvg in fvgs:
            if not fvg.active:
                continue

            fvg.age += 1

            # MarketState flip against FVG
            if (
                market_state in ("TREND", "REVERSAL")
                and fvg.market_state in ("TREND", "REVERSAL")
                and market_state != fvg.market_state
            ):
                fvg.active = False
                fvg.invalidated_reason = "STATE_FLIP"
                continue

            if hasattr(candle, "get"):
                close = candle.get("close")
                high = candle.get("high", close)
                low = candle.get("low", close)
            else:
                close = float(candle)
                high = close
                low = close

            # Full mitigation / acceptance through level
            if fvg.direction == "bullish" and close is not None and close <= fvg.lower:
                fvg.active = False
                fvg.retired = True
                fvg.closed_reason = "FULL_MITIGATION"
                continue
            if fvg.direction == "bearish" and close is not None and close >= fvg.upper:
                fvg.active = False
                fvg.retired = True
                fvg.closed_reason = "FULL_MITIGATION"
                continue

            # Repricing interaction (confidence decay only)
            if market_state == "REPRICING":
                if fvg.direction == "bullish" and low is not None and low <= fvg.upper:
                    fvg.confidence -= interaction_penalty
                    fvg.interactions += 1
                    fvg.taps = fvg.interactions
                if fvg.direction == "bearish" and high is not None and high >= fvg.lower:
                    fvg.confidence -= interaction_penalty
                    fvg.interactions += 1
                    fvg.taps = fvg.interactions

            if fvg.confidence < min_confidence:
                fvg.active = False
                fvg.invalidated_reason = "CONFIDENCE"

class OrderBlockDetector:
    @staticmethod
    def detect_order_blocks(
        df,
        market_state_series,
        htf_trend,
        timeframe,
        displacement_threshold=0.6,
        require_inefficiency=True,
        htf_mid_series=None,
        cisd_series=None,
    ):
        obs = []

        if df is None or len(df) < 2:
            return obs
        if market_state_series is None:
            print("[WARN] market_state missing; no OBs generated.")
            return obs
        if htf_trend not in ("BULLISH", "BEARISH"):
            print("[WARN] HTF trend unknown; no OBs generated.")
            return obs

        if htf_mid_series is None and "trend_high" in df.columns and "trend_low" in df.columns:
            htf_mid_series = (df["trend_high"] + df["trend_low"]) / 2.0

        def _get_value(series, idx, default=None):
            if series is None:
                return default
            if hasattr(series, "iloc"):
                if len(series) == 0:
                    return default
                if idx < len(series):
                    return series.iloc[idx]
                return series.iloc[-1]
            try:
                if idx < len(series):
                    return series[idx]
                return series[-1]
            except Exception:
                return default

        last_swing_high = None
        last_swing_low = None

        for i in range(1, len(df)):
            prev_idx = i - 1
            if "swing_high" in df.columns and bool(df.iloc[prev_idx].get("swing_high", False)):
                last_swing_high = df.iloc[prev_idx]["high"]
            if "swing_low" in df.columns and bool(df.iloc[prev_idx].get("swing_low", False)):
                last_swing_low = df.iloc[prev_idx]["low"]

            candle = df.iloc[i]
            rng = candle["high"] - candle["low"]
            if rng <= 0:
                continue
            body = abs(candle["close"] - candle["open"])
            displacement = body / rng
            if displacement < displacement_threshold:
                continue

            market_state = _get_value(market_state_series, i)
            if market_state not in ("TREND", "REPRICING", "REVERSAL"):
                continue

            cisd_ok = True
            if market_state == "REVERSAL":
                cisd_ok = bool(_get_value(cisd_series, i, default=False))
            if not cisd_ok:
                continue

            mss = bool(df.iloc[i]["MSS"]) if "MSS" in df.columns else False
            bos_bull = bool(df.iloc[i]["BOS_bullish"]) if "BOS_bullish" in df.columns else False
            bos_bear = bool(df.iloc[i]["BOS_bearish"]) if "BOS_bearish" in df.columns else False

            bullish_disp = candle["close"] > candle["open"]
            bearish_disp = candle["close"] < candle["open"]
            if not bullish_disp and not bearish_disp:
                continue

            structure_break = False
            if bullish_disp and last_swing_high is not None and candle["close"] > last_swing_high:
                structure_break = True
            if bearish_disp and last_swing_low is not None and candle["close"] < last_swing_low:
                structure_break = True

            if not (mss or bos_bull or bos_bear or structure_break):
                continue

            if market_state != "REVERSAL":
                if bullish_disp and htf_trend != "BULLISH":
                    continue
                if bearish_disp and htf_trend != "BEARISH":
                    continue

            ob_idx = i - 1
            ob_candle = df.iloc[ob_idx]
            ob_bearish = ob_candle["close"] < ob_candle["open"]
            ob_bullish = ob_candle["close"] > ob_candle["open"]

            if bullish_disp and not ob_bearish:
                continue
            if bearish_disp and not ob_bullish:
                continue

            if require_inefficiency:
                if i >= len(df) - 1:
                    continue
                prev_high = df.iloc[i - 1]["high"]
                prev_low = df.iloc[i - 1]["low"]
                next_low = df.iloc[i + 1]["low"]
                next_high = df.iloc[i + 1]["high"]
                if bullish_disp and not (next_low > prev_high):
                    continue
                if bearish_disp and not (next_high < prev_low):
                    continue

            mid = _get_value(htf_mid_series, ob_idx, default=None)
            if mid is None or pd.isna(mid):
                continue

            if bullish_disp and not (ob_candle["high"] <= mid):
                continue
            if bearish_disp and not (ob_candle["low"] >= mid):
                continue

            if bullish_disp:
                direction = "bullish"
            else:
                direction = "bearish"

            disp_score = (displacement - displacement_threshold) / max(1e-9, 1.0 - displacement_threshold)
            disp_score = max(0.0, min(1.0, disp_score))
            trend_score = 1.0 if (
                (direction == "bullish" and htf_trend == "BULLISH")
                or (direction == "bearish" and htf_trend == "BEARISH")
            ) else (0.6 if market_state == "REVERSAL" else 0.0)
            struct_score = 1.0

            confidence = 0.4 * disp_score + 0.3 * trend_score + 0.3 * struct_score
            confidence = max(0.0, min(1.0, confidence))

            print(
                "[OB] MarketState: "
                f"{market_state} | Displacement: {displacement:.2f} | "
                f"Order Block Created: {direction} {ob_candle['low']:.2f}-{ob_candle['high']:.2f} | "
                f"Confidence: {confidence:.2f}"
            )

            obs.append(
                OrderBlock(
                    id=uuid.uuid4().hex,
                    created_at=ob_idx,
                    timeframe=timeframe,
                    direction=direction,
                    lower=ob_candle["low"],
                    upper=ob_candle["high"],
                    market_state=market_state,
                    displacement_index=i,
                    confidence=confidence,
                    interactions=0,
                    active=True,
                )
            )

        return obs

    @staticmethod
    def annotate_order_block_lifecycle(
        obs,
        df,
        market_state_series=None,
        min_confidence=0.50,
        interaction_penalty=0.10,
        cisd_bullish_series=None,
        cisd_bearish_series=None,
    ):
        if df is None or df.empty:
            return obs
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        last_idx = len(df) - 1
        min_confidence = max(0.0, float(min_confidence))
        interaction_penalty = max(0.0, float(interaction_penalty))

        def _get_value(series, idx, default=None):
            if series is None:
                return default
            if hasattr(series, "iloc"):
                if len(series) == 0:
                    return default
                if idx < len(series):
                    return series.iloc[idx]
                return series.iloc[-1]
            try:
                if idx < len(series):
                    return series[idx]
                return series[-1]
            except Exception:
                return default

        for ob in obs:
            ob.repriced_at = None
            ob.mitigated_at = None
            ob.display_end = None
            ob.retired = False
            ob.closed_reason = None
            ob.invalidated_reason = None
            ob.interactions = 0
            ob.active = True

            if ob.confidence < min_confidence:
                ob.active = False
                ob.invalidated_reason = "CONFIDENCE"
                continue

            start = (ob.created_at or 0) + 1
            if start > last_idx:
                ob.display_end = last_idx
                continue

            end_at = last_idx

            for j in range(start, len(df)):
                state = _get_value(market_state_series, j)
                if ob.direction == "bullish" and _get_value(cisd_bearish_series, j, default=False):
                    ob.active = False
                    ob.invalidated_reason = "CISD_AGAINST"
                    end_at = j
                    break
                if ob.direction == "bearish" and _get_value(cisd_bullish_series, j, default=False):
                    ob.active = False
                    ob.invalidated_reason = "CISD_AGAINST"
                    end_at = j
                    break

                if (
                    state in ("TREND", "REPRICING", "REVERSAL")
                    and ob.market_state in ("TREND", "REPRICING", "REVERSAL")
                    and state != ob.market_state
                ):
                    ob.active = False
                    ob.invalidated_reason = "STATE_FLIP"
                    end_at = j
                    break

                close = closes[j]
                if ob.direction == "bullish" and close < ob.lower:
                    ob.active = False
                    ob.retired = True
                    ob.closed_reason = "FULL_MITIGATION"
                    ob.mitigated_at = j
                    end_at = j
                    break
                if ob.direction == "bearish" and close > ob.upper:
                    ob.active = False
                    ob.retired = True
                    ob.closed_reason = "FULL_MITIGATION"
                    ob.mitigated_at = j
                    end_at = j
                    break

                if state == "REPRICING":
                    if ob.direction == "bullish" and lows[j] <= ob.upper:
                        if ob.repriced_at is None:
                            ob.repriced_at = j
                        ob.confidence -= interaction_penalty
                        ob.interactions += 1
                    if ob.direction == "bearish" and highs[j] >= ob.lower:
                        if ob.repriced_at is None:
                            ob.repriced_at = j
                        ob.confidence -= interaction_penalty
                        ob.interactions += 1

                    if ob.confidence < min_confidence:
                        ob.active = False
                        ob.invalidated_reason = "CONFIDENCE"
                        end_at = j
                        break

            ob.display_end = end_at
        return obs

    @staticmethod
    def update_order_block_state(
        ob,
        candle,
        min_confidence=0.50,
        interaction_penalty=0.10,
    ):
        if not ob.active:
            return ob

        if hasattr(candle, "get"):
            close = candle.get("close")
            high = candle.get("high", close)
            low = candle.get("low", close)
        else:
            close = float(candle)
            high = close
            low = close

        if ob.direction == "bullish":
            if low is not None and low <= ob.upper:
                ob.confidence -= interaction_penalty
                ob.interactions += 1
            if close is not None and close < ob.lower:
                ob.active = False
                ob.retired = True
                ob.closed_reason = "FULL_MITIGATION"
        else:
            if high is not None and high >= ob.lower:
                ob.confidence -= interaction_penalty
                ob.interactions += 1
            if close is not None and close > ob.upper:
                ob.active = False
                ob.retired = True
                ob.closed_reason = "FULL_MITIGATION"

        if ob.confidence < min_confidence:
            ob.active = False
            ob.invalidated_reason = "CONFIDENCE"

        return ob

class Visualizer:
    """Handles plotting of market structure using Plotly."""
    def _add_swing_leg(self, fig, fvgs, row=None):
        """
        Draw only the 50% midpoint level used for discount/premium.
        (Leg line removed per discounted-zone-only view.)
        """
        seen = set()
        for fvg in fvgs:
            if fvg.ref_high_idx is None or fvg.ref_low_idx is None:
                continue
            key = (fvg.ref_high_idx, fvg.ref_low_idx, fvg.ref_high, fvg.ref_low)
            if key in seen:
                continue
            seen.add(key)

            x0 = fvg.ref_high_idx
            x1 = fvg.ref_low_idx
            y0 = fvg.ref_high
            y1 = fvg.ref_low
            if x0 is None or x1 is None or y0 is None or y1 is None:
                continue
            if x0 == x1:
                continue

            # 50% midline only
            if fvg.mid_level is not None:
                x_left = min(x0, x1)
                x_right = max(x0, x1)
                mid_kwargs = dict(
                    type="line",
                    x0=x_left,
                    y0=fvg.mid_level,
                    x1=x_right,
                    y1=fvg.mid_level,
                    line=dict(color="rgba(255, 215, 0, 0.4)", width=1, dash="dash"),
                    layer="below",
                )
                if row is None:
                    fig.add_shape(**mid_kwargs)
                else:
                    fig.add_shape(**mid_kwargs, row=row, col=1)
    def _add_fvg_rectangles(self, fig, fvgs, row=None, x_end=None):
        """
        Overlay Fair Value Gaps as shaded rectangles.
        """
        for fvg in fvgs:
            color = (
                "rgba(0, 255, 0, 0.20)" if fvg.direction == "bullish"
                else "rgba(255, 0, 0, 0.20)"
            )

            x0 = fvg.created_at
            if fvg.display_end is None:
                continue
            x1 = fvg.display_end
            if x_end is not None:
                x1 = min(x1, x_end)
            if x1 < x0:
                continue
            y0, y1 = sorted((fvg.lower, fvg.upper))

            if row is None:
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                )
            else:
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                    row=row,
                    col=1,
                )

    def _add_order_block_rectangles(self, fig, obs, row=None, x_end=None):
        """
        Overlay Order Blocks as shaded rectangles.
        """
        for ob in obs:
            if ob.display_end is None:
                continue

            color = (
                "rgba(57, 255, 20, 0.25)" if ob.direction == "bullish"
                else "rgba(255, 0, 102, 0.25)"
            )

            x0 = ob.created_at
            x1 = ob.display_end
            if x_end is not None:
                x1 = min(x1, x_end)
            if x1 < x0:
                continue
            y0, y1 = sorted((ob.lower, ob.upper))

            if row is None:
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                )
            else:
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                    row=row,
                    col=1,
                )

    def plot_structure_plotly(self, df, title, html_file, fvgs=None, obs=None):
        """Plot single timeframe structure."""
        seq_index = list(range(len(df)))
        time_labels = df["time"].dt.strftime('%Y-%m-%d %H:%M').tolist()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=seq_index,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            customdata=time_labels,
            hovertext=[f"<b>{label}</b><br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
                      for label, o, h, l, c in zip(time_labels, df["open"], df["high"], df["low"], df["close"])],
            hoverinfo='text', name="Price"
        ))
        if fvgs:
            self._add_fvg_rectangles(fig, fvgs, x_end=len(df) - 1)
        if obs:
            self._add_order_block_rectangles(fig, obs, x_end=len(df) - 1)


        # Swing Highs
        swing_high_indices = [seq_index[i] for i in range(len(df)) if df.iloc[i]["swing_high"]]
        fig.add_trace(go.Scatter(x=swing_high_indices, y=df.loc[df.swing_high, "high"],
                                 mode="markers", marker=dict(size=8, color="blue"), name="Swing High"))

        # Swing Lows
        swing_low_indices = [seq_index[i] for i in range(len(df)) if df.iloc[i]["swing_low"]]
        fig.add_trace(go.Scatter(x=swing_low_indices, y=df.loc[df.swing_low, "low"],
                                 mode="markers", marker=dict(size=8, color="orange"), name="Swing Low"))

        # BOS Bullish
        bos_bull_indices = [seq_index[i] for i in range(len(df)) if df.iloc[i]["BOS_bullish"]]
        fig.add_trace(go.Scatter(x=bos_bull_indices, y=df.loc[df.BOS_bullish, "structure_level"],
                                 mode="text", text="BOS Bull", textposition="top center",
                                 textfont=dict(size=10, color="green"), name="BOS Bullish"))

        # BOS Bearish
        bos_bear_indices = [seq_index[i] for i in range(len(df)) if df.iloc[i]["BOS_bearish"]]
        fig.add_trace(go.Scatter(x=bos_bear_indices, y=df.loc[df.BOS_bearish, "structure_level"],
                                 mode="text", text="BOS Bear", textposition="bottom center",
                                 textfont=dict(size=10, color="red"), name="BOS Bearish"))

        # CHoCH Bullish
        choch_bull_indices = [seq_index[i] for i in range(len(df)) if df.iloc[i]["CHOCH_bullish"]]
        fig.add_trace(go.Scatter(x=choch_bull_indices, y=df.loc[df.CHOCH_bullish, "structure_level"],
                                 mode="text", text="CHOCH Bull", textposition="top center",
                                 textfont=dict(size=10, color="lime"), name="CHOCH Bullish"))

        # CHoCH Bearish
        choch_bear_indices = [seq_index[i] for i in range(len(df)) if df.iloc[i]["CHOCH_bearish"]]
        fig.add_trace(go.Scatter(x=choch_bear_indices, y=df.loc[df.CHOCH_bearish, "structure_level"],
                                 mode="text", text="CHOCH Bear", textposition="bottom center",
                                 textfont=dict(size=10, color="orange"), name="CHOCH Bearish"))

        label_frequency = max(1, len(seq_index) // 15)
        tickvals = list(range(0, len(seq_index), label_frequency))
        ticktext = [time_labels[i] if i < len(time_labels) else '' for i in tickvals]

        fig.update_layout(title=title, xaxis_title='Time (gaps removed)', yaxis_title='Price',
                          xaxis_rangeslider_visible=False, xaxis=dict(tickvals=tickvals, ticktext=ticktext, tickangle=-45),
                          template="plotly_dark", height=800, hovermode='x unified')
        fig.write_html(html_file)

    def plot_multi_tf_plotly(self, df_1m, df_5m, title, html_file, fvgs_1m=None, fvgs_5m=None):
        """Plot multi-timeframe structure."""
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, subplot_titles=("1-Minute Chart", "5-Minute Chart"))

        # 1m Chart
        seq_index_1m = list(range(len(df_1m)))
        time_labels_1m = df_1m["time"].dt.strftime('%Y-%m-%d %H:%M').tolist()
        fig.add_trace(go.Candlestick(x=seq_index_1m, open=df_1m["open"], high=df_1m["high"], low=df_1m["low"], close=df_1m["close"],
                                     customdata=time_labels_1m, hovertext=[f"<b>{label}</b><br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
                                                                           for label, o, h, l, c in zip(time_labels_1m, df_1m["open"], df_1m["high"], df_1m["low"], df_1m["close"])],
                                     hoverinfo='text', name="1m Price"), row=1, col=1)

        # Add markers for 1m (similar to single plot, but simplified)
        swing_high_1m = [seq_index_1m[i] for i in range(len(df_1m)) if df_1m.iloc[i]["swing_high"]]
        fig.add_trace(go.Scatter(x=swing_high_1m, y=df_1m.loc[df_1m.swing_high, "high"], mode="markers", marker=dict(size=6, color="blue"), name="1m Swing High"), row=1, col=1)
        swing_low_1m = [seq_index_1m[i] for i in range(len(df_1m)) if df_1m.iloc[i]["swing_low"]]
        fig.add_trace(go.Scatter(x=swing_low_1m, y=df_1m.loc[df_1m.swing_low, "low"], mode="markers", marker=dict(size=6, color="orange"), name="1m Swing Low"), row=1, col=1)

        # BOS 1m Bullish
        bos_bull_indices_1m = [seq_index_1m[i] for i in range(len(df_1m)) if df_1m.iloc[i]["BOS_bullish"]]
        fig.add_trace(go.Scatter(x=bos_bull_indices_1m, y=df_1m.loc[df_1m.BOS_bullish, "structure_level"],
                                 mode="text", text="BOS Bull", textposition="top center",
                                 textfont=dict(size=10, color="green"), name="1m BOS Bullish"), row=1, col=1)

        # BOS 1m Bearish
        bos_bear_indices_1m = [seq_index_1m[i] for i in range(len(df_1m)) if df_1m.iloc[i]["BOS_bearish"]]
        fig.add_trace(go.Scatter(x=bos_bear_indices_1m, y=df_1m.loc[df_1m.BOS_bearish, "structure_level"],
                                 mode="text", text="BOS Bear", textposition="bottom center",
                                 textfont=dict(size=10, color="red"), name="1m BOS Bearish"), row=1, col=1)

        # CHoCH 1m Bullish
        choch_bull_indices_1m = [seq_index_1m[i] for i in range(len(df_1m)) if df_1m.iloc[i]["CHOCH_bullish"]]
        fig.add_trace(go.Scatter(x=choch_bull_indices_1m, y=df_1m.loc[df_1m.CHOCH_bullish, "structure_level"],
                                 mode="text", text="CHOCH Bull", textposition="top center",
                                 textfont=dict(size=10, color="lime"), name="1m CHoCH Bullish"), row=1, col=1)

        # CHoCH 1m Bearish
        choch_bear_indices_1m = [seq_index_1m[i] for i in range(len(df_1m)) if df_1m.iloc[i]["CHOCH_bearish"]]
        fig.add_trace(go.Scatter(x=choch_bear_indices_1m, y=df_1m.loc[df_1m.CHOCH_bearish, "structure_level"],
                                 mode="text", text="CHOCH Bear", textposition="bottom center",
                                 textfont=dict(size=10, color="orange"), name="1m CHoCH Bearish"), row=1, col=1)
        if fvgs_1m:
            self._add_fvg_rectangles(fig, fvgs_1m, row=1, x_end=len(df_1m) - 1)

        # 5m Chart
        seq_index_5m = list(range(len(df_5m)))
        time_labels_5m = df_5m["time"].dt.strftime('%Y-%m-%d %H:%M').tolist()
        fig.add_trace(go.Candlestick(x=seq_index_5m, open=df_5m["open"], high=df_5m["high"], low=df_5m["low"], close=df_5m["close"],
                                     customdata=time_labels_5m, hovertext=[f"<b>{label}</b><br>Open: {o:.2f}<br>High: {h:.2f}<br>Low: {l:.2f}<br>Close: {c:.2f}"
                                                                           for label, o, h, l, c in zip(time_labels_5m, df_5m["open"], df_5m["high"], df_5m["low"], df_5m["close"])],
                                     hoverinfo='text', name="5m Price"), row=2, col=1)

        swing_high_5m = [seq_index_5m[i] for i in range(len(df_5m)) if df_5m.iloc[i]["swing_high"]]
        fig.add_trace(go.Scatter(x=swing_high_5m, y=df_5m.loc[df_5m.swing_high, "high"], mode="markers", marker=dict(size=6, color="blue"), name="5m Swing High"), row=2, col=1)
        swing_low_5m = [seq_index_5m[i] for i in range(len(df_5m)) if df_5m.iloc[i]["swing_low"]]
        fig.add_trace(go.Scatter(x=swing_low_5m, y=df_5m.loc[df_5m.swing_low, "low"], mode="markers", marker=dict(size=6, color="orange"), name="5m Swing Low"), row=2, col=1)

        # BOS 5m Bullish
        bos_bull_indices_5m = [seq_index_5m[i] for i in range(len(df_5m)) if df_5m.iloc[i]["BOS_bullish"]]
        fig.add_trace(go.Scatter(x=bos_bull_indices_5m, y=df_5m.loc[df_5m.BOS_bullish, "structure_level"],
                                 mode="text", text="BOS Bull", textposition="top center",
                                 textfont=dict(size=10, color="green"), name="5m BOS Bullish"), row=2, col=1)

        # BOS 5m Bearish
        bos_bear_indices_5m = [seq_index_5m[i] for i in range(len(df_5m)) if df_5m.iloc[i]["BOS_bearish"]]
        fig.add_trace(go.Scatter(x=bos_bear_indices_5m, y=df_5m.loc[df_5m.BOS_bearish, "structure_level"],
                                 mode="text", text="BOS Bear", textposition="bottom center",
                                 textfont=dict(size=10, color="red"), name="5m BOS Bearish"), row=2, col=1)

        # CHoCH 5m Bullish
        choch_bull_indices_5m = [seq_index_5m[i] for i in range(len(df_5m)) if df_5m.iloc[i]["CHOCH_bullish"]]
        fig.add_trace(go.Scatter(x=choch_bull_indices_5m, y=df_5m.loc[df_5m.CHOCH_bullish, "structure_level"],
                                 mode="text", text="CHOCH Bull", textposition="top center",
                                 textfont=dict(size=10, color="lime"), name="5m CHoCH Bullish"), row=2, col=1)

        # CHoCH 5m Bearish
        choch_bear_indices_5m = [seq_index_5m[i] for i in range(len(df_5m)) if df_5m.iloc[i]["CHOCH_bearish"]]
        fig.add_trace(go.Scatter(x=choch_bear_indices_5m, y=df_5m.loc[df_5m.CHOCH_bearish, "structure_level"],
                                 mode="text", text="CHOCH Bear", textposition="bottom center",
                                 textfont=dict(size=10, color="orange"), name="5m CHoCH Bearish"), row=2, col=1)
        if fvgs_5m:
            self._add_fvg_rectangles(fig, fvgs_5m, row=2, x_end=len(df_5m) - 1)
        fig.update_layout(title=title, height=1200, template="plotly_dark", hovermode='x unified')

        label_frequency_1m = max(1, len(seq_index_1m) // 15)
        tickvals_1m = list(range(0, len(seq_index_1m), label_frequency_1m))
        ticktext_1m = [time_labels_1m[i] if i < len(time_labels_1m) else '' for i in tickvals_1m]
        fig.update_xaxes(tickvals=tickvals_1m, ticktext=ticktext_1m, tickangle=-45, row=1, col=1, title_text='Time (1m, gaps removed)')

        label_frequency_5m = max(1, len(seq_index_5m) // 15)
        tickvals_5m = list(range(0, len(seq_index_5m), label_frequency_5m))
        ticktext_5m = [time_labels_5m[i] if i < len(time_labels_5m) else '' for i in tickvals_5m]
        fig.update_xaxes(tickvals=tickvals_5m, ticktext=ticktext_5m, tickangle=-45, row=2, col=1, title_text='Time (5m, gaps removed)')

        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Price', row=2, col=1)

        fig.write_html(html_file)
def compute_cisd(df):
    df = df.copy()

    # Displacement metrics (derived if missing)
    if "displacement_ratio" not in df.columns:
        rng = (df["high"] - df["low"]).to_numpy()
        body = (df["close"] - df["open"]).abs().to_numpy()
        ratio = np.zeros(len(df))
        valid_rng = rng > 0
        ratio[valid_rng] = body[valid_rng] / rng[valid_rng]
        df["displacement_ratio"] = ratio
    if "displacement_valid" not in df.columns:
        df["displacement_valid"] = df["displacement_ratio"] >= 0.6
    if "displacement_direction" not in df.columns:
        df["displacement_direction"] = np.where(
            df["close"] > df["open"],
            "bullish",
            np.where(df["close"] < df["open"], "bearish", "neutral"),
        )

    # Acceptance confirmation (derived if possible)
    if "acceptance_confirmed" not in df.columns and "acceptance_count" in df.columns:
        df["acceptance_confirmed"] = df["acceptance_count"] >= 2

    required_cols = [
        "HTF_structure_break_bullish",
        "HTF_structure_break_bearish",
        "HTF_external_sellside_taken",
        "HTF_external_buyside_taken",
        "acceptance_confirmed",
        "bayesian_posterior_reversal",
        "confidence_score",
        "displacement_valid",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[WARN] Missing CISD columns: {', '.join(missing)} (CISD disabled)")

    if "HTF_structure_break_bullish" not in df.columns:
        df["HTF_structure_break_bullish"] = False
    if "HTF_structure_break_bearish" not in df.columns:
        df["HTF_structure_break_bearish"] = False
    if "HTF_external_sellside_taken" not in df.columns:
        df["HTF_external_sellside_taken"] = False
    if "HTF_external_buyside_taken" not in df.columns:
        df["HTF_external_buyside_taken"] = False
    if "acceptance_confirmed" not in df.columns:
        df["acceptance_confirmed"] = False
    if "bayesian_posterior_reversal" not in df.columns:
        df["bayesian_posterior_reversal"] = 0.0
    if "confidence_score" not in df.columns:
        df["confidence_score"] = 0.0

    df["HTF_external_liquidity_taken"] = (
        df["HTF_external_sellside_taken"].astype(bool)
        | df["HTF_external_buyside_taken"].astype(bool)
    )
    df["HTF_structure_break"] = (
        df["HTF_structure_break_bullish"].astype(bool)
        | df["HTF_structure_break_bearish"].astype(bool)
    )

    base_gate = (
        df["HTF_external_liquidity_taken"].astype(bool)
        & df["HTF_structure_break"].astype(bool)
        & df["displacement_valid"].astype(bool)
        & df["acceptance_confirmed"].astype(bool)
        & (df["bayesian_posterior_reversal"] >= 0.75)
        & (df["confidence_score"] >= 0.75)
    )
    cisd = base_gate & base_gate.shift(1, fill_value=False)

    df["CISD"] = cisd.astype(bool)
    df["CISD_bullish"] = df["CISD"] & df["HTF_structure_break_bullish"].astype(bool)
    df["CISD_bearish"] = df["CISD"] & df["HTF_structure_break_bearish"].astype(bool)
    return df

def assign_market_state(df):
    df = df.copy()
    if "CISD" in df.columns:
        df["market_state"] = np.where(df["CISD"], "REVERSAL", np.where(df["MSS"], "REPRICING", "TREND"))
    else:
        df["market_state"] = np.where(df["MSS"], "REPRICING", "TREND")
    if "swing_trend" in df.columns:
        df["trend_state"] = df["swing_trend"]
    return df

class Main:
    """Main class to orchestrate the entire analysis and visualization process."""

    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.timeframe_builder = TimeframeBuilder(config)
        self.structure_analyzer = StructureAnalyzer(config)
        self.visualizer = Visualizer()

    @staticmethod
    def _dedupe_fvgs_by_candle(fvgs):
        """Keep only one FVG per candle (prefer non-inverse, then larger range)."""
        best_by_idx = {}
        for fvg in fvgs:
            idx = fvg.created_at
            if idx is None:
                continue
            existing = best_by_idx.get(idx)
            if existing is None:
                best_by_idx[idx] = fvg
                continue
            existing_range = abs(existing.upper - existing.lower)
            new_range = abs(fvg.upper - fvg.lower)
            if existing.inverse and not fvg.inverse:
                best_by_idx[idx] = fvg
            elif existing.inverse == fvg.inverse and new_range > existing_range:
                best_by_idx[idx] = fvg
        return [best_by_idx[k] for k in sorted(best_by_idx)]

    def run(self):
        """Run the complete analysis pipeline."""
        print("Extracting data...")
        self.data_manager.extract_data()

        print("Loading parquet files...")
        df = self.data_manager.load_parquet_data()
        if df.empty:
            print("No data loaded.")
            return
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")

        df_ticks = self.data_manager.preprocess_ticks(df)
        tf_data = self.timeframe_builder.build_timeframes(df_ticks)

        df_1m = tf_data["1m"].copy()
        df_5m = tf_data["5m"].copy()
        df_15m = tf_data["15m"].copy()
        df_4h = tf_data["4h"].copy()

        htf_trend = self.structure_analyzer.compute_htf_trend(df_4h)
        if htf_trend is None:
            htf_trend = "BULLISH"
        print(f"[INFO] 4H HTF Trend: {htf_trend}")

        # Analyze each timeframe
        df_1m = self.structure_analyzer.detect_swings(df_1m, self.config.lookback_1m)
        df_1m = self.structure_analyzer.compute_swing_trend(df_1m)
        df_1m = self.structure_analyzer.classify_swing_hierarchy(df_1m)
        df_1m = self.structure_analyzer.detect_bos_choch(df_1m, htf_trend)
        df_1m = compute_cisd(df_1m)
        df_1m = assign_market_state(df_1m)
        fvgs_1m = FairValueGapDetector.detect_fvg(
            df_1m,
            market_state_series=df_1m["market_state"],
            timeframe="1m",
            htf_trend=htf_trend,
            min_range=self.config.fvg_min_range,
            displacement_threshold=self.config.fvg_body_ratio,
        )
        fvgs_1m = FairValueGapDetector.annotate_mitigations(
            fvgs_1m,
            df_1m,
            market_state_series=df_1m["market_state"],
            min_confidence=self.config.fvg_min_confidence,
            interaction_penalty=self.config.fvg_interaction_penalty,
        )
        inv_fvgs_1m = []
        if self.config.plot_inverse_fvgs:
            inv_fvgs_1m = FairValueGapDetector.detect_inverse_fvgs(
                df_1m,
                fvgs_1m,
                min_range=self.config.fvg_min_range,
            )
            inv_fvgs_1m = FairValueGapDetector.annotate_mitigations(
                inv_fvgs_1m,
                df_1m,
                market_state_series=df_1m["market_state"],
                min_confidence=self.config.fvg_min_confidence,
                interaction_penalty=self.config.fvg_interaction_penalty,
            )


        df_5m = self.structure_analyzer.detect_swings(df_5m, self.config.lookback_5m)
        df_5m = self.structure_analyzer.compute_swing_trend(df_5m)
        df_5m = self.structure_analyzer.classify_swing_hierarchy(df_5m)
        df_5m = self.structure_analyzer.detect_bos_choch(df_5m, htf_trend)
        df_5m = compute_cisd(df_5m)
        df_5m = assign_market_state(df_5m)
        fvgs_5m = FairValueGapDetector.detect_fvg(
            df_5m,
            market_state_series=df_5m["market_state"],
            timeframe="5m",
            htf_trend=htf_trend,
            min_range=self.config.fvg_min_range,
            displacement_threshold=self.config.fvg_body_ratio,
        )
        fvgs_5m = FairValueGapDetector.annotate_mitigations(
            fvgs_5m,
            df_5m,
            market_state_series=df_5m["market_state"],
            min_confidence=self.config.fvg_min_confidence,
            interaction_penalty=self.config.fvg_interaction_penalty,
        )
        obs_5m = OrderBlockDetector.detect_order_blocks(
            df_5m,
            market_state_series=df_5m["market_state"],
            htf_trend=htf_trend,
            timeframe="5m",
            displacement_threshold=self.config.fvg_body_ratio,
            require_inefficiency=True,
            htf_mid_series=(df_5m["trend_high"] + df_5m["trend_low"]) / 2.0,
            cisd_series=df_5m["CISD"] if "CISD" in df_5m.columns else None,
        )
        obs_5m = OrderBlockDetector.annotate_order_block_lifecycle(
            obs_5m,
            df_5m,
            market_state_series=df_5m["market_state"],
            min_confidence=self.config.ob_min_confidence,
            interaction_penalty=self.config.ob_interaction_penalty,
            cisd_bullish_series=df_5m["CISD_bullish"] if "CISD_bullish" in df_5m.columns else None,
            cisd_bearish_series=df_5m["CISD_bearish"] if "CISD_bearish" in df_5m.columns else None,
        )
        inv_fvgs_5m = []
        if self.config.plot_inverse_fvgs:
            inv_fvgs_5m = FairValueGapDetector.detect_inverse_fvgs(
                df_5m,
                fvgs_5m,
                min_range=self.config.fvg_min_range,
            )
            inv_fvgs_5m = FairValueGapDetector.annotate_mitigations(
                inv_fvgs_5m,
                df_5m,
                market_state_series=df_5m["market_state"],
                min_confidence=self.config.fvg_min_confidence,
                interaction_penalty=self.config.fvg_interaction_penalty,
            )

        df_15m = self.structure_analyzer.detect_swings(df_15m, self.config.lookback_15m)
        df_15m = self.structure_analyzer.compute_swing_trend(df_15m)
        df_15m = self.structure_analyzer.classify_swing_hierarchy(df_15m)
        df_15m = self.structure_analyzer.detect_bos_choch(df_15m, htf_trend)
        df_15m = compute_cisd(df_15m)
        df_15m = assign_market_state(df_15m)
        fvgs_15m = FairValueGapDetector.detect_fvg(
            df_15m,
            market_state_series=df_15m["market_state"],
            timeframe="15m",
            htf_trend=htf_trend,
            min_range=self.config.fvg_min_range,
            displacement_threshold=self.config.fvg_body_ratio,
        )
        fvgs_15m = FairValueGapDetector.annotate_mitigations(
            fvgs_15m,
            df_15m,
            market_state_series=df_15m["market_state"],
            min_confidence=self.config.fvg_min_confidence,
            interaction_penalty=self.config.fvg_interaction_penalty,
        )
        inv_fvgs_15m = []
        if self.config.plot_inverse_fvgs:
            inv_fvgs_15m = FairValueGapDetector.detect_inverse_fvgs(
                df_15m,
                fvgs_15m,
                min_range=self.config.fvg_min_range,
            )
            inv_fvgs_15m = FairValueGapDetector.annotate_mitigations(
                inv_fvgs_15m,
                df_15m,
                market_state_series=df_15m["market_state"],
                min_confidence=self.config.fvg_min_confidence,
                interaction_penalty=self.config.fvg_interaction_penalty,
            )
        fvgs_1m_plot = self._dedupe_fvgs_by_candle(fvgs_1m + inv_fvgs_1m)
        fvgs_5m_plot = self._dedupe_fvgs_by_candle(fvgs_5m + inv_fvgs_5m)
        fvgs_15m_plot = self._dedupe_fvgs_by_candle(fvgs_15m + inv_fvgs_15m)

        # Plot based on config
        if self.config.plot_15m:
            self.visualizer.plot_structure_plotly(df_15m, f"15m Structure | HTF = {htf_trend}", "structure_validation_15m.html", fvgs=fvgs_15m_plot)
        if self.config.plot_1m_5m:
            self.visualizer.plot_multi_tf_plotly(df_1m, df_5m, f"1m & 5m Structure | HTF = {htf_trend}", "structure_validation_1m_5m.html", fvgs_1m=fvgs_1m_plot, fvgs_5m=fvgs_5m_plot)
        if self.config.plot_1m:
            self.visualizer.plot_structure_plotly(df_1m, f"1m Structure | HTF = {htf_trend}", "structure_validation_1m.html", fvgs=fvgs_1m_plot)
        if self.config.plot_5m:
            self.visualizer.plot_structure_plotly(
                df_5m,
                f"5m Structure | HTF = {htf_trend}",
                "structure_validation_5m.html",
                fvgs=fvgs_5m_plot,
                obs=obs_5m,
            )

        print("[DONE] Plots generated based on configuration.")

if __name__ == "__main__":
    config = Config(zip_files=['nifty data.zip', 'extracted_data/parquet_out (2).zip'])
    main = Main(config)
    main.run()
