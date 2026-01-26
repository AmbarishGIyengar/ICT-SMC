from dataclasses import dataclass
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
    plot_15m: bool = True
    plot_1m_5m: bool = False
    plot_1m: bool = True
    plot_5m: bool = True

class DataManager:
    """Handles data extraction, loading, and preprocessing."""

    def __init__(self, config: Config):
        self.config = config

    def extract_data(self):
        """Extract zip files to the specified directory if not already extracted."""
        extract_path = Path(self.config.extract_dir)
        if not extract_path.exists():
            extract_path.mkdir(parents=True)
            for zip_path in self.config.zip_files:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.config.extract_dir)

    def load_parquet_data(self):
        """Load all parquet files from the data directory into a single DataFrame."""
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
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            return df
        return pd.DataFrame()

    def preprocess_ticks(self, df):
        """Preprocess tick data: ensure datetime, sort, filter market hours, keep required columns."""
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        market_open = pd.to_datetime("09:15:00").time()
        market_close = pd.to_datetime("15:30:00").time()
        df = df[
            (df["time"].dt.time >= market_open) &
            (df["time"].dt.time <= market_close)
        ]
        df = df[["time", "ltp"]]
        df = df[
            (df["time"] >= self.config.date_start) &
            (df["time"] <= self.config.date_end)
        ]
        return df

class TimeframeBuilder:
    """Builds OHLC timeframes from tick data."""

    def __init__(self, config: Config):
        self.config = config

    def resample_ohlc(self, df, timeframe):
        """Resample tick data into OHLC candles for the given timeframe."""
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

        for i in range(lookback, len(df) - lookback):
            window_high = df["high"].iloc[i-lookback:i+lookback+1].max()
            window_low = df["low"].iloc[i-lookback:i+lookback+1].min()
            is_high = df["high"].iloc[i] == window_high
            is_low = df["low"].iloc[i] == window_low

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

    def classify_swing_hierarchy(self, df):
        """Classify swings into external and internal based on trend."""
        df = df.copy()
        df["swing_label"] = None
        external_high = external_low = None
        external_high_idx = external_low_idx = None
        current_trend = None

        swing_points = []
        for i in range(len(df)):
            if df.loc[i, "swing_high"]:
                swing_points.append(("HIGH", i))
            elif df.loc[i, "swing_low"]:
                swing_points.append(("LOW", i))

            if len(swing_points) >= 2:
                first, second = swing_points[0], swing_points[1]
                if first[0] == "LOW" and second[0] == "HIGH":
                    external_low = df.loc[first[1], "low"]
                    external_high = df.loc[second[1], "high"]
                    external_low_idx, external_high_idx = first[1], second[1]
                    current_trend = "BULLISH"
                elif first[0] == "HIGH" and second[0] == "LOW":
                    external_high = df.loc[first[1], "high"]
                    external_low = df.loc[second[1], "low"]
                    external_high_idx, external_low_idx = first[1], second[1]
                    current_trend = "BEARISH"

                if current_trend:
                    df.loc[external_high_idx, "swing_label"] = "EXTERNAL_HIGH"
                    df.loc[external_low_idx, "swing_label"] = "EXTERNAL_LOW"
                    break

        for i in range(max(external_high_idx or 0, external_low_idx or 0) + 1, len(df)):
            if df.loc[i, "swing_high"]:
                price = df.loc[i, "high"]
                if current_trend == "BULLISH" and price > external_high:
                    df.loc[i, "swing_label"] = "EXTERNAL_HIGH"
                    external_high = price
                else:
                    df.loc[i, "swing_label"] = "INTERNAL_HIGH"
            elif df.loc[i, "swing_low"]:
                price = df.loc[i, "low"]
                if current_trend == "BEARISH" and price < external_low:
                    df.loc[i, "swing_label"] = "EXTERNAL_LOW"
                    external_low = price
                else:
                    df.loc[i, "swing_label"] = "INTERNAL_LOW"

        return df

    def detect_bos_choch(self, df, initial_trend, disp_threshold=0.6):
        df = df.copy()

        # Outputs
        df["BOS_bullish"] = False
        df["BOS_bearish"] = False
        df["CHOCH_bullish"] = False
        df["CHOCH_bearish"] = False
        df["MSS"] = False
        df["structure_level"] = np.nan
        df["structure_type"] = None
        df["effective_trend"] = initial_trend

        # -----------------------------
        # Structure anchors (SWING-BASED ONLY)
        # -----------------------------
        external_high = None
        external_low = None
        internal_high = None
        internal_low = None

        # -----------------------------
        # State
        # -----------------------------
        trend = initial_trend                  # BULLISH / BEARISH
        phase = "NORMAL"                       # NORMAL / CHOCH_SEEN
        bos_allowed = True                     # Rule 4: One BOS per leg

        for i in range(len(df)):

            high = df.loc[i, "high"]
            low = df.loc[i, "low"]
            open_ = df.loc[i, "open"]
            close = df.loc[i, "close"]

            # -----------------------------
            # Acceptance
            # -----------------------------
            rng = high - low
            if rng == 0:
                continue

            accepted = abs(close - open_) / rng >= disp_threshold

            # -----------------------------
            # Update structure anchors from swings based on current trend
            # -----------------------------
            if df.loc[i, "swing_high"]:
                if external_high is None or high > external_high:
                    external_high = high
                    bos_allowed = True  # Reset BOS allowance for new leg
                else:
                    internal_high = high

            elif df.loc[i, "swing_low"]:
                if external_low is None or low < external_low:
                    external_low = low
                    bos_allowed = True  # Reset BOS allowance for new leg
                else:
                    internal_low = low

            if (external_high is None and trend == "BULLISH") or (external_low is None and trend == "BEARISH"):
                continue

            # =============================
            # STATE: NORMAL
            # =============================
            if phase == "NORMAL":

                # ---- BOS (continuation) - Rules 1,2,3,4,5
                if bos_allowed and trend == "BULLISH" and close > external_high and accepted:
                    df.loc[i, "BOS_bullish"] = True
                    df.loc[i, "structure_level"] = external_high
                    df.loc[i, "structure_type"] = "BOS"
                    bos_allowed = False  # Disable until new leg

                elif bos_allowed and trend == "BEARISH" and close < external_low and accepted:
                    df.loc[i, "BOS_bearish"] = True
                    df.loc[i, "structure_level"] = external_low
                    df.loc[i, "structure_type"] = "BOS"
                    bos_allowed = False  # Disable until new leg

                # ---- CHoCH (trend reversal signal)
                elif (
                    trend == "BULLISH"
                    and internal_low is not None
                    and close < internal_low
                ):
                    df.loc[i, "CHOCH_bearish"] = True
                    df.loc[i, "structure_level"] = internal_low
                    df.loc[i, "structure_type"] = "CHOCH"
                    phase = "CHOCH_SEEN"

                elif (
                    trend == "BEARISH"
                    and internal_high is not None
                    and close > internal_high
                ):
                    df.loc[i, "CHOCH_bullish"] = True
                    df.loc[i, "structure_level"] = internal_high
                    df.loc[i, "structure_type"] = "CHOCH"
                    phase = "CHOCH_SEEN"

            # =============================
            # STATE: CHOCH_SEEN
            # =============================
            elif phase == "CHOCH_SEEN":

                # ---- MSS (external confirmation)
                if trend == "BULLISH" and close < external_low and accepted:
                    df.loc[i, "MSS"] = True
                    df.loc[i, "structure_level"] = external_low
                    df.loc[i, "structure_type"] = "MSS"
                    trend = "BEARISH"
                    phase = "NORMAL"
                    bos_allowed = True  # Reset BOS allowance for new leg after MSS
                    internal_high = external_high
                    external_high = None

                elif trend == "BEARISH" and close > external_high and accepted:
                    df.loc[i, "MSS"] = True
                    df.loc[i, "structure_level"] = external_high
                    df.loc[i, "structure_type"] = "MSS"
                    trend = "BULLISH"
                    phase = "NORMAL"
                    bos_allowed = True  # Reset BOS allowance for new leg after MSS
                    internal_low = external_low
                    external_low = None

            # -----------------------------
            # Persist state
            # -----------------------------
            df.loc[i, "effective_trend"] = trend
            df.loc[i, "structure_state"] = phase

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

class Visualizer:
    """Handles plotting of market structure using Plotly."""

    def plot_structure_plotly(self, df, title, html_file):
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

    def plot_multi_tf_plotly(self, df_1m, df_5m, title, html_file):
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

class Main:
    """Main class to orchestrate the entire analysis and visualization process."""

    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.timeframe_builder = TimeframeBuilder(config)
        self.structure_analyzer = StructureAnalyzer(config)
        self.visualizer = Visualizer()

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
        df_1m = self.structure_analyzer.classify_swing_hierarchy(df_1m)
        df_1m = self.structure_analyzer.detect_bos_choch(df_1m, htf_trend)

        df_5m = self.structure_analyzer.detect_swings(df_5m, self.config.lookback_5m)
        df_5m = self.structure_analyzer.classify_swing_hierarchy(df_5m)
        df_5m = self.structure_analyzer.detect_bos_choch(df_5m, htf_trend)

        df_15m = self.structure_analyzer.detect_swings(df_15m, self.config.lookback_15m)
        df_15m = self.structure_analyzer.classify_swing_hierarchy(df_15m)
        df_15m = self.structure_analyzer.detect_bos_choch(df_15m, htf_trend)

        # Plot based on config
        if self.config.plot_15m:
            self.visualizer.plot_structure_plotly(df_15m, f"15m Structure | HTF = {htf_trend}", "structure_validation_15m.html")
        if self.config.plot_1m_5m:
            self.visualizer.plot_multi_tf_plotly(df_1m, df_5m, f"1m & 5m Structure | HTF = {htf_trend}", "structure_validation_1m_5m.html")
        if self.config.plot_1m:
            self.visualizer.plot_structure_plotly(df_1m, f"1m Structure | HTF = {htf_trend}", "structure_validation_1m.html")
        if self.config.plot_5m:
            self.visualizer.plot_structure_plotly(df_5m, f"5m Structure | HTF = {htf_trend}", "structure_validation_5m.html")

        print("[DONE] Plots generated based on configuration.")

if __name__ == "__main__":
    config = Config(zip_files=['nifty data.zip', 'extracted_data/parquet_out (2).zip'])
    main = Main(config)
    main.run()