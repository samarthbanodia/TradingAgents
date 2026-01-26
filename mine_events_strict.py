#!/usr/bin/env python3
"""
Event Miner v2.1 (Strict) - Two-Stage Intraday Event Detection

Fixes in v2.1:
- RTH-only filtering (09:30-16:00 ET) by default
- Bar-count based windows (not time deltas)
- Gap checking to discard events with missing bars
- Fixed plotting using bar indices
- Clusters do not cross trading day boundaries

Usage examples:
    python mine_events_strict.py --input_path ohlcv_5min.parquet --out_dir out --make_plots
    python mine_events_strict.py --include_extended_hours --out_dir out
    python mine_events_strict.py --confirm_abs_bp 80 --confirm_ret_z 5.0 --out_dir out

Author: Event Miner v2.1
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Optional matplotlib import for plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Event Miner v2.1 (Strict) - Two-stage intraday event detection"
    )

    # Input/Output
    parser.add_argument("--input_path", type=str, default="ohlcv_5min.parquet",
                        help="Path to input file (parquet or csv)")
    parser.add_argument("--out_dir", type=str, default="./out",
                        help="Output directory")

    # Trading hours filter
    parser.add_argument("--include_extended_hours", action="store_true",
                        help="Include extended hours (04:00-20:00 ET). Default: RTH only (09:30-16:00 ET)")

    # Stage 1: Candidate thresholds (high recall, loose)
    parser.add_argument("--cand_ret_z", type=float, default=3.0,
                        help="Stage 1: z-score threshold for return spike")
    parser.add_argument("--cand_vol_mult", type=float, default=4.0,
                        help="Stage 1: volume multiplier threshold")
    parser.add_argument("--cand_range_mult", type=float, default=3.0,
                        help="Stage 1: range multiplier threshold")
    parser.add_argument("--cand_impact_mult", type=float, default=4.0,
                        help="Stage 1: impact multiplier threshold")
    parser.add_argument("--cand_abs_bp", type=float, default=30.0,
                        help="Stage 1: minimum absolute move in basis points")

    # Stage 2: Confirmation thresholds (strict)
    parser.add_argument("--confirm_abs_bp", type=float, default=60.0,
                        help="Stage 2: minimum absolute move in basis points")
    parser.add_argument("--confirm_ret_z", type=float, default=4.0,
                        help="Stage 2: z-score threshold for confirmation")
    parser.add_argument("--confirm_vol_mult", type=float, default=6.0,
                        help="Stage 2: volume multiplier for liquidity confirmation")
    parser.add_argument("--confirm_range_mult", type=float, default=4.0,
                        help="Stage 2: range multiplier for liquidity confirmation")
    parser.add_argument("--confirm_impact_mult", type=float, default=6.0,
                        help="Stage 2: impact multiplier for liquidity confirmation")
    parser.add_argument("--persistence_ratio", type=float, default=0.5,
                        help="Stage 2: ret_6 must be >= this fraction of peak ret_3")

    # Clustering
    parser.add_argument("--cluster_gap", type=int, default=2,
                        help="Max gap (bars) to merge candidates into same cluster")

    # Window sizes (in bars)
    parser.add_argument("--event_bars", type=int, default=4,
                        help="Number of bars for event window (before and including t0)")
    parser.add_argument("--val_bars", type=int, default=12,
                        help="Number of bars for validation window (after t0)")

    # Gap check
    parser.add_argument("--max_gap_minutes", type=float, default=10.0,
                        help="Max allowed gap between consecutive bars in windows (minutes)")

    # Label thresholds
    parser.add_argument("--cont_thresh", type=float, default=0.003,
                        help="Continuation threshold (decimal, 0.003 = 0.30%%)")
    parser.add_argument("--rev_thresh", type=float, default=0.003,
                        help="Reversal threshold (decimal, 0.003 = 0.30%%)")

    # Baseline window
    parser.add_argument("--baseline_window", type=int, default=78,
                        help="Rolling window for baselines (~1 trading day at 5-min)")

    # Plotting
    parser.add_argument("--make_plots", action="store_true",
                        help="Generate event plots (PNG)")
    parser.add_argument("--plot_bars_context", type=int, default=24,
                        help="Number of bars before/after t0 to include in plot")

    return parser.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from parquet or CSV.

    Handles both ISO string and epoch timestamps. Falls back from
    parquet to CSV if parquet not found.
    """
    path = Path(input_path)

    # Try parquet first, then CSV fallback
    if path.suffix == ".parquet" and path.exists():
        print(f"Loading parquet: {path}")
        df = pd.read_parquet(path)
    elif path.suffix == ".csv" and path.exists():
        print(f"Loading CSV: {path}")
        df = pd.read_csv(path)
    elif path.suffix == ".parquet" and not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            print(f"Parquet not found, falling back to CSV: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Neither {path} nor {csv_path} found")
    else:
        raise FileNotFoundError(f"File not found: {path}")

    # Validate required columns
    required_cols = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamp robustly
    if df["timestamp"].dtype == "object" or df["timestamp"].dtype.name.startswith("datetime"):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif np.issubdtype(df["timestamp"].dtype, np.integer):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif np.issubdtype(df["timestamp"].dtype, np.floating):
        max_ts = df["timestamp"].max()
        unit = "ms" if max_ts > 1e11 else "s"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Sort and deduplicate
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["ticker", "timestamp"], keep="last")

    print(f"Loaded {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df


def filter_trading_hours(df: pd.DataFrame, include_extended: bool = False) -> pd.DataFrame:
    """
    Filter to trading hours only.

    Args:
        df: DataFrame with UTC timestamps
        include_extended: If True, include 04:00-20:00 ET. If False, RTH only (09:30-16:00 ET)

    Returns:
        Filtered DataFrame
    """
    print(f"Filtering to {'extended hours (04:00-20:00 ET)' if include_extended else 'RTH only (09:30-16:00 ET)'}...")

    df = df.copy()

    # Convert to NY time
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")

    # Extract time components
    df["ny_weekday"] = df["timestamp_ny"].dt.weekday  # 0=Mon, 6=Sun
    df["ny_time"] = df["timestamp_ny"].dt.time
    df["ny_date"] = df["timestamp_ny"].dt.date

    # Filter weekdays (Mon-Fri)
    weekday_mask = df["ny_weekday"] <= 4

    # Define time bounds
    if include_extended:
        # Extended hours: 04:00 - 20:00 ET
        from datetime import time as dt_time
        start_time = dt_time(4, 0)
        end_time = dt_time(20, 0)
    else:
        # RTH only: 09:30 - 16:00 ET
        from datetime import time as dt_time
        start_time = dt_time(9, 30)
        end_time = dt_time(16, 0)

    time_mask = (df["ny_time"] >= start_time) & (df["ny_time"] <= end_time)

    # Apply filters
    filtered_df = df[weekday_mask & time_mask].copy()

    # Keep ny_date for clustering (to prevent cross-day clusters)
    # Drop temporary columns except ny_date
    filtered_df = filtered_df.drop(columns=["timestamp_ny", "ny_weekday", "ny_time"])

    rows_removed = len(df) - len(filtered_df)
    print(f"Filtered: {len(filtered_df):,} rows kept, {rows_removed:,} removed")

    return filtered_df


def compute_features(df: pd.DataFrame, baseline_window: int = 78) -> pd.DataFrame:
    """
    Compute rolling features per ticker with longer baseline window.

    Features:
    - log_close, ret_1, ret_3, ret_6
    - rv_78: rolling std of ret_1 (1 trading day baseline)
    - vol_med_78: rolling median of volume
    - range, range_med_78
    - impact, impact_med_78
    - Derived: z_ret_3, abs_ret_3_bp, volume_mult, range_mult, impact_mult
    """
    print(f"Computing features (baseline window = {baseline_window} bars)...")

    df = df.copy()
    bw = baseline_window

    # Initialize columns
    feature_cols = [
        "log_close", "ret_1", "ret_3", "ret_6",
        "rv_78", "vol_med_78", "range", "range_med_78",
        "impact", "impact_med_78",
        "z_ret_3", "abs_ret_3_bp", "volume_mult", "range_mult", "impact_mult"
    ]
    for col in feature_cols:
        df[col] = np.nan

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        idx = df.loc[mask].index

        close = df.loc[idx, "close"]
        high = df.loc[idx, "high"]
        low = df.loc[idx, "low"]
        volume = df.loc[idx, "volume"]

        # Log close and returns
        log_close = np.log(close)
        ret_1 = log_close.diff(1)
        ret_3 = log_close.diff(3)
        ret_6 = log_close.diff(6)

        # Rolling volatility (std of 1-bar returns over baseline window)
        rv = ret_1.rolling(window=bw, min_periods=bw).std()

        # Rolling median volume
        vol_med = volume.rolling(window=bw, min_periods=bw).median()

        # Range normalized by close
        bar_range = (high - low) / close
        range_med = bar_range.rolling(window=bw, min_periods=bw).median()

        # Impact (Amihud-style)
        safe_volume = volume.clip(lower=1)
        impact = ret_1.abs() / safe_volume
        impact_med = impact.rolling(window=bw, min_periods=bw).median()

        # Derived metrics
        sqrt3 = math.sqrt(3)
        z_ret_3 = ret_3.abs() / (rv * sqrt3)

        # Absolute return in basis points: abs(exp(ret_3) - 1) * 10000
        abs_ret_3_bp = (np.exp(ret_3.abs()) - 1) * 10000

        # Multipliers
        volume_mult = volume / vol_med
        range_mult = bar_range / range_med
        impact_mult = impact / impact_med

        # Assign back
        df.loc[idx, "log_close"] = log_close
        df.loc[idx, "ret_1"] = ret_1
        df.loc[idx, "ret_3"] = ret_3
        df.loc[idx, "ret_6"] = ret_6
        df.loc[idx, "rv_78"] = rv
        df.loc[idx, "vol_med_78"] = vol_med
        df.loc[idx, "range"] = bar_range
        df.loc[idx, "range_med_78"] = range_med
        df.loc[idx, "impact"] = impact
        df.loc[idx, "impact_med_78"] = impact_med
        df.loc[idx, "z_ret_3"] = z_ret_3
        df.loc[idx, "abs_ret_3_bp"] = abs_ret_3_bp
        df.loc[idx, "volume_mult"] = volume_mult
        df.loc[idx, "range_mult"] = range_mult
        df.loc[idx, "impact_mult"] = impact_mult

    valid_count = df["rv_78"].notna().sum()
    print(f"Features computed. Valid baseline bars: {valid_count:,}")
    return df


def stage1_candidate_flags(
    df: pd.DataFrame,
    cand_ret_z: float = 3.0,
    cand_vol_mult: float = 4.0,
    cand_range_mult: float = 3.0,
    cand_impact_mult: float = 4.0,
    cand_abs_bp: float = 30.0
) -> pd.DataFrame:
    """
    Stage 1: Flag candidate bars with loose gating (high recall).

    A bar becomes a candidate if:
    - abs_ret_3_bp >= cand_abs_bp (minimum absolute move)
    AND ANY of:
    - z_ret_3 > cand_ret_z
    - volume_mult > cand_vol_mult
    - range_mult > cand_range_mult
    - impact_mult > cand_impact_mult
    """
    print(f"Stage 1: Flagging candidates (ret_z>{cand_ret_z}, vol>{cand_vol_mult}, "
          f"range>{cand_range_mult}, impact>{cand_impact_mult}, abs_bp>={cand_abs_bp})...")

    df = df.copy()

    # Minimum absolute move gate
    abs_gate = df["abs_ret_3_bp"] >= cand_abs_bp

    # Individual trigger conditions
    trig_ret = df["z_ret_3"] > cand_ret_z
    trig_vol = df["volume_mult"] > cand_vol_mult
    trig_range = df["range_mult"] > cand_range_mult
    trig_impact = df["impact_mult"] > cand_impact_mult

    # Any trigger fires
    any_trigger = trig_ret | trig_vol | trig_range | trig_impact

    # Candidate = absolute gate AND any trigger
    df["candidate"] = (abs_gate & any_trigger).fillna(False)

    # Store individual flags for analysis
    df["cand_trig_ret"] = trig_ret.fillna(False)
    df["cand_trig_vol"] = trig_vol.fillna(False)
    df["cand_trig_range"] = trig_range.fillna(False)
    df["cand_trig_impact"] = trig_impact.fillna(False)

    cand_count = df["candidate"].sum()
    print(f"Stage 1 candidates: {cand_count:,} bars")

    return df


def cluster_candidates(df: pd.DataFrame, cluster_gap: int = 2) -> pd.DataFrame:
    """
    Cluster nearby candidate bars into event clusters.

    Candidates within cluster_gap bars are merged into the same cluster.
    Clusters do NOT cross trading day boundaries (based on ny_date).

    Returns DataFrame with cluster_id column (-1 if not in a cluster).
    """
    print(f"Clustering candidates (gap <= {cluster_gap} bars, no cross-day)...")

    df = df.copy()
    df["cluster_id"] = -1

    cluster_counter = 0

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_df = df.loc[mask].reset_index(drop=False)
        ticker_df = ticker_df.rename(columns={"index": "orig_idx"})

        # Find candidate bars with their positions and dates
        candidate_rows = ticker_df[ticker_df["candidate"]].copy()

        if candidate_rows.empty:
            continue

        # Group by date first to prevent cross-day clusters
        for ny_date, date_group in candidate_rows.groupby("ny_date"):
            if date_group.empty:
                continue

            # Get positional indices within ticker_df for this date
            positions = []
            for _, row in date_group.iterrows():
                # Find position in ticker_df
                pos = ticker_df[ticker_df["orig_idx"] == row["orig_idx"]].index[0]
                positions.append((pos, row["orig_idx"]))

            if not positions:
                continue

            positions.sort(key=lambda x: x[0])

            # Cluster within this date
            current_start_pos = positions[0][0]
            current_end_pos = positions[0][0]
            current_orig_indices = [positions[0][1]]

            for pos, orig_idx in positions[1:]:
                if pos - current_end_pos <= cluster_gap + 1:
                    # Extend cluster
                    current_end_pos = pos
                    current_orig_indices.append(orig_idx)
                else:
                    # Finalize cluster
                    for oi in current_orig_indices:
                        df.loc[oi, "cluster_id"] = cluster_counter
                    cluster_counter += 1

                    # Start new cluster
                    current_start_pos = pos
                    current_end_pos = pos
                    current_orig_indices = [orig_idx]

            # Finalize last cluster for this date
            for oi in current_orig_indices:
                df.loc[oi, "cluster_id"] = cluster_counter
            cluster_counter += 1

    num_clusters = df[df["cluster_id"] >= 0]["cluster_id"].nunique()
    print(f"Candidate clusters: {num_clusters:,}")

    return df


def check_window_gaps(timestamps: pd.Series, max_gap_minutes: float) -> bool:
    """
    Check if any gap between consecutive timestamps exceeds max_gap_minutes.

    Returns True if window is valid (no gaps exceed threshold).
    """
    if len(timestamps) < 2:
        return True

    timestamps = timestamps.sort_values()
    gaps = timestamps.diff().dropna()

    # Convert to minutes
    gap_minutes = gaps.dt.total_seconds() / 60

    return gap_minutes.max() <= max_gap_minutes


def check_same_trading_day(timestamps: pd.Series) -> bool:
    """
    Check if all timestamps are on the same trading day (NY timezone).

    Returns True if all bars are on the same day.
    """
    if len(timestamps) < 2:
        return True

    ny_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    return ny_dates.nunique() == 1


def stage2_confirm_clusters(
    df: pd.DataFrame,
    confirm_abs_bp: float = 60.0,
    confirm_ret_z: float = 4.0,
    confirm_vol_mult: float = 6.0,
    confirm_range_mult: float = 4.0,
    confirm_impact_mult: float = 6.0,
    persistence_ratio: float = 0.5
) -> list:
    """
    Stage 2: Confirm clusters with strict gating.

    A cluster is confirmed if:
    1) Absolute move gate: max abs_ret_3_bp in cluster >= confirm_abs_bp
    2) Strength gate: max z_ret_3 in cluster >= confirm_ret_z
    3) Liquidity/attention: at least ONE of:
       - max volume_mult >= confirm_vol_mult
       - max range_mult >= confirm_range_mult
       - max impact_mult >= confirm_impact_mult
    4) Persistence: either cluster_len >= 2 OR
       abs(ret_6 at cluster_end) >= persistence_ratio * peak_ret_3

    Returns list of confirmed cluster info dicts.
    """
    print(f"Stage 2: Confirming clusters (abs_bp>={confirm_abs_bp}, ret_z>={confirm_ret_z}, "
          f"vol>={confirm_vol_mult} OR range>={confirm_range_mult} OR impact>={confirm_impact_mult})...")

    cluster_ids = df[df["cluster_id"] >= 0]["cluster_id"].unique()
    confirmed_clusters = []

    for cluster_id in cluster_ids:
        cluster_mask = df["cluster_id"] == cluster_id
        cluster_df = df[cluster_mask]

        if cluster_df.empty:
            continue

        ticker = cluster_df["ticker"].iloc[0]

        # Cluster metrics
        peak_abs_ret_3_bp = cluster_df["abs_ret_3_bp"].max()
        peak_z_ret_3 = cluster_df["z_ret_3"].max()
        peak_volume_mult = cluster_df["volume_mult"].max()
        peak_range_mult = cluster_df["range_mult"].max()
        peak_impact_mult = cluster_df["impact_mult"].max()

        cluster_len = len(cluster_df)

        # t0 is the FIRST bar in cluster (by timestamp)
        t0_idx = cluster_df["timestamp"].idxmin()
        t0 = cluster_df.loc[t0_idx, "timestamp"]
        cluster_end_idx = cluster_df["timestamp"].idxmax()

        # Peak ret_3 (absolute value)
        peak_ret_3_abs = cluster_df["ret_3"].abs().max()

        # ret_6 at cluster end for persistence check
        ret_6_at_end = cluster_df.loc[cluster_end_idx, "ret_6"]
        if pd.isna(ret_6_at_end):
            ret_6_at_end = 0

        # === CONFIRMATION GATES ===

        # Gate 1: Absolute move
        gate1_abs = peak_abs_ret_3_bp >= confirm_abs_bp

        # Gate 2: Strength (z-score)
        gate2_strength = peak_z_ret_3 >= confirm_ret_z

        # Gate 3: Liquidity/attention (any one)
        gate3_liquidity = (
            peak_volume_mult >= confirm_vol_mult or
            peak_range_mult >= confirm_range_mult or
            peak_impact_mult >= confirm_impact_mult
        )

        # Gate 4: Persistence (multi-bar OR ret_6 persists)
        persistence_check = abs(ret_6_at_end) >= persistence_ratio * peak_ret_3_abs
        gate4_persistence = cluster_len >= 2 or persistence_check

        # All gates must pass
        confirmed = gate1_abs and gate2_strength and gate3_liquidity and gate4_persistence

        if confirmed:
            # Direction from ret_3 at t0
            ret_3_at_t0 = cluster_df.loc[t0_idx, "ret_3"]
            if pd.isna(ret_3_at_t0) or ret_3_at_t0 == 0:
                ret_6_at_t0 = cluster_df.loc[t0_idx, "ret_6"]
                if pd.isna(ret_6_at_t0) or ret_6_at_t0 == 0:
                    # Fallback to price change
                    close_t0 = cluster_df.loc[t0_idx, "close"]
                    # Find previous bar in full df
                    ticker_mask = df["ticker"] == ticker
                    ticker_df = df[ticker_mask].sort_values("timestamp").reset_index(drop=True)
                    t0_pos_list = ticker_df[ticker_df["timestamp"] == t0].index.tolist()
                    if t0_pos_list and t0_pos_list[0] > 0:
                        t0_pos = t0_pos_list[0]
                        prev_close = ticker_df.iloc[t0_pos - 1]["close"]
                        direction = 1 if close_t0 >= prev_close else -1
                    else:
                        direction = 1
                else:
                    direction = 1 if ret_6_at_t0 >= 0 else -1
            else:
                direction = 1 if ret_3_at_t0 >= 0 else -1

            confirmed_clusters.append({
                "cluster_id": cluster_id,
                "ticker": ticker,
                "t0": t0,
                "t0_orig_idx": t0_idx,
                "cluster_end_orig_idx": cluster_end_idx,
                "direction": direction,
                "cluster_len": cluster_len,
                "peak_abs_ret_3_bp": peak_abs_ret_3_bp,
                "peak_z_ret_3": peak_z_ret_3,
                "peak_volume_mult": peak_volume_mult,
                "peak_range_mult": peak_range_mult,
                "peak_impact_mult": peak_impact_mult,
            })

    print(f"Stage 2 confirmed: {len(confirmed_clusters):,} clusters")
    return confirmed_clusters


def extract_events(
    df: pd.DataFrame,
    confirmed_clusters: list,
    event_bars: int = 4,
    val_bars: int = 12,
    max_gap_minutes: float = 10.0,
    cont_thresh: float = 0.003,
    rev_thresh: float = 0.003
) -> pd.DataFrame:
    """
    Extract event records from confirmed clusters using BAR COUNT windows.

    For each confirmed cluster at position i (t0):
    - Event window: df.iloc[i-event_bars+1 : i+1] (event_bars bars ending at t0)
    - Validation window: df.iloc[i+1 : i+1+val_bars] (val_bars bars after t0)

    Windows are validated for:
    1) Same trading day (no overnight gaps)
    2) Max gap between consecutive bars (no missing data)
    """
    print(f"Extracting events (event_bars={event_bars}, val_bars={val_bars}, max_gap={max_gap_minutes}min)...")

    events = []
    skipped_history = 0
    skipped_future = 0
    skipped_event_day = 0
    skipped_val_day = 0
    skipped_event_gap = 0
    skipped_val_gap = 0

    for cluster_info in confirmed_clusters:
        ticker = cluster_info["ticker"]
        t0 = cluster_info["t0"]
        direction = cluster_info["direction"]

        # Get ticker data as a contiguous array with positional indexing
        ticker_mask = df["ticker"] == ticker
        ticker_df = df[ticker_mask].sort_values("timestamp").reset_index(drop=True)

        # Find t0 position in ticker_df
        t0_matches = ticker_df[ticker_df["timestamp"] == t0].index.tolist()
        if not t0_matches:
            continue
        t0_pos = t0_matches[0]

        # Check window availability
        if t0_pos < event_bars - 1:
            skipped_history += 1
            continue
        if t0_pos + val_bars >= len(ticker_df):
            skipped_future += 1
            continue

        # Event window: bars [t0_pos - event_bars + 1, t0_pos] inclusive
        event_start_pos = t0_pos - event_bars + 1
        event_end_pos = t0_pos
        event_window_df = ticker_df.iloc[event_start_pos:event_end_pos + 1]

        # Validation window: bars [t0_pos + 1, t0_pos + val_bars] inclusive
        val_start_pos = t0_pos + 1
        val_end_pos = t0_pos + val_bars
        val_window_df = ticker_df.iloc[val_start_pos:val_end_pos + 1]

        # Check event window is on same trading day
        if not check_same_trading_day(event_window_df["timestamp"]):
            skipped_event_day += 1
            continue

        # Check validation window is on same trading day
        if not check_same_trading_day(val_window_df["timestamp"]):
            skipped_val_day += 1
            continue

        # Gap check for event window (within same day, check for missing bars)
        if not check_window_gaps(event_window_df["timestamp"], max_gap_minutes):
            skipped_event_gap += 1
            continue

        # Gap check for validation window
        if not check_window_gaps(val_window_df["timestamp"], max_gap_minutes):
            skipped_val_gap += 1
            continue

        # Extract timestamps
        event_window_start = event_window_df["timestamp"].iloc[0]
        event_window_end = event_window_df["timestamp"].iloc[-1]
        val_window_start = val_window_df["timestamp"].iloc[0]
        val_window_end = val_window_df["timestamp"].iloc[-1]

        # Returns
        close_event_start = event_window_df["close"].iloc[0]
        close_t0 = event_window_df["close"].iloc[-1]
        close_val_end = val_window_df["close"].iloc[-1]

        event_return = (close_t0 / close_event_start) - 1
        forward_return_60m = (close_val_end / close_t0) - 1

        # Label proxy
        if direction == 1:
            if forward_return_60m > cont_thresh:
                label_proxy = "continuation"
            elif forward_return_60m < -rev_thresh:
                label_proxy = "reversal"
            else:
                label_proxy = "unclear"
        else:
            if forward_return_60m < -cont_thresh:
                label_proxy = "continuation"
            elif forward_return_60m > rev_thresh:
                label_proxy = "reversal"
            else:
                label_proxy = "unclear"

        event_id = f"{ticker}_{t0.strftime('%Y%m%d_%H%M%S')}"

        events.append({
            "event_id": event_id,
            "ticker": ticker,
            "t0_utc": t0,
            "t0_pos": t0_pos,  # Store position for plotting
            "direction": direction,
            "cluster_len": cluster_info["cluster_len"],
            "peak_abs_ret_3_bp": cluster_info["peak_abs_ret_3_bp"],
            "peak_z_ret_3": cluster_info["peak_z_ret_3"],
            "peak_volume_mult": cluster_info["peak_volume_mult"],
            "peak_range_mult": cluster_info["peak_range_mult"],
            "peak_impact_mult": cluster_info["peak_impact_mult"],
            "event_window_start": event_window_start,
            "event_window_end": event_window_end,
            "val_window_start": val_window_start,
            "val_window_end": val_window_end,
            "event_return": event_return,
            "forward_return_60m": forward_return_60m,
            "label_proxy": label_proxy,
        })

    events_df = pd.DataFrame(events)

    print(f"Events extracted: {len(events_df):,}")
    print(f"  Skipped (not enough history): {skipped_history}")
    print(f"  Skipped (not enough future): {skipped_future}")
    print(f"  Skipped (event window crosses day): {skipped_event_day}")
    print(f"  Skipped (val window crosses day): {skipped_val_day}")
    print(f"  Skipped (event window gap): {skipped_event_gap}")
    print(f"  Skipped (val window gap): {skipped_val_gap}")

    return events_df


def make_event_plot(
    df: pd.DataFrame,
    event: pd.Series,
    out_dir: Path,
    plot_bars_context: int = 24
) -> None:
    """
    Generate a plot for a single event using BAR INDEX for x-axis.

    Uses sequential bar indices to avoid blank regions from overnight gaps.
    X-axis shows bar labels with timestamps at key points.
    """
    if not HAS_MATPLOTLIB:
        return

    ticker = event["ticker"]
    t0 = event["t0_utc"]
    t0_pos = event["t0_pos"]
    event_start = event["event_window_start"]
    event_end = event["event_window_end"]
    val_start = event["val_window_start"]
    val_end = event["val_window_end"]
    direction = event["direction"]

    # Get ticker data
    ticker_mask = df["ticker"] == ticker
    ticker_df = df[ticker_mask].sort_values("timestamp").reset_index(drop=True)

    # Plot window by BAR INDEX (not time)
    plot_start_pos = max(0, t0_pos - plot_bars_context)
    plot_end_pos = min(len(ticker_df), t0_pos + plot_bars_context + 1)

    plot_df = ticker_df.iloc[plot_start_pos:plot_end_pos].copy().reset_index(drop=True)

    if len(plot_df) < 5:
        return

    # Use sequential integers for x-axis (bar index)
    x_vals = np.arange(len(plot_df))

    # Find positions of key points in plot_df
    t0_plot_idx = None
    event_start_idx = None
    event_end_idx = None
    val_start_idx = None
    val_end_idx = None

    for i, row in plot_df.iterrows():
        ts = row["timestamp"]
        if ts == t0:
            t0_plot_idx = i
        if ts == event_start:
            event_start_idx = i
        if ts == event_end:
            event_end_idx = i
        if ts == val_start:
            val_start_idx = i
        if ts == val_end:
            val_end_idx = i

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot close price using bar indices
    ax.plot(x_vals, plot_df["close"].values, "b-", linewidth=1.5, label="Close", zorder=5)

    # Shade event window
    if event_start_idx is not None and event_end_idx is not None:
        color = "green" if direction == 1 else "red"
        ax.axvspan(event_start_idx - 0.5, event_end_idx + 0.5,
                   alpha=0.3, color=color, label="Event Window", zorder=1)

    # Shade validation window
    if val_start_idx is not None and val_end_idx is not None:
        ax.axvspan(val_start_idx - 0.5, val_end_idx + 0.5,
                   alpha=0.2, color="blue", label="Validation Window", zorder=1)

    # Mark t0
    if t0_plot_idx is not None:
        ax.axvline(t0_plot_idx, color="black", linestyle="--", linewidth=2,
                   label=f"t0 (dir={direction:+d})", zorder=10)

    # Title with stats
    t0_ny = t0.tz_convert("America/New_York")
    ax.set_title(
        f"{ticker} - Event at {t0_ny.strftime('%Y-%m-%d %H:%M')} ET\n"
        f"Peak: {event['peak_abs_ret_3_bp']:.0f}bp | z={event['peak_z_ret_3']:.1f} | "
        f"vol_mult={event['peak_volume_mult']:.1f}x | fwd_ret={event['forward_return_60m']*100:.2f}%",
        fontsize=10
    )
    ax.set_xlabel("Bar (5-min intervals)")
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Create x-tick labels showing time at every ~6 bars
    tick_step = max(1, len(plot_df) // 8)
    tick_positions = list(range(0, len(plot_df), tick_step))
    tick_labels = []
    for pos in tick_positions:
        ts = plot_df.iloc[pos]["timestamp"]
        ts_ny = ts.tz_convert("America/New_York")
        tick_labels.append(ts_ny.strftime('%m-%d\n%H:%M'))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    # Add text annotation
    event_ret = event["event_return"] * 100
    fwd_ret = event["forward_return_60m"] * 100
    label = event["label_proxy"]

    text = (f"Event Return: {event_ret:.2f}%\n"
            f"Fwd Return: {fwd_ret:.2f}%\n"
            f"Label: {label}\n"
            f"Cluster: {event['cluster_len']} bars")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()

    # Save
    plot_dir = out_dir / "plots" / ticker
    plot_dir.mkdir(parents=True, exist_ok=True)

    filename = f"event_{t0.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(plot_dir / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: Path,
    plot_bars_context: int = 24
) -> None:
    """Generate plots for all events using bar-index based context."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots")
        return

    print(f"Generating {len(events_df)} plots (context: {plot_bars_context} bars each side)...")

    for i, (_, event) in enumerate(events_df.iterrows()):
        make_event_plot(df, event, out_dir, plot_bars_context)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(events_df)} plots")

    print(f"Plots saved to {out_dir / 'plots'}")


def write_summary(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: Path,
    args
) -> dict:
    """Generate and write summary statistics."""
    total_bars = len(df)
    candidate_bars = df["candidate"].sum() if "candidate" in df.columns else 0
    candidate_clusters = df[df["cluster_id"] >= 0]["cluster_id"].nunique() if "cluster_id" in df.columns else 0
    confirmed_events = len(events_df)

    # Per ticker
    per_ticker = {}
    if len(events_df) > 0:
        per_ticker = events_df.groupby("ticker").size().to_dict()

    # Label distribution
    label_dist = {}
    if len(events_df) > 0 and "label_proxy" in events_df.columns:
        label_dist = events_df["label_proxy"].value_counts().to_dict()

    # Distribution stats
    dist_stats = {}
    if len(events_df) > 0:
        dist_stats = {
            "peak_abs_ret_3_bp": {
                "median": float(events_df["peak_abs_ret_3_bp"].median()),
                "p95": float(events_df["peak_abs_ret_3_bp"].quantile(0.95)),
                "max": float(events_df["peak_abs_ret_3_bp"].max()),
            },
            "peak_volume_mult": {
                "median": float(events_df["peak_volume_mult"].median()),
                "p95": float(events_df["peak_volume_mult"].quantile(0.95)),
                "max": float(events_df["peak_volume_mult"].max()),
            },
            "peak_z_ret_3": {
                "median": float(events_df["peak_z_ret_3"].median()),
                "p95": float(events_df["peak_z_ret_3"].quantile(0.95)),
                "max": float(events_df["peak_z_ret_3"].max()),
            },
        }

    summary = {
        "total_bars_loaded": int(total_bars),
        "trading_hours": "extended (04:00-20:00 ET)" if args.include_extended_hours else "RTH (09:30-16:00 ET)",
        "stage1_candidate_bars": int(candidate_bars),
        "stage1_candidate_clusters": int(candidate_clusters),
        "stage2_confirmed_events": int(confirmed_events),
        "per_ticker_event_counts": per_ticker,
        "label_distribution": label_dist,
        "distribution_stats": dist_stats,
        "thresholds": {
            "cand_ret_z": args.cand_ret_z,
            "cand_vol_mult": args.cand_vol_mult,
            "confirm_abs_bp": args.confirm_abs_bp,
            "confirm_ret_z": args.confirm_ret_z,
        }
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total bars (after filtering):  {total_bars:,}")
    print(f"Trading hours:                 {summary['trading_hours']}")
    print(f"Stage 1 candidate bars:        {candidate_bars:,}")
    print(f"Stage 1 candidate clusters:    {candidate_clusters:,}")
    print(f"Stage 2 confirmed events:      {confirmed_events:,}")

    print("\nPer-ticker event counts:")
    for ticker, count in sorted(per_ticker.items()):
        print(f"  {ticker}: {count}")

    print("\nLabel distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count}")

    if dist_stats:
        print("\nDistribution stats:")
        print(f"  peak_abs_ret_3_bp: median={dist_stats['peak_abs_ret_3_bp']['median']:.1f}, "
              f"p95={dist_stats['peak_abs_ret_3_bp']['p95']:.1f}, max={dist_stats['peak_abs_ret_3_bp']['max']:.1f}")
        print(f"  peak_volume_mult:  median={dist_stats['peak_volume_mult']['median']:.1f}, "
              f"p95={dist_stats['peak_volume_mult']['p95']:.1f}, max={dist_stats['peak_volume_mult']['max']:.1f}")
        print(f"  peak_z_ret_3:      median={dist_stats['peak_z_ret_3']['median']:.1f}, "
              f"p95={dist_stats['peak_z_ret_3']['p95']:.1f}, max={dist_stats['peak_z_ret_3']['max']:.1f}")

    # Top 10 events
    if len(events_df) > 0:
        print("\n" + "-" * 70)
        print("TOP 10 EVENTS by peak_abs_ret_3_bp:")
        print("-" * 70)
        top10 = events_df.nlargest(10, "peak_abs_ret_3_bp")[
            ["ticker", "t0_utc", "direction", "peak_abs_ret_3_bp", "peak_z_ret_3",
             "peak_volume_mult", "forward_return_60m", "label_proxy"]
        ]
        for _, row in top10.iterrows():
            print(f"  {row['ticker']} @ {row['t0_utc'].strftime('%Y-%m-%d %H:%M')} | "
                  f"dir={row['direction']:+d} | {row['peak_abs_ret_3_bp']:.0f}bp | "
                  f"z={row['peak_z_ret_3']:.1f} | vol_x={row['peak_volume_mult']:.1f} | "
                  f"fwd={row['forward_return_60m']*100:.2f}% | {row['label_proxy']}")

    print("=" * 70)

    # Write JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    return summary


def main():
    """Main entry point."""
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Event Miner v2.1 (Strict) - Two-Stage Detection")
    print("=" * 70)
    print(f"Input: {args.input_path}")
    print(f"Output: {out_dir}")
    print(f"Trading hours: {'Extended (04:00-20:00 ET)' if args.include_extended_hours else 'RTH only (09:30-16:00 ET)'}")
    print(f"\nStage 1 (Candidate) thresholds:")
    print(f"  ret_z > {args.cand_ret_z}, vol_mult > {args.cand_vol_mult}, "
          f"range_mult > {args.cand_range_mult}, impact_mult > {args.cand_impact_mult}")
    print(f"  abs_bp >= {args.cand_abs_bp}")
    print(f"\nStage 2 (Confirm) thresholds:")
    print(f"  abs_bp >= {args.confirm_abs_bp}, ret_z >= {args.confirm_ret_z}")
    print(f"  liquidity: vol >= {args.confirm_vol_mult} OR range >= {args.confirm_range_mult} "
          f"OR impact >= {args.confirm_impact_mult}")
    print(f"\nWindows: event={args.event_bars} bars, val={args.val_bars} bars, max_gap={args.max_gap_minutes}min")
    print(f"Baseline window: {args.baseline_window} bars")
    print("=" * 70 + "\n")

    # Load data
    df = load_data(args.input_path)

    # Filter to trading hours
    df = filter_trading_hours(df, include_extended=args.include_extended_hours)

    # Re-sort and reset index after filtering
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # Compute features
    df = compute_features(df, baseline_window=args.baseline_window)

    # Stage 1: Candidate flagging
    df = stage1_candidate_flags(
        df,
        cand_ret_z=args.cand_ret_z,
        cand_vol_mult=args.cand_vol_mult,
        cand_range_mult=args.cand_range_mult,
        cand_impact_mult=args.cand_impact_mult,
        cand_abs_bp=args.cand_abs_bp
    )

    # Cluster candidates (no cross-day)
    df = cluster_candidates(df, cluster_gap=args.cluster_gap)

    # Stage 2: Confirm clusters
    confirmed_clusters = stage2_confirm_clusters(
        df,
        confirm_abs_bp=args.confirm_abs_bp,
        confirm_ret_z=args.confirm_ret_z,
        confirm_vol_mult=args.confirm_vol_mult,
        confirm_range_mult=args.confirm_range_mult,
        confirm_impact_mult=args.confirm_impact_mult,
        persistence_ratio=args.persistence_ratio
    )

    # Extract events with gap checking
    events_df = extract_events(
        df,
        confirmed_clusters,
        event_bars=args.event_bars,
        val_bars=args.val_bars,
        max_gap_minutes=args.max_gap_minutes,
        cont_thresh=args.cont_thresh,
        rev_thresh=args.rev_thresh
    )

    # Save events (drop t0_pos before saving)
    if len(events_df) > 0:
        events_path = out_dir / "events.csv"
        save_df = events_df.drop(columns=["t0_pos"], errors="ignore")
        save_df.to_csv(events_path, index=False)
        print(f"\nEvents written to {events_path}")
    else:
        print("\nNo events found!")

    # Generate plots
    if args.make_plots and len(events_df) > 0:
        generate_plots(df, events_df, out_dir, plot_bars_context=args.plot_bars_context)

    # Write summary
    write_summary(df, events_df, out_dir, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
