#!/usr/bin/env python3
"""
Event Miner v1 - Intraday Regime Attribution

Detects abnormal intraday events from 5-minute OHLCV data using rolling
baselines and configurable thresholds. Outputs event windows for analysis.

Usage examples:
    python mine_events.py --input_path ohlcv_5min.parquet --out_dir out --make_plots
    python mine_events.py --input_path ohlcv_5min.csv --out_dir out
    python mine_events.py --ret_spike_mult 3.0 --vol_spike_mult 4.0 --out_dir out

Author: Event Miner v1
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
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Event Miner v1 - Detect abnormal intraday events from OHLCV data"
    )

    # Input/Output
    parser.add_argument(
        "--input_path", type=str, default="ohlcv_5min.parquet",
        help="Path to input file (parquet or csv). Default: ohlcv_5min.parquet"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./out",
        help="Output directory. Default: ./out"
    )
    parser.add_argument(
        "--tz", type=str, default="UTC",
        help="Timezone for display (internal is always UTC). Default: UTC"
    )

    # Trigger thresholds
    parser.add_argument(
        "--ret_spike_mult", type=float, default=2.5,
        help="Return spike multiplier (vs rv_20 * sqrt(3)). Default: 2.5"
    )
    parser.add_argument(
        "--vol_spike_mult", type=float, default=3.0,
        help="Volume spike multiplier (vs vol_med_20). Default: 3.0"
    )
    parser.add_argument(
        "--range_spike_mult", type=float, default=2.0,
        help="Range expansion multiplier (vs range_med_20). Default: 2.0"
    )
    parser.add_argument(
        "--impact_spike_mult", type=float, default=3.0,
        help="Impact spike multiplier (vs impact_med_20). Default: 3.0"
    )

    # Window sizes
    parser.add_argument(
        "--event_bars", type=int, default=4,
        help="Number of bars for event window (before t0). Default: 4 (20 min)"
    )
    parser.add_argument(
        "--val_bars", type=int, default=12,
        help="Number of bars for validation window (after t0). Default: 12 (60 min)"
    )
    parser.add_argument(
        "--cluster_gap", type=int, default=2,
        help="Max gap (bars) to merge triggers into same cluster. Default: 2"
    )

    # Label thresholds
    parser.add_argument(
        "--cont_thresh", type=float, default=0.003,
        help="Continuation threshold (decimal). Default: 0.003 (0.3%%)"
    )
    parser.add_argument(
        "--rev_thresh", type=float, default=0.003,
        help="Reversal threshold (decimal). Default: 0.003 (0.3%%)"
    )

    # Plotting
    parser.add_argument(
        "--make_plots", action="store_true",
        help="Generate event plots (PNG)"
    )
    parser.add_argument(
        "--plot_hours_before", type=float, default=2.0,
        help="Hours before t0 to include in plot. Default: 2.0"
    )
    parser.add_argument(
        "--plot_hours_after", type=float, default=2.0,
        help="Hours after t0 to include in plot. Default: 2.0"
    )

    return parser.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from parquet or CSV.

    Prefers parquet if available; falls back to CSV.
    Handles both ISO string and epoch timestamps.

    Args:
        input_path: Path to data file

    Returns:
        DataFrame with parsed timestamps, sorted by ticker and timestamp
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
        # Try CSV fallback
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

    # Parse timestamp
    if df["timestamp"].dtype == "object" or df["timestamp"].dtype.name.startswith("datetime"):
        # ISO string or already datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif np.issubdtype(df["timestamp"].dtype, np.integer):
        # Epoch milliseconds (common from Polygon)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif np.issubdtype(df["timestamp"].dtype, np.floating):
        # Epoch seconds or milliseconds
        max_ts = df["timestamp"].max()
        unit = "ms" if max_ts > 1e11 else "s"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
    else:
        # Try generic parsing
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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling features per ticker.

    Features computed:
    - ret_1: 1-bar log return
    - ret_3: 3-bar log return (~15 min)
    - rv_20: Rolling std of ret_1 over 20 bars
    - vol_med_20: Rolling median of volume over 20 bars
    - range: (high - low) / close
    - range_med_20: Rolling median of range over 20 bars
    - impact: abs(ret_1) / max(volume, 1)
    - impact_med_20: Rolling median of impact over 20 bars

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added feature columns
    """
    print("Computing rolling features...")

    # Work on a copy
    df = df.copy()

    # Log price for returns
    df["log_close"] = np.log(df["close"])

    # Initialize feature columns
    df["ret_1"] = np.nan
    df["ret_3"] = np.nan
    df["rv_20"] = np.nan
    df["vol_med_20"] = np.nan
    df["range"] = np.nan
    df["range_med_20"] = np.nan
    df["impact"] = np.nan
    df["impact_med_20"] = np.nan

    # Compute per ticker
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        idx = df.loc[mask].index

        log_close = df.loc[idx, "log_close"]
        volume = df.loc[idx, "volume"]
        high = df.loc[idx, "high"]
        low = df.loc[idx, "low"]
        close = df.loc[idx, "close"]

        # Returns
        ret_1 = log_close.diff(1)
        ret_3 = log_close.diff(3)

        # Rolling volatility (std of 1-bar returns)
        rv_20 = ret_1.rolling(window=20, min_periods=20).std()

        # Rolling median volume
        vol_med_20 = volume.rolling(window=20, min_periods=20).median()

        # Range normalized by close
        bar_range = (high - low) / close
        range_med_20 = bar_range.rolling(window=20, min_periods=20).median()

        # Impact (Amihud-style)
        # Avoid division by zero
        safe_volume = volume.clip(lower=1)
        impact = ret_1.abs() / safe_volume
        impact_med_20 = impact.rolling(window=20, min_periods=20).median()

        # Assign back
        df.loc[idx, "ret_1"] = ret_1
        df.loc[idx, "ret_3"] = ret_3
        df.loc[idx, "rv_20"] = rv_20
        df.loc[idx, "vol_med_20"] = vol_med_20
        df.loc[idx, "range"] = bar_range
        df.loc[idx, "range_med_20"] = range_med_20
        df.loc[idx, "impact"] = impact
        df.loc[idx, "impact_med_20"] = impact_med_20

    # Drop temporary column
    df = df.drop(columns=["log_close"])

    print(f"Features computed. Valid rv_20 bars: {df['rv_20'].notna().sum():,}")
    return df


def flag_triggers(
    df: pd.DataFrame,
    ret_spike_mult: float = 2.5,
    vol_spike_mult: float = 3.0,
    range_spike_mult: float = 2.0,
    impact_spike_mult: float = 3.0
) -> pd.DataFrame:
    """
    Flag bars that meet trigger conditions.

    Trigger conditions (any triggers the bar):
    A) Return spike: abs(ret_3) > ret_spike_mult * rv_20 * sqrt(3)
    B) Volume spike: volume > vol_spike_mult * vol_med_20
    C) Range expansion: range > range_spike_mult * range_med_20
    D) Impact spike: impact > impact_spike_mult * impact_med_20

    Args:
        df: DataFrame with computed features
        ret_spike_mult: Multiplier for return spike threshold
        vol_spike_mult: Multiplier for volume spike threshold
        range_spike_mult: Multiplier for range expansion threshold
        impact_spike_mult: Multiplier for impact spike threshold

    Returns:
        DataFrame with trigger flag columns
    """
    print("Flagging trigger conditions...")

    df = df.copy()

    # Compute thresholds
    sqrt3 = math.sqrt(3)

    # Return spike: |ret_3| > mult * rv_20 * sqrt(3)
    # sqrt(3) scales the 1-bar vol to 3-bar
    ret_threshold = ret_spike_mult * df["rv_20"] * sqrt3
    df["trig_ret"] = df["ret_3"].abs() > ret_threshold

    # Volume spike: volume > mult * vol_med_20
    df["trig_vol"] = df["volume"] > vol_spike_mult * df["vol_med_20"]

    # Range expansion: range > mult * range_med_20
    df["trig_range"] = df["range"] > range_spike_mult * df["range_med_20"]

    # Impact spike: impact > mult * impact_med_20
    df["trig_impact"] = df["impact"] > impact_spike_mult * df["impact_med_20"]

    # Combined trigger (any condition)
    df["triggered"] = (
        df["trig_ret"] | df["trig_vol"] | df["trig_range"] | df["trig_impact"]
    )

    # Handle NaN baselines (no trigger if baseline is NaN)
    df["triggered"] = df["triggered"].fillna(False)

    triggered_count = df["triggered"].sum()
    print(f"Total triggered bars: {triggered_count:,}")

    return df


def cluster_triggers(df: pd.DataFrame, cluster_gap: int = 2) -> pd.DataFrame:
    """
    Merge nearby triggered bars into event clusters.

    Triggered bars within cluster_gap bars of each other are merged
    into the same cluster. Each cluster gets a unique cluster_id.

    Args:
        df: DataFrame with trigger flags
        cluster_gap: Maximum gap (bars) to merge into same cluster

    Returns:
        DataFrame with cluster_id column
    """
    print(f"Clustering triggers (gap <= {cluster_gap} bars)...")

    df = df.copy()
    df["cluster_id"] = -1  # -1 means not in a cluster

    cluster_counter = 0

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_df = df.loc[mask].copy()
        ticker_idx = ticker_df.index.tolist()

        triggered_positions = [
            i for i, idx in enumerate(ticker_idx)
            if ticker_df.loc[idx, "triggered"]
        ]

        if not triggered_positions:
            continue

        # Group consecutive/nearby triggers
        current_cluster_start = triggered_positions[0]
        current_cluster_end = triggered_positions[0]

        for pos in triggered_positions[1:]:
            # If within gap, extend cluster
            if pos - current_cluster_end <= cluster_gap + 1:
                current_cluster_end = pos
            else:
                # Finalize previous cluster
                for p in range(current_cluster_start, current_cluster_end + 1):
                    if p < len(ticker_idx):
                        idx = ticker_idx[p]
                        if ticker_df.loc[idx, "triggered"]:
                            df.loc[idx, "cluster_id"] = cluster_counter
                cluster_counter += 1

                # Start new cluster
                current_cluster_start = pos
                current_cluster_end = pos

        # Finalize last cluster
        for p in range(current_cluster_start, current_cluster_end + 1):
            if p < len(ticker_idx):
                idx = ticker_idx[p]
                if ticker_df.loc[idx, "triggered"]:
                    df.loc[idx, "cluster_id"] = cluster_counter
        cluster_counter += 1

    num_clusters = df[df["cluster_id"] >= 0]["cluster_id"].nunique()
    print(f"Total event clusters: {num_clusters:,}")

    return df


def extract_events(
    df: pd.DataFrame,
    event_bars: int = 4,
    val_bars: int = 12,
    cont_thresh: float = 0.003,
    rev_thresh: float = 0.003
) -> pd.DataFrame:
    """
    Extract event records from clustered triggers.

    For each cluster:
    - t0 is the timestamp of the first bar in the cluster
    - Direction is sign of ret_3 at t0
    - Event window: event_bars before t0 (inclusive)
    - Validation window: val_bars after t0

    Args:
        df: DataFrame with cluster_id
        event_bars: Number of bars for event window
        val_bars: Number of bars for validation window
        cont_thresh: Threshold for continuation label
        rev_thresh: Threshold for reversal label

    Returns:
        DataFrame with one row per event
    """
    print("Extracting events...")

    events = []
    cluster_ids = df[df["cluster_id"] >= 0]["cluster_id"].unique()

    for cluster_id in cluster_ids:
        cluster_mask = df["cluster_id"] == cluster_id
        cluster_df = df[cluster_mask]

        if cluster_df.empty:
            continue

        ticker = cluster_df["ticker"].iloc[0]
        t0 = cluster_df["timestamp"].min()
        t0_idx = cluster_df["timestamp"].idxmin()

        # Get full ticker data
        ticker_mask = df["ticker"] == ticker
        ticker_df = df[ticker_mask].sort_values("timestamp").reset_index(drop=True)

        # Find position of t0 in ticker data
        t0_pos = ticker_df[ticker_df["timestamp"] == t0].index
        if len(t0_pos) == 0:
            continue
        t0_pos = t0_pos[0]

        # Check if we have enough bars for windows
        if t0_pos < event_bars:
            continue  # Not enough history
        if t0_pos + val_bars >= len(ticker_df):
            continue  # Not enough future

        # Event window: [t0 - event_bars, t0]
        event_start_pos = t0_pos - event_bars
        event_end_pos = t0_pos
        event_window_start = ticker_df.iloc[event_start_pos]["timestamp"]
        event_window_end = ticker_df.iloc[event_end_pos]["timestamp"]

        # Validation window: (t0, t0 + val_bars]
        val_start_pos = t0_pos + 1
        val_end_pos = t0_pos + val_bars
        val_window_start = ticker_df.iloc[val_start_pos]["timestamp"]
        val_window_end = ticker_df.iloc[val_end_pos]["timestamp"]

        # Direction from ret_3 at t0
        ret_3_at_t0 = cluster_df.loc[t0_idx, "ret_3"]
        if pd.isna(ret_3_at_t0) or ret_3_at_t0 == 0:
            # Fallback: use ret_1
            ret_1_at_t0 = cluster_df.loc[t0_idx, "ret_1"]
            direction = 1 if ret_1_at_t0 >= 0 else -1
        else:
            direction = 1 if ret_3_at_t0 >= 0 else -1

        # Cluster statistics
        cluster_len = len(cluster_df)
        max_abs_ret_3 = cluster_df["ret_3"].abs().max()

        # Volume multiple
        vol_mult = cluster_df["volume"] / cluster_df["vol_med_20"]
        max_volume_mult = vol_mult.max() if not vol_mult.isna().all() else np.nan

        # Range multiple
        range_mult = cluster_df["range"] / cluster_df["range_med_20"]
        max_range_mult = range_mult.max() if not range_mult.isna().all() else np.nan

        # Impact multiple
        impact_mult = cluster_df["impact"] / cluster_df["impact_med_20"]
        max_impact_mult = impact_mult.max() if not impact_mult.isna().all() else np.nan

        # Event return: close[t0] / close[event_start] - 1
        close_event_start = ticker_df.iloc[event_start_pos]["close"]
        close_t0 = ticker_df.iloc[event_end_pos]["close"]
        event_return = (close_t0 / close_event_start) - 1

        # Forward return 60m: close[val_end] / close[t0] - 1
        close_val_end = ticker_df.iloc[val_end_pos]["close"]
        forward_return_60m = (close_val_end / close_t0) - 1

        # Label proxy
        if direction == 1:
            # Positive event
            if forward_return_60m > cont_thresh:
                label_proxy = "continuation"
            elif forward_return_60m < -rev_thresh:
                label_proxy = "reversal"
            else:
                label_proxy = "unclear"
        else:
            # Negative event
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
            "direction": direction,
            "cluster_len": cluster_len,
            "max_abs_ret_3": max_abs_ret_3,
            "max_volume_mult": max_volume_mult,
            "max_range_mult": max_range_mult,
            "max_impact_mult": max_impact_mult,
            "event_window_start": event_window_start,
            "event_window_end": event_window_end,
            "val_window_start": val_window_start,
            "val_window_end": val_window_end,
            "event_return": event_return,
            "forward_return_60m": forward_return_60m,
            "label_proxy": label_proxy,
        })

    events_df = pd.DataFrame(events)
    print(f"Events extracted: {len(events_df):,} (after window checks)")

    return events_df


def make_event_plot(
    df: pd.DataFrame,
    event: pd.Series,
    out_dir: Path,
    hours_before: float = 2.0,
    hours_after: float = 2.0
) -> None:
    """
    Generate a plot for a single event.

    Shows close price with t0, event window, and validation window marked.

    Args:
        df: Full DataFrame with OHLCV data
        event: Series with event details
        out_dir: Output directory
        hours_before: Hours before t0 to include
        hours_after: Hours after t0 to include
    """
    if not HAS_MATPLOTLIB:
        return

    ticker = event["ticker"]
    t0 = event["t0_utc"]
    event_start = event["event_window_start"]
    event_end = event["event_window_end"]
    val_start = event["val_window_start"]
    val_end = event["val_window_end"]
    direction = event["direction"]

    # Get ticker data
    ticker_df = df[df["ticker"] == ticker].copy()
    ticker_df = ticker_df.sort_values("timestamp")

    # Define plot range
    plot_start = t0 - pd.Timedelta(hours=hours_before)
    plot_end = t0 + pd.Timedelta(hours=hours_after)

    plot_df = ticker_df[
        (ticker_df["timestamp"] >= plot_start) &
        (ticker_df["timestamp"] <= plot_end)
    ]

    if len(plot_df) < 5:
        return  # Not enough data

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(plot_df["timestamp"], plot_df["close"], "b-", linewidth=1, label="Close")

    # Mark event window (shaded green/red based on direction)
    color = "green" if direction == 1 else "red"
    ax.axvspan(event_start, event_end, alpha=0.2, color=color, label="Event Window")

    # Mark validation window (shaded gray)
    ax.axvspan(val_start, val_end, alpha=0.1, color="blue", label="Validation Window")

    # Mark t0
    ax.axvline(t0, color="black", linestyle="--", linewidth=2, label=f"t0 (dir={direction})")

    # Format
    ax.set_title(f"{ticker} - Event at {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Close Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    # Add annotations
    event_ret = event["event_return"] * 100
    fwd_ret = event["forward_return_60m"] * 100
    label = event["label_proxy"]

    text = f"Event Return: {event_ret:.2f}%\nFwd Return 60m: {fwd_ret:.2f}%\nLabel: {label}"
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()

    # Save
    plot_dir = out_dir / "plots" / ticker
    plot_dir.mkdir(parents=True, exist_ok=True)

    filename = f"event_{t0.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(plot_dir / filename, dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_plots(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: Path,
    hours_before: float = 2.0,
    hours_after: float = 2.0
) -> None:
    """
    Generate plots for all events.

    Args:
        df: Full DataFrame with OHLCV data
        events_df: DataFrame with event records
        out_dir: Output directory
        hours_before: Hours before t0 to include
        hours_after: Hours after t0 to include
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots")
        return

    print(f"Generating {len(events_df)} plots...")

    for i, (_, event) in enumerate(events_df.iterrows()):
        make_event_plot(df, event, out_dir, hours_before, hours_after)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(events_df)} plots")

    print(f"Plots saved to {out_dir / 'plots'}")


def write_summary(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: Path
) -> dict:
    """
    Generate and write summary statistics.

    Args:
        df: Full DataFrame with features
        events_df: DataFrame with event records
        out_dir: Output directory

    Returns:
        Dictionary with summary statistics
    """
    total_bars = len(df)
    triggered_bars = df["triggered"].sum() if "triggered" in df.columns else 0
    total_clusters = df[df["cluster_id"] >= 0]["cluster_id"].nunique() if "cluster_id" in df.columns else 0
    events_kept = len(events_df)

    # Per ticker counts
    if len(events_df) > 0:
        per_ticker = events_df.groupby("ticker").size().to_dict()
    else:
        per_ticker = {}

    # Label distribution
    if len(events_df) > 0 and "label_proxy" in events_df.columns:
        label_dist = events_df["label_proxy"].value_counts().to_dict()
    else:
        label_dist = {}

    summary = {
        "total_bars_loaded": int(total_bars),
        "total_triggered_bars": int(triggered_bars),
        "total_event_clusters": int(total_clusters),
        "events_kept_after_window_checks": int(events_kept),
        "per_ticker_event_counts": per_ticker,
        "label_distribution": label_dist,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total bars loaded:           {total_bars:,}")
    print(f"Total triggered bars:        {triggered_bars:,}")
    print(f"Total event clusters:        {total_clusters:,}")
    print(f"Events kept (after checks):  {events_kept:,}")
    print("\nPer-ticker event counts:")
    for ticker, count in sorted(per_ticker.items()):
        print(f"  {ticker}: {count}")
    print("\nLabel distribution:")
    for label, count in sorted(label_dist.items()):
        print(f"  {label}: {count}")
    print("=" * 60)

    # Write JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    return summary


def main():
    """Main entry point."""
    args = parse_args()

    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Event Miner v1")
    print("=" * 60)
    print(f"Input: {args.input_path}")
    print(f"Output: {out_dir}")
    print(f"Thresholds: ret={args.ret_spike_mult}, vol={args.vol_spike_mult}, "
          f"range={args.range_spike_mult}, impact={args.impact_spike_mult}")
    print(f"Windows: event={args.event_bars} bars, val={args.val_bars} bars")
    print("=" * 60 + "\n")

    # Load data
    df = load_data(args.input_path)

    # Compute features
    df = compute_features(df)

    # Flag triggers
    df = flag_triggers(
        df,
        ret_spike_mult=args.ret_spike_mult,
        vol_spike_mult=args.vol_spike_mult,
        range_spike_mult=args.range_spike_mult,
        impact_spike_mult=args.impact_spike_mult
    )

    # Cluster triggers
    df = cluster_triggers(df, cluster_gap=args.cluster_gap)

    # Extract events
    events_df = extract_events(
        df,
        event_bars=args.event_bars,
        val_bars=args.val_bars,
        cont_thresh=args.cont_thresh,
        rev_thresh=args.rev_thresh
    )

    # Save events CSV
    if len(events_df) > 0:
        events_path = out_dir / "events.csv"
        events_df.to_csv(events_path, index=False)
        print(f"\nEvents written to {events_path}")
    else:
        print("\nNo events found!")

    # Generate plots if requested
    if args.make_plots and len(events_df) > 0:
        generate_plots(
            df, events_df, out_dir,
            hours_before=args.plot_hours_before,
            hours_after=args.plot_hours_after
        )

    # Write summary
    write_summary(df, events_df, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
