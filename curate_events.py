#!/usr/bin/env python3
"""
Event Curator - Multi-Factor Quality Selection for Intraday Regime Attribution

Curates high-quality intraday spike events from candidates using:
1) Multi-factor quality scoring (magnitude, z-score, attention, liquidity, shape)
2) Regular Trading Hours (RTH) filtering
3) Per-ticker quota enforcement (min/max coverage)
4) Archetype diversity (info-driven, panic, macro spillover, idiosyncratic)

Produces a balanced, high-quality curated set for regime attribution research.

Usage:
    python curate_events.py --candidates out_strict/events.csv --out_dir curated
    python curate_events.py --target_total 150 --stock_min 12 --stock_max 25
    python curate_events.py --include_extended_hours --out_dir curated_extended

Author: Event Curator v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Timezone handling
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ticker classification
STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMD", "NFLX", "PLTR"]
ETF_TICKERS = ["SPY", "QQQ", "XLK"]
ALL_TICKERS = STOCK_TICKERS + ETF_TICKERS

# Regular Trading Hours (ET)
RTH_START_HOUR = 9
RTH_START_MIN = 30
RTH_END_HOUR = 16
RTH_END_MIN = 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Curate high-quality intraday spike events with balanced coverage"
    )

    # Input/Output
    parser.add_argument("--candidates", type=str, default="out_strict/events.csv",
                        help="Path to candidate events CSV")
    parser.add_argument("--ohlcv", type=str, default="ohlcv_5min.parquet",
                        help="Path to OHLCV data (parquet or csv)")
    parser.add_argument("--out_dir", type=str, default="./curated",
                        help="Output directory")

    # Target counts
    parser.add_argument("--target_total", type=int, default=120,
                        help="Target total number of curated events")

    # Per-ticker quotas for stocks
    parser.add_argument("--stock_min", type=int, default=10,
                        help="Minimum events per stock ticker")
    parser.add_argument("--stock_max", type=int, default=20,
                        help="Maximum events per stock ticker")

    # Per-ticker quotas for ETFs
    parser.add_argument("--etf_min", type=int, default=5,
                        help="Minimum events per ETF ticker")
    parser.add_argument("--etf_max", type=int, default=15,
                        help="Maximum events per ETF ticker")

    # RTH settings
    parser.add_argument("--include_extended_hours", action="store_true",
                        help="Include events outside regular trading hours")
    parser.add_argument("--max_gap_minutes", type=int, default=30,
                        help="Max allowed gap in event/val window (minutes)")

    # Score weights (for tuning)
    parser.add_argument("--w_mag", type=float, default=0.25,
                        help="Weight for magnitude score")
    parser.add_argument("--w_z", type=float, default=0.20,
                        help="Weight for z-score")
    parser.add_argument("--w_attn", type=float, default=0.20,
                        help="Weight for attention (volume/range)")
    parser.add_argument("--w_liq", type=float, default=0.15,
                        help="Weight for liquidity stress (impact)")
    parser.add_argument("--w_shape", type=float, default=0.20,
                        help="Weight for event shape quality")

    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_candidates(path: str) -> pd.DataFrame:
    """Load candidate events from CSV."""
    print(f"Loading candidates from {path}...")
    df = pd.read_csv(path)

    # Parse timestamps
    time_cols = ["t0_utc", "event_window_start", "event_window_end",
                 "val_window_start", "val_window_end"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    print(f"Loaded {len(df)} candidate events")
    return df


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV data for archetype analysis."""
    path = Path(path)

    if path.suffix == ".parquet" and path.exists():
        df = pd.read_parquet(path)
    elif path.suffix == ".csv" and path.exists():
        df = pd.read_csv(path)
    elif path.suffix == ".parquet" and not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            print(f"Warning: OHLCV file not found, archetype tagging will be limited")
            return None
    else:
        print(f"Warning: OHLCV file not found at {path}")
        return None

    # Parse timestamp
    if "timestamp" in df.columns:
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif not hasattr(df["timestamp"].dt, "tz") or df["timestamp"].dt.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


# ============================================================================
# RTH FILTERING
# ============================================================================

def convert_to_et(ts):
    """Convert UTC timestamp to America/New_York."""
    if HAS_PYTZ:
        et_tz = pytz.timezone("America/New_York")
        return ts.astimezone(et_tz)
    else:
        # Fallback: assume EST (-5 hours) - not DST aware
        return ts - pd.Timedelta(hours=5)


def is_within_rth(ts) -> bool:
    """Check if timestamp is within Regular Trading Hours (9:30-16:00 ET)."""
    ts_et = convert_to_et(ts)

    # Check weekday (Mon=0, Fri=4)
    if ts_et.weekday() > 4:
        return False

    # Check time
    time_minutes = ts_et.hour * 60 + ts_et.minute
    rth_start = RTH_START_HOUR * 60 + RTH_START_MIN  # 9:30 = 570
    rth_end = RTH_END_HOUR * 60 + RTH_END_MIN        # 16:00 = 960

    return rth_start <= time_minutes <= rth_end


def check_window_continuity(event, ohlcv_df, max_gap_minutes: int = 30) -> bool:
    """
    Check if event and validation windows are continuous (no large gaps).

    Returns True if windows are clean, False if there are overnight gaps
    or missing data that would make the event unreliable.
    """
    if ohlcv_df is None:
        return True  # Can't check, assume OK

    ticker = event["ticker"]
    event_start = event["event_window_start"]
    val_end = event["val_window_end"]

    # Get ticker data in the window
    ticker_df = ohlcv_df[
        (ohlcv_df["ticker"] == ticker) &
        (ohlcv_df["timestamp"] >= event_start) &
        (ohlcv_df["timestamp"] <= val_end)
    ].sort_values("timestamp")

    if len(ticker_df) < 2:
        return False

    # Check for large gaps
    timestamps = ticker_df["timestamp"].values
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 'm')
        if gap > max_gap_minutes:
            return False

    return True


def filter_rth_events(
    df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    include_extended: bool = False,
    max_gap_minutes: int = 30
) -> pd.DataFrame:
    """
    Filter events to those within Regular Trading Hours with clean windows.

    Args:
        df: Candidate events DataFrame
        ohlcv_df: OHLCV data for continuity checks
        include_extended: If True, skip RTH filtering
        max_gap_minutes: Maximum allowed gap in windows
    """
    print(f"\nFiltering for RTH (include_extended={include_extended})...")

    if include_extended:
        # Only filter for window continuity
        mask = df.apply(
            lambda row: check_window_continuity(row, ohlcv_df, max_gap_minutes),
            axis=1
        )
    else:
        # Filter for RTH AND window continuity
        def is_valid_event(row):
            # Check t0 is in RTH
            if not is_within_rth(row["t0_utc"]):
                return False

            # Check event window start is in RTH
            if not is_within_rth(row["event_window_start"]):
                return False

            # Check val window end is in RTH
            if not is_within_rth(row["val_window_end"]):
                return False

            # Check window continuity
            if not check_window_continuity(row, ohlcv_df, max_gap_minutes):
                return False

            return True

        mask = df.apply(is_valid_event, axis=1)

    df_filtered = df[mask].copy()
    print(f"RTH filter: {len(df)} -> {len(df_filtered)} events")

    return df_filtered


# ============================================================================
# QUALITY SCORING
# ============================================================================

def robust_normalize(series: pd.Series) -> pd.Series:
    """
    Robust normalization using median and IQR.
    Maps to roughly [0, 1] with most values, but allows outliers to exceed 1.
    """
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isna(iqr):
        # Fallback to min-max if IQR is zero
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    # Scale so median maps to 0.5 and IQR spans ~0.25 to ~0.75
    normalized = 0.5 + (series - median) / (2 * iqr)
    return normalized.clip(0, 1)


def compute_quality_scores(
    df: pd.DataFrame,
    w_mag: float = 0.25,
    w_z: float = 0.20,
    w_attn: float = 0.20,
    w_liq: float = 0.15,
    w_shape: float = 0.20
) -> pd.DataFrame:
    """
    Compute multi-factor quality score for each event.

    Factors:
    1) score_mag: Peak absolute return in basis points (15-min move)
    2) score_z: Z-score of the move relative to baseline volatility
    3) score_attn: Market attention = max(volume_mult, range_mult)
    4) score_liq: Liquidity stress = impact_mult
    5) score_shape: Event shape quality (multi-bar, persistence)

    Uses per-ticker robust normalization to prevent volatile stocks from dominating.
    """
    print("\nComputing quality scores...")

    df = df.copy()

    # Initialize score columns
    df["score_mag"] = 0.0
    df["score_z"] = 0.0
    df["score_attn"] = 0.0
    df["score_liq"] = 0.0
    df["score_shape"] = 0.0

    # Compute per-ticker normalized scores
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_df = df.loc[mask]

        if len(ticker_df) < 2:
            # Not enough data for normalization, use global
            continue

        # 1) Magnitude score: peak_abs_ret_3_bp
        if "peak_abs_ret_3_bp" in ticker_df.columns:
            df.loc[mask, "score_mag"] = robust_normalize(ticker_df["peak_abs_ret_3_bp"])

        # 2) Z-score: peak_z_ret_3
        if "peak_z_ret_3" in ticker_df.columns:
            df.loc[mask, "score_z"] = robust_normalize(ticker_df["peak_z_ret_3"])

        # 3) Attention: max(volume_mult, range_mult)
        vol_mult = ticker_df.get("peak_volume_mult", pd.Series(0, index=ticker_df.index))
        range_mult = ticker_df.get("peak_range_mult", pd.Series(0, index=ticker_df.index))
        attn = pd.concat([vol_mult, range_mult], axis=1).max(axis=1)
        df.loc[mask, "score_attn"] = robust_normalize(attn)

        # 4) Liquidity stress: impact_mult
        if "peak_impact_mult" in ticker_df.columns:
            df.loc[mask, "score_liq"] = robust_normalize(ticker_df["peak_impact_mult"])

        # 5) Shape score: cluster_len and forward behavior
        shape_score = pd.Series(0.5, index=ticker_df.index)

        # Reward multi-bar events
        if "cluster_len" in ticker_df.columns:
            cluster_bonus = (ticker_df["cluster_len"] >= 2).astype(float) * 0.3
            shape_score = shape_score + cluster_bonus

        # Reward clear forward behavior (non-unclear label)
        if "label_proxy" in ticker_df.columns:
            clear_label = (ticker_df["label_proxy"] != "unclear").astype(float) * 0.2
            shape_score = shape_score + clear_label

        df.loc[mask, "score_shape"] = shape_score.clip(0, 1)

    # Compute composite score
    df["quality_score"] = (
        w_mag * df["score_mag"] +
        w_z * df["score_z"] +
        w_attn * df["score_attn"] +
        w_liq * df["score_liq"] +
        w_shape * df["score_shape"]
    )

    print(f"Quality scores computed. Range: {df['quality_score'].min():.3f} - {df['quality_score'].max():.3f}")

    return df


# ============================================================================
# ARCHETYPE TAGGING
# ============================================================================

def tag_archetypes(df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each event with an archetype based on market conditions.

    Archetypes:
    - info_driven: High volume + high move + tends to continue
    - panic: High impact + down move + tends to reverse
    - macro_spillover: Stock moves with SPY/QQQ in same direction
    - idiosyncratic: Stock moves while SPY/QQQ relatively flat

    This is a heuristic tagging for diversity selection, not ground truth.
    """
    print("\nTagging event archetypes...")

    df = df.copy()
    df["archetype"] = "mixed"

    # Pre-compute SPY/QQQ returns around each timestamp for correlation check
    etf_returns = {}
    if ohlcv_df is not None:
        for etf in ["SPY", "QQQ"]:
            etf_df = ohlcv_df[ohlcv_df["ticker"] == etf].copy()
            if len(etf_df) > 0:
                etf_df = etf_df.sort_values("timestamp").set_index("timestamp")
                etf_df["ret_15m"] = np.log(etf_df["close"]).diff(3)  # 3 bars = 15 min
                etf_returns[etf] = etf_df["ret_15m"]

    for idx, row in df.iterrows():
        ticker = row["ticker"]
        t0 = row["t0_utc"]
        direction = row.get("direction", 0)

        vol_mult = row.get("peak_volume_mult", 0)
        impact_mult = row.get("peak_impact_mult", 0)
        abs_ret_bp = row.get("peak_abs_ret_3_bp", 0)
        label = row.get("label_proxy", "unclear")

        # Skip ETFs for archetype (they define the market)
        if ticker in ETF_TICKERS:
            df.loc[idx, "archetype"] = "market"
            continue

        # Get SPY/QQQ return around t0
        spy_ret = 0
        qqq_ret = 0
        if "SPY" in etf_returns and len(etf_returns["SPY"]) > 0:
            # Find nearest timestamp
            spy_idx = etf_returns["SPY"].index.get_indexer([t0], method="nearest")
            if len(spy_idx) > 0 and spy_idx[0] >= 0 and spy_idx[0] < len(etf_returns["SPY"]):
                spy_ret = etf_returns["SPY"].iloc[spy_idx[0]]
                if pd.isna(spy_ret):
                    spy_ret = 0

        if "QQQ" in etf_returns and len(etf_returns["QQQ"]) > 0:
            qqq_idx = etf_returns["QQQ"].index.get_indexer([t0], method="nearest")
            if len(qqq_idx) > 0 and qqq_idx[0] >= 0 and qqq_idx[0] < len(etf_returns["QQQ"]):
                qqq_ret = etf_returns["QQQ"].iloc[qqq_idx[0]]
                if pd.isna(qqq_ret):
                    qqq_ret = 0

        etf_move = (spy_ret + qqq_ret) / 2 if (spy_ret != 0 or qqq_ret != 0) else 0
        etf_move_bp = abs(etf_move) * 10000

        # Archetype heuristics
        archetype = "mixed"

        # Info-driven: High volume, big move, continuation
        if vol_mult > 100 and abs_ret_bp > 80 and label == "continuation":
            archetype = "info_driven"

        # Panic: High impact, down move, reversal
        elif impact_mult > 10 and direction == -1 and label == "reversal":
            archetype = "panic"

        # Macro spillover: Stock moves with market (both up or both down significantly)
        elif etf_move_bp > 20:
            stock_dir = 1 if direction > 0 else -1
            etf_dir = 1 if etf_move > 0 else -1
            if stock_dir == etf_dir:
                archetype = "macro_spillover"

        # Idiosyncratic: Stock moves while market flat
        elif etf_move_bp < 10 and abs_ret_bp > 60:
            archetype = "idiosyncratic"

        df.loc[idx, "archetype"] = archetype

    # Print archetype distribution
    arch_counts = df["archetype"].value_counts()
    print("Archetype distribution:")
    for arch, count in arch_counts.items():
        print(f"  {arch}: {count}")

    return df


# ============================================================================
# QUOTA-BASED SELECTION
# ============================================================================

def select_with_quotas(
    df: pd.DataFrame,
    target_total: int = 120,
    stock_min: int = 10,
    stock_max: int = 20,
    etf_min: int = 5,
    etf_max: int = 15
) -> pd.DataFrame:
    """
    Select events enforcing per-ticker quotas for balanced coverage.

    Algorithm:
    1) For each ticker, rank events by quality_score
    2) Select min_per_ticker events for each ticker (guarantee coverage)
    3) Fill remaining slots by global quality_score, respecting max caps
    4) Within each ticker, try to include archetype diversity

    Returns curated DataFrame.
    """
    print(f"\nSelecting with quotas (target={target_total})...")
    print(f"  Stocks: min={stock_min}, max={stock_max}")
    print(f"  ETFs: min={etf_min}, max={etf_max}")

    df = df.copy()
    selected_indices = set()

    # Track per-ticker selections
    ticker_selections = {t: [] for t in df["ticker"].unique()}

    # Step 1: Guarantee minimum per ticker
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].sort_values("quality_score", ascending=False)

        if ticker in ETF_TICKERS:
            min_events = etf_min
        else:
            min_events = stock_min

        # Select top events up to minimum
        available = min(len(ticker_df), min_events)

        # Try to get archetype diversity within minimum selection
        archetypes_selected = set()
        selected_for_ticker = []

        # First pass: one from each archetype if available
        for arch in ["info_driven", "panic", "macro_spillover", "idiosyncratic", "mixed", "market"]:
            arch_df = ticker_df[
                (ticker_df["archetype"] == arch) &
                (~ticker_df.index.isin(selected_for_ticker))
            ]
            if len(arch_df) > 0 and len(selected_for_ticker) < available:
                best_idx = arch_df.index[0]
                selected_for_ticker.append(best_idx)
                archetypes_selected.add(arch)

        # Second pass: fill remaining with best quality
        remaining = ticker_df[~ticker_df.index.isin(selected_for_ticker)]
        for idx in remaining.index:
            if len(selected_for_ticker) >= available:
                break
            selected_for_ticker.append(idx)

        ticker_selections[ticker] = selected_for_ticker
        selected_indices.update(selected_for_ticker)

    # Step 2: Fill to target using global quality, respecting max caps
    remaining_slots = target_total - len(selected_indices)

    if remaining_slots > 0:
        # Get unselected events sorted by quality
        unselected = df[~df.index.isin(selected_indices)].sort_values(
            "quality_score", ascending=False
        )

        for idx, row in unselected.iterrows():
            if remaining_slots <= 0:
                break

            ticker = row["ticker"]

            # Check max cap
            if ticker in ETF_TICKERS:
                max_events = etf_max
            else:
                max_events = stock_max

            if len(ticker_selections[ticker]) >= max_events:
                continue

            # Add this event
            ticker_selections[ticker].append(idx)
            selected_indices.add(idx)
            remaining_slots -= 1

    # Create curated DataFrame
    curated_df = df.loc[list(selected_indices)].copy()
    curated_df = curated_df.sort_values(["ticker", "quality_score"], ascending=[True, False])

    print(f"Selected {len(curated_df)} events")

    # Print per-ticker counts
    print("\nPer-ticker selection:")
    for ticker in sorted(curated_df["ticker"].unique()):
        count = len(curated_df[curated_df["ticker"] == ticker])
        ticker_type = "ETF" if ticker in ETF_TICKERS else "STOCK"
        print(f"  {ticker} ({ticker_type}): {count}")

    return curated_df


# ============================================================================
# OUTPUT
# ============================================================================

def write_outputs(
    curated_df: pd.DataFrame,
    all_candidates_df: pd.DataFrame,
    out_dir: Path
) -> None:
    """Write curated events and summary statistics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write curated events
    curated_path = out_dir / "curated_events.csv"
    curated_df.to_csv(curated_path, index=False)
    print(f"\nCurated events written to {curated_path}")

    # Compute summary statistics
    summary = {
        "total_candidates": len(all_candidates_df),
        "curated_count": len(curated_df),
        "per_ticker": curated_df.groupby("ticker").size().to_dict(),
        "archetype_distribution": curated_df["archetype"].value_counts().to_dict(),
        "label_distribution": curated_df["label_proxy"].value_counts().to_dict() if "label_proxy" in curated_df.columns else {},
        "quality_score_stats": {
            "min": float(curated_df["quality_score"].min()),
            "median": float(curated_df["quality_score"].median()),
            "max": float(curated_df["quality_score"].max()),
        },
        "peak_abs_ret_3_bp_stats": {
            "min": float(curated_df["peak_abs_ret_3_bp"].min()),
            "median": float(curated_df["peak_abs_ret_3_bp"].median()),
            "max": float(curated_df["peak_abs_ret_3_bp"].max()),
        },
    }

    # Write summary JSON
    summary_path = out_dir / "curation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CURATION SUMMARY")
    print("=" * 70)
    print(f"Total candidates: {summary['total_candidates']}")
    print(f"Curated events:   {summary['curated_count']}")

    print("\nPer-ticker counts:")
    for ticker, count in sorted(summary["per_ticker"].items()):
        ticker_type = "ETF" if ticker in ETF_TICKERS else "STOCK"
        print(f"  {ticker} ({ticker_type}): {count}")

    print("\nArchetype distribution:")
    for arch, count in sorted(summary["archetype_distribution"].items()):
        print(f"  {arch}: {count}")

    if summary["label_distribution"]:
        print("\nLabel distribution:")
        for label, count in sorted(summary["label_distribution"].items()):
            print(f"  {label}: {count}")

    print("\nQuality score range: "
          f"{summary['quality_score_stats']['min']:.3f} - "
          f"{summary['quality_score_stats']['max']:.3f} "
          f"(median: {summary['quality_score_stats']['median']:.3f})")

    print(f"\nPeak return range: "
          f"{summary['peak_abs_ret_3_bp_stats']['min']:.0f}bp - "
          f"{summary['peak_abs_ret_3_bp_stats']['max']:.0f}bp "
          f"(median: {summary['peak_abs_ret_3_bp_stats']['median']:.0f}bp)")

    # Top 10 curated events
    print("\n" + "-" * 70)
    print("TOP 10 CURATED EVENTS by quality_score:")
    print("-" * 70)
    top10 = curated_df.nlargest(10, "quality_score")
    for _, row in top10.iterrows():
        print(f"  {row['ticker']:5s} @ {row['t0_utc'].strftime('%Y-%m-%d %H:%M')} | "
              f"score={row['quality_score']:.3f} | "
              f"{row['peak_abs_ret_3_bp']:.0f}bp | "
              f"arch={row['archetype']:15s} | "
              f"{row.get('label_proxy', 'n/a')}")

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("Event Curator - Multi-Factor Quality Selection")
    print("=" * 70)
    print(f"Candidates: {args.candidates}")
    print(f"OHLCV: {args.ohlcv}")
    print(f"Output: {out_dir}")
    print(f"Target: {args.target_total} events")
    print(f"Stock quotas: min={args.stock_min}, max={args.stock_max}")
    print(f"ETF quotas: min={args.etf_min}, max={args.etf_max}")
    print(f"Include extended hours: {args.include_extended_hours}")
    print("=" * 70)

    # Load data
    candidates_df = load_candidates(args.candidates)
    ohlcv_df = load_ohlcv(args.ohlcv)

    # Filter for RTH
    filtered_df = filter_rth_events(
        candidates_df,
        ohlcv_df,
        include_extended=args.include_extended_hours,
        max_gap_minutes=args.max_gap_minutes
    )

    if len(filtered_df) == 0:
        print("ERROR: No events passed RTH filter!")
        return

    # Compute quality scores
    scored_df = compute_quality_scores(
        filtered_df,
        w_mag=args.w_mag,
        w_z=args.w_z,
        w_attn=args.w_attn,
        w_liq=args.w_liq,
        w_shape=args.w_shape
    )

    # Tag archetypes
    tagged_df = tag_archetypes(scored_df, ohlcv_df)

    # Select with quotas
    curated_df = select_with_quotas(
        tagged_df,
        target_total=args.target_total,
        stock_min=args.stock_min,
        stock_max=args.stock_max,
        etf_min=args.etf_min,
        etf_max=args.etf_max
    )

    # Write outputs
    write_outputs(curated_df, candidates_df, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
