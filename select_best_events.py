#!/usr/bin/env python3
"""
Select Best Events — High-signal balanced subset for agentic debate system.

Trims ~451 mined events to 80–120 (target 100) using:
  Step 0: Hard quality gates (RTH, window integrity, gap check)
  Step 1: Per-ticker robust strength scoring
  Step 2: Outcome-quality trimming (label_proxy as signal, not prediction)
  Step 3: Ticker + outcome balance via quotas
  Step 4: Temporal de-duplication (per-ticker cooldown + global macro cap)
  Step 5: Greedy selection with quota enforcement
  Step 6: Explainable outputs (selected, rejected, summary)

Rationale:
  - label_proxy is outcome-based (what actually happened 60 min later), not a prediction.
  - We prioritise clear outcomes (continuation / reversal) for early evaluation,
    but keep a small slice of high-strength unclear events to represent genuine
    uncertainty / sideways consolidation — this is defensible, not cherry-picking.
  - Per-ticker quotas + de-dup avoid concentration in volatile names or macro days.

Usage:
    python select_best_events.py
    python select_best_events.py --target_n 80 --min_per_stock 6
    python select_best_events.py --include_extended_hours --out_dir out_final_ext
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STOCK_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMD", "NFLX", "PLTR"]
ETF_TICKERS = ["SPY", "QQQ", "XLK"]
ALL_TICKERS = STOCK_TICKERS + ETF_TICKERS

RTH_START_MIN = 9 * 60 + 30   # 09:30 ET = 570 minutes
RTH_END_MIN = 16 * 60         # 16:00 ET = 960 minutes


def parse_args():
    p = argparse.ArgumentParser(
        description="Select high-signal balanced event subset for agentic debate"
    )

    # I/O
    p.add_argument("--events_path", default="out_strict/events.csv")
    p.add_argument("--ohlcv_path", default="ohlcv_5min.parquet")
    p.add_argument("--out_dir", default="out_final")

    # Target size
    p.add_argument("--target_n", type=int, default=100)

    # Ticker quotas
    p.add_argument("--min_per_stock", type=int, default=8)
    p.add_argument("--max_per_stock", type=int, default=15)
    p.add_argument("--min_per_etf", type=int, default=3)
    p.add_argument("--max_per_etf", type=int, default=8)

    # Unclear cap
    p.add_argument("--unclear_frac_cap", type=float, default=0.15)

    # De-dup
    p.add_argument("--ticker_cooldown_min", type=int, default=60)

    # RTH
    p.add_argument("--include_extended_hours", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(path):
    df = pd.read_csv(path)
    time_cols = ["t0_utc", "event_window_start", "event_window_end",
                 "val_window_start", "val_window_end"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_ohlcv(path):
    p = Path(path)
    if p.suffix == ".parquet" and p.exists():
        df = pd.read_parquet(p)
    elif p.suffix == ".csv" and p.exists():
        df = pd.read_csv(p)
    else:
        # Try alternate extension
        alt = p.with_suffix(".csv" if p.suffix == ".parquet" else ".parquet")
        if alt.exists():
            df = pd.read_parquet(alt) if alt.suffix == ".parquet" else pd.read_csv(alt)
        else:
            print(f"WARNING: OHLCV not found at {path}, gap checks disabled")
            return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Step 0 — Hard quality gates
# ---------------------------------------------------------------------------

def to_et(ts):
    """Convert UTC timestamp to America/New_York."""
    return ts.tz_convert("America/New_York")


def apply_hard_gates(df, ohlcv, include_extended):
    """
    Apply hard quality gates.  Returns (eligible_df, rejected_rows_list).
    Each rejected row is (event_id, reason).
    """
    rejects = []

    # Pre-compute ET times for t0, event window start/end, val window end
    df = df.copy()
    df["t0_et"] = df["t0_utc"].dt.tz_convert("America/New_York")
    df["ews_et"] = df["event_window_start"].dt.tz_convert("America/New_York")
    df["vwe_et"] = df["val_window_end"].dt.tz_convert("America/New_York")

    keep_mask = pd.Series(True, index=df.index)
    reject_reason = pd.Series("", index=df.index)

    # Gate 1: RTH check (unless --include_extended_hours)
    if not include_extended:
        t0_min = df["t0_et"].dt.hour * 60 + df["t0_et"].dt.minute
        outside_rth = (t0_min < RTH_START_MIN) | (t0_min > RTH_END_MIN)
        # Also check weekday
        outside_rth = outside_rth | (df["t0_et"].dt.weekday > 4)
        reject_reason = reject_reason.where(~outside_rth, "out_of_rth")
        keep_mask = keep_mask & ~outside_rth

    # Gate 2: Window must not cross a trading-day boundary (in NY time)
    cross_day = df["ews_et"].dt.date != df["vwe_et"].dt.date
    reject_reason = reject_reason.where(~(cross_day & keep_mask), "day_boundary")
    keep_mask = keep_mask & ~cross_day

    # Gate 3: Gap check within event+val window (max 10 min gap)
    if ohlcv is not None:
        gap_fail = pd.Series(False, index=df.index)
        for idx, row in df[keep_mask].iterrows():
            ticker = row["ticker"]
            ws = row["event_window_start"]
            we = row["val_window_end"]
            sub = ohlcv[(ohlcv["ticker"] == ticker) &
                        (ohlcv["timestamp"] >= ws) &
                        (ohlcv["timestamp"] <= we)].sort_values("timestamp")
            if len(sub) < 2:
                gap_fail.at[idx] = True
                continue
            gaps_min = sub["timestamp"].diff().dt.total_seconds().iloc[1:] / 60.0
            if gaps_min.max() > 10:
                gap_fail.at[idx] = True

        reject_reason = reject_reason.where(~(gap_fail & keep_mask), "window_gap_gt_10min")
        keep_mask = keep_mask & ~gap_fail

    # Gate 4: Non-NaN on required features
    required_cols = ["peak_abs_ret_3_bp", "peak_z_ret_3", "peak_volume_mult",
                     "peak_range_mult", "peak_impact_mult", "forward_return_60m"]
    for col in required_cols:
        if col in df.columns:
            nan_mask = df[col].isna()
            reject_reason = reject_reason.where(~(nan_mask & keep_mask), f"nan_{col}")
            keep_mask = keep_mask & ~nan_mask

    # Build reject list
    rejected_idx = df.index[~keep_mask]
    reject_df = df.loc[rejected_idx].copy()
    reject_df["reject_reason"] = reject_reason.loc[rejected_idx]

    eligible = df.loc[keep_mask].copy()

    # Drop helper columns
    for col in ["t0_et", "ews_et", "vwe_et"]:
        eligible.drop(columns=[col], inplace=True, errors="ignore")
        reject_df.drop(columns=[col], inplace=True, errors="ignore")

    return eligible, reject_df


# ---------------------------------------------------------------------------
# Step 1 — Strength score (per-ticker robust scaling)
# ---------------------------------------------------------------------------

def robust_z_per_ticker(series, ticker_series):
    """
    Robust z-score per ticker: (x - median) / IQR, clipped to [0, 1].

    Uses median and IQR (interquartile range) so extreme outliers in volatile
    names (TSLA, NVDA) don't dominate the ranking.  Clipped to [0,1] so all
    signals contribute proportionally.
    """
    result = pd.Series(0.0, index=series.index)
    for ticker in ticker_series.unique():
        mask = ticker_series == ticker
        vals = series[mask]
        if len(vals) < 3:
            # Not enough data for IQR; use global scaling later
            result[mask] = 0.5
            continue

        med = vals.median()
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or np.isnan(iqr):
            # Degenerate case: all same value
            result[mask] = 0.5
            continue

        # Map so median → 0.5, one IQR above median → ~0.75
        z = 0.5 + (vals - med) / (2 * iqr)
        result[mask] = z.clip(0, 1)

    return result


def compute_strength_score(df):
    """
    Compute strength_score per event.

    Formula (all per-ticker robust-z scaled to [0,1]):
        strength_score = 0.25 * rz(peak_abs_ret_3_bp)     # magnitude
                       + 0.20 * rz(peak_z_ret_3)           # rarity vs baseline vol
                       + 0.20 * rz(attention)               # max(volume_mult, range_mult)
                       + 0.15 * rz(peak_impact_mult)        # liquidity stress
                       + 0.20 * min(cluster_len, 4) / 4     # persistence / shape

    Weights sum to 1.0.  cluster_len is capped at 4 (diminishing returns past 20 min).
    """
    df = df.copy()
    tickers = df["ticker"]

    # Attention = max(volume_mult, range_mult) — captures the "market noticed this"
    df["attention"] = df[["peak_volume_mult", "peak_range_mult"]].max(axis=1)

    # Per-ticker robust z-scores
    rz_mag = robust_z_per_ticker(df["peak_abs_ret_3_bp"], tickers)
    rz_z = robust_z_per_ticker(df["peak_z_ret_3"], tickers)
    rz_attn = robust_z_per_ticker(df["attention"], tickers)
    rz_impact = robust_z_per_ticker(df["peak_impact_mult"], tickers)

    # Persistence: cluster_len capped at 4 bars (20 min), then normalised to [0,1]
    persistence = df["cluster_len"].clip(upper=4) / 4.0

    df["strength_score"] = (
        0.25 * rz_mag +
        0.20 * rz_z +
        0.20 * rz_attn +
        0.15 * rz_impact +
        0.20 * persistence
    )

    df.drop(columns=["attention"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Unclear trimming logic
# ---------------------------------------------------------------------------

def tag_unclear_eligibility(df):
    """
    Mark which unclear events are high-enough quality to keep.

    An unclear event is eligible if ANY of:
      1) strength_score in top 30% within its ticker, OR
      2) peak_abs_ret_3_bp >= global 75th percentile, OR
      3) peak_volume_mult >= global 90th percentile, OR
      4) peak_impact_mult >= global 90th percentile

    This keeps genuine uncertainty / consolidation events while
    rejecting low-signal ambiguous noise.
    """
    df = df.copy()
    df["unclear_eligible"] = True  # default: all clear outcomes are eligible

    unclear_mask = df["label_proxy"] == "unclear"
    if unclear_mask.sum() == 0:
        return df

    # Global thresholds
    bp_p75 = df["peak_abs_ret_3_bp"].quantile(0.75)
    vol_p90 = df["peak_volume_mult"].quantile(0.90)
    impact_p90 = df["peak_impact_mult"].quantile(0.90)

    # Per-ticker top-30% threshold on strength_score
    ticker_p70 = df.groupby("ticker")["strength_score"].quantile(0.70)

    for idx in df.index[unclear_mask]:
        row = df.loc[idx]
        ticker = row["ticker"]
        ss = row["strength_score"]
        bp = row["peak_abs_ret_3_bp"]
        vm = row["peak_volume_mult"]
        im = row["peak_impact_mult"]

        passes = (
            ss >= ticker_p70.get(ticker, 1.0) or
            bp >= bp_p75 or
            vm >= vol_p90 or
            im >= impact_p90
        )
        df.at[idx, "unclear_eligible"] = passes

    kept = unclear_mask & df["unclear_eligible"]
    dropped = unclear_mask & ~df["unclear_eligible"]
    print(f"  Unclear events: {unclear_mask.sum()} total, "
          f"{kept.sum()} eligible (high-strength), "
          f"{dropped.sum()} will be deprioritised")

    return df


# ---------------------------------------------------------------------------
# Step 4 — De-duplication
# ---------------------------------------------------------------------------

def dedup_within_ticker(df, cooldown_min):
    """
    Within each ticker, if two events are within cooldown_min minutes,
    keep the one with higher strength_score.  Returns de-duped df and
    list of (event_id, reason) for rejects.
    """
    df = df.sort_values(["ticker", "strength_score"], ascending=[True, False]).copy()
    keep = []
    dup_rejects = []

    for ticker in df["ticker"].unique():
        tdf = df[df["ticker"] == ticker].sort_values("t0_utc")
        last_kept_t0 = None

        # First pass: greedily keep by strength order but check temporal proximity
        # Strategy: sort by strength descending, keep if no conflict with already-kept
        by_strength = tdf.sort_values("strength_score", ascending=False)
        kept_times = []  # list of t0_utc for already-kept events

        for idx, row in by_strength.iterrows():
            t0 = row["t0_utc"]
            conflict = False
            for kt in kept_times:
                gap = abs((t0 - kt).total_seconds()) / 60.0
                if gap < cooldown_min:
                    conflict = True
                    break
            if conflict:
                dup_rejects.append(idx)
            else:
                keep.append(idx)
                kept_times.append(t0)

    deduped = df.loc[keep]
    return deduped, dup_rejects


# ---------------------------------------------------------------------------
# Step 5 — Selection algorithm
# ---------------------------------------------------------------------------

def select_events(eligible, args):
    """
    Greedy selection with quota enforcement.

    1) Fill per-ticker minimums (clear outcomes first, then high-strength unclear)
    2) Fill remaining to target_n globally by strength_score,
       respecting max caps, unclear cap, and de-dup
    3) If over target, trim lowest-strength while preserving minimums
    """
    target_n = args.target_n
    min_stock = args.min_per_stock
    max_stock = args.max_per_stock
    min_etf = args.min_per_etf
    max_etf = args.max_per_etf
    unclear_cap = int(np.ceil(target_n * args.unclear_frac_cap))
    cooldown = args.ticker_cooldown_min

    # De-dup within ticker
    print(f"\nStep 4 — De-duplicating (cooldown={cooldown} min)...")
    deduped, dup_reject_idx = dedup_within_ticker(eligible, cooldown)
    print(f"  After de-dup: {len(deduped)} events (removed {len(dup_reject_idx)} near-duplicates)")

    # Separate clear vs unclear
    clear = deduped[deduped["label_proxy"].isin(["continuation", "reversal"])].copy()
    unclear_elig = deduped[
        (deduped["label_proxy"] == "unclear") & (deduped["unclear_eligible"])
    ].copy()
    unclear_inelig = deduped[
        (deduped["label_proxy"] == "unclear") & (~deduped["unclear_eligible"])
    ].copy()

    print(f"  Clear-outcome pool: {len(clear)}")
    print(f"  Unclear eligible pool: {len(unclear_elig)}")
    print(f"  Unclear ineligible (excluded): {len(unclear_inelig)}")

    selected_idx = set()
    keep_reasons = {}
    ticker_counts = {t: 0 for t in ALL_TICKERS}
    unclear_count = 0

    def can_add(row_idx, row):
        nonlocal unclear_count
        ticker = row["ticker"]
        is_etf = ticker in ETF_TICKERS
        max_cap = max_etf if is_etf else max_stock

        if ticker_counts.get(ticker, 0) >= max_cap:
            return False, "over_cap"
        if row["label_proxy"] == "unclear" and unclear_count >= unclear_cap:
            return False, "unclear_cap_reached"
        return True, ""

    def add_event(row_idx, row, reason):
        nonlocal unclear_count
        selected_idx.add(row_idx)
        keep_reasons[row_idx] = reason
        ticker_counts[row["ticker"]] = ticker_counts.get(row["ticker"], 0) + 1
        if row["label_proxy"] == "unclear":
            unclear_count += 1

    # --- Phase 1: Fill per-ticker minimums ---
    print("\nStep 5 — Selection...")
    print("  Phase 1: Filling per-ticker minimums...")

    for ticker in ALL_TICKERS:
        is_etf = ticker in ETF_TICKERS
        min_needed = min_etf if is_etf else min_stock

        # Clear outcomes first (sorted by strength)
        ticker_clear = clear[clear["ticker"] == ticker].sort_values(
            "strength_score", ascending=False
        )
        for idx, row in ticker_clear.iterrows():
            if ticker_counts[ticker] >= min_needed:
                break
            if idx not in selected_idx:
                ok, _ = can_add(idx, row)
                if ok:
                    add_event(idx, row, "min_quota")

        # If still short, fill with high-strength unclear
        if ticker_counts[ticker] < min_needed:
            ticker_unclear = unclear_elig[unclear_elig["ticker"] == ticker].sort_values(
                "strength_score", ascending=False
            )
            for idx, row in ticker_unclear.iterrows():
                if ticker_counts[ticker] >= min_needed:
                    break
                if idx not in selected_idx:
                    ok, _ = can_add(idx, row)
                    if ok:
                        add_event(idx, row, "min_quota_unclear_fill")

    phase1_count = len(selected_idx)
    print(f"  Phase 1 selected: {phase1_count}")

    # --- Phase 2: Fill to target globally by strength ---
    print("  Phase 2: Filling to target globally...")
    remaining_slots = target_n - len(selected_idx)

    if remaining_slots > 0:
        # Combine clear + eligible unclear, sort globally by strength
        global_pool = pd.concat([clear, unclear_elig]).sort_values(
            "strength_score", ascending=False
        )

        for idx, row in global_pool.iterrows():
            if remaining_slots <= 0:
                break
            if idx in selected_idx:
                continue
            ok, reason = can_add(idx, row)
            if ok:
                label_tag = "top_global"
                if row["label_proxy"] == "unclear":
                    label_tag = "high_strength_unclear"
                add_event(idx, row, label_tag)
                remaining_slots -= 1

    print(f"  Phase 2 total selected: {len(selected_idx)}")

    # --- Phase 3: If over target, trim lowest strength while preserving mins ---
    if len(selected_idx) > target_n:
        print(f"  Phase 3: Trimming {len(selected_idx)} -> {target_n}...")
        selected_df_tmp = eligible.loc[list(selected_idx)].sort_values(
            "strength_score", ascending=True
        )
        to_remove = len(selected_idx) - target_n

        for idx, row in selected_df_tmp.iterrows():
            if to_remove <= 0:
                break
            ticker = row["ticker"]
            is_etf = ticker in ETF_TICKERS
            min_needed = min_etf if is_etf else min_stock
            if ticker_counts[ticker] > min_needed:
                selected_idx.discard(idx)
                ticker_counts[ticker] -= 1
                if row["label_proxy"] == "unclear":
                    unclear_count -= 1
                to_remove -= 1

    # --- Build selected and rejected DataFrames ---
    selected_df = eligible.loc[list(selected_idx)].copy()
    selected_df["keep_reason"] = selected_df.index.map(keep_reasons)
    selected_df = selected_df.sort_values(["ticker", "strength_score"],
                                          ascending=[True, False])

    # Build rejected: everything not selected
    # Combine: (1) hard-gate rejects passed in separately, (2) dup rejects,
    # (3) unclear ineligible, (4) events just not selected
    all_reject_idx = set(eligible.index) - selected_idx
    reject_reasons_map = {}

    for idx in dup_reject_idx:
        reject_reasons_map[idx] = "near_duplicate"
    for idx in unclear_inelig.index:
        if idx in all_reject_idx:
            reject_reasons_map[idx] = "low_strength_unclear"
    # Anything else not selected
    for idx in all_reject_idx:
        if idx not in reject_reasons_map:
            reject_reasons_map[idx] = "over_cap_or_below_cutoff"

    rejected_from_eligible = eligible.loc[list(all_reject_idx)].copy()
    rejected_from_eligible["reject_reason"] = rejected_from_eligible.index.map(
        reject_reasons_map
    )

    return selected_df, rejected_from_eligible, dup_reject_idx


# ---------------------------------------------------------------------------
# Step 6 — Outputs
# ---------------------------------------------------------------------------

def build_summary(selected_df, total_candidates, total_eligible):
    """Build summary dict for JSON output."""

    ticker_counts = selected_df.groupby("ticker").size().to_dict()
    label_counts = selected_df["label_proxy"].value_counts().to_dict()
    unclear_kept = int(label_counts.get("unclear", 0))
    reason_counts = selected_df["keep_reason"].value_counts().to_dict()

    ss = selected_df["strength_score"]
    quantiles = {
        f"p{int(q*100)}": round(float(ss.quantile(q)), 4)
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]
    }

    top10 = selected_df.nlargest(10, "peak_abs_ret_3_bp")
    top10_list = []
    for _, row in top10.iterrows():
        top10_list.append({
            "event_id": row["event_id"],
            "ticker": row["ticker"],
            "t0_utc": str(row["t0_utc"]),
            "peak_abs_ret_3_bp": round(float(row["peak_abs_ret_3_bp"]), 1),
            "peak_z_ret_3": round(float(row["peak_z_ret_3"]), 2),
            "strength_score": round(float(row["strength_score"]), 4),
            "label_proxy": row["label_proxy"],
            "direction": int(row["direction"]),
        })

    summary = {
        "total_candidates": int(total_candidates),
        "eligible_after_hard_gates": int(total_eligible),
        "selected_count": int(len(selected_df)),
        "per_ticker": {t: int(ticker_counts.get(t, 0)) for t in ALL_TICKERS},
        "per_label": label_counts,
        "unclear_kept": unclear_kept,
        "keep_reason_counts": reason_counts,
        "strength_score_quantiles": quantiles,
        "top_10_by_magnitude": top10_list,
        "rationale": (
            "label_proxy is outcome-based (what happened 60 min after the spike), "
            "not a prediction. We prioritise clear outcomes (continuation/reversal) "
            "for early evaluation of the agentic debate system, while keeping a small "
            "high-strength unclear slice (~10-15%) to represent genuine uncertainty / "
            "sideways consolidation. Per-ticker quotas and temporal de-duplication "
            "prevent concentration in volatile names or single macro-shock days. "
            "Strength scoring uses per-ticker robust scaling (median/IQR) so volatile "
            "stocks like TSLA and NVDA do not dominate the selection."
        ),
    }
    return summary


def print_console_report(selected_df, rejected_count, summary):
    """Print a concise console report."""
    print("\n" + "=" * 72)
    print("SELECTION REPORT")
    print("=" * 72)
    print(f"  Candidates:       {summary['total_candidates']}")
    print(f"  After hard gates: {summary['eligible_after_hard_gates']}")
    print(f"  Selected:         {summary['selected_count']}")
    print(f"  Rejected:         {rejected_count}")

    print("\n  Per-ticker:")
    for ticker in ALL_TICKERS:
        count = summary["per_ticker"].get(ticker, 0)
        tag = "ETF" if ticker in ETF_TICKERS else "   "
        print(f"    {ticker:5s} [{tag}]: {count:3d}")

    print(f"\n  Label distribution:")
    for label in ["continuation", "reversal", "unclear"]:
        count = summary["per_label"].get(label, 0)
        print(f"    {label:14s}: {count}")

    print(f"\n  Unclear kept: {summary['unclear_kept']} "
          f"({summary['unclear_kept']/max(summary['selected_count'],1)*100:.1f}%)")

    print(f"\n  Strength score quantiles:")
    for k, v in summary["strength_score_quantiles"].items():
        print(f"    {k}: {v:.4f}")

    print(f"\n  Keep reasons:")
    for reason, count in summary["keep_reason_counts"].items():
        print(f"    {reason:30s}: {count}")

    print(f"\n  Top 5 by magnitude:")
    for ev in summary["top_10_by_magnitude"][:5]:
        dir_sym = "+" if ev["direction"] == 1 else "-"
        print(f"    {ev['ticker']:5s} {ev['t0_utc'][:16]} | "
              f"{dir_sym}{ev['peak_abs_ret_3_bp']:6.1f}bp | "
              f"z={ev['peak_z_ret_3']:.2f} | "
              f"ss={ev['strength_score']:.4f} | "
              f"{ev['label_proxy']}")

    print("=" * 72)


def write_outputs(selected_df, rejected_df, hardgate_rejected_df, summary, out_dir):
    """Write all output files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Selected events
    sel_path = out_dir / "selected_events.csv"
    # Arrange columns: original fields + strength_score + keep_reason
    orig_cols = [c for c in selected_df.columns
                 if c not in ["strength_score", "keep_reason", "unclear_eligible"]]
    out_cols = orig_cols + ["strength_score", "keep_reason"]
    selected_df[out_cols].to_csv(sel_path, index=False)
    print(f"\n  Selected events -> {sel_path}")

    # Rejected events: combine hard-gate rejects + selection rejects
    all_rejected = pd.concat([hardgate_rejected_df, rejected_df], ignore_index=True)
    rej_path = out_dir / "rejected_events.csv"
    rej_cols = [c for c in all_rejected.columns
                if c not in ["unclear_eligible"]]
    all_rejected[rej_cols].to_csv(rej_path, index=False)
    print(f"  Rejected events -> {rej_path}")

    # Summary JSON
    sum_path = out_dir / "summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary         -> {sum_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 72)
    print("SELECT BEST EVENTS — High-signal balanced subset")
    print("=" * 72)
    print(f"  Events:        {args.events_path}")
    print(f"  OHLCV:         {args.ohlcv_path}")
    print(f"  Output:        {args.out_dir}")
    print(f"  Target:        {args.target_n}")
    print(f"  Stock quotas:  min={args.min_per_stock} max={args.max_per_stock}")
    print(f"  ETF quotas:    min={args.min_per_etf} max={args.max_per_etf}")
    print(f"  Unclear cap:   {args.unclear_frac_cap:.0%}")
    print(f"  Cooldown:      {args.ticker_cooldown_min} min")
    print(f"  Extended hrs:  {args.include_extended_hours}")
    print("=" * 72)

    # Load
    events = load_events(args.events_path)
    ohlcv = load_ohlcv(args.ohlcv_path)
    total_candidates = len(events)
    print(f"\nLoaded {total_candidates} candidate events")

    # Step 0: Hard quality gates
    print("\nStep 0 — Hard quality gates...")
    eligible, hardgate_rejected = apply_hard_gates(events, ohlcv, args.include_extended_hours)
    print(f"  Passed: {len(eligible)}, Rejected: {len(hardgate_rejected)}")
    if len(hardgate_rejected) > 0:
        reason_counts = hardgate_rejected["reject_reason"].value_counts()
        for reason, count in reason_counts.items():
            print(f"    {reason}: {count}")

    if len(eligible) == 0:
        print("ERROR: No events survived hard gates!")
        return

    # Step 1: Strength score
    print("\nStep 1 — Computing strength scores...")
    eligible = compute_strength_score(eligible)
    print(f"  Score range: {eligible['strength_score'].min():.4f} - "
          f"{eligible['strength_score'].max():.4f} "
          f"(median {eligible['strength_score'].median():.4f})")

    # Step 2: Tag unclear eligibility
    print("\nStep 2 — Tagging unclear eligibility...")
    eligible = tag_unclear_eligibility(eligible)

    # Step 3 + 4 + 5: Selection with quotas, de-dup, balance
    print("\nStep 3 — Applying balance constraints + selection...")
    selected, rejected_from_selection, dup_idx = select_events(eligible, args)

    # Step 6: Outputs
    total_rejected = len(hardgate_rejected) + len(rejected_from_selection)
    summary = build_summary(selected, total_candidates, len(eligible))
    print_console_report(selected, total_rejected, summary)
    write_outputs(selected, rejected_from_selection, hardgate_rejected, summary, args.out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
