"""
Detailed Backtest PnL — per-trade analysis with per-ticker transaction costs.

Takes the ml_stage_v2 results JSON (which contains holdout predictions) and
the events CSV (which has forward_return_60m) to compute a full backtest.

Outputs:
  - Per-trade PnL table (CSV)
  - Aggregate stats (JSON): Sharpe, drawdown, win rate, edge vs breakeven
  - Equity curve plot (PNG)

Cost model (round-trip, basis points):
  Source: Frazzini, Israel, Moskowitz (AQR 2018) + retail spread surveys 2024.

Usage:
    python -m eval.backtest_pnl
    python -m eval.backtest_pnl \\
        --results out_eval_extended/ml_stage_v2_results.json \\
        --jsonl   out_agents_extended/ip_outputs_v3.jsonl \\
        --events_csv out_extended_final/selected_events.csv \\
        --out_dir out_eval_extended
"""

import argparse
import json
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COSTS_BPS = {
    "AAPL": 12, "MSFT": 12, "SPY": 10, "QQQ": 10, "XLK": 10,
    "NVDA": 17, "TSLA": 18, "META": 15, "AMD": 17, "NFLX": 16,
    "PLTR": 28, "AMZN": 15, "GOOGL": 15, "JPM": 12, "COIN": 30,
}
DEFAULT_COST = 20

LABEL_DEC = {0: "continuation", 1: "reversal", 2: "unclear"}
LABEL_ENC  = {"continuation": 0, "reversal": 1, "unclear": 2}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "error" not in r:
                    records.append(r)
            except json.JSONDecodeError:
                pass
    return records


def load_events_csv(path):
    by_id = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_id[row["event_id"]] = row
    return by_id


def sharpe(pnl_arr, events_per_year):
    if len(pnl_arr) < 2:
        return 0.0
    return float(pnl_arr.mean() / (pnl_arr.std() + 1e-9) * np.sqrt(events_per_year))


def max_drawdown(pnl_arr):
    cum  = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def calmar(pnl_arr, events_per_year):
    mdd = max_drawdown(pnl_arr)
    if mdd == 0:
        return float("inf")
    ann_return = float(pnl_arr.mean() * events_per_year)
    return ann_return / mdd


def breakeven_accuracy(cost_bps, abs_ret_bps):
    """Minimum accuracy needed to be profitable given cost and return magnitude."""
    if abs_ret_bps <= 0:
        return 1.0
    return (cost_bps + abs_ret_bps) / (2 * abs_ret_bps)


def main():
    parser = argparse.ArgumentParser(description="Standalone backtest PnL analysis")
    parser.add_argument("--results",    default="out_agents_v3b/ml_stage_v2_results.json",
                        help="ml_stage_v2 results JSON (contains holdout_preds by label)")
    parser.add_argument("--jsonl",      default="out_agents_v3b/ip_outputs_v3.jsonl",
                        help="JSONL with agent outputs (for holdout event records)")
    parser.add_argument("--events_csv", default="out_final/selected_events.csv")
    parser.add_argument("--out_dir",    default="out_agents_v3b")
    parser.add_argument("--holdout_frac", type=float, default=0.20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading data...")
    with open(args.results) as f:
        results = json.load(f)

    events_by_id = load_events_csv(args.events_csv)
    records_all  = load_jsonl(args.jsonl)

    # Sort records by timestamp (same as ml_stage_v2)
    def parse_ts(r):
        try:
            ts = pd.Timestamp(r.get("t0_utc", ""))
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts
        except Exception:
            return pd.Timestamp("2000-01-01", tz="UTC")

    records_all.sort(key=parse_ts)

    # Drop unclear events (same filter as feature matrix)
    records_clear = [r for r in records_all if r.get("true_label_proxy", "") in ("continuation", "reversal")]
    n = len(records_clear)

    # Holdout slice (last holdout_frac by time — same split as ml_stage_v2)
    split_point = int(n * (1 - args.holdout_frac))
    rec_holdout = records_clear[split_point:]
    log.info(f"Holdout records: {len(rec_holdout)}")

    # ── Get predictions from results JSON ─────────────────────────────────────
    holdout_preds_by_config = results.get("holdout_preds", {})

    # If ml_stage_v2 didn't embed preds, try to derive from results (fallback)
    # We'll use the "full_stack" config if available
    if not holdout_preds_by_config:
        log.warning("No holdout_preds in results JSON — will use v3b ensemble labels from JSONL")
        preds = [LABEL_ENC.get(r.get("ensemble_label", "unclear"), 2) for r in rec_holdout]
        holdout_preds_by_config = {"v3b_ensemble": preds}

    # ── Per-config backtest ────────────────────────────────────────────────────
    all_bt = {}

    for config_name, preds in holdout_preds_by_config.items():
        if len(preds) != len(rec_holdout):
            log.warning(f"  {config_name}: pred count {len(preds)} ≠ holdout count {len(rec_holdout)}, skipping")
            continue

        log.info(f"\n{'='*55}")
        log.info(f"Backtest: {config_name}")
        log.info(f"{'='*55}")

        trades = []
        skipped = 0

        for record, pred in zip(rec_holdout, preds):
            event_id  = record.get("event_id", "")
            ticker    = record.get("ticker", "")
            direction = int(record.get("direction", 1))
            true_lbl  = record.get("true_label_proxy", "")

            ev = events_by_id.get(event_id)
            if ev is None:
                skipped += 1
                continue

            try:
                fwd_ret = float(ev.get("forward_return_60m", 0) or 0)
            except (TypeError, ValueError):
                skipped += 1
                continue

            pred_lbl = LABEL_DEC.get(int(pred), "unclear")
            if pred_lbl == "unclear":
                skipped += 1
                continue

            pred_dir  = +1 if pred_lbl == "continuation" else -1
            trade_dir = direction * pred_dir   # +1 = long, -1 = short

            gross_bps = fwd_ret * 10000
            cost_bps  = COSTS_BPS.get(ticker, DEFAULT_COST)
            pnl_bps   = trade_dir * gross_bps - cost_bps
            correct   = (pred_lbl == true_lbl)

            trades.append({
                "event_id":   event_id,
                "ticker":     ticker,
                "t0_utc":     record.get("t0_utc", ""),
                "direction":  direction,
                "true_label": true_lbl,
                "pred_label": pred_lbl,
                "correct":    correct,
                "trade_dir":  trade_dir,
                "gross_bps":  round(gross_bps, 1),
                "cost_bps":   cost_bps,
                "pnl_bps":    round(pnl_bps, 1),
            })

        if not trades:
            log.warning(f"  No tradeable events. Skipped: {skipped}")
            continue

        log.info(f"  Tradeable events : {len(trades)}  (skipped: {skipped})")

        pnl_arr = np.array([t["pnl_bps"] for t in trades])
        abs_ret = np.abs([t["gross_bps"] for t in trades])

        # Event date span for annualization
        try:
            ts_min = pd.Timestamp(trades[0]["t0_utc"])
            ts_max = pd.Timestamp(trades[-1]["t0_utc"])
            span_years = max((ts_max - ts_min).days / 365.25, 0.1)
        except Exception:
            span_years = 1.0
        events_yr = len(trades) / span_years

        win_rate   = float((pnl_arr > 0).mean())
        ann_sharpe = sharpe(pnl_arr, events_yr)
        max_dd     = max_drawdown(pnl_arr)
        cal_ratio  = calmar(pnl_arr, events_yr)
        med_abs    = float(np.median(abs_ret))
        be_acc     = breakeven_accuracy(np.mean([t["cost_bps"] for t in trades]), med_abs)

        # Per-ticker breakdown
        by_ticker = defaultdict(list)
        for t in trades:
            by_ticker[t["ticker"]].append(t["pnl_bps"])
        ticker_summary = {
            tk: {
                "n": len(v),
                "mean_pnl": round(float(np.mean(v)), 1),
                "win_rate": round(float((np.array(v) > 0).mean()), 3),
            }
            for tk, v in sorted(by_ticker.items())
        }

        log.info(f"  Win rate         : {win_rate:.1%}")
        log.info(f"  Mean PnL         : {pnl_arr.mean():+.1f} bps/trade")
        log.info(f"  Median PnL       : {float(np.median(pnl_arr)):+.1f} bps/trade")
        log.info(f"  Total PnL        : {pnl_arr.sum():+.0f} bps")
        log.info(f"  Ann. Sharpe      : {ann_sharpe:+.3f}")
        log.info(f"  Max Drawdown     : {max_dd:.0f} bps")
        log.info(f"  Calmar ratio     : {cal_ratio:.2f}")
        log.info(f"  Median |return|  : {med_abs:.1f} bps")
        log.info(f"  Breakeven acc    : {be_acc:.1%}")
        log.info(f"  Edge/trade       : {pnl_arr.mean():+.1f} bps  (need > 0 to be profitable)")

        viable = "✓ VIABLE" if pnl_arr.mean() > 0 and ann_sharpe > 0.3 else \
                 "⚠ MARGINAL" if pnl_arr.mean() > 0 else "✗ NOT PROFITABLE"
        log.info(f"  Verdict          : {viable}")

        # Per-ticker log
        log.info("\n  Per-ticker results:")
        for tk, s in ticker_summary.items():
            log.info(f"    {tk:6s}: n={s['n']:4d} | mean={s['mean_pnl']:+6.1f} bps | win={s['win_rate']:.1%}")

        all_bt[config_name] = {
            "n_trades":           len(trades),
            "n_skipped":          skipped,
            "win_rate":           round(win_rate, 4),
            "mean_pnl_bps":       round(float(pnl_arr.mean()), 2),
            "median_pnl_bps":     round(float(np.median(pnl_arr)), 2),
            "std_pnl_bps":        round(float(pnl_arr.std()), 2),
            "total_pnl_bps":      round(float(pnl_arr.sum()), 1),
            "annualized_sharpe":  round(ann_sharpe, 3),
            "max_drawdown_bps":   round(max_dd, 1),
            "calmar_ratio":       round(cal_ratio, 3),
            "events_per_year":    round(events_yr, 1),
            "span_years":         round(span_years, 2),
            "median_abs_ret_bps": round(med_abs, 1),
            "breakeven_accuracy": round(be_acc, 4),
            "verdict":            viable,
            "by_ticker":          ticker_summary,
            "pnl_series":         [t["pnl_bps"] for t in trades],
        }

        # Save trade CSV
        trade_csv_path = out_dir / f"backtest_trades_{config_name}.csv"
        keys = ["event_id","ticker","t0_utc","direction","true_label","pred_label",
                "correct","trade_dir","gross_bps","cost_bps","pnl_bps"]
        with open(trade_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(trades)
        log.info(f"\n  Trade log → {trade_csv_path}")

        # Equity curve plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cum = np.cumsum(pnl_arr)
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})

            axes[0].plot(cum, color="#2196F3", linewidth=1.2)
            axes[0].axhline(0, color="gray", linewidth=0.7, linestyle="--")
            axes[0].fill_between(range(len(cum)), cum, 0, where=(cum >= 0), alpha=0.15, color="green")
            axes[0].fill_between(range(len(cum)), cum, 0, where=(cum < 0),  alpha=0.15, color="red")
            axes[0].set_ylabel("Cumulative PnL (bps)")
            axes[0].set_title(
                f"Equity Curve — {config_name} | Sharpe {ann_sharpe:+.2f} | Win {win_rate:.1%} | "
                f"Mean {pnl_arr.mean():+.1f} bps/trade",
                fontsize=10,
            )
            axes[0].spines[["top", "right"]].set_visible(False)

            # Drawdown
            peak = np.maximum.accumulate(cum)
            dd   = peak - cum
            axes[1].fill_between(range(len(dd)), dd, color="red", alpha=0.4)
            axes[1].set_ylabel("Drawdown (bps)")
            axes[1].set_xlabel("Trade number")
            axes[1].spines[["top", "right"]].set_visible(False)

            plt.tight_layout()
            fig_path = out_dir / f"equity_curve_{config_name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"  Equity curve → {fig_path}")
        except Exception as e:
            log.warning(f"Could not save equity curve: {e}")

    # ── Save aggregate results ─────────────────────────────────────────────────
    out_path = out_dir / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(all_bt, f, indent=2, default=str)
    log.info(f"\nAll backtest results → {out_path}")

    # ── Cross-config comparison ────────────────────────────────────────────────
    if len(all_bt) > 1:
        log.info(f"\n{'='*55}")
        log.info("CROSS-CONFIG COMPARISON")
        log.info(f"{'='*55}")
        log.info(f"  {'Config':<20} {'Sharpe':>8} {'MeanPnL':>10} {'WinRate':>9} {'MaxDD':>8} {'Verdict'}")
        log.info(f"  {'-'*70}")
        for cfg, s in all_bt.items():
            log.info(
                f"  {cfg:<20} {s['annualized_sharpe']:>+8.3f} "
                f"{s['mean_pnl_bps']:>+9.1f}bps "
                f"{s['win_rate']:>8.1%} "
                f"{s['max_drawdown_bps']:>7.0f}bps "
                f"  {s['verdict']}"
            )


if __name__ == "__main__":
    main()
