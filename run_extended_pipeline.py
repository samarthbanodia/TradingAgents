"""
Extended Pipeline Orchestrator — N≥1000, 15 tickers, 2022-2025.

Runs all stages in order. Each stage checks whether its output already
exists and skips if complete (resume-friendly). Stages can also be run
individually by passing --stage <name>.

Stages:
  1  fetch       Fetch OHLCV (15 tickers, 2022-2025, monthly batches)
  2  mine        Mine spike events from extended OHLCV
  3  select      Select N=1200 balanced events
  4  news        Build news packets (Polygon News API)
  5  agents      Run V3b agents (3 LLM calls per event)
  6  leakage     Falsification audit (shuffled label null model)
  7  ml          ML evaluation (temporal holdout, ablations, McNemar, backtest)
  8  backtest    Standalone detailed backtest with equity curve

Usage:
    python run_extended_pipeline.py            # run all stages
    python run_extended_pipeline.py --stage fetch
    python run_extended_pipeline.py --stage ml
    python run_extended_pipeline.py --dry_run  # print commands only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
OHLCV_EXT      = "ohlcv_5min_extended.parquet"
EVENTS_STRICT  = "out_extended_strict/events.csv"
EVENTS_FINAL   = "out_extended_final/selected_events.csv"
NEWS_PACKETS   = "out_extended_news/packets"
AGENTS_JSONL   = "out_agents_extended/ip_outputs_v3.jsonl"
EVAL_DIR       = "out_eval_extended"

TARGET_N       = 800    # event selection target (~24 months × 15 tickers)
MIN_STOCK      = 40     # min events per stock ticker
MAX_STOCK      = 80     # max events per stock ticker
MIN_ETF        = 15     # min events per ETF
MAX_ETF        = 35     # max events per ETF


def run(cmd, dry_run=False, label=""):
    if label:
        print(f"\n{'-'*60}")
        print(f"  {label}")
        print(f"{'-'*60}")
    print(f"  $ {cmd}")
    if dry_run:
        return True
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        return False
    print(f"\n  Done in {elapsed:.0f}s")
    return True


def stage_fetch(dry_run):
    if Path(OHLCV_EXT).exists():
        print(f"  [fetch] {OHLCV_EXT} exists — skipping (delete to re-fetch)")
        return True
    return run(
        f"python fetch_ohlcv_extended.py --out {OHLCV_EXT}",
        dry_run=dry_run,
        label="STAGE 1: Fetch extended OHLCV (15 tickers, 2022-2025)",
    )


def stage_mine(dry_run):
    if Path(EVENTS_STRICT).exists():
        print(f"  [mine] {EVENTS_STRICT} exists — skipping")
        return True
    return run(
        f"python mine_events_strict.py "
        f"--input_path {OHLCV_EXT} "
        f"--out_dir out_extended_strict",
        dry_run=dry_run,
        label="STAGE 2: Mine spike events from extended OHLCV",
    )


def stage_select(dry_run):
    if Path(EVENTS_FINAL).exists():
        print(f"  [select] {EVENTS_FINAL} exists — skipping")
        return True
    return run(
        f"python select_best_events.py "
        f"--events_path {EVENTS_STRICT} "
        f"--ohlcv_path {OHLCV_EXT} "
        f"--out_dir out_extended_final "
        f"--target_n {TARGET_N} "
        f"--min_per_stock {MIN_STOCK} "
        f"--max_per_stock {MAX_STOCK} "
        f"--min_per_etf {MIN_ETF} "
        f"--max_per_etf {MAX_ETF}",
        dry_run=dry_run,
        label=f"STAGE 3: Select {TARGET_N} balanced events",
    )


def stage_news(dry_run):
    # Check if packets dir has at least some files
    packets = Path(NEWS_PACKETS)
    if packets.exists() and len(list(packets.rglob("*.json"))) >= 100:
        print(f"  [news] {NEWS_PACKETS} appears populated — skipping. Delete to re-fetch.")
        return True
    return run(
        f"python build_news_packets.py "
        f"--events_csv {EVENTS_FINAL} "
        f"--out_dir out_extended_news",
        dry_run=dry_run,
        label="STAGE 4: Build news packets (Polygon News API)",
    )


def stage_agents(dry_run):
    if Path(AGENTS_JSONL).exists():
        # Count lines to detect partial run
        with open(AGENTS_JSONL) as f:
            n = sum(1 for _ in f)
        if n >= TARGET_N * 0.95:
            print(f"  [agents] {AGENTS_JSONL} has {n} records (≥95% of {TARGET_N}) — skipping")
            return True
        print(f"  [agents] {AGENTS_JSONL} has {n} records — resuming")

    return run(
        f"python run_ip_v3.py "
        f"--events_csv {EVENTS_FINAL} "
        f"--packets_dir {NEWS_PACKETS} "
        f"--ohlcv_path {OHLCV_EXT} "
        f"--out_dir out_agents_extended "
        f"--resume",
        dry_run=dry_run,
        label="STAGE 5: Run V3b agents (3 LLM calls per event)",
    )


def stage_leakage(dry_run):
    out = Path(EVAL_DIR) / "falsification_audit.json"
    if out.exists():
        print(f"  [leakage] {out} exists — skipping")
        return True
    return run(
        f"python -m eval.falsification_audit "
        f"--jsonl {AGENTS_JSONL} "
        f"--events_csv {EVENTS_FINAL} "
        f"--ohlcv {OHLCV_EXT} "
        f"--out_dir {EVAL_DIR} "
        f"--n_runs 20",
        dry_run=dry_run,
        label="STAGE 6: Falsification audit (shuffled label null model)",
    )


def stage_ml(dry_run):
    out = Path(EVAL_DIR) / "ml_stage_v2_results.json"
    if out.exists():
        print(f"  [ml] {out} exists — skipping (delete to re-run)")
        return True
    return run(
        f"python -m eval.ml_stage_v2 "
        f"--jsonl {AGENTS_JSONL} "
        f"--events_csv {EVENTS_FINAL} "
        f"--ohlcv {OHLCV_EXT} "
        f"--out_dir {EVAL_DIR} "
        f"--holdout_frac 0.20 "
        f"--n_splits 5 "
        f"--null_model_runs 10",
        dry_run=dry_run,
        label="STAGE 7: ML evaluation (temporal holdout, ablations, McNemar, backtest)",
    )


def stage_backtest(dry_run):
    out = Path(EVAL_DIR) / "backtest_results.json"
    if out.exists():
        print(f"  [backtest] {out} exists — skipping")
        return True
    results_json = str(Path(EVAL_DIR) / "ml_stage_v2_results.json")
    return run(
        f"python -m eval.backtest_pnl "
        f"--results {results_json} "
        f"--jsonl {AGENTS_JSONL} "
        f"--events_csv {EVENTS_FINAL} "
        f"--out_dir {EVAL_DIR}",
        dry_run=dry_run,
        label="STAGE 8: Detailed backtest PnL with equity curves",
    )


STAGE_MAP = {
    "fetch":    stage_fetch,
    "mine":     stage_mine,
    "select":   stage_select,
    "news":     stage_news,
    "agents":   stage_agents,
    "leakage":  stage_leakage,
    "ml":       stage_ml,
    "backtest": stage_backtest,
}
STAGE_ORDER = ["fetch", "mine", "select", "news", "agents", "leakage", "ml", "backtest"]


def main():
    parser = argparse.ArgumentParser(description="Extended pipeline orchestrator")
    parser.add_argument("--stage",   choices=list(STAGE_MAP.keys()) + ["all"], default="all")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    print("=" * 60)
    print("TradingAgents Extended Pipeline  N>=1000")
    print("=" * 60)
    print(f"  OHLCV    : {OHLCV_EXT}")
    print(f"  Events   : {EVENTS_FINAL}")
    print(f"  Agents   : {AGENTS_JSONL}")
    print(f"  Eval out : {EVAL_DIR}")
    print(f"  Target N : {TARGET_N}")
    if args.dry_run:
        print("  [DRY RUN - commands will be printed but not executed]")
    print()

    stages = STAGE_ORDER if args.stage == "all" else [args.stage]

    for stage_name in stages:
        fn = STAGE_MAP[stage_name]
        ok = fn(dry_run=args.dry_run)
        if not ok and not args.dry_run:
            print(f"\nPipeline failed at stage: {stage_name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All requested stages complete.")
    print("=" * 60)
    print(f"\nKey output files:")
    print(f"  Events selected  : {EVENTS_FINAL}")
    print(f"  Agent outputs    : {AGENTS_JSONL}")
    print(f"  ML results       : {EVAL_DIR}/ml_stage_v2_results.json")
    print(f"  Falsification    : {EVAL_DIR}/falsification_audit.json")
    print(f"  Backtest results : {EVAL_DIR}/backtest_results.json")
    print(f"  Equity curves    : {EVAL_DIR}/equity_curve_*.png")
    print(f"  Feature plot     : {EVAL_DIR}/feature_importance_v2.png")


if __name__ == "__main__":
    main()
