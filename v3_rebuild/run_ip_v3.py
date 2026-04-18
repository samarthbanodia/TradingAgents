"""
V3 Direct-Label Ensemble Pipeline
==================================

Key changes vs V2.1:
  - 3 LLM calls/event (Micro, News, Macro) — no Skeptic, no Judge
  - Agents output direct labels (continuation/reversal) not I/P scores
  - RETuning evidence-scoring prompt format (enumerate FOR/AGAINST before committing)
  - WHO→WHOM→WHAT causal schema for Micro agent
  - Weighted math ensemble with direction heuristic (4th voter)
  - Asymmetric reversal prior: 60% for large-z intraday spikes
  - Time-of-day conditioned direction weight
  - 7 new V3 microstructure features (idio residual, semivariance, signed vol, etc.)
  - Records include all agent scores + v3_features for downstream ML (LightGBM)

Usage:
    python run_ip_v3.py --limit 5          # smoke test
    python run_ip_v3.py                    # all 100 events
    python run_ip_v3.py --resume           # continue after interrupt
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from providers import OpenAIProvider, AnthropicProvider, DeepSeekProvider
from agents.briefs_v3 import (
    build_open_baselines,
    build_technical_brief_v3,
    build_news_brief_v3,
    build_macro_brief_v3,
    compute_v3_features,
    _is_opening_bell,
    _get_open_adjusted,
)
from agents.prompts_v3 import (
    MICROSTRUCTURE_SYSTEM_V3,
    NEWS_SYSTEM_V3,
    MACRO_SYSTEM_V3,
)
from eval.ensemble import (
    run_ensemble,
    validate_agent_output_v3,
    run_evaluation_v3,
    print_summary_v3,
)

ETF_TICKERS = {"SPY", "QQQ", "XLK"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Cost Estimation ───────────────────────────────────────────────────────────

_PRICE_PER_1M = {
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"input": 2.50,  "output": 10.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "deepseek-chat":             {"input": 0.27,  "output": 1.10},
    "gemini-2.0-flash-lite":     {"input": 0.075, "output": 0.30},
}

# V3: 3 calls/event (no Skeptic, no Judge)
_CALL_PROFILE_V3 = [
    ("micro",  "micro",  1100, 350),
    ("news",   "news",   1200, 250),
    ("macro",  "macro",   950, 250),
]


def log_cost_estimate(providers_map, n_events):
    W = 66
    logger.info("=" * W)
    logger.info("UPFRONT COST ESTIMATE — V3  (3 calls/event; no Skeptic/Judge)")
    logger.info(f"  {'Call':<10} {'Model':<32} {'$/event':>9}")
    logger.info("-" * W)
    total_per_event = 0.0
    for call_name, prov_key, in_tok, out_tok in _CALL_PROFILE_V3:
        model = providers_map[prov_key].model
        rates = _PRICE_PER_1M.get(model, {"input": 2.00, "output": 8.00})
        cost  = (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000
        total_per_event += cost
        logger.info(f"  {call_name:<10} {model:<32} ${cost:.5f}")
    logger.info("-" * W)
    logger.info(f"  {'Per-event total':<44} ${total_per_event:.5f}")
    logger.info(f"  {'Run total ({} events)'.format(n_events):<44} ${total_per_event * n_events:.4f}")
    logger.info("")
    logger.info("  V2.1 was 5 calls/event ($0.88) — V3 saves 2 calls/event (40% fewer LLM calls)")
    logger.info("=" * W)


# ── Provider Initialization ───────────────────────────────────────────────────

def init_providers(args):
    openai_key    = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_key  = os.getenv("DEEPSEEK_API_KEY")

    available = [k for k, v in [("openai", openai_key), ("anthropic", anthropic_key), ("deepseek", deepseek_key)] if v]
    if not available:
        logger.error("No API keys found. Set OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY in .env")
        sys.exit(1)
    logger.info(f"Available providers: {available}")

    providers = {}

    # Micro → OpenAI gpt-4o-mini (structured numeric pattern recognition)
    if openai_key:
        providers["micro"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    elif anthropic_key:
        logger.warning("OpenAI key missing → using Anthropic for Micro")
        providers["micro"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)
    else:
        providers["micro"] = DeepSeekProvider(args.deepseek_model, deepseek_key, args.temperature, args.max_tokens)

    # News → Anthropic Claude Haiku (text/prose reasoning)
    if anthropic_key:
        providers["news"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)
    elif openai_key:
        logger.warning("Anthropic key missing → using OpenAI for News")
        providers["news"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    else:
        providers["news"] = DeepSeekProvider(args.deepseek_model, deepseek_key, args.temperature, args.max_tokens)

    # Macro → DeepSeek (different model family for diversity)
    if deepseek_key:
        providers["macro"] = DeepSeekProvider(args.deepseek_model, deepseek_key, args.temperature, args.max_tokens)
    elif openai_key:
        logger.warning("DeepSeek key missing → using OpenAI for Macro")
        providers["macro"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    else:
        providers["macro"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)

    return providers


# ── I/O Helpers ───────────────────────────────────────────────────────────────

def load_events(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_packet(packets_dir, event_id, ticker):
    path = os.path.join(packets_dir, ticker, f"{event_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"event": {}, "news": {"ticker_news": [], "no_news_found": True, "macro_news": {}}}


def load_ohlcv(path):
    if not path or not os.path.exists(path):
        logger.warning(f"OHLCV not found at {path} — V3 microstructure features will be null")
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        logger.info(f"Loaded OHLCV: {len(df):,} rows, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load OHLCV: {e}")
        return None


def load_processed_ids(jsonl_path):
    ids = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["event_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Core Event Processor ──────────────────────────────────────────────────────

def process_event_v3(row, packet, ohlcv_df, baselines, providers, news_lookback_minutes):
    """
    V3 pipeline for a single event:
    1. Build V3 technical brief (V2 base + 7 new features)
    2. Build news + macro briefs (unchanged from V2)
    3. Call Micro, News, Macro agents (3 LLM calls, no Skeptic/Judge)
    4. Run weighted ensemble with direction heuristic
    5. Return rich record for evaluation + downstream ML
    """
    event_id = row["event_id"]
    ticker   = row["ticker"]
    t0_str   = row["t0_utc"]
    t_start  = time.time()

    is_bell = _is_opening_bell(t0_str)
    dir_int = int(row["direction"])
    is_up   = dir_int == 1
    is_etf  = ticker in ETF_TICKERS

    # Open-adjusted signals (for briefs + record)
    adj_vol, adj_range = _get_open_adjusted(row, ohlcv_df, baselines)

    # ── Build briefs ──
    tech_brief, v3_features = build_technical_brief_v3(row, ohlcv_df, baselines)
    news_brief, has_pre_event_news = build_news_brief_v3(row, packet, news_lookback_minutes)
    macro_brief = build_macro_brief_v3(row, packet, ohlcv_df, news_lookback_minutes)

    # ── Micro agent ──
    logger.info(f"  [Micro] {providers['micro'].model}...")
    raw_micro = providers["micro"].call(MICROSTRUCTURE_SYSTEM_V3, tech_brief)
    micro_out = validate_agent_output_v3(raw_micro, "micro")

    # ── News agent ──
    logger.info(f"  [News]  {providers['news'].model}...")
    raw_news = providers["news"].call(NEWS_SYSTEM_V3, news_brief)
    news_out = validate_agent_output_v3(raw_news, "news")

    # ── Macro agent ──
    logger.info(f"  [Macro] {providers['macro'].model}...")
    raw_macro = providers["macro"].call(MACRO_SYSTEM_V3, macro_brief)
    macro_out = validate_agent_output_v3(raw_macro, "macro")

    # ── Weighted ensemble ──
    ens = run_ensemble(micro_out, news_out, macro_out, row, v3_features)

    elapsed = round(time.time() - t_start, 1)
    logger.info(
        f"  Done {elapsed}s → "
        f"micro={micro_out['label']}({micro_out['confidence']:.2f}) "
        f"news={news_out['label']}({news_out['confidence']:.2f}) "
        f"macro={macro_out['label']}({macro_out['confidence']:.2f}) "
        f"→ ensemble={ens['ensemble_label']}({ens['ensemble_score']:.3f})"
    )

    return {
        # Identity
        "event_id":            event_id,
        "ticker":              ticker,
        "t0_utc":              t0_str,
        "direction":           dir_int,
        "true_label_proxy":    row.get("label_proxy", ""),
        # Context flags
        "has_pre_event_news":  has_pre_event_news,
        "is_opening_bell":     is_bell,
        "is_etf":              is_etf,
        "is_up_spike":         is_up,
        # Agent outputs (full dicts for downstream ML)
        "micro":               micro_out,
        "news":                news_out,
        "macro":               macro_out,
        # Flat agent fields (for CSV convenience)
        "micro_label":         micro_out["label"],
        "micro_conf":          micro_out["confidence"],
        "micro_cont_score":    micro_out["continuation_score"],
        "micro_rev_score":     micro_out["reversal_score"],
        "micro_who":           micro_out.get("who", ""),
        "news_label":          news_out["label"],
        "news_conf":           news_out["confidence"],
        "news_cont_score":     news_out["continuation_score"],
        "news_rev_score":      news_out["reversal_score"],
        "macro_label":         macro_out["label"],
        "macro_conf":          macro_out["confidence"],
        "macro_cont_score":    macro_out["continuation_score"],
        "macro_rev_score":     macro_out["reversal_score"],
        # V3 features (for LightGBM ML stage)
        "v3_features":         v3_features,
        # Ensemble
        "ensemble_label":      ens["ensemble_label"],
        "ensemble_score":      ens["ensemble_score"],
        "ensemble_votes":      ens["votes"],
        "dir_weight":          ens["dir_weight"],
        "dir_prior":           ens["dir_prior"],
        "is_macro_driven":     ens["is_macro_driven"],
        # Legacy compat (for eval)
        "open_adj_vol_mult":   adj_vol,
        "open_adj_range_mult": adj_range,
        "elapsed_s":           elapsed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V3 Direct-Label Ensemble Pipeline")
    parser.add_argument("--events_csv",            default="out_final/selected_events.csv")
    parser.add_argument("--packets_dir",           default="out_news/packets")
    parser.add_argument("--ohlcv_path",            default="ohlcv_5min.parquet")
    parser.add_argument("--out_dir",               default="out_agents_v3")
    parser.add_argument("--openai_model",          default="gpt-4o-mini")
    parser.add_argument("--anthropic_model",       default="claude-haiku-4-5-20251001")
    parser.add_argument("--deepseek_model",        default="deepseek-chat")
    parser.add_argument("--temperature",           type=float, default=0.3)
    parser.add_argument("--max_tokens",            type=int,   default=1024)
    parser.add_argument("--news_lookback_minutes", type=int,   default=180)
    parser.add_argument("--limit",  type=int,   default=None, help="Process only first N events")
    parser.add_argument("--resume", action="store_true",      help="Skip already-processed events")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.events_csv):
        logger.error(f"Events CSV not found: {args.events_csv}")
        sys.exit(1)

    jsonl_path = os.path.join(args.out_dir, "ip_outputs_v3.jsonl")
    csv_path   = os.path.join(args.out_dir, "ip_outputs_v3.csv")
    eval_path  = os.path.join(args.out_dir, "eval_summary_v3.json")

    if not args.resume and os.path.exists(jsonl_path):
        logger.warning(f"Clearing {jsonl_path} (use --resume to continue)")
        open(jsonl_path, "w").close()

    events   = load_events(args.events_csv)
    logger.info(f"Loaded {len(events)} events from {args.events_csv}")

    ohlcv_df = load_ohlcv(args.ohlcv_path)

    if ohlcv_df is not None:
        logger.info("Building per-ticker per-slot open baselines...")
        baselines = build_open_baselines(ohlcv_df)
        logger.info(f"Baselines ready for {len(baselines)} tickers")
    else:
        baselines = {}
        logger.warning("OHLCV unavailable — V3 features and open-adj will be null")

    # Pre-event news coverage stats
    if os.path.exists(args.packets_dir):
        from agents.briefs_v2 import _filter_pre_event_news, _parse_t0
        events_with_pre_news = 0
        for row in events:
            t0 = _parse_t0(row["t0_utc"])
            pkt = load_packet(args.packets_dir, row["event_id"], row["ticker"])
            arts = pkt.get("news", {}).get("ticker_news", [])
            if _filter_pre_event_news(arts, t0, args.news_lookback_minutes):
                events_with_pre_news += 1
        logger.info(
            f"Pre-event news coverage: {events_with_pre_news}/{len(events)} events "
            f"have ≥1 pre-event article"
        )

    # Resume logic
    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(jsonl_path)
        logger.info(f"Resume: {len(processed_ids)} events already processed")

    if args.limit:
        events = events[:args.limit]
        logger.info(f"Limiting to {len(events)} events")

    providers = init_providers(args)
    log_cost_estimate(providers, len(events) - len(processed_ids))

    # Load existing records for resume
    all_records = []
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total   = len(events)
    n_to_do = sum(1 for r in events if r["event_id"] not in processed_ids)
    n_done  = skipped = failed = 0
    event_times = []

    logger.info(f"Events to process: {n_to_do}  (skipping {total - n_to_do} done)")

    try:
        for idx, row in enumerate(events, 1):
            eid    = row["event_id"]
            ticker = row["ticker"]

            if eid in processed_ids:
                skipped += 1
                continue

            logger.info(f"[{idx}/{total}] {eid} ({ticker}) label={row.get('label_proxy','?')}")
            packet     = load_packet(args.packets_dir, eid, ticker)
            t_ev_start = time.time()

            try:
                record = process_event_v3(
                    row, packet, ohlcv_df, baselines, providers, args.news_lookback_minutes
                )
                append_jsonl(jsonl_path, record)
                all_records.append(record)
            except Exception as e:
                logger.error(f"  FAILED {eid}: {e}", exc_info=True)
                failed += 1
                append_jsonl(jsonl_path, {
                    "event_id":         eid,
                    "ticker":           ticker,
                    "error":            str(e),
                    "true_label_proxy": row.get("label_proxy", ""),
                })

            event_times.append(time.time() - t_ev_start)
            n_done += 1
            avg_t = sum(event_times) / len(event_times)
            eta   = avg_t * (n_to_do - n_done)
            logger.info(f"[{idx}/{total}] ETA ~{eta:.0f}s remaining (avg {avg_t:.1f}s/event)")

    except KeyboardInterrupt:
        logger.info("Interrupted — saving partial results...")

    logger.info(f"Run complete. processed={n_done}, skipped={skipped}, failed={failed}")

    # ── Write CSV summary ──
    valid_records = [r for r in all_records if "error" not in r]
    if valid_records:
        csv_fields = [
            "event_id", "ticker", "t0_utc", "direction",
            "has_pre_event_news", "is_opening_bell", "is_etf", "is_up_spike",
            "micro_label", "micro_conf", "micro_cont_score", "micro_rev_score", "micro_who",
            "news_label",  "news_conf",  "news_cont_score",  "news_rev_score",
            "macro_label", "macro_conf", "macro_cont_score", "macro_rev_score",
            "ensemble_label", "ensemble_score",
            "is_macro_driven", "dir_weight", "dir_prior",
            "open_adj_vol_mult", "open_adj_range_mult",
            "true_label_proxy", "elapsed_s",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in valid_records:
                writer.writerow(r)
        logger.info(f"CSV written: {csv_path}  ({len(valid_records)} rows)")

    # ── Evaluation ──
    eval_records = [r for r in all_records if "error" not in r]
    if eval_records:
        logger.info(f"\nRunning evaluation on {len(eval_records)} valid records...")
        results = run_evaluation_v3(eval_records, eval_path)
        logger.info(f"Eval written: {eval_path}")
        logger.info("\n" + "=" * 66)
        logger.info("STRATEGY COMPARISON (V3)")
        logger.info("=" * 66)
        print_summary_v3(results)

        # Key comparison vs baselines
        sc        = results.get("strategy_comparison", {})
        v3_acc    = sc.get("v3_ensemble",         {}).get("accuracy_3way")
        dir_acc   = sc.get("direction_heuristic",  {}).get("accuracy_3way")
        v2_baseline = 0.163
        v1_baseline = 0.352

        logger.info("\n" + "-" * 66)
        logger.info("Baselines:")
        logger.info(f"  V2.1 full pipeline     : {v2_baseline:.3f} (16.3%)  ← must beat")
        logger.info(f"  V1  full pipeline      : {v1_baseline:.3f} (35.2%)")
        if dir_acc is not None:
            logger.info(f"  Direction heuristic    : {dir_acc:.3f}  ← primary target to beat")
        if v3_acc is not None:
            beat_v2  = "✓" if v3_acc > v2_baseline else "✗"
            beat_v1  = "✓" if v3_acc > v1_baseline else "✗"
            beat_dir = "✓" if (dir_acc is not None and v3_acc > dir_acc) else "✗"
            logger.info(f"  V3 ensemble            : {v3_acc:.3f}  beat V2? {beat_v2}  beat V1? {beat_v1}  beat dir? {beat_dir}")
        logger.info("-" * 66)
    else:
        logger.warning("No valid records for evaluation")


if __name__ == "__main__":
    main()
