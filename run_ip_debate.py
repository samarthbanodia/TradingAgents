"""
I-P Decomposition Multi-Agent Debate System
=============================================
Main orchestrator: loads events, runs 3 agents + skeptic + revision + judge,
writes incremental JSONL and final CSV + eval summary.

Usage:
    python run_ip_debate.py --limit 3          # test on 3 events
    python run_ip_debate.py                     # all 100 events
    python run_ip_debate.py --resume            # skip already-processed events
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from providers import OpenAIProvider, AnthropicProvider, GeminiProvider, DeepSeekProvider
from agents.briefs import (
    build_technical_brief,
    build_news_brief,
    build_macro_brief,
    build_combined_brief,
    _is_opening_bell,
)
from agents.prompts import (
    MICROSTRUCTURE_SYSTEM,
    NEWS_SYSTEM,
    MACRO_SYSTEM,
    SKEPTIC_SYSTEM,
    REVISION_SYSTEM,
    JUDGE_SYSTEM,
    format_skeptic_input,
    format_revision_input,
    format_judge_input,
)
from eval.metrics import run_evaluation

ETF_TICKERS = {"SPY", "QQQ", "XLK"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_events(csv_path: str) -> list[dict]:
    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def load_packet(packets_dir: str, event_id: str, ticker: str) -> dict:
    """Load the news packet JSON for an event."""
    path = os.path.join(packets_dir, ticker, f"{event_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"event": {}, "news": {"ticker_news": [], "no_news_found": True, "macro_news": {}}}


def load_ohlcv(path: str):
    """Optionally load OHLCV parquet. Returns None if unavailable."""
    if not path or not os.path.exists(path):
        logger.warning(f"OHLCV file not found at {path}, co-move features will be null")
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        logger.info(f"Loaded OHLCV: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Failed to load OHLCV: {e}")
        return None


def load_processed_ids(jsonl_path: str) -> set:
    """Read existing JSONL to find already-processed event IDs."""
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


def append_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def compute_disagreement(r2_micro: dict, r2_news: dict, r2_macro: dict) -> float:
    """Mean pairwise L1 distance of (I,P) vectors across 3 agents."""
    agents = [r2_micro, r2_news, r2_macro]
    pairs = [(0, 1), (0, 2), (1, 2)]
    total = 0.0
    for i, j in pairs:
        di = abs(agents[i].get("I_score", 0.5) - agents[j].get("I_score", 0.5))
        dp = abs(agents[i].get("P_score", 0.5) - agents[j].get("P_score", 0.5))
        total += di + dp
    return round(total / 3, 4)


def compute_round2_shift(r1s: list[dict], r2s: list[dict]) -> float:
    """Average absolute change in I and P from Round 1 to Round 2 across agents."""
    shifts = []
    for r1, r2 in zip(r1s, r2s):
        di = abs(r2.get("I_score", 0.5) - r1.get("I_score", 0.5))
        dp = abs(r2.get("P_score", 0.5) - r1.get("P_score", 0.5))
        shifts.append((di + dp) / 2)
    return round(sum(shifts) / len(shifts), 4) if shifts else 0.0


def derive_dominant_class(final_I: float, final_P: float) -> tuple[str, str]:
    """Apply dominance rules to get (dominant_class, policy_behavior)."""
    if final_I > final_P + 0.10:
        return "info", "continuation"
    if final_P > final_I + 0.10:
        return "panic", "reversal"
    if abs(final_I - final_P) <= 0.10 and max(final_I, final_P) >= 0.40:
        return "mixed", "unclear"
    return "noise", "unclear"


# ── Validation helpers ────────────────────────────────────────────────────────

def _clamp01(v) -> float:
    """Clamp a value to [0, 1], return 0.5 on error."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def validate_agent_output(raw: dict, role: str) -> dict:
    """Ensures required fields are present and valid; clamps scores; truncates lists."""
    raw["I_score"] = _clamp01(raw.get("I_score", 0.5))
    raw["P_score"] = _clamp01(raw.get("P_score", 0.5))
    raw["confidence"] = _clamp01(raw.get("confidence", 0.5))
    evidence = raw.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    raw["evidence"] = evidence[:3]
    counterfactuals = raw.get("counterfactuals", [])
    if not isinstance(counterfactuals, list):
        counterfactuals = [str(counterfactuals)] if counterfactuals else []
    raw["counterfactuals"] = counterfactuals[:2]
    return raw


def validate_skeptic_output(raw: dict) -> dict:
    """Ensures skeptic output fields are present and valid."""
    critique = raw.get("critique_points", [])
    if not isinstance(critique, list):
        critique = [str(critique)] if critique else []
    raw["critique_points"] = critique[:4]
    qfa = raw.get("questions_for_each_agent", {})
    if not isinstance(qfa, dict):
        qfa = {}
    for key in ("microstructure", "news", "macro"):
        val = qfa.get(key, [])
        if not isinstance(val, list):
            val = [str(val)] if val else []
        qfa[key] = val[:2]
    raw["questions_for_each_agent"] = qfa
    adj = raw.get("suggested_adjustments", {})
    if not isinstance(adj, dict):
        adj = {}
    try:
        adj["I_delta"] = max(-0.3, min(0.3, float(adj.get("I_delta", 0.0) or 0.0)))
    except (TypeError, ValueError):
        adj["I_delta"] = 0.0
    try:
        adj["P_delta"] = max(-0.3, min(0.3, float(adj.get("P_delta", 0.0) or 0.0)))
    except (TypeError, ValueError):
        adj["P_delta"] = 0.0
    raw["suggested_adjustments"] = adj
    try:
        cd = max(-0.4, min(0.2, float(raw.get("confidence_delta", 0.0) or 0.0)))
    except (TypeError, ValueError):
        cd = 0.0
    raw["confidence_delta"] = cd
    return raw


def validate_judge_output(raw: dict) -> dict:
    """Ensures judge output is valid; falls back to code-computed class if invalid."""
    raw["final_I"] = _clamp01(raw.get("final_I", 0.5))
    raw["final_P"] = _clamp01(raw.get("final_P", 0.5))
    raw["final_confidence"] = _clamp01(raw.get("final_confidence", 0.5))
    dc = raw.get("dominant_class", "")
    pb = raw.get("policy_behavior", "")
    if dc not in {"info", "panic", "mixed", "noise"} or pb not in {"continuation", "reversal", "unclear"}:
        dc_comp, pb_comp = derive_dominant_class(raw["final_I"], raw["final_P"])
        raw["dominant_class"] = dc_comp
        raw["policy_behavior"] = pb_comp
    return raw


# ── Cost estimation ──────────────────────────────────────────────────────────

# Approximate retail pricing per 1M tokens (input, output) as of early 2026
_PRICE_PER_1M = {
    "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
    "gpt-4o":                    {"input": 2.50,  "output": 10.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "deepseek-chat":             {"input": 0.27,  "output": 1.10},
    "gemini-2.0-flash-lite":     {"input": 0.075, "output": 0.30},
}

# (call_label, provider_key, est_input_tokens, est_output_tokens)
_CALL_PROFILE = [
    ("micro_r1",  "micro",    900,  300),
    ("news_r1",   "news",    1100,  300),
    ("macro_r1",  "macro",    900,  300),
    ("skeptic",   "skeptic", 2000,  500),
    ("micro_r2",  "micro",   1100,  300),
    ("news_r2",   "news",    1100,  300),
    ("macro_r2",  "macro",   1100,  300),
    ("judge",     "judge",   1800,  400),
]


def log_cost_estimate(providers: dict, n_events: int):
    """Print per-call and total cost estimates before the run starts."""
    W = 58
    logger.info("=" * W)
    logger.info("UPFRONT COST ESTIMATE (approximate retail pricing)")
    logger.info(f"  {'Call':<12} {'Model':<30} {'$/event':>8}")
    logger.info("-" * W)
    total_per_event = 0.0
    provider_totals: dict[str, float] = {}
    for call_name, prov_key, in_tok, out_tok in _CALL_PROFILE:
        model = providers[prov_key].model
        rates = _PRICE_PER_1M.get(model, {"input": 2.00, "output": 8.00})
        cost = (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000
        total_per_event += cost
        provider_totals[model] = provider_totals.get(model, 0.0) + cost
        logger.info(f"  {call_name:<12} {model:<30} ${cost:.5f}")
    logger.info("-" * W)
    logger.info(f"  {'Per-event total':<42} ${total_per_event:.5f}")
    logger.info(f"  {'Run total ({} events)'.format(n_events):<42} ${total_per_event * n_events:.4f}")
    logger.info("")
    logger.info("  By provider (run total):")
    for model, cost in sorted(provider_totals.items(), key=lambda x: -x[1] * n_events):
        logger.info(f"    {model:<32} ${cost * n_events:.4f}")
    logger.info("=" * W)


# ── Provider initialization ─────────────────────────────────────────────────

def init_providers(args) -> dict:
    """Initialize LLM providers from env vars. Fallback to any available key."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    available = []
    if openai_key:
        available.append("openai")
    if anthropic_key:
        available.append("anthropic")
    if google_key:
        available.append("gemini")
    if deepseek_key:
        available.append("deepseek")

    if not available:
        logger.error("No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    logger.info(f"Available providers: {available}")

    providers = {}

    # Microstructure → OpenAI
    if openai_key:
        providers["micro"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    elif anthropic_key:
        logger.warning("OpenAI key missing, using Anthropic for Microstructure agent")
        providers["micro"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)
    else:
        providers["micro"] = DeepSeekProvider(args.deepseek_model, deepseek_key, args.temperature, args.max_tokens)

    # News → Anthropic
    if anthropic_key:
        providers["news"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)
    elif openai_key:
        logger.warning("Anthropic key missing, using OpenAI for News agent")
        providers["news"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    else:
        providers["news"] = DeepSeekProvider(args.deepseek_model, deepseek_key, args.temperature, args.max_tokens)

    # Macro → DeepSeek (preferred over Gemini), fallback chain: Gemini → OpenAI → Anthropic
    if deepseek_key:
        providers["macro"] = DeepSeekProvider(args.deepseek_model, deepseek_key, args.temperature, args.max_tokens)
    elif google_key:
        providers["macro"] = GeminiProvider(args.gemini_model, google_key, args.temperature, args.max_tokens)
    elif openai_key:
        logger.warning("DeepSeek/Google key missing, using OpenAI for Macro agent")
        providers["macro"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    else:
        providers["macro"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)

    # Skeptic → Anthropic
    if anthropic_key:
        providers["skeptic"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)
    elif openai_key:
        logger.warning("Anthropic key missing, using OpenAI for Skeptic")
        providers["skeptic"] = OpenAIProvider(args.openai_model, openai_key, args.temperature, args.max_tokens)
    else:
        providers["skeptic"] = GeminiProvider(args.gemini_model, google_key, args.temperature, args.max_tokens)

    # Judge → OpenAI (gpt-4o)
    if openai_key:
        providers["judge"] = OpenAIProvider(args.judge_model, openai_key, args.temperature, args.max_tokens)
    elif anthropic_key:
        logger.warning("OpenAI key missing, using Anthropic for Judge")
        providers["judge"] = AnthropicProvider(args.anthropic_model, anthropic_key, args.temperature, args.max_tokens)
    else:
        providers["judge"] = GeminiProvider(args.gemini_model, google_key, args.temperature, args.max_tokens)

    return providers


# ── Core debate loop ─────────────────────────────────────────────────────────

def process_event(row: dict, packet: dict, ohlcv_df, providers: dict) -> dict:
    """Run the full I-P debate for a single event. Returns the output record."""
    event_id = row["event_id"]
    ticker = row["ticker"]
    t0 = time.time()

    # Build briefs
    tech_brief = build_technical_brief(row, ohlcv_df)
    news_brief = build_news_brief(row, packet)
    macro_brief = build_macro_brief(row, packet, ohlcv_df)
    combined_brief = build_combined_brief(row, packet)

    # ── Round 1 ──
    logger.info(f"  [R1] Microstructure ({providers['micro'].model})...")
    r1_micro = validate_agent_output(providers["micro"].call(MICROSTRUCTURE_SYSTEM, tech_brief), "microstructure")

    logger.info(f"  [R1] News ({providers['news'].model})...")
    r1_news = validate_agent_output(providers["news"].call(NEWS_SYSTEM, news_brief), "news")

    logger.info(f"  [R1] Macro ({providers['macro'].model})...")
    r1_macro = validate_agent_output(providers["macro"].call(MACRO_SYSTEM, macro_brief), "macro")

    # ── Skeptic ──
    logger.info(f"  [SK] Skeptic ({providers['skeptic'].model})...")
    skeptic_input = format_skeptic_input(r1_micro, r1_news, r1_macro, combined_brief)
    skeptic_out = validate_skeptic_output(providers["skeptic"].call(SKEPTIC_SYSTEM, skeptic_input))

    # ── Round 2 ──
    logger.info(f"  [R2] Microstructure revision...")
    r2_micro = validate_agent_output(providers["micro"].call(
        REVISION_SYSTEM, format_revision_input(r1_micro, skeptic_out, "microstructure")
    ), "microstructure")

    logger.info(f"  [R2] News revision...")
    r2_news = validate_agent_output(providers["news"].call(
        REVISION_SYSTEM, format_revision_input(r1_news, skeptic_out, "news")
    ), "news")

    logger.info(f"  [R2] Macro revision...")
    r2_macro = validate_agent_output(providers["macro"].call(
        REVISION_SYSTEM, format_revision_input(r1_macro, skeptic_out, "macro")
    ), "macro")

    # ── Judge ──
    logger.info(f"  [JG] Judge ({providers['judge'].model})...")
    judge_input = format_judge_input(r2_micro, r2_news, r2_macro, skeptic_out)
    judge_out = validate_judge_output(providers["judge"].call(JUDGE_SYSTEM, judge_input))

    # ── Derived fields ──
    final_I = judge_out["final_I"]
    final_P = judge_out["final_P"]
    dominant_class = judge_out["dominant_class"]
    policy_behavior = judge_out["policy_behavior"]

    top_rationale = judge_out.get("rationale", [""])[0] if judge_out.get("rationale") else ""
    is_etf = ticker in ETF_TICKERS

    disagreement = compute_disagreement(r2_micro, r2_news, r2_macro)
    r2_shift = compute_round2_shift([r1_micro, r1_news, r1_macro], [r2_micro, r2_news, r2_macro])

    news_data = packet.get("news", {})
    has_news = bool(news_data.get("ticker_news")) and not news_data.get("no_news_found", False)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"  Done in {elapsed}s → I={final_I:.2f} P={final_P:.2f} → {policy_behavior}")

    return {
        "event_id": event_id,
        "ticker": ticker,
        "t0_utc": row["t0_utc"],
        "direction": int(row["direction"]),
        "label_proxy": row["label_proxy"],
        "true_label_proxy": row.get("label_proxy", ""),
        "has_news": has_news,
        "is_opening_bell": _is_opening_bell(row["t0_utc"]),
        "is_etf": is_etf,
        # Round 1
        "r1_micro": r1_micro,
        "r1_news": r1_news,
        "r1_macro": r1_macro,
        # Skeptic
        "skeptic": skeptic_out,
        # Round 2
        "r2_micro": r2_micro,
        "r2_news": r2_news,
        "r2_macro": r2_macro,
        # Judge
        "judge": judge_out,
        # Final scores
        "final_I": final_I,
        "final_P": final_P,
        "final_confidence": judge_out["final_confidence"],
        "dominant_class": dominant_class,
        "policy_behavior": policy_behavior,
        "top_rationale": top_rationale,
        "rationale": judge_out.get("rationale", []),
        "what_would_change_mind": judge_out.get("what_would_change_mind", []),
        # Derived
        "disagreement_score": disagreement,
        "round2_shift": r2_shift,
        "elapsed_s": elapsed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="I-P Decomposition Multi-Agent Debate")
    parser.add_argument("--events_csv", default="out_final/selected_events.csv")
    parser.add_argument("--packets_dir", default="out_news/packets")
    parser.add_argument("--ohlcv_path", default="ohlcv_5min.parquet")
    parser.add_argument("--out_dir", default="out_agents")
    parser.add_argument("--openai_model", default="gpt-4o-mini")
    parser.add_argument("--anthropic_model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--gemini_model", default="gemini-2.0-flash-lite")
    parser.add_argument("--deepseek_model", default="deepseek-chat")
    parser.add_argument("--judge_model", default="gpt-4o",
                        help="Judge model; gpt-4o recommended for synthesis quality")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N events")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed events")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Startup validation
    if not os.path.exists(args.events_csv):
        logger.error(f"Events CSV not found: {args.events_csv}")
        sys.exit(1)
    if not os.path.exists(args.packets_dir):
        logger.warning(f"Packets directory not found: {args.packets_dir}")
    for _key_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"):
        if not os.getenv(_key_name):
            logger.warning(f"Missing API key: {_key_name}")

    jsonl_path = os.path.join(args.out_dir, "ip_outputs.jsonl")
    csv_path = os.path.join(args.out_dir, "ip_outputs.csv")
    eval_path = os.path.join(args.out_dir, "eval_summary.json")

    # Fresh run: clear stale output so we don't accumulate duplicate records
    if not args.resume:
        if os.path.exists(jsonl_path):
            logger.warning(f"Clearing existing {jsonl_path} (use --resume to continue a prior run)")
            open(jsonl_path, "w").close()

    # Load data
    events = load_events(args.events_csv)
    logger.info(f"Loaded {len(events)} events from {args.events_csv}")

    ohlcv_df = load_ohlcv(args.ohlcv_path)

    # Resume logic
    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(jsonl_path)
        logger.info(f"Resume mode: {len(processed_ids)} event IDs already in JSONL (will be skipped)")
        # Warn about error records — they are skipped but not retried
        error_ids = set()
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        try:
                            _rec = json.loads(_line)
                            if "error" in _rec:
                                error_ids.add(_rec["event_id"])
                        except (json.JSONDecodeError, KeyError):
                            pass
        if error_ids:
            logger.warning(
                f"  {len(error_ids)} events previously FAILED and will be skipped: {sorted(error_ids)}\n"
                f"  Remove their lines from {jsonl_path} and re-run with --resume to retry them."
            )

    # Apply limit
    if args.limit:
        events = events[:args.limit]
        logger.info(f"Limiting to {len(events)} events")

    # Init providers
    providers = init_providers(args)
    log_cost_estimate(providers, len(events))

    # Process events
    all_records = []
    # Load existing records for eval
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total = len(events)
    # n_to_do: only events not already in the JSONL (accurate ETA on resume)
    n_to_do = sum(1 for row in events if row["event_id"] not in processed_ids)
    n_done = 0
    skipped = 0
    failed = 0
    event_elapsed_times: list[float] = []
    logger.info(f"Events to process this session: {n_to_do} (skipping {total - n_to_do} already done)")

    try:
        for idx, row in enumerate(events, 1):
            event_id = row["event_id"]
            ticker = row["ticker"]

            if event_id in processed_ids:
                skipped += 1
                continue

            logger.info(f"[{idx}/{total}] Processing {event_id}...")

            packet = load_packet(args.packets_dir, event_id, ticker)
            t_event_start = time.time()

            try:
                record = process_event(row, packet, ohlcv_df, providers)
                append_jsonl(jsonl_path, record)
                all_records.append(record)
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                failed += 1
                # Write a minimal error record so we can resume past it
                error_record = {
                    "event_id": event_id,
                    "ticker": ticker,
                    "error": str(e),
                    "label_proxy": row.get("label_proxy"),
                }
                append_jsonl(jsonl_path, error_record)

            event_elapsed_times.append(time.time() - t_event_start)
            n_done += 1
            avg_t = sum(event_elapsed_times) / len(event_elapsed_times)
            events_remaining = n_to_do - n_done
            logger.info(f"[{idx}/{total}] ETA ~{avg_t * events_remaining:.0f}s remaining")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Saving partial results...")

    logger.info(f"Done. Processed={len(all_records)}, Skipped={skipped}, Failed={failed}")

    # ── Write CSV summary ──
    if all_records:
        csv_fields = [
            "event_id", "ticker", "t0_utc", "direction",
            "has_news", "is_opening_bell", "is_etf",
            "final_I", "final_P", "final_confidence",
            "dominant_class", "policy_behavior",
            "disagreement_score", "round2_shift",
            "top_rationale", "true_label_proxy",
            "elapsed_s",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in all_records:
                if "error" not in r:
                    writer.writerow(r)
        logger.info(f"CSV written to {csv_path}")

    # ── Run evaluation ──
    eval_records = [r for r in all_records if "error" not in r]
    if eval_records:
        results = run_evaluation(eval_records, eval_path)
        logger.info(f"Eval written to {eval_path}")
        logger.info(f"  3-way accuracy: {results.get('accuracy_3way', '?')}")
        logger.info(f"  Clear-subset accuracy: {results.get('accuracy_clear_subset', '?')}")
        logger.info(f"  Brier score: {results.get('brier_score', '?')}")
    else:
        logger.warning("No successful records to evaluate")


if __name__ == "__main__":
    main()
