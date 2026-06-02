"""
Constants stability sweep for V3 ensemble.

Varies each of the 14 hardcoded ensemble constants by ±10% and ±20%,
re-runs ensemble predictions on stored JSONL agent outputs (no LLM calls),
and reports how much accuracy changes.

A reviewer concern is "manual overfitting to 85 events." This sweep shows
whether the constants are robust or fragile.

Usage:
    python -m eval.constants_sweep
    python -m eval.constants_sweep --jsonl out_agents_v3b/ip_outputs_v3.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Baseline constants (must mirror eval/ensemble.py exactly) ─────────────────

BASELINE = {
    # Agent vote weights
    "w_micro":          0.35,
    "w_news":           0.20,
    "w_macro":          0.15,
    "w_direction":      0.30,
    # Unclear threshold
    "unclear_thresh":   0.35,
    # Direction weight — time-of-day
    "dir_w_early":      0.22,   # first 30-60 min (< 10h ET)
    "dir_w_late":       0.38,   # last 2 hours   (>= 14h ET)
    # Reversal prior
    "prior_high_z":     0.62,   # |z| > 3.0
    "prior_low_z":      0.55,   # |z| <= 3.0
    "z_threshold":      3.00,   # z boundary
    "macro_discount":   0.85,   # prior *= this if macro-driven
    "down_discount":    0.82,   # prior *= this for DOWN spikes
    # Prior clamps
    "prior_max":        0.72,
    "prior_min":        0.42,
}

# Hour thresholds for time-of-day (integer, ±10% rounded)
HOUR_EARLY = 10
HOUR_LATE  = 14


# ── Ensemble re-implementation (parametric) ───────────────────────────────────

def _parse_t0(t0_str):
    import pandas as pd
    try:
        ts = pd.Timestamp(t0_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def _run_ensemble_parametric(record, c):
    """Re-run ensemble for one record using constant dict c."""
    import pytz

    direction  = int(record.get("direction", 0))
    is_up      = direction == 1
    peak_z     = float(record.get("peak_z_ret_3") or 2.0)
    is_macro   = bool(record.get("is_macro_driven", False))

    # Time-of-day weight
    t0_et_hour = 12
    try:
        t0 = _parse_t0(record["t0_utc"])
        if t0 is not None:
            t0_et_hour = t0.tz_convert(pytz.timezone("America/New_York")).hour
    except Exception:
        pass

    if t0_et_hour < HOUR_EARLY:
        dir_w = c["dir_w_early"]
    elif t0_et_hour >= HOUR_LATE:
        dir_w = c["dir_w_late"]
    else:
        dir_w = c["w_direction"]

    # Reversal prior
    prior = c["prior_high_z"] if abs(peak_z) > c["z_threshold"] else c["prior_low_z"]
    if is_macro:
        prior *= c["macro_discount"]
    if not is_up:
        prior *= c["down_discount"]
    prior = min(c["prior_max"], max(c["prior_min"], prior))

    dir_label = "reversal" if is_up else "continuation"
    votes = {"continuation": 0.0, "reversal": 0.0}
    votes[dir_label]                               += dir_w * prior
    votes["continuation" if is_up else "reversal"] += dir_w * (1.0 - prior)

    for key, w in [("micro", c["w_micro"]), ("news", c["w_news"]), ("macro", c["w_macro"])]:
        lbl  = record.get(f"{key}_label", "")
        conf = float(record.get(f"{key}_conf") or 0.5)
        conf = max(0.0, min(1.0, conf))
        if lbl in votes:
            votes[lbl] += w * conf

    top_label = max(votes, key=votes.get)
    top_score = votes[top_label]
    return "unclear" if top_score < c["unclear_thresh"] else top_label


def accuracy_for_constants(records, c):
    """Compute 3-way accuracy and clear-only accuracy for constant dict c."""
    correct_all = correct_clear = n_clear = n = 0
    for r in records:
        true = r.get("true_label_proxy", "")
        if not true:
            continue
        pred = _run_ensemble_parametric(r, c)
        n += 1
        if pred == true:
            correct_all += 1
        if true != "unclear":
            n_clear += 1
            if pred == true:
                correct_clear += 1
    acc     = correct_all   / n       if n       else 0.0
    acc_clr = correct_clear / n_clear if n_clear else 0.0
    return round(acc, 4), round(acc_clr, 4)


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(records, deltas=(0.90, 0.95, 1.05, 1.10)):
    base_acc, base_clr = accuracy_for_constants(records, BASELINE)
    logger.info(f"Baseline — 3-way: {base_acc:.4f} ({base_acc*100:.1f}%)  "
                f"clear: {base_clr:.4f} ({base_clr*100:.1f}%)")

    results = {}
    for const_name, base_val in BASELINE.items():
        row = {"baseline": base_val, "variants": {}}
        for mult in deltas:
            variant_val = round(base_val * mult, 6)
            c = {**BASELINE, const_name: variant_val}
            acc, clr = accuracy_for_constants(records, c)
            delta_pp = round((acc - base_acc) * 100, 2)
            row["variants"][f"×{mult:.2f}"] = {
                "value": variant_val,
                "acc":   acc,
                "clr":   clr,
                "delta_pp": delta_pp,
            }
        results[const_name] = row

    return base_acc, base_clr, results


def print_sweep_table(base_acc, results):
    header = (f"{'Constant':<18} {'Baseline':>9} "
              f"{'×0.90 Δpp':>10} {'×0.95 Δpp':>10} "
              f"{'×1.05 Δpp':>10} {'×1.10 Δpp':>10}  "
              f"{'max|Δ|':>8}")
    logger.info(f"\n{'='*85}")
    logger.info("CONSTANTS STABILITY SWEEP (V3 ensemble, 3-way accuracy)")
    logger.info(f"Baseline accuracy: {base_acc*100:.1f}%")
    logger.info(f"{'='*85}")
    logger.info(header)
    logger.info("─" * 85)

    for name, row in results.items():
        bv = row["baseline"]
        deltas = [row["variants"][k]["delta_pp"] for k in ["×0.90", "×0.95", "×1.05", "×1.10"]]
        max_abs = max(abs(d) for d in deltas)
        flag = "  ← SENSITIVE" if max_abs >= 2.0 else ""
        logger.info(
            f"  {name:<16} {bv:>9.4f} "
            f"{deltas[0]:>+9.2f}pp {deltas[1]:>+9.2f}pp "
            f"{deltas[2]:>+9.2f}pp {deltas[3]:>+9.2f}pp  "
            f"{max_abs:>7.2f}pp{flag}"
        )

    logger.info("─" * 85)
    max_impact = max(
        max(abs(v["delta_pp"]) for v in row["variants"].values())
        for row in results.values()
    )
    logger.info(f"  Largest single-constant effect: {max_impact:.2f}pp")
    if max_impact < 2.0:
        logger.info("  → All constants stable within ±10% perturbation ✓")
    elif max_impact < 5.0:
        logger.info("  → Mild sensitivity on flagged constants — review individually")
    else:
        logger.info("  → High sensitivity detected — constants may be over-tuned ✗")
    logger.info(f"{'='*85}")


def main():
    parser = argparse.ArgumentParser(description="V3 ensemble constants stability sweep")
    parser.add_argument("--jsonl",   default="out_agents_v3b/ip_outputs_v3.jsonl")
    parser.add_argument("--out_dir", default="out_agents_v3b")
    args = parser.parse_args()

    records = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "error" not in r and r.get("true_label_proxy"):
                    records.append(r)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(records)} valid records from {args.jsonl}")

    base_acc, base_clr, results = run_sweep(records)
    print_sweep_table(base_acc, results)

    # Save JSON
    out = {
        "baseline_acc":      base_acc,
        "baseline_acc_clr":  base_clr,
        "constants":         BASELINE,
        "sweep":             results,
    }
    out_path = Path(args.out_dir) / "constants_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Sweep results saved → {out_path}")


if __name__ == "__main__":
    main()
