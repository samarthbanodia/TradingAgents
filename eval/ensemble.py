"""
V3 Weighted Ensemble + Evaluation.

Ensemble design (research-backed):
- 3 LLM agents vote with confidence weighting
- Direction heuristic as 4th voter (non-LLM, decorrelated)
- Asymmetric reversal prior: large-z intraday events revert ~60% (arXiv:2501.16772)
- Time-of-day conditioned direction weight (reversal strongest last 2h)
- Macro-driven flag reduces reversal prior (OFI/VAR research)
- Down-spike direction weight reduced (leverage asymmetry)
- Unclear is ensemble-only (top vote < UNCLEAR_THRESHOLD)

Evaluation:
- Compares V3 ensemble against all V2 baselines + new strategies
- Tracks agent-level accuracy for diagnostic purposes
"""

import json
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# ── Ensemble weights ──────────────────────────────────────────────────────────

BASE_WEIGHTS = {
    "micro":     0.35,
    "news":      0.20,
    "macro":     0.15,
    "direction": 0.30,   # time-of-day conditioned below
}

UNCLEAR_THRESHOLD = 0.35   # top vote must exceed this or ensemble returns "unclear"


# ── Direction heuristic parameters ───────────────────────────────────────────

def _direction_weight(t0_et_hour):
    """
    Time-of-day conditioned direction heuristic weight.
    Reversal is strongest in last 2 hours; weaker at the open.
    """
    if t0_et_hour < 10:          # first 30–60 min (price discovery, momentum)
        return 0.22
    elif t0_et_hour >= 14:       # last 2 hours (strong reversion regime)
        return 0.38
    else:
        return BASE_WEIGHTS["direction"]


def _reversal_prior(peak_z, is_macro_driven, is_up_spike):
    """
    How confident the direction heuristic is in the reversal direction.

    Base: 60% for large-z (>3.0), 55% for moderate z.
    Adjustments:
    - Macro-driven event: reversal less likely (market absorbs info)
    - Down spike: down→continuation is weaker than up→reversal (leverage asymmetry)
    """
    if abs(peak_z) > 3.0:
        prior = 0.62
    else:
        prior = 0.55

    if is_macro_driven:
        prior *= 0.85   # market-wide move less likely to revert

    if not is_up_spike:
        prior *= 0.82   # down→continuation is weaker empirically

    return round(min(0.72, max(0.42, prior)), 4)


# ── Core ensemble function ────────────────────────────────────────────────────

def run_ensemble(micro_out, news_out, macro_out, row, v3_features=None):
    """
    V3 weighted ensemble.

    Parameters
    ----------
    micro_out, news_out, macro_out : dict
        Validated agent outputs with keys: label, confidence
    row : dict
        Event row with keys: direction, peak_z_ret_3, t0_utc
    v3_features : dict, optional
        V3 computed features (for is_macro_driven, spy_zscore_t0)

    Returns
    -------
    dict with keys:
        ensemble_label   : str
        ensemble_score   : float
        votes            : dict  {continuation: float, reversal: float}
        dir_weight       : float
        dir_prior        : float
        is_macro_driven  : bool
        meta             : dict
    """
    import pytz
    from agents.briefs_v2 import _parse_t0

    if v3_features is None:
        v3_features = {}

    direction    = int(row.get("direction", 0))
    is_up        = direction == 1
    peak_z       = float(row.get("peak_z_ret_3", 2.0))
    is_macro     = bool(v3_features.get("is_macro_driven", False))

    # Time-of-day
    t0_et_hour = 12   # default mid-session
    try:
        t0 = _parse_t0(row["t0_utc"])
        if t0 is not None:
            tz = pytz.timezone("America/New_York")
            t0_et_hour = t0.tz_convert(tz).hour
    except Exception:
        pass

    # Direction heuristic parameters
    dir_w   = _direction_weight(t0_et_hour)
    prior   = _reversal_prior(peak_z, is_macro, is_up)

    # Direction label: up→reversal, down→continuation (base-rate supported)
    dir_label = "reversal" if is_up else "continuation"

    # Initialize vote accumulators
    votes = {"continuation": 0.0, "reversal": 0.0}

    # Direction heuristic vote (uses prior as its confidence)
    votes[dir_label]                              += dir_w * prior
    votes["continuation" if is_up else "reversal"] += dir_w * (1.0 - prior)

    # Agent votes (confidence-weighted)
    for agent_out, weight in [
        (micro_out, BASE_WEIGHTS["micro"]),
        (news_out,  BASE_WEIGHTS["news"]),
        (macro_out, BASE_WEIGHTS["macro"]),
    ]:
        lbl  = agent_out.get("label", "")
        conf = float(agent_out.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        if lbl in votes:
            votes[lbl] += weight * conf
        # If label is invalid/missing, skip (don't pollute votes)

    top_label = max(votes, key=votes.get)
    top_score = round(votes[top_label], 4)

    if top_score < UNCLEAR_THRESHOLD:
        final_label = "unclear"
    else:
        final_label = top_label

    return {
        "ensemble_label":  final_label,
        "ensemble_score":  top_score,
        "votes":           {k: round(v, 4) for k, v in votes.items()},
        "dir_weight":      round(dir_w, 3),
        "dir_prior":       round(prior, 3),
        "is_macro_driven": is_macro,
        "meta": {
            "t0_et_hour":   t0_et_hour,
            "is_up_spike":  is_up,
            "peak_z":       peak_z,
            "unclear_threshold": UNCLEAR_THRESHOLD,
        },
    }


# ── Validation helpers ────────────────────────────────────────────────────────

def _clamp(x, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return 0.5


def validate_agent_output_v3(raw, role):
    """
    Validate and normalize a V3 agent output dict.

    - label must be "continuation" or "reversal"
    - confidence recomputed from scores if inconsistent
    - Fields clamped/coerced to valid ranges
    """
    if not isinstance(raw, dict):
        raw = {}

    # Scores
    cont_score = int(raw.get("continuation_score", 0) or 0)
    rev_score  = int(raw.get("reversal_score",     0) or 0)
    raw["continuation_score"] = max(0, min(20, cont_score))
    raw["reversal_score"]     = max(0, min(20, rev_score))

    # Label
    lbl = raw.get("label", "")
    if lbl not in ("continuation", "reversal"):
        # Derive from scores
        if cont_score > rev_score:
            lbl = "continuation"
        else:
            lbl = "reversal"
        raw["label"] = lbl

    # Confidence — prefer score-derived if stated confidence is missing/extreme
    stated_conf = raw.get("confidence")
    total = cont_score + rev_score
    if total > 0:
        score_conf = (max(cont_score, rev_score) - min(cont_score, rev_score)) / (total + 0.01)
        score_conf = round(_clamp(score_conf, 0.50, 0.95), 4)
    else:
        score_conf = 0.55

    if stated_conf is None:
        raw["confidence"] = score_conf
    else:
        raw["confidence"] = round(_clamp(stated_conf, 0.50, 0.95), 4)

    # WHO/WHOM/WHAT (Micro only — OK if missing for News/Macro)
    for field in ("who", "whom", "what"):
        if field not in raw:
            raw[field] = ""

    # key_reason
    if not raw.get("key_reason"):
        raw["key_reason"] = ""

    # counterfactuals
    cf = raw.get("counterfactuals", [])
    raw["counterfactuals"] = (cf if isinstance(cf, list) else [str(cf)])[:2]

    return raw


# ── Strategy predictions (for evaluation) ────────────────────────────────────

def _agent_label(out):
    """Extract label from a V3 agent output dict."""
    lbl = out.get("label", "")
    return lbl if lbl in ("continuation", "reversal") else "unclear"


def _strategy_predictions_v3(records):
    """
    For each strategy, compute {event_id: predicted_label}.
    """
    preds = {
        "micro_v3":            {},
        "news_v3":             {},
        "macro_v3":            {},
        "v3_majority":         {},
        "v3_ensemble":         {},
        "direction_heuristic": {},
        "always_reversal":     {},
        "always_continuation": {},
    }

    for r in records:
        eid = r.get("event_id")
        if not eid or not r.get("true_label_proxy"):
            continue

        direction = int(r.get("direction", 0))
        preds["always_reversal"][eid]     = "reversal"
        preds["always_continuation"][eid] = "continuation"
        preds["direction_heuristic"][eid] = "reversal" if direction == 1 else "continuation"

        # Pipeline ensemble output
        el = r.get("ensemble_label")
        if el:
            preds["v3_ensemble"][eid] = el

        # Individual agent outputs
        m = r.get("micro")
        n = r.get("news")
        k = r.get("macro")

        if m:
            preds["micro_v3"][eid] = _agent_label(m)
        if n:
            preds["news_v3"][eid]  = _agent_label(n)
        if k:
            preds["macro_v3"][eid] = _agent_label(k)

        if m and n and k:
            agent_labels = [_agent_label(m), _agent_label(n), _agent_label(k)]
            cnt = Counter(agent_labels)
            top_cnt = cnt.most_common(1)[0][1]
            preds["v3_majority"][eid] = (
                cnt.most_common(1)[0][0] if top_cnt > 1 else "unclear"
            )

    return preds


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def _accuracy(eid_to_pred, eid_to_true, subset_eids=None):
    eids = set(eid_to_pred) & set(eid_to_true)
    if subset_eids is not None:
        eids &= subset_eids

    correct_all = correct_clear = n_clear = 0
    for eid in eids:
        pred, true = eid_to_pred[eid], eid_to_true[eid]
        if pred == true:
            correct_all += 1
        if true != "unclear":
            n_clear += 1
            if pred == true:
                correct_clear += 1

    n = len(eids)
    return (
        round(correct_all   / n,       4) if n       else None,
        round(correct_clear / n_clear, 4) if n_clear else None,
        n_clear, n,
    )


def _confusion_matrix(eid_to_pred, eid_to_true):
    labels = ["continuation", "reversal", "unclear"]
    cm = {t: {p: 0 for p in labels} for t in labels}
    for eid in set(eid_to_pred) & set(eid_to_true):
        t, p = eid_to_true[eid], eid_to_pred[eid]
        if t in cm and p in cm:
            cm[t][p] += 1
    return cm


# ── Main evaluation function ──────────────────────────────────────────────────

def run_evaluation_v3(records, out_path):
    """
    Evaluate V3 output records. Writes JSON to out_path, returns results dict.
    """
    valid = [r for r in records if r.get("true_label_proxy")]
    if not valid:
        results = {"error": "No valid records"}
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    eid_to_true = {r["event_id"]: r["true_label_proxy"] for r in valid}

    results = {
        "total_events":       len(valid),
        "label_distribution": dict(Counter(eid_to_true.values())),
    }

    preds = _strategy_predictions_v3(valid)

    strategy_names = [
        ("micro_v3",            "Micro V3 alone"),
        ("news_v3",             "News V3 alone"),
        ("macro_v3",            "Macro V3 alone"),
        ("v3_majority",         "V3 majority vote (3 agents)"),
        ("v3_ensemble",         "V3 ensemble (agents + direction heuristic)"),
        ("direction_heuristic", "Heuristic: up=reversal down=cont"),
        ("always_reversal",     "Heuristic: always reversal"),
        ("always_continuation", "Heuristic: always continuation"),
    ]

    strategy_results = {}
    for key, name in strategy_names:
        p = preds[key]
        acc, acc_clr, n_clr, n = _accuracy(p, eid_to_true)
        cm = _confusion_matrix(p, eid_to_true)
        strategy_results[key] = {
            "name":                    name,
            "n":                       n,
            "accuracy_3way":           acc,
            "accuracy_clear_subset":   acc_clr,
            "clear_subset_n":          n_clr,
            "confusion_matrix":        cm,
            "prediction_distribution": dict(Counter(p.values())),
        }

    results["strategy_comparison"] = strategy_results

    # ── V3 ensemble detailed breakdown ──
    ens_preds = preds["v3_ensemble"]
    ens_eids  = set(ens_preds) & set(eid_to_true)
    ens_recs  = [r for r in valid if r["event_id"] in ens_eids]

    # Ensemble score calibration
    scored_recs = [r for r in ens_recs
                   if eid_to_true.get(r["event_id"]) in ("continuation", "reversal")]
    if scored_recs:
        brier_sum = 0.0
        for r in scored_recs:
            votes  = r.get("ensemble_votes", r.get("votes", {}))
            p_rev  = _clamp(votes.get("reversal", 0.5))
            actual = 1.0 if eid_to_true[r["event_id"]] == "reversal" else 0.0
            brier_sum += (p_rev - actual) ** 2
        results["brier_score"] = round(brier_sum / len(scored_recs), 4)
        results["brier_n"]     = len(scored_recs)

    # Accuracy by is_macro_driven, is_opening_bell, direction
    for flag in ("is_macro_driven", "is_opening_bell", "has_pre_event_news"):
        groups = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in ens_recs:
            k   = str(r.get(flag, "unknown"))
            pb  = ens_preds.get(r["event_id"])
            tru = eid_to_true.get(r["event_id"])
            if pb and tru:
                groups[k]["total"] += 1
                if pb == tru:
                    groups[k]["correct"] += 1
        results[f"accuracy_by_{flag}"] = {
            k: {"accuracy": round(g["correct"] / g["total"], 4) if g["total"] else None,
                "n": g["total"]}
            for k, g in groups.items()
        }

    # Accuracy by direction
    dir_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in ens_recs:
        try:
            d = int(r.get("direction", 0))
        except (ValueError, TypeError):
            d = 0
        k   = "up" if d == 1 else "down"
        pb  = ens_preds.get(r["event_id"])
        tru = eid_to_true.get(r["event_id"])
        if pb and tru:
            dir_groups[k]["total"] += 1
            if pb == tru:
                dir_groups[k]["correct"] += 1
    results["accuracy_by_direction"] = {
        k: {"accuracy": round(g["correct"] / g["total"], 4) if g["total"] else None, "n": g["total"]}
        for k, g in dir_groups.items()
    }

    # Agent agreement vs ensemble accuracy
    for r in ens_recs:
        m = r.get("micro")
        n_ = r.get("news")
        k_ = r.get("macro")
        if m and n_ and k_:
            labels = [_agent_label(m), _agent_label(n_), _agent_label(k_)]
            unique = len(set(labels))
            r["_agent_agreement"] = "full" if unique == 1 else ("partial" if unique == 2 else "disagree")

    agree_groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in ens_recs:
        lvl = r.get("_agent_agreement", "unknown")
        pb  = ens_preds.get(r["event_id"])
        tru = eid_to_true.get(r["event_id"])
        if pb and tru:
            agree_groups[lvl]["total"] += 1
            if pb == tru:
                agree_groups[lvl]["correct"] += 1
    results["accuracy_by_agent_agreement"] = {
        k: {"accuracy": round(g["correct"] / g["total"], 4) if g["total"] else None, "n": g["total"]}
        for k, g in agree_groups.items()
    }

    # Summary table
    results["summary_table"] = _build_summary_table(strategy_results)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"V3 evaluation written to {out_path}")
    return results


def _build_summary_table(strategy_results):
    header = f"{'Strategy':<40} {'N':>5} {'3-way':>7} {'Clear':>7}"
    rows   = [header, "-" * len(header)]
    order  = [
        "micro_v3", "news_v3", "macro_v3",
        "v3_majority", "v3_ensemble",
        "direction_heuristic", "always_reversal", "always_continuation",
    ]
    for key in order:
        if key not in strategy_results:
            continue
        sr    = strategy_results[key]
        name  = sr["name"][:39]
        n     = sr.get("n") or 0
        acc   = sr.get("accuracy_3way")
        acc_c = sr.get("accuracy_clear_subset")
        rows.append(f"{name:<40} {n:>5} "
                    f"{f'{acc:.3f}' if acc is not None else '  n/a':>7} "
                    f"{f'{acc_c:.3f}' if acc_c is not None else '  n/a':>7}")
    return rows


def print_summary_v3(results):
    """Pretty-print V3 eval summary."""
    for line in results.get("summary_table", []):
        logger.info(line)

    brier = results.get("brier_score")
    if brier is not None:
        logger.info(f"\nBrier score (V3 ensemble): {brier}")

    for flag in ("is_macro_driven", "is_opening_bell", "has_pre_event_news"):
        grp = results.get(f"accuracy_by_{flag}", {})
        if grp:
            logger.info(f"\nAccuracy by {flag}:")
            for k, v in grp.items():
                logger.info(f"  {flag}={k}: {v.get('accuracy')} (n={v.get('n')})")

    by_dir = results.get("accuracy_by_direction", {})
    if by_dir:
        logger.info("\nAccuracy by direction:")
        for k, v in by_dir.items():
            logger.info(f"  direction={k}: {v.get('accuracy')} (n={v.get('n')})")

    by_agr = results.get("accuracy_by_agent_agreement", {})
    if by_agr:
        logger.info("\nAccuracy by agent agreement:")
        for k, v in by_agr.items():
            logger.info(f"  {k}: {v.get('accuracy')} (n={v.get('n')})")
