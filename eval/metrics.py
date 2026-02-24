"""Evaluation metrics: accuracy, Brier score, confusion matrix, calibration."""

import json
import numpy as np
from collections import defaultdict


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def run_evaluation(records: list, out_path: str):
    """Compute all evaluation metrics and write to JSON.

    Each record should have: label_proxy, policy_behavior, final_I, final_P,
    final_confidence, has_news, is_opening_bell, disagreement_score
    """
    results = {}

    # Filter valid records
    valid = [r for r in records if r.get("label_proxy") and r.get("policy_behavior")]
    if not valid:
        results["error"] = "No valid records for evaluation"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    results["total_events"] = len(valid)

    # --- 3-way accuracy ---
    correct_3way = sum(1 for r in valid if r["policy_behavior"] == r["label_proxy"])
    results["accuracy_3way"] = round(correct_3way / len(valid), 4)

    # --- Clear-subset accuracy (exclude true unclear) ---
    clear = [r for r in valid if r["label_proxy"] != "unclear"]
    if clear:
        correct_clear = sum(1 for r in clear if r["policy_behavior"] == r["label_proxy"])
        results["accuracy_clear_subset"] = round(correct_clear / len(clear), 4)
        results["clear_subset_n"] = len(clear)
    else:
        results["accuracy_clear_subset"] = None

    # --- Confusion matrix ---
    labels = ["continuation", "reversal", "unclear", "noise"]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    for r in valid:
        true = r["label_proxy"]
        pred = r["policy_behavior"]
        if true in labels and pred in labels:
            confusion[true][pred] += 1
    results["confusion_matrix"] = confusion

    # --- Brier score ---
    # p_reversal = clamp(final_P - final_I + 0.5, 0, 1)
    brier_records = [r for r in valid if r["label_proxy"] in ("continuation", "reversal")]
    if brier_records:
        brier_sum = 0.0
        for r in brier_records:
            p_rev = _clamp(r.get("final_P", 0.5) - r.get("final_I", 0.5) + 0.5)
            actual = 1.0 if r["label_proxy"] == "reversal" else 0.0
            brier_sum += (p_rev - actual) ** 2
        results["brier_score"] = round(brier_sum / len(brier_records), 4)
        results["brier_n"] = len(brier_records)
    else:
        results["brier_score"] = None

    # --- Calibration bins ---
    if brier_records:
        bins = defaultdict(lambda: {"count": 0, "reversals": 0, "sum_p": 0.0})
        bin_edges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        for r in brier_records:
            p_rev = _clamp(r.get("final_P", 0.5) - r.get("final_I", 0.5) + 0.5)
            actual = 1 if r["label_proxy"] == "reversal" else 0
            for lo, hi in bin_edges:
                if lo <= p_rev < hi:
                    key = f"{lo:.1f}-{hi:.1f}"
                    bins[key]["count"] += 1
                    bins[key]["reversals"] += actual
                    bins[key]["sum_p"] += p_rev
                    break
        calibration = {}
        for key, b in sorted(bins.items()):
            calibration[key] = {
                "n": b["count"],
                "observed_reversal_rate": round(b["reversals"] / b["count"], 4) if b["count"] else None,
                "mean_predicted_p": round(b["sum_p"] / b["count"], 4) if b["count"] else None,
            }
        results["calibration_bins"] = calibration

    # --- Accuracy by has_news ---
    for flag_field in ("has_news", "is_opening_bell"):
        groups = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in valid:
            key = str(r.get(flag_field, "unknown"))
            groups[key]["total"] += 1
            if r["policy_behavior"] == r["label_proxy"]:
                groups[key]["correct"] += 1
        acc_by = {}
        for k, g in groups.items():
            acc_by[k] = {"accuracy": round(g["correct"] / g["total"], 4), "n": g["total"]}
        results[f"accuracy_by_{flag_field}"] = acc_by

    # --- Disagreement vs accuracy (quintiles) ---
    scored = [r for r in valid if r.get("disagreement_score") is not None]
    if len(scored) >= 5:
        scored_sorted = sorted(scored, key=lambda r: r["disagreement_score"])
        q_size = len(scored_sorted) // 5
        quintiles = {}
        for qi in range(5):
            start = qi * q_size
            end = start + q_size if qi < 4 else len(scored_sorted)
            bucket = scored_sorted[start:end]
            correct = sum(1 for r in bucket if r["policy_behavior"] == r["label_proxy"])
            quintiles[f"Q{qi+1}"] = {
                "accuracy": round(correct / len(bucket), 4),
                "n": len(bucket),
                "disagreement_range": [
                    round(bucket[0]["disagreement_score"], 4),
                    round(bucket[-1]["disagreement_score"], 4),
                ],
            }
        results["disagreement_vs_accuracy"] = quintiles

    # Write
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
