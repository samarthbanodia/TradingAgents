"""
Rigorous selective prediction (supervisor suggestion #1). FREE — reuses v4 outputs.

Protocol:
  - Evidence score per event = ensemble vote margin (|cont - rev|), i.e. agent conviction.
  - Coverage-risk curve on the TEST (holdout) period at 10/20/30/50/100% coverage:
        agent accuracy (ensemble_label) on the covered slice, vs
        (a) always-continuation on the same slice,
        (b) microstructure-only selector (pick same coverage by spike magnitude, predict
            the direction heuristic) — the "did the LLM help us SELECT?" baseline.
  - Honesty guard: pick the evidence threshold for ~20% coverage on the VALIDATION
    (train) period, apply ONCE to the TEST period, report the single number.
  - Bootstrap 95% CIs on the agent accuracy (small n).
"""
import warnings
import numpy as np
import pandas as pd
from harness import load_events, HOLDOUT_START

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def dirheur(d):
    return "reversal" if d == 1 else "continuation"


def boot(mask_correct, n=2000, seed=0):
    x = np.asarray(mask_correct, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return (np.percentile(s, 2.5), np.percentile(s, 97.5))


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    ev = ev[ev["true_label_proxy"].isin(["continuation", "reversal"])].copy()
    ev["y"] = ev["true_label_proxy"]
    ev["mag"] = ev["open_adj_range_mult"].fillna(0)  # microstructure spike-size signal
    val = ev[ev["t0_utc"] < HOLDOUT_START].copy()    # validation period
    test = ev[ev["t0_utc"] >= HOLDOUT_START].copy()  # test period
    print(f"validation n={len(val)} | test n={len(test)} | "
          f"test always-continue acc = {(test.y=='continuation').mean():.3f}\n")

    print("=== COVERAGE–RISK CURVE on TEST (rank by agent evidence = ensemble margin) ===")
    print(f"{'cov':>5}{'k':>5}{'agent_acc':>11}{'95% CI':>16}{'always_cont':>12}{'micro_sel':>10}{'agent-best_base':>16}")
    d = test.sort_values("mag", ascending=False).reset_index(drop=True)  # for micro selector
    t = test.sort_values("ens_margin", ascending=False).reset_index(drop=True)
    n = len(test)
    for c in [0.1, 0.2, 0.3, 0.5, 1.0]:
        k = max(1, int(round(c * n)))
        sub = t.iloc[:k]
        agent = (sub.ensemble_label == sub.y).mean()
        lo, hi = boot((sub.ensemble_label == sub.y).values)
        always = (sub.y == "continuation").mean()
        # microstructure-only selector: top-k by spike magnitude, predict direction heuristic
        msub = d.iloc[:k]
        micro = (pd.Series([dirheur(x) for x in msub.direction], index=msub.index) == msub.y).mean()
        best_base = max(always, micro)
        print(f"{c:>5.1f}{k:>5}{agent:>11.3f}  [{lo:.3f},{hi:.3f}]{always:>12.3f}{micro:>10.3f}{agent-best_base:>16.3f}")

    print("\n=== HONEST validation->test single-threshold (pick ~20% coverage on validation) ===")
    thr = val["ens_margin"].quantile(0.80)  # threshold giving ~top-20% conviction on validation
    acted = test[test["ens_margin"] >= thr]
    if len(acted):
        agent = (acted.ensemble_label == acted.y).mean()
        always = (acted.y == "continuation").mean()
        lo, hi = boot((acted.ensemble_label == acted.y).values)
        print(f"threshold(margin)={thr:.3f} chosen on validation")
        print(f"test events acted on: {len(acted)}/{len(test)} ({len(acted)/len(test):.0%} coverage)")
        print(f"agent acc on acted = {agent:.3f}  CI=[{lo:.3f},{hi:.3f}]  |  always-continue on same = {always:.3f}")
        print(f"=> agent beats always-continue by {agent-always:+.3f}")
    else:
        print("no test events cleared the validation threshold")

    print("\nINTERPRETATION: a positive result needs agent_acc to clearly beat BOTH always_cont")
    print("and micro_sel at some coverage, with the CI not overlapping the baseline.")


if __name__ == "__main__":
    main()
