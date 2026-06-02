"""
WS1 - Horizon sweep.  Where does the agent signal live?

For each horizon, relabel all 800 events (leak-free) and, on the holdout
(>=2026-01-06, clear labels only), compare:
  - continuation base rate / majority-class accuracy   (trivial)
  - direction heuristic (up->reversal, down->continuation)  (their hardest baseline)
  - V3 ensemble label accuracy
  - each agent (micro / news / macro) alone
  - ensemble LIFT over the best trivial baseline

No LLM calls; agent predictions are reused as-is (they targeted 60min, so longer
horizons test whether that reasoning transfers — interpret as exploratory).
"""
import warnings
import numpy as np
import pandas as pd
from harness import load_events, load_ohlcv, build_ohlcv_index, relabel, HOLDOUT_START

warnings.filterwarnings("ignore", category=UserWarning)

HORIZONS = [30, 60, 120, "EOD", "NEXT_OPEN", "T+1D", "T+2D"]


def dir_heuristic(direction):
    # up spike (1) -> reversal ; down spike (0) -> continuation
    return "reversal" if direction == 1 else "continuation"


def acc(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    return (pred == truth).mean() if len(truth) else np.nan


def main():
    ev = load_events()
    idx = build_ohlcv_index(load_ohlcv())

    rows = []
    for h in HORIZONS:
        labs, frs = [], []
        for _, r in ev.iterrows():
            lab, fr = relabel(idx, r["ticker"], r["t0_utc"], r["direction"], h)
            labs.append(lab); frs.append(fr)
        d = ev.copy()
        d["y"] = labs
        # holdout, clear labels only
        ho = d[(d["t0_utc"] >= HOLDOUT_START) & d["y"].isin(["continuation", "reversal"])]
        n = len(ho)
        if n == 0:
            continue
        base_cont = (ho["y"] == "continuation").mean()
        majority = max(base_cont, 1 - base_cont)
        dh = acc([dir_heuristic(x) for x in ho["direction"]], ho["y"])
        ens = acc(ho["ensemble_label"], ho["y"])
        mic = acc(ho["micro_label"], ho["y"])
        nws = acc(ho["news_label"], ho["y"])
        mac = acc(ho["macro_label"], ho["y"])
        unclear_frac = (pd.Series(labs)[d["t0_utc"] >= HOLDOUT_START] == "unclear").mean()
        rows.append({
            "horizon": str(h), "n_ho": n, "cont_base": round(base_cont, 3),
            "majority": round(majority, 3), "dir_heur": round(dh, 3),
            "ensemble": round(ens, 3), "micro": round(mic, 3),
            "news": round(nws, 3), "macro": round(mac, 3),
            "ens_lift_vs_trivial": round(ens - max(majority, dh), 3),
            "unclear_frac": round(unclear_frac, 2),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== WS1 HORIZON SWEEP (holdout, clear labels) ===")
    print(out.to_string(index=False))
    print("\nReading guide: 'ens_lift_vs_trivial' > 0 means the agent ensemble beats the")
    print("best of {majority-class, direction-heuristic} at that horizon. Positive and")
    print("growing with horizon = the signal lives away from the 60-min HFT regime.")
    import os
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ws1_horizon_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
