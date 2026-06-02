"""
WS2 - Selective prediction: is there a high-precision subset?

The unconditional ensemble is reversal-biased and loses to the majority baseline
(WS1). But selective prediction asks a different question: when the agents are
*confident / unanimous*, are they right often enough to trade, and does selecting
on agent signal beat selecting the same fraction of events by a trivial rule?

On the holdout (clear labels), at chosen horizons:
  1. Per-class precision/recall for ensemble + each agent (does 'reversal' fire precisely?).
  2. Unanimous-agreement subset accuracy vs full-set accuracy.
  3. Accuracy-coverage curves: rank by a confidence signal, sweep coverage, and compare
     agent-selection accuracy to EQUAL-COVERAGE direction-heuristic selection.
     (The credibility test: agent selection must beat trivial selection at equal coverage.)
"""
import warnings
import numpy as np
import pandas as pd
from harness import load_events, load_ohlcv, build_ohlcv_index, relabel, HOLDOUT_START

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 220)


def dir_heur(direction):
    return "reversal" if direction == 1 else "continuation"


def build(horizon):
    ev = load_events()
    idx = build_ohlcv_index(load_ohlcv())
    labs = []
    for _, r in ev.iterrows():
        lab, _ = relabel(idx, r["ticker"], r["t0_utc"], r["direction"], horizon)
        labs.append(lab)
    ev["y"] = labs
    ho = ev[(ev["t0_utc"] >= HOLDOUT_START) & ev["y"].isin(["continuation", "reversal"])].copy()
    ho["dir_pred"] = ho["direction"].map(dir_heur)
    return ho


def per_class(ho, col, name):
    rows = []
    for cls in ["continuation", "reversal"]:
        fired = ho[ho[col] == cls]
        prec = (fired["y"] == cls).mean() if len(fired) else np.nan
        support_pred = len(fired)
        actual = ho[ho["y"] == cls]
        rec = (actual[col] == cls).mean() if len(actual) else np.nan
        rows.append({"model": name, "class": cls, "precision": round(prec, 3),
                     "recall": round(rec, 3), "n_predicted": support_pred,
                     "n_actual": len(actual)})
    return rows


def coverage_curve(ho, score_col, pred_col, ascending=False, steps=(0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)):
    """Rank events by score_col; at each coverage take top-k and report accuracy of
    pred_col on that subset, vs equal-coverage direction-heuristic accuracy."""
    d = ho.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    n = len(d)
    rows = []
    for c in steps:
        k = max(1, int(round(c * n)))
        sub = d.iloc[:k]
        agent_acc = (sub[pred_col] == sub["y"]).mean()
        dir_acc = (sub["dir_pred"] == sub["y"]).mean()
        cont_base = (sub["y"] == "continuation").mean()
        rows.append({"coverage": c, "k": k,
                     "agent_acc": round(agent_acc, 3),
                     "dirheur_acc_sameK": round(dir_acc, 3),
                     "majority_sameK": round(max(cont_base, 1 - cont_base), 3),
                     "agent_minus_dir": round(agent_acc - dir_acc, 3)})
    return pd.DataFrame(rows)


def main():
    for horizon in [60, "T+2D"]:
        ho = build(horizon)
        print("\n" + "=" * 78)
        print(f"HORIZON = {horizon}  | holdout clear n={len(ho)}  "
              f"| cont base={round((ho['y']=='continuation').mean(),3)}")
        print("=" * 78)

        print("\n-- Per-class precision/recall (does a class fire precisely?) --")
        rows = []
        for col, nm in [("ensemble_label", "ensemble"), ("micro_label", "micro"),
                        ("news_label", "news"), ("macro_label", "macro"),
                        ("dir_pred", "dir_heur")]:
            rows += per_class(ho, col, nm)
        print(pd.DataFrame(rows).to_string(index=False))

        print("\n-- Unanimous-agreement subset vs full --")
        uni = ho[ho["unanimous"]]
        print(f"   unanimous n={len(uni)} ({len(uni)/len(ho):.0%} coverage) | "
              f"ensemble acc on unanimous={ (uni['ensemble_label']==uni['y']).mean():.3f} | "
              f"dir-heur on same={ (uni['dir_pred']==uni['y']).mean():.3f} | "
              f"full ensemble acc={ (ho['ensemble_label']==ho['y']).mean():.3f}")
        # also: agreement among micro+macro only (news is weakest)
        mm = ho[ho["micro_label"] == ho["macro_label"]]
        print(f"   micro==macro n={len(mm)} ({len(mm)/len(ho):.0%}) | "
              f"that-label acc={ (mm['micro_label']==mm['y']).mean():.3f} | "
              f"dir-heur same={ (mm['dir_pred']==mm['y']).mean():.3f}")

        print("\n-- Accuracy-coverage: rank by ensemble margin (|cont-rev votes|), predict ensemble_label --")
        print(coverage_curve(ho, "ens_margin", "ensemble_label", ascending=False).to_string(index=False))

        print("\n-- Accuracy-coverage: rank by agreement count then margin, predict majority-vote label --")
        ho["maj_label"] = np.where(ho["n_rev_votes"] >= 2, "reversal", "continuation")
        ho["agree_score"] = ho["agree_count"] + ho["ens_margin"]  # 3-agree ranked above 2-agree
        print(coverage_curve(ho, "agree_score", "maj_label", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
