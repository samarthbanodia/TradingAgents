"""
Full-pipeline audit — hunt for things silently breaking the result. FREE.

Checks:
  1. LABEL HARSHNESS: distribution of the 60-min move size. The label uses a 0.3%
     threshold. If most 'continuation'/'reversal' events are tiny near-zero net moves,
     we're forcing labels onto NOISE — no method can beat that.
  2. DECISIVE vs MARGINAL: does agent accuracy improve on events whose outcome is a
     big decisive move (real signal) vs marginal (<0.5%) moves (noise)?
  3. EVENT INDEPENDENCE: are the 800 events clustered on a few market-wide days
     (correlated => effective N << 800)?
  4. TICKER MIX: how concentrated / homogeneous is the universe?
  5. SPIKE GENUINENESS: are the mined spikes actually large, or marginal/artefacts?
  6. TIME-OF-DAY: opening-bell concentration (different dynamics).
  7. UNCLEAR: how many, and how sensitive the split is to the threshold.
"""
import warnings
import numpy as np
import pandas as pd
from harness import load_events, load_ohlcv, build_ohlcv_index, forward_return, HOLDOUT_START

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    idx = build_ohlcv_index(load_ohlcv())
    # 60-min signed forward return + magnitude
    fr = [forward_return(idx, r.ticker, r.t0_utc, 60)[0] for r in ev.itertuples()]
    ev["fwd60"] = fr
    ev["absfwd"] = np.abs(ev["fwd60"])
    cl = ev[ev.true_label_proxy.isin(["continuation", "reversal"])].copy()

    print("=== 1) LABEL HARSHNESS: 60-min move-size distribution (clear events) ===")
    for lo, hi, name in [(0, .003, "<0.3% (BELOW threshold?!)"), (.003, .005, "0.3-0.5% (marginal)"),
                         (.005, .01, "0.5-1%"), (.01, .02, "1-2%"), (.02, 1, ">2% (decisive)")]:
        m = ((cl.absfwd >= lo) & (cl.absfwd < hi))
        print(f"  {name:26} {m.sum():4d}  ({m.mean():5.1%})")
    print(f"  median |60-min move| = {cl.absfwd.median():.3%}")
    print("  -> if a big chunk sits in 0.3-0.5%, those labels are basically coin-flips on noise.")

    print("\n=== 2) DECISIVE vs MARGINAL: agent accuracy by outcome magnitude (holdout) ===")
    ho = cl[cl.t0_utc >= HOLDOUT_START]
    for lo, hi, name in [(0, .005, "marginal (<0.5%)"), (.005, .01, "0.5-1%"), (.01, 1, "decisive (>1%)")]:
        s = ho[(ho.absfwd >= lo) & (ho.absfwd < hi)]
        if len(s) == 0:
            continue
        acc = (s.ensemble_label == s.true_label_proxy).mean()
        base = max((s.true_label_proxy == "continuation").mean(), (s.true_label_proxy == "reversal").mean())
        print(f"  {name:18} n={len(s):3}  agent_acc={acc:.3f}  base={base:.3f}  lift={acc-base:+.3f}  "
              f"cont_rate={ (s.true_label_proxy=='continuation').mean():.2f}")
    print("  -> if agents do better on decisive moves, the noise labels were hiding real skill.")

    print("\n=== 3) EVENT INDEPENDENCE: clustering on market-wide days ===")
    ev["day"] = ev.t0_utc.dt.date
    perday = ev.groupby("day").size().sort_values(ascending=False)
    print(f"  {len(ev)} events on {ev['day'].nunique()} distinct days | median {perday.median():.0f}/day, max {perday.max()}/day")
    print(f"  top-10 days hold {perday.head(10).sum()} events ({perday.head(10).sum()/len(ev):.0%}) "
          f"-> these co-move, so effective independent N is much smaller than {len(ev)}")

    print("\n=== 4) TICKER MIX ===")
    print("  " + ev.ticker.value_counts().to_dict().__str__())

    print("\n=== 5) SPIKE GENUINENESS (mined spike size) ===")
    mag = ev["open_adj_range_mult"].astype(float)
    print(f"  open_adj_range_mult: median={mag.median():.2f}x  p10={mag.quantile(.1):.2f}x  p90={mag.quantile(.9):.2f}x  min={mag.min():.2f}x")
    print(f"  events with range_mult < 2x (weak 'spikes'): {(mag<2).sum()} ({(mag<2).mean():.0%})")

    print("\n=== 6) TIME-OF-DAY (UTC hour of t0) ===")
    ev["hr"] = ev.t0_utc.dt.hour
    print("  " + ev.hr.value_counts().sort_index().to_dict().__str__())
    open_evt = ev.is_opening_bell.fillna(False)
    print(f"  opening-bell events: {open_evt.sum()} ({open_evt.mean():.0%})")

    print("\n=== 7) UNCLEAR sensitivity ===")
    n_unclear = (ev.true_label_proxy == "unclear").sum()
    print(f"  unclear at 0.3% threshold: {n_unclear} ({n_unclear/len(ev):.0%})")
    for thr in [0.003, 0.005, 0.01]:
        unclear = (ev.absfwd < thr).sum()
        print(f"  if threshold were {thr:.1%}: ~{unclear} events ({unclear/len(ev):.0%}) would be 'unclear/noise'")


if __name__ == "__main__":
    main()
