"""
Does NEWS condition the overreaction signal? + surface real events to fact-check our news.

1) Interaction: among HIGH-overreaction spikes (top idio_resid tercile), do catalyst-backed
   ones CONTINUE while catalyst-less ones REVERT? (the LLM-regime-filter hypothesis)
2) Print specific high-overreaction events with the news WE gave the agent, so we can look
   up what actually happened that day and judge if our news feed is adequate.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from harness import load_events, HOLDOUT_START

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKT = os.path.join(HERE, "out_extended_news", "packets")
CAT = ["earnings","guidance","upgrade","downgrade","fda","approval","lawsuit","acquisi","merger",
       "ceo","resign","buyback","dividend","recall","investigat","beats","misses","raises","cuts",
       "partnership","contract","price target","analyst"]


def pkt(ticker, eid):
    p = os.path.join(PKT, ticker, f"{eid}.json")
    if not os.path.exists(p):
        return 0, []
    d = json.load(open(p)).get("news", {})
    arts = d.get("ticker_news", []) or []
    titles = [(a.get("published_utc","")[:16], a.get("title","")) for a in arts]
    return d.get("ticker_news_count", len(arts)), titles


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    info = [pkt(r.ticker, r.event_id) for r in ev.itertuples()]
    ev["art_cnt"] = [x[0] for x in info]
    ev["titles"] = [x[1] for x in info]
    ev["cat_hits"] = [sum(k in " ".join(t for _, t in ti).lower() for k in CAT) for ti in ev["titles"]]
    cl = ev[ev.true_label_proxy.isin(["continuation","reversal"])].copy()

    print("=== 1) NEWS x OVERREACTION INTERACTION (all clear events) ===")
    hi = cl[cl.idio_resid_bp.astype(float) >= cl.idio_resid_bp.astype(float).quantile(2/3)]
    print(f"high-overreaction subset (top idio_resid tercile) n={len(hi)} | "
          f"reversal rate overall = {(hi.true_label_proxy=='reversal').mean():.3f}")
    for has, name in [(hi.cat_hits >= 1, "HAS catalyst news"), (hi.cat_hits == 0, "NO catalyst news")]:
        s = hi[has]
        print(f"   high-overreaction & {name:18} n={len(s):3}  reversal={ (s.true_label_proxy=='reversal').mean():.3f}  "
              f"continuation={ (s.true_label_proxy=='continuation').mean():.3f}")
    # 2x2 for completeness
    print("   (hypothesis: catalyst-backed overreactions CONTINUE; catalyst-less ones REVERT)")

    print("\n=== 2) SPECIFIC HIGH-OVERREACTION EVENTS — what news did WE give the agent? ===")
    top = cl.assign(absidio=cl.idio_resid_bp.abs()).sort_values("absidio", ascending=False).head(8)
    for r in top.itertuples():
        print(f"\n  {r.event_id}  ({r.ticker}, {r.t0_utc.date()})  idio_resid={r.idio_resid_bp:+.0f}bp  "
              f"dir={'UP' if r.direction==1 else 'DOWN'}  OUTCOME={r.true_label_proxy}")
        print(f"    our news packet: {r.art_cnt} articles, catalyst-words={r.cat_hits}")
        if r.titles:
            for ts, ti in r.titles[:4]:
                print(f"      [{ts}] {ti[:95]}")
        else:
            print("      (no ticker news in packet)")


if __name__ == "__main__":
    main()
