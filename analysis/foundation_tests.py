"""
Foundation tests before any rebuild. FREE — existing data only.

A) NEWS MATERIALITY: does the AMOUNT/strength of news (not the binary flag) separate
   continuation vs reversal? (tests the CEO-death intuition properly)
B) MICROSTRUCTURE / OVERREACTION signals -> reversal: do spike magnitude, order-flow
   proxies, idiosyncratic move, etc. predict reversal on the holdout better than base rate?
   (the 'is the quant signal even there' gate)
C) NON-OPENING-BELL subset: does anything behave differently off the noisy open?
"""
import json, os, warnings
import numpy as np
import pandas as pd
from harness import load_events, load_ohlcv, build_ohlcv_index, forward_return, HOLDOUT_START

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKT = os.path.join(HERE, "out_extended_news", "packets")
CATALYST = ["earnings", "guidance", "upgrade", "downgrade", "fda", "approval", "lawsuit",
            "acquisi", "merger", "ceo", "resign", "buyback", "dividend", "recall", "investigat",
            "beats", "misses", "raises", "cuts", "partnership", "contract"]


def article_info(ticker, event_id):
    p = os.path.join(PKT, ticker, f"{event_id}.json")
    if not os.path.exists(p):
        return 0, 0
    d = json.load(open(p)).get("news", {})
    arts = d.get("ticker_news", []) or []
    cnt = d.get("ticker_news_count", len(arts))
    text = " ".join((a.get("title", "") + " " + a.get("description", "")) for a in arts).lower()
    cat = sum(k in text for k in CATALYST)
    return cnt, cat


def revrate(df):
    return (df.true_label_proxy == "reversal").mean()


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    ev["art_cnt"], ev["cat_hits"] = zip(*[article_info(r.ticker, r.event_id) for r in ev.itertuples()])
    cl = ev[ev.true_label_proxy.isin(["continuation", "reversal"])].copy()
    base_cont = (cl.true_label_proxy == "continuation").mean()
    print(f"clear events={len(cl)} | overall continuation rate={base_cont:.3f}\n")

    print("=== A) NEWS MATERIALITY (does amount/strength of news separate the classes?) ===")
    print("by article count:")
    for lo, hi, name in [(0, 1, "0 articles"), (1, 2, "1 article"), (2, 3, "2 articles"), (3, 99, ">=3 articles")]:
        s = cl[(cl.art_cnt >= lo) & (cl.art_cnt < hi)]
        if len(s): print(f"  {name:13} n={len(s):3}  continuation={ (s.true_label_proxy=='continuation').mean():.3f}")
    print("by catalyst-keyword hits in headlines (earnings/M&A/CEO/FDA/etc.):")
    for lo, hi, name in [(0, 1, "0 catalyst words"), (1, 3, "1-2 catalyst words"), (3, 99, ">=3 catalyst words")]:
        s = cl[(cl.cat_hits >= lo) & (cl.cat_hits < hi)]
        if len(s): print(f"  {name:18} n={len(s):3}  continuation={ (s.true_label_proxy=='continuation').mean():.3f}")
    # strong-catalyst AND fresh (>=3 catalyst words) vs no-news
    strong = cl[cl.cat_hits >= 2]; nonews = cl[cl.art_cnt == 0]
    print(f"strong-catalyst (>=2 words, n={len(strong)}): continuation={ (strong.true_label_proxy=='continuation').mean():.3f}  "
          f"| no-news (n={len(nonews)}): continuation={ (nonews.true_label_proxy=='continuation').mean():.3f}")

    print("\n=== B) MICROSTRUCTURE / OVERREACTION SIGNALS -> reversal (HOLDOUT) ===")
    ho = cl[cl.t0_utc >= HOLDOUT_START].copy()
    base_rev = revrate(ho)
    print(f"holdout n={len(ho)} | base reversal rate={base_rev:.3f}  (a signal 'works' if top vs bottom tercile differ)")
    sigs = ["adj_zscore_tod", "open_adj_range_mult", "open_adj_vol_mult", "signed_vol_ratio",
            "vpin_proxy", "idio_resid_bp", "spread_bp", "rs_ratio"]
    print(f"{'signal':20}{'botT_rev':>9}{'topT_rev':>9}{'spread':>8}")
    for s in sigs:
        x = pd.to_numeric(ho[s], errors="coerce")
        d = ho.assign(v=x).dropna(subset=["v"])
        if d.v.nunique() < 5: continue
        try:
            d["tert"] = pd.qcut(d.v, 3, labels=["bot", "mid", "top"], duplicates="drop")
        except Exception:
            continue
        bot = revrate(d[d.tert == "bot"]); top = revrate(d[d.tert == "top"])
        print(f"{s:20}{bot:>9.3f}{top:>9.3f}{top-bot:>+8.3f}")

    print("\n=== C) NON-OPENING-BELL subset ===")
    off = cl[cl.is_opening_bell != True]
    offho = off[off.t0_utc >= HOLDOUT_START]
    print(f"off-open events: {len(off)} (vs {len(cl)} total) | holdout off-open n={len(offho)}")
    if len(offho):
        acc = (offho.ensemble_label == offho.true_label_proxy).mean()
        b = max((offho.true_label_proxy=='continuation').mean(), (offho.true_label_proxy=='reversal').mean())
        print(f"  off-open: continuation rate={ (offho.true_label_proxy=='continuation').mean():.3f} | "
              f"agent acc={acc:.3f} | baseline={b:.3f} | lift={acc-b:+.3f}")
        # best microstructure signal off-open
        for s in ["open_adj_range_mult", "signed_vol_ratio", "idio_resid_bp"]:
            x = pd.to_numeric(offho[s], errors="coerce"); d = offho.assign(v=x).dropna(subset=["v"])
            if d.v.nunique() >= 5:
                d["tert"] = pd.qcut(d.v, 3, labels=["b","m","t"], duplicates="drop")
                print(f"    {s}: bot-tert rev={revrate(d[d.tert=='b']):.3f} top-tert rev={revrate(d[d.tert=='t']):.3f}")


if __name__ == "__main__":
    main()
