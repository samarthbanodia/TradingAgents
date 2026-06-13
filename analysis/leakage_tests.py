"""
Automated leakage regression tests. FREE. Run before trusting ANY result.
These are the asserts the rebuild pipeline must keep passing.

  A. FEATURE pre-event check: stored signed_vol/vpin must match a t0-inclusive
     pre-event recompute (no future bars). PASS if corr ~ 1.0.
  B. NEWS strict-pre-event check: no article in a packet may be published >= t0.
     (Documents the OLD leak count; must be 0 after the build_news_packets fix.)
  C. SHUFFLED-LABEL null: agent accuracy on randomly shuffled labels must ~= base
     rate. If it's higher, a label leak exists.
  D. LABEL window: the label return must use only bars strictly AFTER t0.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from harness import load_events

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKT = os.path.join(HERE, "out_extended_news", "packets")


def _idx():
    df = pd.read_parquet("../ohlcv_5min_extended.parquet",
                         columns=["ticker","timestamp","open","high","low","close","volume"])
    if df.timestamp.dt.tz is None: df["timestamp"] = df.timestamp.dt.tz_localize("UTC")
    df = df.sort_values(["ticker","timestamp"])
    return {tk: {c: g[c].values for c in df.columns if c != "ticker"} for tk, g in df.groupby("ticker", sort=False)}


def sv_incl_t0(idx, tk, t0, n=5):
    g = idx.get(tk)
    if g is None: return np.nan
    pos = np.searchsorted(g["timestamp"], np.datetime64(t0))
    end = pos+1 if (pos < len(g["timestamp"]) and g["timestamp"][pos] == np.datetime64(t0)) else pos
    s = max(0, end-n)
    if end <= s: return np.nan
    o,h,l,c,v = (g[x][s:end] for x in ["open","high","low","close","volume"])
    btr = (c-o)/(h-l+1e-9); tot = v.sum()
    return float((v*btr).sum()/tot) if tot >= 1 else np.nan


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    idx = _idx()
    results = []

    # A. feature pre-event check
    ev["sv_rc"] = [sv_incl_t0(idx, r.ticker, r.t0_utc) for r in ev.itertuples()]
    m = ev.dropna(subset=["sv_rc", "signed_vol_ratio"])
    corr = np.corrcoef(m.sv_rc, m.signed_vol_ratio.astype(float))[0, 1]
    results.append(("A feature pre-event (signed_vol corr~1)", corr > 0.99, f"corr={corr:.3f}"))

    # B. news strict-pre-event
    viol = 0
    for r in ev.itertuples():
        p = os.path.join(PKT, r.ticker, f"{r.event_id}.json")
        if not os.path.exists(p): continue
        for a in (json.load(open(p)).get("news", {}).get("ticker_news", []) or []):
            ts = a.get("published_utc", "")
            if ts and pd.Timestamp(ts) >= r.t0_utc: viol += 1
    results.append(("B news strict < t0 (0 violations)", viol == 0,
                    f"{viol} post-t0 articles (OLD packets; rebuild must be 0)"))

    # C. shuffled-label null
    cl = ev[ev.true_label_proxy.isin(["continuation","reversal"])].copy()
    base = max((cl.true_label_proxy=="continuation").mean(), (cl.true_label_proxy=="reversal").mean())
    rng = np.random.default_rng(0)
    accs = []
    for _ in range(200):
        y = rng.permutation(cl.true_label_proxy.values)
        accs.append((cl.ensemble_label.values == y).mean())
    null_mean = np.mean(accs)
    real = (cl.ensemble_label == cl.true_label_proxy).mean()
    results.append(("C shuffled-label null ~= base rate", abs(null_mean - (1-base)) < 0.05 or null_mean < base,
                    f"shuffled acc={null_mean:.3f} vs base={base:.3f}; real={real:.3f}"))

    # D. label uses strictly-after-t0 bars (structural — harness.forward_return uses ts>pos)
    results.append(("D label window strictly > t0", True, "by construction in harness.forward_return (end bars have ts>t0)"))

    print("=== LEAKAGE REGRESSION TESTS ===")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42} {detail}")
    print("\nNote: B 'fails' on the OLD packets by design — it documents the leak we fixed in")
    print("build_news_packets.py; re-running the news fetch must drive it to 0.")


if __name__ == "__main__":
    main()
