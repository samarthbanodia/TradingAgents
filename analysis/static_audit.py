"""
Static data-integrity audit. FREE — no LLM calls, existing data only.

1) FEATURE LEAKAGE SWEEP: recompute order-flow features strictly from pre-event bars
   and compare to the STORED values (low correlation => stored used post-event data).
2) CATALYST-TIMESTAMP CHECK: scan news packets for any article published AT/AFTER t0
   (a post-event article leaking into the input window).
3) PER-EVENT AUDIT SHEET: for a stratified sample, document event time, spike path,
   catalyst + its timestamp, label window/outcome -> analysis/event_audit_sheet.md
"""
import json, os, warnings
import numpy as np
import pandas as pd
from harness import load_events


def fwd_ret_60(idx, ticker, t0):
    g = idx.get(ticker)
    if g is None: return np.nan
    ts = g["timestamp"]; pos = np.searchsorted(ts, np.datetime64(t0))
    if pos >= len(ts) or ts[pos] != np.datetime64(t0):
        pos -= 1
    if pos < 0: return np.nan
    c0 = g["close"][pos]; t0d = pd.Timestamp(t0)
    target = np.datetime64(t0d + pd.Timedelta(minutes=60))
    same_day = pd.to_datetime(ts).normalize() == t0d.normalize().tz_localize(None) if False else None
    cand = np.where((ts <= target) & (np.arange(len(ts)) >= pos))[0]
    return (g["close"][cand[-1]]/c0 - 1.0) if len(cand) else np.nan

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKT = os.path.join(HERE, "out_extended_news", "packets")
PARQUET = "../ohlcv_5min_extended.parquet"


def build_full_index(path):
    df = pd.read_parquet(path, columns=["ticker","timestamp","open","high","low","close","volume"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.sort_values(["ticker","timestamp"])
    return {tk: {c: g[c].values for c in ["timestamp","open","high","low","close","volume"]}
            for tk, g in df.groupby("ticker", sort=False)}


def pre_bars(idx, ticker, t0, n=5):
    g = idx.get(ticker)
    if g is None: return None
    pos = np.searchsorted(g["timestamp"], np.datetime64(t0))
    s, e = max(0, pos-n), pos
    return None if e <= s else {c: g[c][s:e] for c in g}


def signed_vol_clean(idx, ticker, t0):
    b = pre_bars(idx, ticker, t0)
    if b is None: return np.nan
    btr = (b["close"]-b["open"])/(b["high"]-b["low"]+1e-9)
    tot = b["volume"].sum()
    return float((b["volume"]*btr).sum()/tot) if tot >= 1 else np.nan


def load_pkt(ticker, eid):
    p = os.path.join(PKT, ticker, f"{eid}.json")
    if not os.path.exists(p): return []
    arts = json.load(open(p)).get("news", {}).get("ticker_news", []) or []
    return [(a.get("published_utc",""), a.get("title","")) for a in arts]


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    idx = build_full_index(PARQUET)

    print("=== 1) FEATURE LEAKAGE SWEEP ===")
    ev["sv_clean"] = [signed_vol_clean(idx, r.ticker, r.t0_utc) for r in ev.itertuples()]
    m = ev.dropna(subset=["sv_clean","signed_vol_ratio","vpin_proxy"])
    c_sv = np.corrcoef(m.sv_clean, m.signed_vol_ratio.astype(float))[0,1]
    c_vp = np.corrcoef(m.sv_clean.abs(), m.vpin_proxy.astype(float))[0,1]
    print(f"  signed_vol_ratio: corr(clean pre-event, stored) = {c_sv:.3f}  -> {'LEAK (stored used post-t0 bars)' if c_sv<0.7 else 'ok'}")
    print(f"  vpin_proxy:       corr(|clean|, stored)         = {c_vp:.3f}  -> {'LEAK' if c_vp<0.7 else 'ok'}")
    print("  feature window classification (from code):")
    cls = {"idio_resid_bp":"spike window <=t0  SAFE", "rs_ratio":"pre-event vol  SAFE",
           "beta":"pre-event regression  SAFE", "spread_bp":"t0 bar  SAFE(at-t0)",
           "pre_spike_run_len":"bars < t0  SAFE", "adj_zscore_tod":"at t0  SAFE",
           "spy_zscore_t0":"at t0  SAFE", "signed_vol_ratio":"STORED used cluster bars >=t0  LEAK",
           "vpin_proxy":"STORED used cluster bars >=t0  LEAK", "cluster_len":"post-t0  LEAK (already removed)"}
    for k,v in cls.items(): print(f"    {k:20} {v}")

    print("\n=== 2) CATALYST-TIMESTAMP CHECK (any article published >= t0?) ===")
    viol, total_art, evt_with_post = 0, 0, 0
    for r in ev.itertuples():
        arts = load_pkt(r.ticker, r.event_id)
        has_post = False
        for ts, _ in arts:
            total_art += 1
            if ts:
                try:
                    if pd.Timestamp(ts) >= r.t0_utc:
                        viol += 1; has_post = True
                except Exception:
                    pass
        if has_post: evt_with_post += 1
    print(f"  total articles across packets: {total_art}")
    print(f"  articles published AT/AFTER t0 (LEAK): {viol}")
    print(f"  events with >=1 post-t0 article in their packet: {evt_with_post}/{len(ev)}")
    print("  (any >0 here means the news window logic let future articles into the input)")

    print("\n=== 3) PER-EVENT AUDIT SHEET -> analysis/event_audit_sheet.md ===")
    cl = ev[ev.true_label_proxy.isin(["continuation","reversal"])].copy()
    cl["absidio"] = cl.idio_resid_bp.abs()
    # stratified-ish sample: extremes of overreaction + mix of news/no-news/labels
    samp = pd.concat([cl.nlargest(8,"absidio"), cl.nsmallest(6,"absidio"),
                      cl.sample(min(12,len(cl)), random_state=1)]).drop_duplicates("event_id").head(26)
    lines = ["# Per-Event Audit Sheet (sample)\n",
             "For each: decision time, spike path, catalyst+timestamp, what the agent saw, label window.\n"]
    for r in samp.itertuples():
        fr60 = fwd_ret_60(idx, r.ticker, r.t0_utc)
        arts = load_pkt(r.ticker, r.event_id)
        post = sum(1 for ts,_ in arts if ts and pd.Timestamp(ts) >= r.t0_utc)
        lines.append(f"## {r.event_id}  ({r.ticker})")
        lines.append(f"- decision time t0 (UTC): {r.t0_utc}   dir: {'UP' if r.direction==1 else 'DOWN'}")
        lines.append(f"- spike: idio_resid={r.idio_resid_bp:+.0f}bp, vol_mult={r.open_adj_vol_mult}, range_mult={r.open_adj_range_mult}")
        lines.append(f"- catalyst packet: {len(arts)} articles ({post} AT/AFTER t0 = LEAK if >0)")
        for ts, ti in arts[:3]:
            flag = " <-- POST-t0 LEAK" if (ts and pd.Timestamp(ts) >= r.t0_utc) else ""
            lines.append(f"    [{ts[:16]}] {ti[:80]}{flag}")
        lines.append(f"- label window: t0 -> t0+60min | fwd_ret={fr60:+.3%} | LABEL={r.true_label_proxy}")
        lines.append(f"- agent saw: micro/news/macro features above + {len(arts)} articles; predicted {r.ensemble_label}\n")
    with open("event_audit_sheet.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  wrote {len(samp)}-event sheet. First 2 events:\n")
    print("\n".join(lines[2:20]))


if __name__ == "__main__":
    main()
