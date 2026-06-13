"""
Spike plots for the manual 20-30 event audit. FREE.
One panel per event: price path around t0 (pre-event context + the 60-min label
window), with t0 and t0+60min marked. -> analysis/spike_audit_grid.png
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harness import load_events

warnings.filterwarnings("ignore")


def idx_full():
    df = pd.read_parquet("../ohlcv_5min_extended.parquet", columns=["ticker","timestamp","close"])
    if df.timestamp.dt.tz is None: df["timestamp"] = df.timestamp.dt.tz_localize("UTC")
    df = df.sort_values(["ticker","timestamp"])
    return {tk: {"ts": g.timestamp.values, "close": g.close.values} for tk, g in df.groupby("ticker", sort=False)}


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    idx = idx_full()
    cl = ev[ev.true_label_proxy.isin(["continuation","reversal"])].copy()
    cl["absidio"] = cl.idio_resid_bp.abs()
    samp = pd.concat([cl.nlargest(12, "absidio"), cl.sample(12, random_state=1)]).drop_duplicates("event_id").head(24)

    fig, axes = plt.subplots(4, 6, figsize=(22, 13))
    for ax, r in zip(axes.flat, samp.itertuples()):
        g = idx.get(r.ticker)
        pos = np.searchsorted(g["ts"], np.datetime64(r.t0_utc))
        a, b = max(0, pos-10), min(len(g["ts"]), pos+13)   # ~10 bars pre, 12 bars (60min) post
        c = g["close"][a:b]
        x = np.arange(a, b) - pos
        ax.plot(x, c, lw=1.2, color="#1f77b4")
        ax.axvline(0, color="#d62728", lw=1, ls="--")        # t0
        ax.axvspan(0, 12, color="#2ca02c", alpha=0.08)       # 60-min label window
        col = "#2ca02c" if r.true_label_proxy == "continuation" else "#d62728"
        ax.set_title(f"{r.ticker} {pd.Timestamp(r.t0_utc).date()}\nidio={r.idio_resid_bp:+.0f}bp  {r.true_label_proxy}",
                     fontsize=8, color=col)
        ax.tick_params(labelsize=6); ax.axhline(c[10] if len(c)>10 else c[0], color="grey", lw=0.4, ls=":")
    for ax in axes.flat[len(samp):]:
        ax.axis("off")
    fig.suptitle("Spike audit: price path around t0 (dashed=t0, shaded=60-min label window)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "spike_audit_grid.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out} with {len(samp)} events")


if __name__ == "__main__":
    main()
