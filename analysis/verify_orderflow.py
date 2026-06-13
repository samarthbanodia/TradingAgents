"""
Verify the order-flow -> reversal signal is REAL, not the old data leak. FREE.

- Recompute signed_vol_ratio strictly from the 5 bars BEFORE t0 (leak-free, matching
  eval/feature_matrix.py's fix), and compare to the STORED value (possibly leaky).
- Honest protocol: set tercile cut-points on TRAIN, apply to HOLDOUT, report reversal
  rate per holdout tercile + bootstrap 95% CI on the (top - bottom) spread.
- Do the same for idio_resid_bp (was NOT a leaky feature -> clean control).
"""
import warnings
import numpy as np
import pandas as pd
from harness import load_events, HOLDOUT_START

warnings.filterwarnings("ignore")
HERE_PARQUET = "../ohlcv_5min_extended.parquet"


def build_full_index(path):
    df = pd.read_parquet(path, columns=["ticker", "timestamp", "open", "high", "low", "close", "volume"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.sort_values(["ticker", "timestamp"])
    idx = {}
    for tk, g in df.groupby("ticker", sort=False):
        idx[tk] = {c: g[c].values for c in ["timestamp", "open", "high", "low", "close", "volume"]}
    return idx


def signed_vol_pre(idx, ticker, t0, n=5):
    g = idx.get(ticker)
    if g is None:
        return np.nan
    pos = np.searchsorted(g["timestamp"], np.datetime64(t0))  # first bar >= t0
    s, e = max(0, pos - n), pos                                # n bars strictly before t0
    if e <= s:
        return np.nan
    o, h, l, c, v = (g["open"][s:e], g["high"][s:e], g["low"][s:e], g["close"][s:e], g["volume"][s:e])
    btr = (c - o) / (h - l + 1e-9)
    total = v.sum()
    return float((v * btr).sum() / total) if total >= 1 else np.nan


def boot_spread(df, col, lo_thr, hi_thr, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    vals, lab = df[col].values, (df.true_label_proxy.values == "reversal")
    for _ in range(n):
        i = rng.integers(0, len(df), len(df))
        v, y = vals[i], lab[i]
        bot, top = y[v <= lo_thr], y[v >= hi_thr]
        if len(bot) and len(top):
            out.append(top.mean() - bot.mean())
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def report(df_tr, df_ho, col, label):
    lo_thr = df_tr[col].quantile(1/3)
    hi_thr = df_tr[col].quantile(2/3)
    d = df_ho.dropna(subset=[col])
    bot = d[d[col] <= lo_thr]; top = d[d[col] >= hi_thr]
    rb = (bot.true_label_proxy == "reversal").mean()
    rt = (top.true_label_proxy == "reversal").mean()
    lo, hi = boot_spread(d, col, lo_thr, hi_thr)
    sig = "REAL (CI excludes 0)" if lo > 0 else "not significant"
    print(f"  {label:26} bot-tert rev={rb:.3f} (n={len(bot)})  top-tert rev={rt:.3f} (n={len(top)})  "
          f"spread={rt-rb:+.3f}  CI=[{lo:+.3f},{hi:+.3f}]  -> {sig}")


def main():
    ev = load_events("../out_agents_v4a/ip_outputs_v3.jsonl")
    idx = build_full_index(HERE_PARQUET)
    ev["sv_clean"] = [signed_vol_pre(idx, r.ticker, r.t0_utc) for r in ev.itertuples()]
    cl = ev[ev.true_label_proxy.isin(["continuation", "reversal"])].copy()

    # how different is the clean (pre-event) signal from the stored one?
    both = cl.dropna(subset=["sv_clean", "signed_vol_ratio"])
    corr = np.corrcoef(both.sv_clean, both.signed_vol_ratio.astype(float))[0, 1]
    print(f"clear n={len(cl)} | corr(stored signed_vol, clean pre-event signed_vol) = {corr:.3f}")
    print("(low corr => stored value used post-event bars = leak; clean is the honest one)\n")

    tr = cl[cl.t0_utc < HOLDOUT_START]; ho = cl[cl.t0_utc >= HOLDOUT_START]
    print(f"train n={len(tr)} | holdout n={len(ho)} | holdout base reversal rate="
          f"{(ho.true_label_proxy=='reversal').mean():.3f}")
    print("\nHoldout reversal separation (tercile cuts fixed on TRAIN, bootstrap CI on spread):")
    report(tr, ho, "sv_clean", "signed_vol (CLEAN pre-evt)")
    report(tr, ho, "signed_vol_ratio", "signed_vol (STORED/leaky?)")
    report(tr, ho, "idio_resid_bp", "idio_resid (clean control)")
    print("\nVERDICT: if CLEAN signed_vol still separates with CI>0, the order-flow signal is real.")
    print("If only the STORED one separates, the +33pp was the leak.")


if __name__ == "__main__":
    main()
