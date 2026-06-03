"""
Analysis harness for the ICAIF reframe (WS0).

Reuses existing N=800 agent outputs + OHLCV. No LLM calls.

Provides:
  - load_events(): flat DataFrame of the 800 agent-output records.
  - load_ohlcv(): the 5-min bars, indexed per ticker for fast lookup.
  - forward_return(): leak-free forward return from t0 at an arbitrary horizon,
    matching mine_events_strict.py exactly (close_t0 = close at t0 bar).
  - relabel(): continuation/reversal/unclear at an arbitrary horizon, using the
    same +/-0.3% threshold and direction-signing convention as the miner.
  - temporal_split(): train <= 2026-01-05, holdout >= 2026-01-06 (matches ml_stage_v2).

Run directly for the WS0 correctness gate: recompute the 60-min label and confirm
it reproduces the stored true_label_proxy.
"""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL = os.path.join(HERE, "out_agents_extended", "ip_outputs_v3.jsonl")
PARQUET = os.path.join(HERE, "ohlcv_5min_extended.parquet")

CONT_THRESH = 0.003   # +0.30%
REV_THRESH = 0.003
HOLDOUT_START = pd.Timestamp("2026-01-06", tz="UTC")


# ----------------------------------------------------------------------------- events
def load_events(path=JSONL):
    rows = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # skip malformed/error records (failed agent calls leave a stub)
            if not all(k in r for k in ("t0_utc", "micro_label", "news_label",
                                        "macro_label", "ensemble_label")):
                skipped += 1
                continue
            row = {
                "event_id": r["event_id"],
                "ticker": r["ticker"],
                "t0_utc": pd.Timestamp(r["t0_utc"]),
                "direction": int(r["direction"]),
                "true_label_proxy": r["true_label_proxy"],
                "is_up_spike": r.get("is_up_spike"),
                "is_etf": r.get("is_etf"),
                "is_opening_bell": r.get("is_opening_bell"),
                "has_pre_event_news": r.get("has_pre_event_news"),
                # per-agent
                "micro_label": r["micro_label"], "micro_conf": r["micro_conf"],
                "news_label": r["news_label"], "news_conf": r["news_conf"],
                "macro_label": r["macro_label"], "macro_conf": r["macro_conf"],
                # ensemble
                "ensemble_label": r["ensemble_label"],
                "ensemble_score": r.get("ensemble_score"),
                "ens_cont": r.get("ensemble_votes", {}).get("continuation"),
                "ens_rev": r.get("ensemble_votes", {}).get("reversal"),
                # spike magnitude proxies (for momentum baselines / equal-coverage)
                "open_adj_vol_mult": r.get("open_adj_vol_mult"),
                "open_adj_range_mult": r.get("open_adj_range_mult"),
            }
            f3 = r.get("v3_features", {})
            for k in ("idio_resid_bp", "rs_ratio", "signed_vol_ratio", "vpin_proxy",
                      "spread_bp", "pre_spike_run_len", "adj_zscore_tod",
                      "spy_zscore_t0", "is_macro_driven"):
                row[k] = f3.get(k)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df["t0_utc"].dt.tz is None:
        df["t0_utc"] = df["t0_utc"].dt.tz_localize("UTC")
    # agreement features
    labs = df[["micro_label", "news_label", "macro_label"]]
    df["n_rev_votes"] = (labs == "reversal").sum(axis=1)
    df["n_cont_votes"] = (labs == "continuation").sum(axis=1)
    df["unanimous"] = labs.nunique(axis=1) == 1
    df["agree_count"] = labs.apply(lambda r: r.value_counts().iloc[0], axis=1)  # majority size 2 or 3
    df["mean_conf"] = df[["micro_conf", "news_conf", "macro_conf"]].mean(axis=1)
    df["ens_margin"] = (df["ens_cont"].fillna(0) - df["ens_rev"].fillna(0)).abs()
    return df


# ----------------------------------------------------------------------------- ohlcv
def load_ohlcv(path=PARQUET):
    df = pd.read_parquet(path, columns=["ticker", "timestamp", "open", "close"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return df


def build_ohlcv_index(ohlcv):
    """ticker -> (timestamps ndarray, close ndarray, open ndarray, dates ndarray)."""
    idx = {}
    for tkr, g in ohlcv.groupby("ticker", sort=False):
        ts = g["timestamp"].values
        idx[tkr] = {
            "ts": ts,
            "close": g["close"].values,
            "open": g["open"].values,
            "date": g["timestamp"].dt.date.values,
        }
    return idx


# ----------------------------------------------------------------------------- forward return
def forward_return(idx, ticker, t0, horizon):
    """
    Leak-free signed forward return from t0.

    horizon: int minutes for intraday (e.g. 30, 60, 120), or one of
             {"EOD", "NEXT_OPEN", "T+1D", "T+2D"}.
    Returns (fwd_ret, ok). fwd_ret is raw (unsigned by direction).
    close_t0 = close of the bar AT t0 (matches miner: event_window close.iloc[-1]).
    """
    g = idx.get(ticker)
    if g is None:
        return np.nan, False
    ts = g["ts"]
    # locate t0 bar (exact match expected; fall back to last bar <= t0)
    pos = np.searchsorted(ts, np.datetime64(t0))
    if pos >= len(ts) or ts[pos] != np.datetime64(t0):
        pos = pos - 1
        if pos < 0:
            return np.nan, False
    close_t0 = g["close"][pos]
    t0_date = g["date"][pos]

    if isinstance(horizon, int):
        target = np.datetime64(t0 + pd.Timedelta(minutes=horizon))
        # bars on same trading day, at or before target
        same_day = g["date"] == t0_date
        cand = np.where(same_day & (ts <= target) & (np.arange(len(ts)) >= pos))[0]
        if len(cand) == 0:
            return np.nan, False
        end_pos = cand[-1]
    elif horizon == "EOD":
        cand = np.where(g["date"] == t0_date)[0]
        end_pos = cand[-1]
    elif horizon in ("NEXT_OPEN", "T+1D", "T+2D"):
        future_dates = sorted(set(g["date"][g["date"] > t0_date]))
        if horizon == "NEXT_OPEN":
            if not future_dates:
                return np.nan, False
            d = future_dates[0]
            cand = np.where(g["date"] == d)[0]
            close_end = g["open"][cand[0]]  # next session open
            return (close_end / close_t0) - 1.0, True
        n = 1 if horizon == "T+1D" else 2
        if len(future_dates) < n:
            return np.nan, False
        d = future_dates[n - 1]
        cand = np.where(g["date"] == d)[0]
        end_pos = cand[-1]
    else:
        raise ValueError(horizon)

    close_end = g["close"][end_pos]
    return (close_end / close_t0) - 1.0, True


def relabel(idx, ticker, t0, direction, horizon, cont=CONT_THRESH, rev=REV_THRESH):
    fr, ok = forward_return(idx, ticker, t0, horizon)
    if not ok or np.isnan(fr):
        return "missing", np.nan
    if direction == 1:  # up spike
        lab = "continuation" if fr > cont else "reversal" if fr < -rev else "unclear"
    else:  # down spike
        lab = "continuation" if fr < -cont else "reversal" if fr > rev else "unclear"
    return lab, fr


def temporal_split(df):
    train = df[df["t0_utc"] < HOLDOUT_START].copy()
    holdout = df[df["t0_utc"] >= HOLDOUT_START].copy()
    return train, holdout


# ----------------------------------------------------------------------------- WS0 gate
def main():
    ev = load_events()
    print(f"loaded {len(ev)} events | tickers={ev['ticker'].nunique()} | "
          f"date {ev['t0_utc'].min().date()} -> {ev['t0_utc'].max().date()}")
    print("true label dist:", ev["true_label_proxy"].value_counts().to_dict())
    tr, ho = temporal_split(ev)
    print(f"temporal split: train={len(tr)} holdout={len(ho)} "
          f"(holdout {ho['t0_utc'].min().date()} -> {ho['t0_utc'].max().date()})")

    print("\n=== WS0 correctness gate: recompute 60-min label, compare to stored ===")
    ohlcv = load_ohlcv()
    idx = build_ohlcv_index(ohlcv)
    recomputed, frs = [], []
    for _, r in ev.iterrows():
        lab, fr = relabel(idx, r["ticker"], r["t0_utc"], r["direction"], 60)
        recomputed.append(lab); frs.append(fr)
    ev["relabel_60"] = recomputed
    n_missing = (ev["relabel_60"] == "missing").sum()
    comp = ev[ev["relabel_60"] != "missing"]
    match = (comp["relabel_60"] == comp["true_label_proxy"]).mean()
    print(f"  missing bars: {n_missing}")
    print(f"  60-min relabel vs stored true_label_proxy: {match:.4f} agreement "
          f"(n={len(comp)})")
    mismatch = comp[comp["relabel_60"] != comp["true_label_proxy"]]
    print(f"  mismatches: {len(mismatch)}")
    if len(mismatch):
        print(mismatch[["event_id", "true_label_proxy", "relabel_60"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
