"""
Small CLEAN catalyst-anchored pilot (baseline-first, NO LLM). FREE via yfinance.

Builds leak-free earnings events on a HETEROGENEOUS universe (cap/sector mix) and
answers, with a *quantitative* catalyst-strength proxy (|EPS surprise|) before any
LLM spend:

  Q1: Does catalyst strength help predict continuation-vs-reversal overall?
  Q2 (H1): Among HIGH-overreaction events, do STRONG-catalyst ones CONTINUE while
           WEAK-catalyst ones REVERT?  (catalyst conditions the overreaction signal)

Design (leak-free): decision = end of the earnings reaction day R (earnings already
public). Inputs known at R: abnormal reaction move (overreaction), |surprise|.
Label: market-adjusted forward H-day move, continuation if same sign as reaction.
Chronological frozen holdout. Surprise winsorized (small-denominator blowups).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import yfinance as yf

UNIVERSE = [  # heterogeneous: cap + sector mix, incl. less-covered names
    "AAPL","MSFT","NVDA","GOOGL","AMD","TSLA","NFLX","COIN","PLTR",      # tech/large
    "JPM","JNJ","XOM","WMT","PG","HD","CVX","KO","MRK","CAT","UNH",      # diverse large
    "ROKU","RBLX","SOFI","DKNG","ETSY","HOOD","UPST","PLUG","RUN","ASAN","GTLB","BMBL","FUBO"]  # mid/small
H = 5           # label horizon (trading days)
THETA = 0.0     # dead-zone on abnormal drift sign
SURP_CAP = 50.0 # winsorize |surprise%|
HOLDOUT_DATE = pd.Timestamp("2025-10-01")  # chronological frozen holdout


def boot_ci(mask, n=3000, seed=0):
    x = np.asarray(mask, float)
    if len(x) < 5: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return (x.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5))


def main():
    LO, HI = pd.Timestamp("2023-06-01"), pd.Timestamp("2026-05-01")
    cal = []
    for t in UNIVERSE:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=30)
        except Exception:
            continue
        if df is None or df.empty: continue
        df = df.reset_index()
        for _, r in df.iterrows():
            dt = r["Earnings Date"]
            if pd.Timestamp(LO, tz=dt.tz) <= dt <= pd.Timestamp(HI, tz=dt.tz):
                cal.append({"ticker": t, "dt": dt, "surprise": r.get("Surprise(%)"),
                            "timing": "AMC" if dt.time() >= pd.Timestamp("16:00").time()
                            else "BMO" if dt.time() <= pd.Timestamp("09:30").time() else "INTRA"})
    cal = pd.DataFrame(cal)
    cal["date"] = cal["dt"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)

    px = yf.download(UNIVERSE + ["SPY"], start="2023-01-01", end="2026-06-01",
                     interval="1d", auto_adjust=True, progress=False, group_by="ticker")
    def cl(t):
        try:
            s = px[t]["Close"].dropna(); s.index = pd.to_datetime(s.index).tz_localize(None).normalize(); return s
        except Exception: return None
    spy = cl("SPY")

    rows = []
    for _, e in cal.iterrows():
        s = cl(e["ticker"])
        if s is None: continue
        idx = s.index; d = e["date"]
        R = idx.searchsorted(d) if e["timing"] == "BMO" else idx.searchsorted(d, side="right")
        if R < 1 or R + H >= len(idx): continue
        rr = s.iloc[R]/s.iloc[R-1]-1; sp = (spy.loc[idx[R]]/spy.loc[idx[R-1]]-1) if idx[R] in spy.index and idx[R-1] in spy.index else 0
        abn_react = rr - sp
        dr = s.iloc[R+H]/s.iloc[R]-1; spd = (spy.loc[idx[R+H]]/spy.loc[idx[R]]-1) if idx[R+H] in spy.index and idx[R] in spy.index else 0
        abn_drift = dr - spd
        rdir = 1 if abn_react > 0 else -1
        signed = abn_drift * rdir
        lab = "continuation" if signed > THETA else "reversal" if signed < -THETA else "unclear"
        sv = e["surprise"]
        rows.append({"ticker": e["ticker"], "date": d, "abn_react": abn_react,
                     "overreaction": abs(abn_react), "cat_strength": min(abs(sv), SURP_CAP) if pd.notna(sv) else np.nan,
                     "label": lab})
    df = pd.DataFrame(rows)
    cl2 = df[df.label.isin(["continuation","reversal"])].dropna(subset=["cat_strength"]).copy()
    tr = cl2[cl2.date < HOLDOUT_DATE]; ho = cl2[cl2.date >= HOLDOUT_DATE]
    print(f"events: {len(df)} | clear+surprise: {len(cl2)} | train {len(tr)} / holdout {len(ho)} "
          f"| tickers {cl2.ticker.nunique()}")
    print(f"overall continuation rate: {(cl2.label=='continuation').mean():.3f}\n")

    print("=== Q2 (H1): does catalyst STRENGTH condition the overreaction? ===")
    print("Among HIGH-overreaction events (top tercile), reversal rate by catalyst strength:")
    hi = cl2[cl2.overreaction >= cl2.overreaction.quantile(2/3)]
    cut = hi.cat_strength.median()
    for nm, sub in [("STRONG catalyst (|surp|>med)", hi[hi.cat_strength > cut]),
                    ("WEAK catalyst (|surp|<=med)", hi[hi.cat_strength <= cut])]:
        m, lo, h = boot_ci((sub.label == "reversal").values)
        print(f"  {nm:28} n={len(sub):3}  reversal={m:.3f}  CI=[{lo:.3f},{h:.3f}]")
    print("  H1 predicts: STRONG continues (low reversal), WEAK reverts (high reversal).")

    print("\n=== Q1: does catalyst strength help OVERALL? (correlation w/ continuation) ===")
    cont = (cl2.label == "continuation").astype(int)
    r_str = np.corrcoef(cl2.cat_strength, cont)[0,1]
    r_over = np.corrcoef(cl2.overreaction, cont)[0,1]
    print(f"  corr(catalyst_strength, continuation) = {r_str:+.3f}")
    print(f"  corr(overreaction,       continuation) = {r_over:+.3f}  (the quant baseline signal)")
    # holdout: does strong vs weak catalyst differ in continuation, out-of-sample?
    cutho = tr.cat_strength.median()
    for nm, sub in [("HOLDOUT strong-catalyst", ho[ho.cat_strength > cutho]),
                    ("HOLDOUT weak-catalyst", ho[ho.cat_strength <= cutho])]:
        m, lo, h = boot_ci((sub.label == "continuation").values)
        print(f"  {nm:24} n={len(sub):3}  continuation={m:.3f}  CI=[{lo:.3f},{h:.3f}]")


if __name__ == "__main__":
    main()
