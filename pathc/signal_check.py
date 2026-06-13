"""
Path C GO/NO-GO signal check (free; yfinance only; no LLM).

Fixes the build_events.py label bug (labels on MARKET-ADJUSTED abnormal drift,
not raw drift), expands to ~50 tickers across cap tiers (PEAD should survive in
smaller/less-covered names), and reports every number with bootstrap 95% CIs plus
a power analysis (minimum detectable lift at the achieved N).

Core question: after an earnings reaction, does the ABNORMAL move continue (drift)
more than a coin flip — and is it stronger in small/mid caps? Plus: does the EPS
surprise sign predict the abnormal drift? Both are the cheap baselines an LLM must beat.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

TIERS = {
    "mega":  ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
    "large": ["AMD", "TSLA", "NFLX", "JPM", "AVGO", "ORCL", "CRM", "ADBE"],
    "mid":   ["COIN", "PLTR", "ROKU", "RBLX", "AFRM", "SOFI", "DKNG", "PINS",
              "SNAP", "ETSY", "HOOD", "ABNB", "DASH", "NET", "DDOG", "SNOW", "CRWD", "ZS"],
    "small": ["UPST", "FUBO", "PLUG", "RUN", "ENPH", "FSLR", "RIVN", "LCID", "U",
              "PATH", "AI", "SMCI", "MARA", "RIOT", "CVNA", "BYND", "ASAN", "GTLB"],
}
TIER_OF = {t: tier for tier, lst in TIERS.items() for t in lst}
ALL = [t for lst in TIERS.values() for t in lst]
HORIZONS = [1, 3, 5, 10, 20]
THR = 0.0  # label on sign of abnormal drift (dead-zone 0); robustness: also report at 1.5%
START, END = "2023-01-01", "2026-06-01"


def boot_ci(x, fn=np.mean, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (fn(x), np.percentile(stats, 2.5), np.percentile(stats, 97.5))


def mde(n, p=0.5, alpha=0.05, power=0.8):
    """Minimum detectable lift over baseline p (two-proportion, one-sample approx)."""
    if n < 5:
        return np.nan
    z = 1.96 + 0.84
    return z * np.sqrt(p * (1 - p) / n)


def main():
    # ---- earnings dates + surprise ----
    rows = []
    for t in ALL:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=40)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.reset_index()
        for _, r in df.iterrows():
            dt = r["Earnings Date"]
            if pd.Timestamp(START, tz=dt.tz) <= dt <= pd.Timestamp(END, tz=dt.tz):
                rows.append({"ticker": t, "dt": dt,
                             "timing": "AMC" if dt.time() >= pd.Timestamp("16:00").time()
                             else "BMO" if dt.time() <= pd.Timestamp("09:30").time() else "INTRA",
                             "surprise": r.get("Surprise(%)")})
    cal = pd.DataFrame(rows)
    cal["date"] = cal["dt"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    print(f"earnings events fetched: {len(cal)} across {cal['ticker'].nunique()} tickers")

    # ---- daily bars (+SPY) ----
    px = yf.download(ALL + ["SPY"], start=START, end=END, interval="1d",
                     auto_adjust=True, progress=False, group_by="ticker")
    def closes(t):
        try:
            s = px[t]["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            return s
        except Exception:
            return None
    spy = closes("SPY")

    # ---- build events with market-adjusted abnormal drift ----
    ev = []
    for _, e in cal.iterrows():
        s = closes(e["ticker"])
        if s is None:
            continue
        idx = s.index
        d = e["date"]
        R = idx.searchsorted(d) if e["timing"] == "BMO" else idx.searchsorted(d, side="right")
        if R < 1 or R >= len(idx):
            continue
        rr = s.iloc[R] / s.iloc[R - 1] - 1.0
        rdir = 1 if rr > 0 else -1
        rec = {"ticker": e["ticker"], "tier": TIER_OF[e["ticker"]], "rdir": rdir,
               "surprise": e["surprise"]}
        for N in HORIZONS:
            if R + N < len(idx):
                dr = s.iloc[R + N] / s.iloc[R] - 1.0
                spdr = (spy.loc[idx[R + N]] / spy.loc[idx[R]] - 1.0) if (spy is not None and idx[R] in spy.index and idx[R + N] in spy.index) else 0.0
                abn = dr - spdr
                rec[f"abn_{N}"] = abn
                rec[f"cont_{N}"] = 1 if abn * rdir > THR else 0  # 1=continuation(drift), 0=fade
            else:
                rec[f"abn_{N}"] = np.nan; rec[f"cont_{N}"] = np.nan
        ev.append(rec)
    df = pd.DataFrame(ev)
    print(f"usable events: {len(df)}\n")

    # ---- main result: abnormal-drift continuation rate (vs 50% coin flip) ----
    print("=== ABNORMAL-DRIFT CONTINUATION RATE (market-adjusted; vs 50% baseline) ===")
    print(f"{'N':>3} {'n':>4} {'cont%':>7} {'95% CI':>16} {'MDE@N':>7} {'sig>50%?':>9}")
    for N in HORIZONS:
        col = df[f"cont_{N}"].dropna()
        m, lo, hi = boot_ci(col.values)
        sig = "YES" if lo > 0.5 else "no"
        print(f"{N:>3} {len(col):>4} {m:>7.3f} [{lo:.3f},{hi:.3f}] {mde(len(col)):>7.3f} {sig:>9}")

    # ---- by cap tier at the 5-day horizon ----
    print("\n=== 5-DAY ABNORMAL CONTINUATION BY CAP TIER (PEAD should be stronger small-cap) ===")
    for tier in ["mega", "large", "mid", "small"]:
        col = df[df.tier == tier]["cont_5"].dropna()
        m, lo, hi = boot_ci(col.values)
        print(f"  {tier:6} n={len(col):>3}  cont%={m:.3f}  CI=[{lo:.3f},{hi:.3f}]")

    # ---- surprise-sign -> abnormal-drift direction (the cheap rule the LLM must beat) ----
    print("\n=== EPS-SURPRISE SIGN -> ABNORMAL-DRIFT DIRECTION (vs 50%) ===")
    print(f"{'N':>3} {'n':>4} {'hit%':>7} {'95% CI':>16} {'sig>50%?':>9}")
    for N in HORIZONS:
        sub = df.dropna(subset=["surprise", f"abn_{N}"])
        hits = (np.sign(sub["surprise"]) == np.sign(sub[f"abn_{N}"])).astype(int).values
        m, lo, hi = boot_ci(hits)
        sig = "YES" if lo > 0.5 else "no"
        print(f"{N:>3} {len(hits):>4} {m:>7.3f} [{lo:.3f},{hi:.3f}] {sig:>9}")

    print("\nINTERPRETATION:")
    print(" - 'sig>50%? = YES' (CI excludes 0.50) => a real, exploitable effect exists at this N.")
    print(" - If small-cap cont% is clearly > mega-cap, PEAD survives where theory says it should.")
    print(" - MDE@N = smallest lift over 50% detectable at this N (80% power). If MDE > the effect, underpowered.")


if __name__ == "__main__":
    main()
