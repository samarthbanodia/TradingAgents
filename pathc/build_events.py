"""
Path C step 2 — build labeled earnings-drift events + sanity-check the signal.

Reads out_pathc/earnings_calendar.csv, fetches clean DAILY bars via yfinance
(stocks + SPY for market adjustment), and for each earnings event computes:
  - reaction day R (AMC -> next session; BMO -> same session)
  - reaction_ret = close(R)/close(R-1) - 1   (the earnings-day move)
  - drift_ret(N) = close(R+N)/close(R) - 1    for N in {1,3,5,10,20}
  - abn_drift(N) = drift_ret(N) - SPY drift over same window (market-adjusted)
  - label(N): continuation if drift same sign as reaction (|.|>thr), reversal if
    opposite (|.|>thr), else unclear.

Then prints the key question: IS THERE A PEAD SIGNAL?
  - reaction-direction drift-continuation rate (does the move keep going?)
  - surprise-sign vs drift direction (does the fundamental surprise predict drift?)
Saves out_pathc/earnings_events.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(HERE, "out_pathc", "earnings_calendar.csv")
OUT = os.path.join(HERE, "out_pathc", "earnings_events.csv")

HORIZONS = [1, 3, 5, 10, 20]
THR = 0.01  # 1% drift threshold for the continuation/reversal label


def load_daily(tickers, start, end):
    """ticker -> DataFrame(date-indexed daily close)."""
    out = {}
    data = yf.download(tickers, start=start, end=end, interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker")
    for tk in tickers:
        try:
            sub = data[tk][["Close"]].dropna().copy()
        except Exception:
            continue
        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
        out[tk] = sub["Close"]
    return out


def reaction_pos(daily_index, announce_date, timing):
    """Position in daily index of the reaction day R."""
    dates = daily_index
    d = pd.Timestamp(announce_date).normalize()
    if timing == "BMO":
        # react same session if it's a trading day, else next
        pos = dates.searchsorted(d)
    else:  # AMC / INTRADAY -> next session strictly after d
        pos = dates.searchsorted(d, side="right")
    return pos if pos < len(dates) else None


def main():
    cal = pd.read_csv(CAL)
    cal["announce_dt_et"] = pd.to_datetime(cal["announce_dt_et"], utc=True)
    cal["date"] = cal["announce_dt_et"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    tickers = sorted(cal["ticker"].unique())

    start = (cal["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (cal["date"].max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    daily = load_daily(tickers + ["SPY"], start, end)
    spy = daily.get("SPY")

    rows = []
    for _, e in cal.iterrows():
        tk = e["ticker"]
        s = daily.get(tk)
        if s is None or len(s) == 0:
            continue
        idx = s.index
        R = reaction_pos(idx, e["date"], e["timing"])
        if R is None or R < 1 or R >= len(idx):
            continue
        pre = s.iloc[R - 1]
        cR = s.iloc[R]
        reaction_ret = cR / pre - 1.0
        rdir = 1 if reaction_ret > 0 else -1
        row = {
            "event_id": f"{tk}_{e['date'].date()}",
            "ticker": tk, "announce_date": e["date"].date().isoformat(),
            "timing": e["timing"], "surprise_pct": e["surprise_pct"],
            "reaction_date": idx[R].date().isoformat(),
            "reaction_ret": round(reaction_ret, 4), "reaction_dir": rdir,
        }
        for N in HORIZONS:
            if R + N < len(idx):
                dr = s.iloc[R + N] / cR - 1.0
                # market-adjust
                abn = dr
                if spy is not None:
                    try:
                        sp = spy.reindex(idx)  # align
                        abn = dr - (spy.loc[idx[R + N]] / spy.loc[idx[R]] - 1.0)
                    except Exception:
                        pass
                # label: drift relative to reaction direction
                signed = dr * rdir
                lab = "continuation" if signed > THR else "reversal" if signed < -THR else "unclear"
                row[f"drift_{N}d"] = round(dr, 4)
                row[f"abn_drift_{N}d"] = round(abn, 4)
                row[f"label_{N}d"] = lab
            else:
                row[f"drift_{N}d"] = np.nan
                row[f"abn_drift_{N}d"] = np.nan
                row[f"label_{N}d"] = "missing"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} events -> {OUT}\n")

    print("=== IS THERE A PEAD SIGNAL? (all events) ===")
    print(f"{'N':>3} {'n':>4} {'cont_rate':>10} {'react_dir->drift':>17} {'surprise->drift':>16}")
    for N in HORIZONS:
        sub = df[df[f"label_{N}d"].isin(["continuation", "reversal"])]
        if len(sub) == 0:
            continue
        cont_rate = (sub[f"label_{N}d"] == "continuation").mean()
        # does reaction direction predict drift sign? (continuation rate IS this)
        # does surprise sign predict drift sign?
        ss = sub.dropna(subset=["surprise_pct"])
        drift_dir = np.sign(ss[f"drift_{N}d"])
        surp_dir = np.sign(ss["surprise_pct"])
        surp_hit = (surp_dir == drift_dir).mean()
        print(f"{N:>3} {len(sub):>4} {cont_rate:>10.3f} {cont_rate:>17.3f} {surp_hit:>16.3f}")
    print("\nReading guide:")
    print(" cont_rate / react_dir->drift = P(price keeps drifting in the earnings-day direction).")
    print("   >0.55 means post-earnings drift (momentum) exists and is the baseline to beat.")
    print(" surprise->drift = P(sign of EPS surprise matches drift direction).")
    print("   >0.55 means the fundamental surprise carries signal an LLM could read/augment.")


if __name__ == "__main__":
    main()
