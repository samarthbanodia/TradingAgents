"""
Path C step 1 — fetch earnings calendar (date + surprise + AMC/BMO timing).

Free via yfinance (no key). Saves out_pathc/earnings_calendar.csv with:
  ticker, announce_dt_et, timing (AMC/BMO/INTRADAY), eps_estimate, reported_eps, surprise_pct

AMC/BMO is inferred from the announcement time-of-day (ET):
  >= 16:00 -> AMC (market reacts next session)
  <= 09:30 -> BMO (market reacts same session)
  else     -> INTRADAY (rare; treated as next session downstream)
"""
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out_pathc", "earnings_calendar.csv")

STOCKS = ["AAPL", "AMD", "AMZN", "COIN", "GOOGL", "JPM", "META", "MSFT",
          "NFLX", "NVDA", "PLTR", "TSLA"]
LO = pd.Timestamp("2024-05-23", tz="America/New_York")
HI = pd.Timestamp("2026-04-30", tz="America/New_York")


def timing(dt_et):
    t = dt_et.time()
    if t >= pd.Timestamp("16:00").time():
        return "AMC"
    if t <= pd.Timestamp("09:30").time():
        return "BMO"
    return "INTRADAY"


def main():
    rows = []
    for tk in STOCKS:
        try:
            df = yf.Ticker(tk).get_earnings_dates(limit=24)
        except Exception as e:
            print(f"{tk}: ERROR {str(e)[:80]}")
            continue
        if df is None or df.empty:
            print(f"{tk}: no earnings dates")
            continue
        df = df.reset_index().rename(columns={"Earnings Date": "announce_dt_et",
                                              "EPS Estimate": "eps_estimate",
                                              "Reported EPS": "reported_eps",
                                              "Surprise(%)": "surprise_pct"})
        df = df[(df["announce_dt_et"] >= LO) & (df["announce_dt_et"] <= HI)]
        for _, r in df.iterrows():
            dt = r["announce_dt_et"]
            rows.append({
                "ticker": tk,
                "announce_dt_et": dt.isoformat(),
                "timing": timing(dt),
                "eps_estimate": r.get("eps_estimate"),
                "reported_eps": r.get("reported_eps"),
                "surprise_pct": r.get("surprise_pct"),
            })
    out = pd.DataFrame(rows).sort_values(["ticker", "announce_dt_et"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"saved {len(out)} earnings events -> {OUT}")
    print("timing mix:", out["timing"].value_counts().to_dict())
    print("with surprise:", out["surprise_pct"].notna().sum(), "/", len(out))


if __name__ == "__main__":
    main()
