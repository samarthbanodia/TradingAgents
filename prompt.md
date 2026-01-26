0) What you’re building today

Goal: create a local file (parquet/CSV/SQLite) with columns:

ticker, timestamp, open, high, low, close, volume

for these 11 symbols:

AAPL MSFT NVDA TSLA META AMD NFLX PLTR SPY QQQ XLK

over ~3 months.

Once this exists, you never depend on Polygon again for mining/events.

1) Setup
Install
pip install requests pandas pyarrow python-dateutil

Decide date range

Use a range that includes earnings season, e.g.

2025-10-15 → 2026-01-15

(You can change later; pipeline stays same.)

2) Fetch strategy that works on free tier
The safest approach: one request per ticker per day

No pagination headaches

Very low rate pressure

Easy to resume if it stops

Requests count ≈ 11 tickers × ~60 trading days ≈ 660 (fine on free tier).

3) Copy-paste script (robust + resumable)

This does:

day-by-day fetch

retries on rate-limit/network

appends results

writes a single parquet at the end

import time
import random
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "PASTE_YOUR_POLYGON_KEY"

TICKERS = ["AAPL","MSFT","NVDA","TSLA","META","AMD","NFLX","PLTR","SPY","QQQ","XLK"]

START_DATE = "2025-10-15"
END_DATE   = "2026-01-15"

BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

def daterange(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def fetch_5min_day(ticker: str, day: str, max_retries: int = 6):
    url = f"{BASE_URL}/{ticker}/range/5/minute/{day}/{day}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}

    backoff = 1.0
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 200:
            return r.json()

        # rate-limit or transient errors -> retry
        if r.status_code in (429, 500, 502, 503, 504):
            sleep_s = backoff + random.random()
            time.sleep(sleep_s)
            backoff *= 2
            continue

        # other errors -> don't spam retries
        raise RuntimeError(f"{ticker} {day}: HTTP {r.status_code} {r.text[:200]}")

    raise RuntimeError(f"{ticker} {day}: failed after {max_retries} retries")

rows = []
missing_days = []

for ticker in TICKERS:
    print(f"\n=== {ticker} ===")
    for d in daterange(START_DATE, END_DATE):
        # skip weekends quickly
        if d.weekday() >= 5:
            continue

        day = d.strftime("%Y-%m-%d")
        try:
            data = fetch_5min_day(ticker, day)
        except Exception as e:
            print("ERROR:", e)
            missing_days.append((ticker, day, "request_failed"))
            time.sleep(2)
            continue

        results = data.get("results", [])
        if not results:
            # could be market holiday or symbol not trading
            missing_days.append((ticker, day, "no_results"))
        else:
            for bar in results:
                rows.append({
                    "ticker": ticker,
                    "timestamp": pd.to_datetime(bar["t"], unit="ms", utc=True),
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                })

        # gentle throttle (free tier friendly)
        time.sleep(0.20)

df = pd.DataFrame(rows).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
print("\nTotal rows:", len(df))
df.to_parquet("ohlcv_5min.parquet", index=False)

if missing_days:
    md = pd.DataFrame(missing_days, columns=["ticker","day","reason"])
    md.to_csv("missing_days.csv", index=False)
    print("Wrote missing_days.csv with", len(md), "rows")

print("Saved ohlcv_5min.parquet")

4) Sanity checks (so you trust your dataset)

After saving parquet, run this quick check:

import pandas as pd

df = pd.read_parquet("ohlcv_5min.parquet")
print(df.groupby("ticker").size().sort_values())

# Rough expectation:
# ~78 bars per trading day per ticker
# so ~78 * ~60 ≈ 4680 bars per ticker (ballpark)


If one ticker is wildly low, check missing_days.csv.

5) Common gotchas (and what to do)
“No results” on some days

Usually:

weekends

market holidays

occasional API hiccup

You already log them. If many “no_results” appear on normal weekdays, rerun only those days.

Rate-limited (429)

Your script retries with exponential backoff. If you still get throttled:

increase time.sleep(0.20) to 0.35

6) What you do immediately after this (next step)

Once ohlcv_5min.parquet exists:

Compute rolling baselines (vol + volume) per ticker

Trigger events (return spikes, vol spikes, volume spikes)

Extract event windows (15–20 min) + validation windows (30–60 min)

If you tell me “done, parquet saved”, I’ll give you Event Miner v1 code (with clean thresholds + plotting to curate the golden set).

Polygon vs Tiingo: should you mix?

Start Polygon-only for v0. It’s cleaner and reproducible.

Only switch/add Tiingo if:

Polygon blocks your plan (unlikely at 11 tickers)

you later expand to 50+ tickers

you need a second source to cross-verify

For now: Polygon free is enough.