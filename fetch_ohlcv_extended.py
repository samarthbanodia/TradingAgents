"""
Extended OHLCV Fetcher — 15 tickers, 2022-2025, monthly batch requests.

Key upgrade from fetch_ohlcv.py:
- 4 new tickers: AMZN, GOOGL, JPM, COIN
- Date range: 2022-01-01 → 2025-12-31
- Monthly batch requests (not day-by-day): ~720 API calls vs ~15k
  Polygon allows date ranges in the /v2/aggs endpoint; monthly 5-min data
  is ~1,700 bars, well within the 50,000 result limit.
- Pagination: follows next_url if present (shouldn't be needed for monthly)
- Resume: progress tracked by (ticker, year_month) pairs

Usage:
    python fetch_ohlcv_extended.py
    python fetch_ohlcv_extended.py --out ohlcv_5min_extended.parquet
    python fetch_ohlcv_extended.py --start 2023-01-01 --end 2024-12-31
"""

import os
import sys
import time
import random
import calendar
import argparse
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

def _safe_print(*args, **kwargs):
    """Print that replaces unencodable chars instead of crashing."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe = " ".join(str(a).encode("ascii", errors="replace").decode("ascii") for a in args)
        print(safe, **{k: v for k, v in kwargs.items() if k != "end"})

load_dotenv()

TICKERS = [
    # Original 11
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMD", "NFLX", "PLTR",
    "SPY", "QQQ", "XLK",
    # New 4
    "AMZN", "GOOGL", "JPM", "COIN",
]

DEFAULT_START = "2024-05-01"   # free Polygon tier: ~2 years back from 2026-05
DEFAULT_END   = "2026-04-30"  # conservative end (avoids unresolved label windows)
BASE_URL      = "https://api.polygon.io/v2/aggs/ticker"


# ── API key management (same as fetch_ohlcv.py) ───────────────────────────────

class APIKeyManager:
    def __init__(self):
        self.keys = self._load_keys()
        self.current_index = 0
        self.rate_limited = {}
        self.counts = {}
        if not self.keys:
            raise ValueError("No POLYGON_API_KEY_* found in .env")
        _safe_print(f"Loaded {len(self.keys)} Polygon API keys")

    def _load_keys(self):
        keys = []
        # Support single POLYGON_API_KEY as well as POLYGON_API_KEY_1..20
        single = os.getenv("POLYGON_API_KEY")
        if single and not single.startswith("your_"):
            keys.append(single)
        for i in range(1, 21):
            k = os.getenv(f"POLYGON_API_KEY_{i}")
            if k and not k.startswith("your_") and k not in keys:
                keys.append(k)
        return keys

    def get_key(self):
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.current_index]
            if key in self.rate_limited:
                if now >= self.rate_limited[key]:
                    del self.rate_limited[key]
                    return key
                self.current_index = (self.current_index + 1) % len(self.keys)
                continue
            return key
        if self.rate_limited:
            wait = min(self.rate_limited.values()) - now + 1
            if wait > 0:
                _safe_print(f"All keys rate-limited. Waiting {wait:.1f}s...")
                time.sleep(wait)
            self.rate_limited = {k: v for k, v in self.rate_limited.items() if v > time.time()}
        return self.keys[self.current_index]

    def mark_limited(self, key, cooldown=65):
        self.rate_limited[key] = time.time() + cooldown
        _safe_print(f"  Key {self.keys.index(key)+1} rate-limited → rotating")
        self.current_index = (self.current_index + 1) % len(self.keys)

    def record(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def stats(self):
        _safe_print("\nAPI Key Usage:")
        for i, k in enumerate(self.keys):
            _safe_print(f"  Key {i+1}: {self.counts.get(k, 0)} requests")


# ── Fetch one month of 5-min bars, with pagination ────────────────────────────

def fetch_month(ticker: str, year: int, month: int, key_mgr: APIKeyManager, max_retries=8):
    """Fetch all 5-min bars for ticker in the given calendar month.
    Returns list of raw bar dicts (keys: t, o, h, l, c, v).
    Follows next_url pagination if Polygon paginates the response.
    """
    _, last_day = calendar.monthrange(year, month)
    date_from = f"{year}-{month:02d}-01"
    date_to   = f"{year}-{month:02d}-{last_day:02d}"

    url = f"{BASE_URL}/{ticker}/range/5/minute/{date_from}/{date_to}"
    params_base = {"adjusted": "true", "sort": "asc", "limit": 50000}

    all_results = []
    next_url = None
    page = 0

    while True:
        page += 1
        backoff = 1.0
        for attempt in range(max_retries):
            api_key = key_mgr.get_key()
            if next_url:
                # Pagination: follow next_url (already has apiKey in query string from Polygon)
                req_url    = next_url
                req_params = {"apiKey": api_key}
            else:
                req_url    = url
                req_params = {**params_base, "apiKey": api_key}

            try:
                r = requests.get(req_url, params=req_params, timeout=30)

                if r.status_code == 200:
                    key_mgr.record(api_key)
                    data = r.json()
                    all_results.extend(data.get("results") or [])
                    next_url = data.get("next_url")
                    break  # page fetched OK

                if r.status_code == 429:
                    key_mgr.mark_limited(api_key)
                    continue

                if r.status_code == 401:
                    # Bad key — blacklist it permanently and rotate
                    key_mgr.mark_limited(api_key, cooldown=86400)
                    _safe_print(f"  401 on key {api_key[:8]}... blacklisted")
                    continue

                if r.status_code == 403:
                    # Plan doesn't cover this date range — skip silently
                    return []  # caller treats empty as no-data month

                if r.status_code in (500, 502, 503, 504):
                    t = backoff + random.random()
                    _safe_print(f"  Server {r.status_code}, retry in {t:.1f}s...")
                    time.sleep(t)
                    backoff *= 2
                    continue

                # Non-retriable
                raise RuntimeError(f"{ticker} {date_from}: HTTP {r.status_code} {r.text[:200]}")

            except requests.exceptions.Timeout:
                time.sleep(backoff)
                backoff *= 2
            except requests.exceptions.RequestException as e:
                _safe_print(f"  Network error: {e}")
                time.sleep(backoff)
                backoff *= 2
        else:
            raise RuntimeError(f"{ticker} {date_from}: failed after {max_retries} retries (page {page})")

        if not next_url:
            break  # no more pages

    return all_results


# ── Progress tracking ─────────────────────────────────────────────────────────

def load_progress(progress_file):
    if Path(progress_file).exists():
        df = pd.read_csv(progress_file)
        done = set(zip(df["ticker"].astype(str), df["year_month"].astype(str)))
        _safe_print(f"Resuming: {len(done)} ticker-months already fetched")
        return done
    return set()


def save_progress(progress_file, ticker, year, month):
    ym = f"{year}-{month:02d}"
    with open(progress_file, "a") as f:
        if not Path(progress_file).exists() or os.path.getsize(progress_file) == 0:
            f.write("ticker,year_month\n")
        f.write(f"{ticker},{ym}\n")


# ── Month iterator ────────────────────────────────────────────────────────────

def iter_months(start_date: str, end_date: str):
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date,   "%Y-%m-%d")
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extended OHLCV fetcher (monthly batches)")
    parser.add_argument("--out",      default="ohlcv_5min_extended.parquet")
    parser.add_argument("--progress", default="fetch_progress_extended.csv")
    parser.add_argument("--start",    default=DEFAULT_START)
    parser.add_argument("--end",      default=DEFAULT_END)
    parser.add_argument("--tickers",  nargs="+", default=TICKERS)
    args = parser.parse_args()

    _safe_print("=" * 64)
    _safe_print("Extended OHLCV Fetcher - monthly batch requests")
    _safe_print("=" * 64)
    _safe_print(f"Tickers ({len(args.tickers)}): {', '.join(args.tickers)}")
    _safe_print(f"Range  : {args.start} to {args.end}")
    months = list(iter_months(args.start, args.end))
    total_calls = len(args.tickers) * len(months)
    _safe_print(f"Calls  : {len(months)} months × {len(args.tickers)} tickers = {total_calls} API calls")
    print()

    key_mgr = APIKeyManager()
    done    = load_progress(args.progress)

    rows = []
    if Path(args.out).exists() and done:
        _safe_print(f"Loading existing data from {args.out}...")
        existing = pd.read_parquet(args.out)
        rows = existing.to_dict("records")
        _safe_print(f"  {len(rows)} existing bars loaded")

    total_fetched = 0
    start_wall = time.time()

    for ticker in args.tickers:
        _safe_print(f"\n=== {ticker} ===")
        ticker_bars = 0

        for year, month in months:
            ym_key = f"{year}-{month:02d}"
            if (ticker, ym_key) in done:
                continue

            try:
                bars = fetch_month(ticker, year, month, key_mgr)
            except Exception as e:
                _safe_print(f"  ERROR {ticker} {ym_key}: {e}")
                time.sleep(2)
                # Still mark as done to avoid infinite retry; rerun separately if needed
                save_progress(args.progress, ticker, year, month)
                done.add((ticker, ym_key))
                continue

            for b in bars:
                rows.append({
                    "ticker":    ticker,
                    "timestamp": pd.to_datetime(b["t"], unit="ms", utc=True),
                    "open":  b["o"],
                    "high":  b["h"],
                    "low":   b["l"],
                    "close": b["c"],
                    "volume": int(b["v"]),
                })

            ticker_bars  += len(bars)
            total_fetched += 1

            save_progress(args.progress, ticker, year, month)
            done.add((ticker, ym_key))

            # Light throttle — keys rotate, but be polite
            time.sleep(0.12)

            if total_fetched % 50 == 0:
                elapsed = time.time() - start_wall
                rate    = total_fetched / elapsed
                pct     = total_fetched / max(total_calls, 1) * 100
                _safe_print(f"  {pct:.1f}% | {total_fetched}/{total_calls} calls | {rate:.1f}/s | {len(rows):,} bars")

        _safe_print(f"  {ticker}: {ticker_bars:,} bars fetched")

    # Build and save DataFrame
    df = pd.DataFrame(rows).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    _safe_print(f"\nTotal bars: {len(df):,}")
    df.to_parquet(args.out, index=False)
    _safe_print(f"Saved → {args.out}")

    elapsed = time.time() - start_wall
    _safe_print(f"Completed in {elapsed/60:.1f} minutes")
    key_mgr.stats()

    _safe_print("\n" + "=" * 64)
    _safe_print("Bars per ticker:")
    _safe_print("=" * 64)
    summary = df.groupby("ticker").agg(
        bars=("timestamp", "count"),
        first=("timestamp", "min"),
        last=("timestamp", "max"),
    )
    _safe_print(summary.to_string())
    _safe_print(f"\nExpected: ~78 bars/day x ~252 days/year x 2 years = ~39,312 bars/ticker")

    if Path(args.progress).exists() and total_fetched == total_calls:
        os.remove(args.progress)
        _safe_print("\nProgress file cleaned up.")


if __name__ == "__main__":
    main()
