"""
OHLCV Data Fetcher with API Key Rotation
Fetches 5-minute OHLCV data from Polygon.io for specified tickers.
Supports multiple API keys with automatic rotation on rate limits.
"""

import os
import time
import random
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMD", "NFLX", "PLTR", "SPY", "QQQ", "XLK"]
START_DATE = "2025-10-15"
END_DATE = "2026-01-15"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
OUTPUT_FILE = "ohlcv_5min.parquet"
PROGRESS_FILE = "fetch_progress.csv"

class APIKeyManager:
    """Manages multiple API keys with rotation on rate limits."""

    def __init__(self):
        self.keys = self._load_keys()
        self.current_index = 0
        self.rate_limited_keys = {}  # key -> cooldown_until timestamp
        self.request_counts = {k: 0 for k in self.keys}

        if not self.keys:
            raise ValueError("No API keys found in .env file!")

        print(f"Loaded {len(self.keys)} API keys")

    def _load_keys(self):
        """Load all POLYGON_API_KEY_* from environment."""
        keys = []
        for i in range(1, 20):  # Support up to 20 keys
            key = os.getenv(f"POLYGON_API_KEY_{i}")
            if key and key != f"your_api_key_{i}_here" and not key.startswith("your_"):
                keys.append(key)
        return keys

    def get_key(self):
        """Get the next available API key, rotating if needed."""
        now = time.time()

        # Try to find an available key
        for _ in range(len(self.keys)):
            key = self.keys[self.current_index]

            # Check if key is in cooldown
            if key in self.rate_limited_keys:
                if now >= self.rate_limited_keys[key]:
                    # Cooldown expired
                    del self.rate_limited_keys[key]
                    return key
                else:
                    # Still in cooldown, try next key
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    continue

            return key

        # All keys are rate limited - wait for the soonest one
        if self.rate_limited_keys:
            soonest = min(self.rate_limited_keys.values())
            wait_time = soonest - now + 1
            if wait_time > 0:
                print(f"All keys rate-limited. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            # Clear expired cooldowns
            self.rate_limited_keys = {k: v for k, v in self.rate_limited_keys.items() if v > time.time()}

        return self.keys[self.current_index]

    def mark_rate_limited(self, key, cooldown_seconds=60):
        """Mark a key as rate limited with a cooldown period."""
        self.rate_limited_keys[key] = time.time() + cooldown_seconds
        print(f"Key {self.keys.index(key)+1} rate-limited, rotating to next key...")
        self.current_index = (self.current_index + 1) % len(self.keys)

    def record_request(self, key):
        """Record a successful request for a key."""
        self.request_counts[key] = self.request_counts.get(key, 0) + 1

    def print_stats(self):
        """Print usage statistics."""
        print("\nAPI Key Usage Statistics:")
        for i, key in enumerate(self.keys):
            count = self.request_counts.get(key, 0)
            print(f"  Key {i+1}: {count} requests")


def daterange(start_date: str, end_date: str):
    """Generate dates from start to end."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_5min_day(ticker: str, day: str, key_manager: APIKeyManager, max_retries: int = 8):
    """Fetch 5-minute bars for a single ticker and day."""
    url = f"{BASE_URL}/{ticker}/range/5/minute/{day}/{day}"

    backoff = 1.0
    for attempt in range(max_retries):
        api_key = key_manager.get_key()
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

        try:
            r = requests.get(url, params=params, timeout=30)

            if r.status_code == 200:
                key_manager.record_request(api_key)
                return r.json()

            # Rate limit - rotate key
            if r.status_code == 429:
                key_manager.mark_rate_limited(api_key, cooldown_seconds=60)
                continue

            # Server errors - retry with backoff
            if r.status_code in (500, 502, 503, 504):
                sleep_s = backoff + random.random()
                print(f"Server error {r.status_code}, retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                backoff *= 2
                continue

            # Other errors
            raise RuntimeError(f"{ticker} {day}: HTTP {r.status_code} {r.text[:200]}")

        except requests.exceptions.Timeout:
            print(f"Timeout for {ticker} {day}, retrying...")
            time.sleep(backoff)
            backoff *= 2
            continue
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}, retrying...")
            time.sleep(backoff)
            backoff *= 2
            continue

    raise RuntimeError(f"{ticker} {day}: failed after {max_retries} retries")


def load_progress():
    """Load progress from previous run if exists."""
    if Path(PROGRESS_FILE).exists():
        df = pd.read_csv(PROGRESS_FILE)
        completed = set(zip(df['ticker'], df['day']))
        print(f"Resuming from previous run. {len(completed)} ticker-days already fetched.")
        return completed
    return set()


def save_progress(ticker, day):
    """Save progress for resume capability."""
    with open(PROGRESS_FILE, 'a') as f:
        if not Path(PROGRESS_FILE).exists() or os.path.getsize(PROGRESS_FILE) == 0:
            f.write("ticker,day\n")
        f.write(f"{ticker},{day}\n")


def main():
    print("=" * 60)
    print("OHLCV Data Fetcher with API Key Rotation")
    print("=" * 60)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    # Initialize API key manager
    key_manager = APIKeyManager()

    # Load any existing progress
    completed = load_progress()

    # Load existing data if resuming
    rows = []
    if Path(OUTPUT_FILE).exists() and completed:
        print(f"Loading existing data from {OUTPUT_FILE}...")
        existing_df = pd.read_parquet(OUTPUT_FILE)
        rows = existing_df.to_dict('records')
        print(f"Loaded {len(rows)} existing rows")

    missing_days = []
    total_requests = 0

    # Calculate total expected requests
    trading_days = sum(1 for d in daterange(START_DATE, END_DATE) if d.weekday() < 5)
    total_expected = len(TICKERS) * trading_days
    already_done = len(completed)
    remaining = total_expected - already_done

    print(f"\nTotal expected requests: {total_expected}")
    print(f"Already completed: {already_done}")
    print(f"Remaining: {remaining}")
    print()

    start_time = time.time()

    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        ticker_rows = 0

        for d in daterange(START_DATE, END_DATE):
            # Skip weekends
            if d.weekday() >= 5:
                continue

            day = d.strftime("%Y-%m-%d")

            # Skip if already completed
            if (ticker, day) in completed:
                continue

            try:
                data = fetch_5min_day(ticker, day, key_manager)
                total_requests += 1

            except Exception as e:
                print(f"ERROR: {e}")
                missing_days.append((ticker, day, "request_failed"))
                time.sleep(2)
                continue

            results = data.get("results", [])
            if not results:
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
                ticker_rows += len(results)

            # Save progress
            save_progress(ticker, day)
            completed.add((ticker, day))

            # Progress update every 10 requests
            if total_requests % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_requests / elapsed if elapsed > 0 else 0
                print(f"  Progress: {total_requests}/{remaining} requests, {rate:.1f} req/s, {len(rows)} total bars")

            # Gentle throttle
            time.sleep(0.15)

        print(f"  {ticker}: {ticker_rows} new bars fetched")

    # Create and save DataFrame
    df = pd.DataFrame(rows).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    print(f"\nTotal rows: {len(df)}")
    df.to_parquet(OUTPUT_FILE, index=False)

    # Save missing days
    if missing_days:
        md = pd.DataFrame(missing_days, columns=["ticker", "day", "reason"])
        md.to_csv("missing_days.csv", index=False)
        print(f"Wrote missing_days.csv with {len(md)} rows")

    # Print statistics
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Saved {OUTPUT_FILE}")

    key_manager.print_stats()

    # Sanity check
    print("\n" + "=" * 60)
    print("SANITY CHECK - Bars per ticker:")
    print("=" * 60)
    print(df.groupby("ticker").size().sort_values())
    print("\nExpected: ~78 bars/day * ~60 trading days = ~4680 bars per ticker")

    # Cleanup progress file on successful completion
    if not missing_days:
        if Path(PROGRESS_FILE).exists():
            os.remove(PROGRESS_FILE)
            print("\nProgress file cleaned up (all data fetched successfully)")


if __name__ == "__main__":
    main()
