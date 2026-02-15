"""
build_news_packets.py  –  Attach Polygon.io news to curated events
==================================================================
For each event (ticker, t0_utc) in the curated CSV, fetches relevant news
from Polygon within a configurable time window and saves:
  1. A persistent news cache  (out_dir/news_cache.jsonl)
  2. Per-event JSON packets    (out_dir/packets/{ticker}/{ticker}_{ts}.json)
  3. A summary report          (out_dir/news_summary.json)

Uses 8 Polygon API keys with round-robin rotation + exponential backoff on 429/5xx.

Assumptions:
  - Events CSV has columns: ticker, t0_utc (ISO with tz)
  - .env file has POLYGON_API_KEY_1..8
  - Macro news (SPY, QQQ) is fetched per event window unless disabled
  - Articles are de-duped by article_url, capped at --max_articles per query
"""

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import time

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Key Rotator
# ---------------------------------------------------------------------------

class KeyRotator:
    """Round-robin API key rotator with backoff tracking."""

    def __init__(self):
        load_dotenv()
        self.keys = []
        for i in range(1, 20):
            k = os.getenv(f"POLYGON_API_KEY_{i}", "")
            if k and not k.startswith("your_api_key"):
                self.keys.append(k)
        if not self.keys:
            raise RuntimeError("No valid POLYGON_API_KEY_* found in .env")
        self.idx = 0
        self.usage = {i: 0 for i in range(len(self.keys))}

    def next_key(self):
        key = self.keys[self.idx]
        self.usage[self.idx] += 1
        self.idx = (self.idx + 1) % len(self.keys)
        return key

    def summary(self):
        return {f"key_{i+1}": count for i, count in self.usage.items()}


# ---------------------------------------------------------------------------
# Polygon News Fetcher
# ---------------------------------------------------------------------------

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
MAX_RETRIES = 8


def fetch_news(ticker, start_iso, end_iso, limit, rotator, stats):
    """Fetch all news articles for a ticker within [start, end] window.

    Handles pagination via next_url. Returns list of article dicts.
    """
    articles = []
    params = {
        "ticker": ticker,
        "published_utc.gte": start_iso,
        "published_utc.lte": end_iso,
        "limit": min(limit, 50),
        "order": "asc",
        "apiKey": rotator.next_key(),
    }
    url = POLYGON_NEWS_URL

    while True:
        data = _request_with_retry(url, params, rotator, stats)
        if data is None:
            break

        results = data.get("results", [])
        articles.extend(results)

        # Pagination
        next_url = data.get("next_url")
        if not next_url or len(articles) >= limit:
            break
        # next_url already has query params; just need to add apiKey
        url = next_url
        params = {"apiKey": rotator.next_key()}

    return articles[:limit]


def _request_with_retry(url, params, rotator, stats):
    """Execute GET with exponential backoff on 429 / 5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            stats["errors"] += 1
            print(f"  [NET ERROR] {exc} (attempt {attempt+1})")
            time.sleep(2 ** attempt + random.random())
            params["apiKey"] = rotator.next_key()
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            stats["retries_429"] += 1
            wait = (2 ** attempt) + random.random()
            print(f"  [429] Rate limited. Waiting {wait:.1f}s (attempt {attempt+1})")
            time.sleep(wait)
            params["apiKey"] = rotator.next_key()
            continue

        if resp.status_code >= 500:
            stats["retries_5xx"] += 1
            wait = (2 ** attempt) + random.random()
            print(f"  [5xx] Server error {resp.status_code}. Waiting {wait:.1f}s")
            time.sleep(wait)
            params["apiKey"] = rotator.next_key()
            continue

        # Other client errors (400, 403, etc.) – log and give up
        stats["errors"] += 1
        print(f"  [HTTP {resp.status_code}] {resp.text[:200]}")
        return None

    stats["errors"] += 1
    print(f"  [FAIL] Gave up after {MAX_RETRIES} retries")
    return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(cache_path):
    """Load JSONL cache into a dict keyed by (ticker, start, end)."""
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["ticker"], rec["start_iso"], rec["end_iso"])
                cache[key] = rec["articles"]
    return cache


def append_cache(cache_path, ticker, start_iso, end_iso, articles, key_idx):
    """Append one query result to the JSONL cache."""
    rec = {
        "ticker": ticker,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "fetch_utc": dt.datetime.utcnow().isoformat(),
        "key_idx": key_idx,
        "article_count": len(articles),
        "articles": articles,
    }
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------
# Article cleaning
# ---------------------------------------------------------------------------

def clean_articles(articles, max_articles):
    """De-duplicate by article_url, sort by published_utc, cap at max."""
    seen = set()
    unique = []
    for a in articles:
        url = a.get("article_url", "")
        dedup_key = url if url else (a.get("title", "") + a.get("published_utc", ""))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique.append({
            "published_utc": a.get("published_utc", ""),
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "article_url": a.get("article_url", ""),
            "publisher": a.get("publisher", {}).get("name", "") if isinstance(a.get("publisher"), dict) else str(a.get("publisher", "")),
            "author": a.get("author", ""),
            "tickers": a.get("tickers", []),
            "keywords": a.get("keywords", []),
        })
    unique.sort(key=lambda x: x.get("published_utc", ""))
    truncated = len(unique) > max_articles
    return unique[:max_articles], truncated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Polygon news for curated events")
    parser.add_argument("--events_csv", default="out_final/selected_events.csv")
    parser.add_argument("--out_dir", default="out_news")
    parser.add_argument("--lookback_hours", type=float, default=3.0)
    parser.add_argument("--lookahead_hours", type=float, default=1.0)
    parser.add_argument("--max_articles", type=int, default=30)
    parser.add_argument("--include_macro_news", type=str, default="true")
    parser.add_argument("--macro_tickers", type=str, default="SPY,QQQ")
    parser.add_argument("--force_refresh", action="store_true")
    args = parser.parse_args()

    include_macro = args.include_macro_news.lower() in ("true", "1", "yes")
    macro_tickers = [t.strip() for t in args.macro_tickers.split(",") if t.strip()]

    # Load events
    events_df = pd.read_csv(args.events_csv)
    print(f"Loaded {len(events_df)} events from {args.events_csv}")

    # Setup output dirs
    out_dir = pathlib.Path(args.out_dir)
    packets_dir = out_dir / "packets"
    out_dir.mkdir(parents=True, exist_ok=True)
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Cache
    cache_path = out_dir / "news_cache.jsonl"
    cache = {} if args.force_refresh else load_cache(cache_path)
    print(f"Cache: {len(cache)} entries loaded")

    # Key rotator
    rotator = KeyRotator()
    print(f"API keys available: {len(rotator.keys)}")

    # Stats
    stats = {
        "retries_429": 0,
        "retries_5xx": 0,
        "errors": 0,
        "total_articles": 0,
        "events_with_news": 0,
        "events_no_news": 0,
        "events_failed": 0,
        "articles_per_ticker": {},
    }

    lookback = dt.timedelta(hours=args.lookback_hours)
    lookahead = dt.timedelta(hours=args.lookahead_hours)

    for idx, row in events_df.iterrows():
        ticker = row["ticker"]
        t0_str = row["t0_utc"]
        t0 = pd.Timestamp(t0_str)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")

        start = (t0 - lookback).isoformat()
        end = (t0 + lookahead).isoformat()

        event_id = row.get("event_id", f"{ticker}_{t0.strftime('%Y%m%d_%H%M%S')}")
        print(f"\n[{idx+1}/{len(events_df)}] {event_id}  ({start[:19]} -> {end[:19]})")

        # --- Ticker news ---
        cache_key = (ticker, start, end)
        if cache_key in cache:
            ticker_articles = cache[cache_key]
            print(f"  Ticker news: {len(ticker_articles)} (cached)")
        else:
            ticker_articles = fetch_news(ticker, start, end, args.max_articles, rotator, stats)
            append_cache(cache_path, ticker, start, end, ticker_articles, rotator.idx)
            cache[cache_key] = ticker_articles
            print(f"  Ticker news: {len(ticker_articles)} fetched")
            time.sleep(0.25)  # small courtesy delay between events

        ticker_articles, ticker_truncated = clean_articles(ticker_articles, args.max_articles)

        # --- Macro news ---
        macro_news = {}
        if include_macro:
            for mt in macro_tickers:
                if mt == ticker:
                    # skip if event ticker IS the macro ticker
                    macro_news[mt] = ticker_articles
                    continue
                mk = (mt, start, end)
                if mk in cache:
                    m_articles = cache[mk]
                    print(f"  {mt} news: {len(m_articles)} (cached)")
                else:
                    m_articles = fetch_news(mt, start, end, args.max_articles, rotator, stats)
                    append_cache(cache_path, mt, start, end, m_articles, rotator.idx)
                    cache[mk] = m_articles
                    print(f"  {mt} news: {len(m_articles)} fetched")
                    time.sleep(0.25)
                cleaned, _ = clean_articles(m_articles, args.max_articles)
                macro_news[mt] = cleaned

        # --- Build packet ---
        has_news = len(ticker_articles) > 0
        packet = {
            "event": {
                "event_id": event_id,
                "ticker": ticker,
                "t0_utc": str(t0),
                "direction": int(row.get("direction", 0)),
                "label_proxy": row.get("label_proxy", ""),
                "window": {
                    "start": start,
                    "end": end,
                    "lookback_hours": args.lookback_hours,
                    "lookahead_hours": args.lookahead_hours,
                },
            },
            "news": {
                "ticker_news": ticker_articles,
                "ticker_news_count": len(ticker_articles),
                "ticker_news_truncated": ticker_truncated,
                "no_news_found": not has_news,
            },
        }
        if include_macro:
            packet["news"]["macro_news"] = macro_news

        # Save packet
        ticker_dir = packets_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        ts_safe = t0.strftime("%Y%m%d_%H%M%S")
        packet_path = ticker_dir / f"{ticker}_{ts_safe}.json"
        with open(packet_path, "w", encoding="utf-8") as f:
            json.dump(packet, f, indent=2, default=str)

        # Update stats
        if has_news:
            stats["events_with_news"] += 1
        else:
            stats["events_no_news"] += 1
        stats["total_articles"] += len(ticker_articles)
        stats["articles_per_ticker"][ticker] = stats["articles_per_ticker"].get(ticker, 0) + len(ticker_articles)

    # --- Summary ---
    summary = {
        "total_events": len(events_df),
        "events_with_news": stats["events_with_news"],
        "events_no_news": stats["events_no_news"],
        "total_articles_fetched": stats["total_articles"],
        "articles_per_ticker": stats["articles_per_ticker"],
        "retries_429": stats["retries_429"],
        "retries_5xx": stats["retries_5xx"],
        "errors": stats["errors"],
        "api_key_usage": rotator.summary(),
        "config": {
            "lookback_hours": args.lookback_hours,
            "lookahead_hours": args.lookahead_hours,
            "max_articles": args.max_articles,
            "include_macro_news": include_macro,
            "macro_tickers": macro_tickers,
            "force_refresh": args.force_refresh,
        },
    }

    summary_path = out_dir / "news_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Events processed:  {summary['total_events']}")
    print(f"  With news:         {summary['events_with_news']}")
    print(f"  No news:           {summary['events_no_news']}")
    print(f"  Total articles:    {summary['total_articles_fetched']}")
    print(f"  429 retries:       {summary['retries_429']}")
    print(f"  5xx retries:       {summary['retries_5xx']}")
    print(f"  Errors:            {summary['errors']}")
    print(f"  Summary saved:     {summary_path}")
    print(f"  Packets dir:       {packets_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
