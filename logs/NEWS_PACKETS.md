# News Packets: Build Log & Analysis

**Date:** 2026-02-15
**Script:** `build_news_packets.py`
**Output:** `out_news/`

---

## Run Summary

| Metric | Value |
|--------|-------|
| Events processed | 100 |
| Events with ticker news | 59 (59%) |
| Events with no news | 41 (41%) |
| Total articles fetched | 146 |
| 429 rate-limit retries | 15 (all recovered) |
| 5xx retries | 0 |
| Errors | 0 |
| API keys used | 8 (~31 requests each, perfectly balanced) |

### Articles Per Ticker

| Ticker | Articles | Events | Coverage |
|--------|----------|--------|----------|
| NVDA | 46 | 10 | 100% |
| MSFT | 28 | 11 | 91% |
| AAPL | 20 | 10 | 80% |
| AMD | 16 | 15 | 60% |
| TSLA | 12 | 10 | 60% |
| META | 11 | 8 | 75% |
| NFLX | 5 | 14 | 29% |
| PLTR | 4 | 9 | 33% |
| QQQ | 3 | 6 | 33% |
| SPY | 1 | 3 | 33% |
| XLK | 0 | 4 | 0% |

### Configuration

- Lookback: 3 hours before t0
- Lookahead: 1 hour after t0
- Max articles per query: 30
- Macro news: enabled (SPY, QQQ per event)
- Cache: `out_news/news_cache.jsonl` (233 entries)

---

## Output Structure

```
out_news/
├── news_cache.jsonl          # 233 cached query results
├── news_summary.json         # Run stats
└── packets/                  # Per-event JSON packets
    ├── AAPL/
    │   ├── AAPL_20251031_133000.json
    │   ├── AAPL_20251103_143000.json
    │   └── ...
    ├── AMD/
    ├── META/
    ├── MSFT/
    ├── NFLX/
    ├── NVDA/
    ├── PLTR/
    ├── QQQ/
    ├── SPY/
    ├── TSLA/
    └── XLK/
```

### Packet Schema

```json
{
  "event": {
    "event_id": "NVDA_20251124_144000",
    "ticker": "NVDA",
    "t0_utc": "2025-11-24 14:40:00+00:00",
    "direction": -1,
    "label_proxy": "reversal",
    "window": {
      "start": "2025-11-24T11:40:00+00:00",
      "end": "2025-11-24T15:40:00+00:00",
      "lookback_hours": 3.0,
      "lookahead_hours": 1.0
    }
  },
  "news": {
    "ticker_news": [ { "published_utc", "title", "description", "article_url", "publisher", "author", "tickers", "keywords" } ],
    "ticker_news_count": 12,
    "ticker_news_truncated": false,
    "no_news_found": false,
    "macro_news": {
      "SPY": [...],
      "QQQ": [...]
    }
  }
}
```

---

## No-News Analysis (41 of 100 events)

### Root Causes

#### 1. Opening Bell Dominance (76% of no-news events)
- **31 out of 41** no-news events fire at ~14:00 UTC (10:00 AM ET market open)
- These are gap-open reactions to overnight catalysts (earnings after close, pre-market analyst notes, Asia/Europe spillover)
- The news window (t0 - 3h = 7 AM ET) misses overnight articles that drove the move
- This is expected behavior, not a data quality issue

#### 2. Ticker Coverage Hierarchy

| Coverage | Tickers | No-News Rate |
|----------|---------|-------------|
| Excellent | MSFT (9%), AAPL (20%), META (25%) | Mega-cap analyst coverage |
| Moderate | AMD (40%), TSLA (40%) | Good but not exhaustive |
| Poor | NFLX (71%), PLTR (67%) | Limited specialized coverage |
| Very Poor | QQQ (67%), SPY (67%), XLK (100%) | ETFs = structurally sparse |

#### 3. No-News Events Are Stronger
- No-news avg return: **192.6 bp** vs news events: **169.9 bp** (+13.4%)
- These are genuine spikes driven by technical/momentum/gap factors

#### 4. Day-of-Week Pattern
- Monday: 27.6% no-news (most news flow)
- Friday: 64.3% no-news (2.3x Monday's rate)

### Recommendations
1. **Keep all 41 no-news events** in training data - they're valid signals
2. Add ML features: `is_opening_bell`, `has_news`, `hour_of_day_utc`, `is_etf`
3. For production: extend lookback to -6h for opening-bell events
4. "No catalyst found" is itself informative for regime classification

---

## CLI Reference

```bash
# Default run (uses cache)
python build_news_packets.py

# Force refresh all
python build_news_packets.py --force_refresh

# Custom windows
python build_news_packets.py --lookback_hours 6 --lookahead_hours 2

# Disable macro news
python build_news_packets.py --include_macro_news false

# Custom events file
python build_news_packets.py --events_csv path/to/events.csv --out_dir out_news_custom
```
