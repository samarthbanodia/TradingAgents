# TradingAgents Project Status Log

**Last Updated:** 2026-02-15
**Session Summary:** Built complete intraday event mining pipeline + news attachment

---

## Project Goal
Build a regime-attribution system that detects abnormal intraday events ("outliers") from 5-minute OHLCV data, creates event/validation windows, and labels them for downstream ML training.

---

## Completed Tasks

### 1. Data Collection (COMPLETE)
- **Script:** `fetch_ohlcv.py`
- **Output:** `ohlcv_5min.parquet` and `ohlcv_5min.csv`
- **Data:**
  - 11 tickers: AAPL, MSFT, NVDA, TSLA, META, AMD, NFLX, PLTR, SPY, QQQ, XLK
  - Date range: 2025-10-15 to 2026-01-15 (~3 months)
  - Total rows: 126,719 bars
  - Columns: ticker, timestamp, open, high, low, close, volume
- **Features:**
  - API key rotation (8 keys supported)
  - Automatic retry on rate limits
  - Resume capability via progress file
  - Handles market holidays gracefully

### 2. Event Miner v1 - Loose (COMPLETE)
- **Script:** `mine_events.py`
- **Output:** `out/events.csv`, `out/summary.json`
- **Results:** 7,640 events (too many for manual curation)
- **Method:** Single-stage trigger detection with 20-bar baselines
- **Thresholds:** ret_z > 2.5, vol > 3x, range > 2x, impact > 3x

### 3. Event Miner v2.1 - Strict with RTH (COMPLETE)
- **Script:** `mine_events_strict.py`
- **Output:** `out_rth/events.csv`, `out_rth/summary.json`, `out_rth/plots/`
- **Results:** 34 high-quality events (with lower thresholds)
- **Key Features (v2.1):**
  - RTH-only filtering (09:30-16:00 ET) by default
  - `--include_extended_hours` flag for 04:00-20:00 ET
  - Bar-count based windows (not time deltas)
  - Same-day window validation (no cross-day events)
  - Gap checking for missing bars
  - Fixed plotting with bar indices (no blank regions)
  - Clusters cannot cross trading days

---

## Current File Structure

```
TradingAgents/
├── .env                      # 8 Polygon API keys (KEEP PRIVATE)
├── prompt.md                 # Original instructions
│
├── fetch_ohlcv.py           # Data fetcher with API key rotation
├── convert_to_csv.py        # Parquet to CSV converter
├── mine_events.py           # Event Miner v1 (loose, 7.6k events)
├── mine_events_strict.py    # Event Miner v2.1 (strict, RTH-only)
│
├── ohlcv_5min.parquet       # Main dataset (126,719 rows)
├── ohlcv_5min.csv           # Same data as CSV
├── missing_days.csv         # Market holidays log
│
├── out/                     # Event Miner v1 output
│   ├── events.csv           # 7,640 events
│   └── summary.json
│
├── out_strict/              # Event Miner v2 output (extended hours)
│   ├── events.csv           # 451 events
│   └── summary.json
│
├── out_rth/                 # Event Miner v2.1 output (RTH only)
│   ├── events.csv           # 34 events (with lower thresholds)
│   ├── summary.json
│   └── plots/               # Event visualization PNGs
│       ├── AAPL/
│       ├── AMD/
│       └── ...
│
└── logs/                    # This folder
    ├── PROJECT_STATUS.md    # This file
    ├── DATA_SCHEMA.md       # Data column definitions
    ├── EVENTS_SCHEMA.md     # Events output schema
    ├── NEXT_STEPS.md        # Suggested next tasks
    ├── SESSION_LOG.md       # Session history
    └── README_FOR_NEW_INSTANCE.md
```

---

## Key Statistics

### Dataset
- Total bars: 126,719 (all hours)
- RTH bars: 54,843 (after filtering)
- Tickers: 11
- Date range: 2025-10-15 to 2026-01-15

### Event Miner v2.1 Results (RTH, lower thresholds)
- Confirmed clusters: 263
- Events extracted: 34
- Skipped (event window crosses day): 222
- Skipped (val window crosses day): 7
- Label distribution: 18 continuation, 10 reversal, 6 unclear

### Top Events Captured
| Ticker | Date | Move (bp) | Z-Score | Volume Mult | Label |
|--------|------|-----------|---------|-------------|-------|
| NFLX | 2025-10-21 | 668 | 5.0 | 11.0x | continuation |
| AMD | 2025-11-19 | 379 | 4.3 | 0.4x | reversal |
| AMD | 2025-11-11 | 347 | 4.8 | 13.4x | reversal |
| NVDA | 2025-12-08 | 230 | 5.0 | 9.3x | reversal |
| AMD | 2025-12-02 | 223 | 3.8 | 5.6x | continuation |

---

## Commands Reference

```bash
# RTH only (default, recommended)
python mine_events_strict.py --out_dir out_rth --make_plots

# RTH with lower thresholds (more events)
python mine_events_strict.py --out_dir out_rth --make_plots --confirm_abs_bp 40 --confirm_ret_z 3.5

# Include extended hours
python mine_events_strict.py --include_extended_hours --out_dir out_extended --make_plots

# Stricter thresholds (fewer events)
python mine_events_strict.py --confirm_abs_bp 80 --confirm_ret_z 5.0 --out_dir out_strict
```

---

## v2.1 Fixes Applied

1. **RTH-Only Filtering:** Converts timestamps to America/New_York and keeps only 09:30-16:00 Mon-Fri
2. **Bar-Count Windows:** Event and validation windows defined by bar count, not time deltas
3. **Same-Day Validation:** Windows cannot cross trading day boundaries
4. **Gap Checking:** Events with gaps > 10min between bars are discarded
5. **Fixed Plotting:** Uses bar indices for x-axis to avoid blank overnight regions
6. **No Cross-Day Clusters:** Clustering respects trading day boundaries
