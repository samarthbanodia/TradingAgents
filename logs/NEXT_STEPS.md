# Next Steps for Continuation

This document outlines suggested next steps for continuing the TradingAgents project.

---

## Immediate Options

### Option A: Manual Event Curation
The 451 events are ready for manual review to create a "golden set":
1. Generate plots: `python mine_events_strict.py --out_dir out_strict --make_plots`
2. Review `out_strict/plots/{ticker}/` folders
3. Mark events as "good" or "bad" for training

### Option B: Threshold Tuning
Adjust thresholds to get different event counts:
```bash
# Fewer events (~200-300)
python mine_events_strict.py --confirm_abs_bp 80 --confirm_ret_z 5.0 --out_dir out_strict

# More events (~600-800)
python mine_events_strict.py --confirm_abs_bp 50 --confirm_ret_z 3.5 --out_dir out_strict
```

### Option C: Build ML Pipeline
Use the events for downstream modeling.

---

## Suggested Next Tasks

### 1. Feature Engineering for ML
Extract features from event windows for training:
- Pre-event features (from event window)
- Microstructure features (volume profile, trade intensity)
- Technical indicators (RSI, MACD, Bollinger bands)

**Suggested script:** `extract_features.py`

### 2. Label Refinement
Current labels are proxy-based (forward return threshold). Options:
- Manual labeling of golden set
- Multi-class labels (strong reversal, weak reversal, neutral, continuation)
- Regression target (forward return magnitude)

### 3. Event Classifier
Build a classifier to predict event outcomes:
- Input: Event window features
- Output: Reversal/Continuation/Unclear probability
- Models: XGBoost, LightGBM, or neural network

**Suggested script:** `train_classifier.py`

### 4. Backtesting Framework
Simulate trading based on event predictions:
- Entry at t0, exit at val_window_end
- Risk management (stop loss, position sizing)
- Performance metrics (Sharpe, win rate, max drawdown)

**Suggested script:** `backtest.py`

### 5. Real-Time Event Detection
Adapt the miner for live detection:
- Stream 5-min bars from Polygon WebSocket
- Maintain rolling baselines
- Alert on confirmed events

**Suggested script:** `live_detector.py`

---

## Data Expansion Ideas

### More Tickers
Add more liquid tickers:
```python
# Mega caps
GOOG, AMZN, BRK.B, JPM, V, JNJ, UNH, PG, HD

# High-volatility
COIN, MARA, RIOT, GME, AMC

# Sector ETFs
XLF, XLE, XLV, XLI, XLP
```

### Longer History
Fetch more historical data:
```python
START_DATE = "2024-01-01"  # 2+ years
END_DATE = "2026-01-15"
```

### Higher Frequency
Consider 1-minute bars for more granular analysis (requires Polygon paid tier).

---

## Architecture for Production

```
┌─────────────────────────────────────────────────────────────────┐
│                     TradingAgents Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  Data    │──▶│  Event   │──▶│ Feature  │──▶│   ML     │     │
│  │ Fetcher  │   │  Miner   │   │ Extract  │   │ Predict  │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ Parquet  │   │ Events   │   │ Features │   │ Signals  │     │
│  │  Store   │   │   CSV    │   │   CSV    │   │   CSV    │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                                  │
│                        ┌──────────┐                             │
│                        │ Backtest │                             │
│                        │ / Trade  │                             │
│                        └──────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Run Event Miner
```bash
cd C:\Users\Samarth\Desktop\TradingAgents
python mine_events_strict.py --out_dir out_strict --make_plots
```

### Check Results
```bash
# Event count
wc -l out_strict/events.csv

# View summary
cat out_strict/summary.json

# View top events
head -20 out_strict/events.csv
```

### Fetch Fresh Data (if needed)
```bash
# Delete progress file if exists
del fetch_progress.csv

# Run fetcher
python fetch_ohlcv.py
```

---

## Known Issues / Limitations

1. **Extended hours data:** XLK has fewer bars due to lower ETF extended-hours trading
2. **Baseline warmup:** First 78 bars per ticker have NaN baselines (dropped from analysis)
3. **Label noise:** Proxy labels based on 60-min forward return may not capture true event outcomes
4. **Single-day baseline:** 78-bar window may miss regime changes; consider adaptive baselines

---

## Contact / Resources

- **Polygon API Docs:** https://polygon.io/docs
- **Project Folder:** `C:\Users\Samarth\Desktop\TradingAgents`
- **API Keys:** Stored in `.env` file (8 keys available)
