# Quick Start for New Claude Instance

## TL;DR
This is an intraday event mining project. Data is fetched, events are mined, ready for ML.

---

## Current State: READY FOR NEXT PHASE

### What's Done
1. **Data collected:** 126,719 5-min bars for 11 tickers over 3 months
2. **Events mined:** 451 high-quality events using two-stage detection
3. **Labels assigned:** continuation/reversal/unclear based on 60-min forward return

### What's Next (User Choice)
- Manual event curation with plots
- Feature engineering for ML
- Classifier training
- Backtesting

---

## Quick Commands

```bash
cd C:\Users\Samarth\Desktop\TradingAgents

# See current events
python -c "import pandas as pd; df=pd.read_csv('out_strict/events.csv'); print(f'Events: {len(df)}'); print(df.head())"

# Generate event plots
python mine_events_strict.py --out_dir out_strict --make_plots

# Adjust thresholds
python mine_events_strict.py --confirm_abs_bp 80 --out_dir out_strict  # stricter
python mine_events_strict.py --confirm_abs_bp 50 --out_dir out_strict  # looser
```

---

## Key Files

| File | What It Is |
|------|------------|
| `ohlcv_5min.parquet` | Raw OHLCV data (126k rows) |
| `out_strict/events.csv` | Mined events (451 rows) |
| `mine_events_strict.py` | Main event mining script |
| `.env` | 8 Polygon API keys |
| `logs/` | Full documentation |

---

## Read First
1. `logs/PROJECT_STATUS.md` - Overall status
2. `logs/NEXT_STEPS.md` - Suggested tasks
3. `logs/SESSION_LOG.md` - What was done

---

## Data Schema Quick Reference

**OHLCV:** ticker, timestamp, open, high, low, close, volume

**Events:** event_id, ticker, t0_utc, direction, cluster_len, peak_abs_ret_3_bp, peak_z_ret_3, peak_volume_mult, peak_range_mult, peak_impact_mult, event_window_start, event_window_end, val_window_start, val_window_end, event_return, forward_return_60m, label_proxy

---

## Event Detection Logic

```
Stage 1 (Candidate): High recall
- abs_ret_3 >= 30bp AND (z_ret_3 > 3 OR vol > 4x OR range > 3x OR impact > 4x)

Stage 2 (Confirm): High precision
- abs_ret_3 >= 60bp
- z_ret_3 >= 4.0
- vol >= 6x OR range >= 4x OR impact >= 6x
- cluster_len >= 2 OR ret_6 persists
```

---

## Stats

- **Total events:** 451
- **Label split:** 170 continuation, 137 reversal, 144 unclear
- **Top event:** META 868bp move on 2025-10-29 (earnings)
- **Median move:** 124bp
- **Median volume spike:** 224x
