# Event Selection Report

**Script:** `select_best_events.py`
**Date:** 2026-02-09
**Purpose:** Trim 451 mined events to a high-signal, balanced set of 100 for the agentic debate system.

---

## Pipeline Summary

```
451 candidates (out_strict/events.csv)
  │
  ├─ Step 0: Hard quality gates
  │    ├─ RTH filter (09:30–16:00 ET, Mon–Fri)     → -239
  │    ├─ Window integrity (no cross-day)            → -0
  │    ├─ Gap check (max 10 min within windows)      → -1
  │    └─ Non-NaN required features                  → -0
  │
  ▼
211 eligible
  │
  ├─ Step 1: Strength scoring (per-ticker robust z)
  ├─ Step 2: Unclear eligibility tagging (23/55 pass)
  ├─ Step 4: De-duplication (60 min cooldown)        → -4
  │
  ▼
207 de-duped pool
  │
  ├─ Step 5, Phase 1: Fill per-ticker minimums       → 73 selected
  ├─ Step 5, Phase 2: Fill to target globally        → 100 selected
  │
  ▼
100 selected events (out_final/selected_events.csv)
351 rejected events (out_final/rejected_events.csv)
```

---

## Final Set Balance

| Metric | Value |
|--------|-------|
| **Total selected** | **100** |
| Continuation | 43 |
| Reversal | 42 |
| Unclear (high-strength only) | 15 (15.0%) |

---

## Per-Ticker Coverage

| Ticker | Type | Count |
|--------|------|-------|
| AMD | Stock | 15 |
| NFLX | Stock | 14 |
| MSFT | Stock | 11 |
| AAPL | Stock | 10 |
| NVDA | Stock | 10 |
| TSLA | Stock | 10 |
| PLTR | Stock | 9 |
| META | Stock | 8 |
| QQQ | ETF | 6 |
| XLK | ETF | 4 |
| SPY | ETF | 3 |

---

## Strength Score Distribution

| Quantile | Value |
|----------|-------|
| p10 | 0.6392 |
| p25 | 0.6645 |
| p50 (median) | 0.7299 |
| p75 | 0.7880 |
| p90 | 0.8286 |

---

## Keep Reason Breakdown

| Reason | Count | Description |
|--------|-------|-------------|
| min_quota | 72 | Selected to fill per-ticker minimum (clear outcome) |
| high_strength_unclear | 14 | Unclear label but passed high-strength threshold |
| top_global | 13 | Added in global fill phase by strength ranking |
| min_quota_unclear_fill | 1 | Unclear event used to fill a ticker's minimum quota |

---

## Rejection Reason Breakdown

| Reason | Count | Description |
|--------|-------|-------------|
| out_of_rth | 239 | Event t0 outside 09:30–16:00 ET |
| over_cap_or_below_cutoff | 75 | Ticker at max cap or below strength cutoff |
| low_strength_unclear | 32 | Unclear label and did not meet strength thresholds |
| near_duplicate | 4 | Another event within 60 min for same ticker had higher strength |
| window_gap_gt_10min | 1 | Gap > 10 min detected within event or validation window |

---

## Top 10 Events by Magnitude

| Ticker | Time (UTC) | Dir | Move (bp) | Z-Score | Strength | Label |
|--------|------------|-----|-----------|---------|----------|-------|
| META | 2025-10-29 19:45 | +1 | 867.9 | 5.11 | 0.6947 | reversal |
| NFLX | 2025-10-21 20:00 | -1 | 668.3 | 4.97 | 0.7362 | reversal |
| MSFT | 2025-10-29 19:55 | +1 | 385.9 | 4.29 | 0.6599 | reversal |
| AMD | 2025-11-19 21:00 | +1 | 378.7 | 4.32 | 0.6912 | continuation |
| PLTR | 2025-11-06 14:35 | +1 | 358.0 | 6.68 | 0.8800 | reversal |
| AMD | 2025-11-11 18:25 | -1 | 347.4 | 4.50 | 0.7156 | reversal |
| NFLX | 2025-10-30 19:55 | -1 | 321.4 | 5.14 | 0.7491 | reversal |
| AMD | 2025-11-14 14:30 | -1 | 320.7 | 4.61 | 0.7535 | reversal |
| AMD | 2025-11-21 14:30 | -1 | 312.3 | 4.07 | 0.6695 | continuation |
| AMD | 2026-01-13 14:30 | +1 | 289.4 | 4.33 | 0.8077 | reversal |

---

## Strength Score Formula

All signals are per-ticker robust-z scaled using median and IQR, then clipped to [0, 1]. This prevents volatile names (TSLA, NVDA, AMD) from dominating the ranking.

```
strength_score = 0.25 * rz(peak_abs_ret_3_bp)          # magnitude
               + 0.20 * rz(peak_z_ret_3)                # rarity vs baseline volatility
               + 0.20 * rz(max(volume_mult, range_mult)) # market attention
               + 0.15 * rz(peak_impact_mult)             # liquidity stress
               + 0.20 * min(cluster_len, 4) / 4          # persistence / shape
```

Where `rz(x)` = `clip(0.5 + (x - median) / (2 * IQR), 0, 1)` computed per ticker.

---

## Unclear Event Handling

`label_proxy` is **outcome-based** (what actually happened 60 minutes after the spike), not a prediction. It is used as a quality signal:

- **continuation / reversal** = clear outcome, high-signal for early evaluation
- **unclear** = ambiguous outcome, often low-signal noise but sometimes genuine uncertainty

**Policy:** Do not blindly drop unclear events. Keep an unclear event only if ANY of:
1. `strength_score` is in the top 30% within its ticker, OR
2. `peak_abs_ret_3_bp` >= global 75th percentile, OR
3. `peak_volume_mult` >= global 90th percentile, OR
4. `peak_impact_mult` >= global 90th percentile

**Cap:** Unclear events are capped at 15% of the final set (15 out of 100).

**Result:** 55 unclear events in the eligible pool, 23 passed the strength filter, 15 made it into the final set.

---

## De-Duplication

- **Per-ticker cooldown:** Two events from the same ticker within 60 minutes cannot both be selected; the higher strength_score wins.
- **Result:** 4 near-duplicates removed.

---

## Selection Algorithm

1. Apply hard quality gates (RTH, window integrity, gaps, non-NaN) to get the eligible pool
2. Compute per-ticker robust strength scores
3. Tag unclear events for eligibility (high-strength only)
4. De-duplicate within each ticker (60 min cooldown, keep stronger)
5. **Phase 1:** Fill each ticker's minimum quota (8/stock, 3/ETF) using clear outcomes first, then high-strength unclear if needed
6. **Phase 2:** Fill remaining slots to target (100) globally by strength_score, respecting max caps (15/stock, 8/ETF) and unclear cap (15%)
7. If over target, trim lowest strength while preserving per-ticker minimums

---

## Output Files

| File | Rows | Description |
|------|------|-------------|
| `out_final/selected_events.csv` | 100 | Selected events with `strength_score` and `keep_reason` |
| `out_final/rejected_events.csv` | 351 | All rejected events with `reject_reason` |
| `out_final/summary.json` | — | Full statistics, top 10, and rationale |

---

## CLI Reference

```bash
# Default run (100 events)
python select_best_events.py

# Stricter (80 events, higher bar)
python select_best_events.py --target_n 80 --min_per_stock 6

# Include extended hours
python select_best_events.py --include_extended_hours --out_dir out_final_ext

# All options
python select_best_events.py \
    --events_path out_strict/events.csv \
    --ohlcv_path ohlcv_5min.parquet \
    --out_dir out_final \
    --target_n 100 \
    --min_per_stock 8 --max_per_stock 15 \
    --min_per_etf 3 --max_per_etf 8 \
    --unclear_frac_cap 0.15 \
    --ticker_cooldown_min 60 \
    --include_extended_hours
```
