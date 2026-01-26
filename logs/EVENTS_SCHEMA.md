# Events Output Schema

## Events CSV (`out_strict/events.csv`)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| event_id | string | Unique ID: `{ticker}_{t0_YYYYMMDD_HHMMSS}` |
| ticker | string | Stock symbol |
| t0_utc | datetime | Event trigger timestamp (UTC) |
| direction | int | Event direction: +1 (up) or -1 (down) |
| cluster_len | int | Number of bars in trigger cluster |
| peak_abs_ret_3_bp | float | Maximum absolute 3-bar return in cluster (basis points) |
| peak_z_ret_3 | float | Maximum z-score of 3-bar return in cluster |
| peak_volume_mult | float | Maximum volume multiple in cluster |
| peak_range_mult | float | Maximum range multiple in cluster |
| peak_impact_mult | float | Maximum impact multiple in cluster |
| event_window_start | datetime | Start of event window (20 min before t0) |
| event_window_end | datetime | End of event window (= t0) |
| val_window_start | datetime | Start of validation window (5 min after t0) |
| val_window_end | datetime | End of validation window (60 min after t0) |
| event_return | float | Return from event_window_start to t0 |
| forward_return_60m | float | Return from t0 to val_window_end |
| label_proxy | string | Label: "continuation", "reversal", or "unclear" |

### Label Definitions

| Label | Condition |
|-------|-----------|
| continuation | sign(forward_return_60m) == direction AND abs(forward_return_60m) >= 0.30% |
| reversal | sign(forward_return_60m) != direction AND abs(forward_return_60m) >= 0.30% |
| unclear | Neither condition met |

### Window Definitions

```
                    Event Window (4 bars = 20 min)    Validation Window (12 bars = 60 min)
                    |<-------------------------->|    |<---------------------------------------->|
Time:   ... --|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-- ...
                    ^                         ^       ^                                     ^
                event_start                  t0    val_start                            val_end
```

---

## Two-Stage Detection Logic

### Stage 1: Candidate Flagging (High Recall)

A bar becomes a candidate if:
1. **Minimum absolute move:** `abs_ret_3_bp >= 30 bp`
2. **AND any trigger fires:**
   - Return spike: `z_ret_3 > 3.0`
   - Volume spike: `volume_mult > 4.0`
   - Range expansion: `range_mult > 3.0`
   - Impact spike: `impact_mult > 4.0`

### Stage 2: Confirmation (Strict)

A candidate cluster is confirmed if ALL gates pass:

1. **Absolute move gate:** `max(abs_ret_3_bp) >= 60 bp`
2. **Strength gate:** `max(z_ret_3) >= 4.0`
3. **Liquidity gate (any one):**
   - `max(volume_mult) >= 6.0` OR
   - `max(range_mult) >= 4.0` OR
   - `max(impact_mult) >= 6.0`
4. **Persistence gate:**
   - `cluster_len >= 2` OR
   - `abs(ret_6 at cluster_end) >= 0.5 * peak_ret_3`

---

## CLI Arguments for mine_events_strict.py

### Stage 1 (Candidate) Thresholds
| Argument | Default | Description |
|----------|---------|-------------|
| --cand_ret_z | 3.0 | Z-score threshold |
| --cand_vol_mult | 4.0 | Volume multiplier |
| --cand_range_mult | 3.0 | Range multiplier |
| --cand_impact_mult | 4.0 | Impact multiplier |
| --cand_abs_bp | 30.0 | Minimum absolute move (bp) |

### Stage 2 (Confirm) Thresholds
| Argument | Default | Description |
|----------|---------|-------------|
| --confirm_abs_bp | 60.0 | Minimum absolute move (bp) |
| --confirm_ret_z | 4.0 | Z-score threshold |
| --confirm_vol_mult | 6.0 | Volume multiplier |
| --confirm_range_mult | 4.0 | Range multiplier |
| --confirm_impact_mult | 6.0 | Impact multiplier |
| --persistence_ratio | 0.5 | ret_6 must be >= this * peak_ret_3 |

### Other Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| --input_path | ohlcv_5min.parquet | Input file |
| --out_dir | ./out | Output directory |
| --baseline_window | 78 | Rolling window for baselines |
| --cluster_gap | 2 | Max bars to merge into cluster |
| --event_bars | 4 | Event window size (before t0) |
| --val_bars | 12 | Validation window size (after t0) |
| --cont_thresh | 0.003 | Continuation label threshold |
| --rev_thresh | 0.003 | Reversal label threshold |
| --make_plots | False | Generate PNG plots |

---

## Sample Event Record

```json
{
  "event_id": "META_20251029_194500",
  "ticker": "META",
  "t0_utc": "2025-10-29 19:45:00+00:00",
  "direction": 1,
  "cluster_len": 5,
  "peak_abs_ret_3_bp": 867.9,
  "peak_z_ret_3": 5.1,
  "peak_volume_mult": 9.8,
  "peak_range_mult": 12.3,
  "peak_impact_mult": 4.2,
  "event_window_start": "2025-10-29 19:25:00+00:00",
  "event_window_end": "2025-10-29 19:45:00+00:00",
  "val_window_start": "2025-10-29 19:50:00+00:00",
  "val_window_end": "2025-10-29 20:45:00+00:00",
  "event_return": 0.0623,
  "forward_return_60m": -0.0744,
  "label_proxy": "reversal"
}
```
