# Data Schema Documentation

## OHLCV Data (`ohlcv_5min.parquet` / `ohlcv_5min.csv`)

### Columns
| Column | Type | Description |
|--------|------|-------------|
| ticker | string | Stock symbol (AAPL, MSFT, etc.) |
| timestamp | datetime64[ns, UTC] | Bar timestamp in UTC |
| open | float64 | Opening price |
| high | float64 | High price |
| low | float64 | Low price |
| close | float64 | Closing price |
| volume | float64 | Trading volume |

### Data Properties
- **Frequency:** 5-minute bars
- **Timezone:** UTC
- **Adjusted:** Yes (split and dividend adjusted via Polygon)
- **Source:** Polygon.io API
- **Coverage:** Pre-market, regular hours, after-hours

### Tickers Included
```
AAPL  - Apple Inc.
MSFT  - Microsoft Corporation
NVDA  - NVIDIA Corporation
TSLA  - Tesla Inc.
META  - Meta Platforms Inc.
AMD   - Advanced Micro Devices Inc.
NFLX  - Netflix Inc.
PLTR  - Palantir Technologies Inc.
SPY   - SPDR S&P 500 ETF
QQQ   - Invesco QQQ Trust
XLK   - Technology Select Sector SPDR
```

### Row Counts per Ticker
```
XLK:   8,530 bars
NFLX: 11,414 bars
AAPL: 11,465 bars
MSFT: 11,469 bars
META: 11,480 bars
AMD:  11,850 bars
PLTR: 11,964 bars
SPY:  12,122 bars
QQQ:  12,134 bars
TSLA: 12,145 bars
NVDA: 12,146 bars
```

### Missing Days (Market Holidays)
- 2025-11-27 (Thanksgiving)
- 2025-12-25 (Christmas)
- 2026-01-01 (New Year's Day)

---

## Computed Features (in mine_events_strict.py)

These are computed per-ticker during event mining:

| Feature | Formula | Description |
|---------|---------|-------------|
| log_close | log(close) | Natural log of close price |
| ret_1 | diff(log_close, 1) | 1-bar log return (~5 min) |
| ret_3 | diff(log_close, 3) | 3-bar log return (~15 min) |
| ret_6 | diff(log_close, 6) | 6-bar log return (~30 min) |
| rv_78 | rolling_std(ret_1, 78) | Rolling volatility (~1 trading day) |
| vol_med_78 | rolling_median(volume, 78) | Rolling median volume |
| range | (high - low) / close | Normalized bar range |
| range_med_78 | rolling_median(range, 78) | Rolling median range |
| impact | abs(ret_1) / max(volume, 1) | Amihud-style price impact |
| impact_med_78 | rolling_median(impact, 78) | Rolling median impact |
| z_ret_3 | abs(ret_3) / (rv_78 * sqrt(3)) | Z-score of 3-bar return |
| abs_ret_3_bp | (exp(abs(ret_3)) - 1) * 10000 | Absolute return in basis points |
| volume_mult | volume / vol_med_78 | Volume multiple vs baseline |
| range_mult | range / range_med_78 | Range multiple vs baseline |
| impact_mult | impact / impact_med_78 | Impact multiple vs baseline |

### Baseline Window
- **Window size:** 78 bars (~6.5 hours, roughly 1 trading day)
- **min_periods:** 78 (first 77 bars per ticker have NaN baselines)

---

## Sample Data

```csv
ticker,timestamp,open,high,low,close,volume
AAPL,2025-10-15 08:00:00+00:00,248.23,248.30,248.10,248.17,6321.0
AAPL,2025-10-15 08:05:00+00:00,248.02,248.05,247.97,247.98,3215.0
AAPL,2025-10-15 08:10:00+00:00,247.83,247.89,247.83,247.83,4386.0
```
