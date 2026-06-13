# Decision-Time Feature Table

For every field: its source, the time window it's computed from, whether it is
**available at the decision time t0**, whether it enters the **agent prompt**, and
whether it enters the **ML stage**. Anything "available at t0 = NO" that feeds a
prompt/model is a leak.

**Decision time t0** = the timestamp of the spike bar; its bar is fully formed and
its close is the entry price, so data *at or before* t0 is allowed; anything from a
*later* bar is not. (Generated from `mine_events_strict.py`, `agents/briefs_v3.py`,
`eval/feature_matrix.py`; verified by `analysis/leakage_tests.py`.)

| Field | Source | Time window | Avail @ t0 | In prompt | In ML | Notes / leak status |
|---|---|---|:--:|:--:|:--:|---|
| `t0_utc`, `ticker`, `direction` | miner | at t0 | ✅ | ✅ | ✅ | event identity |
| `peak_z_ret_3`, `peak_abs_ret_3_bp` | OHLCV | ≤ t0 (spike 3-bar) | ✅ | ✅ | ✅ | spike magnitude |
| `peak_volume_mult`, `peak_range_mult` | OHLCV | ≤ t0 | ✅ | ✅ | ✅ | spike volume/range |
| `open_adj_vol_mult`, `open_adj_range_mult` | OHLCV + baselines | ≤ t0 | ✅ | ✅ | ✅ | opening-bell-adjusted spike size |
| `beta` | OHLCV regression | 25 bars **< t0** | ✅ | ✅ | ✅ | pre-event |
| `idio_resid_bp` | OHLCV | spike 3-bar − β·SPY 15m, ≤ t0 | ✅ | ✅ | ✅ | **clean overreaction signal** (CI-significant) |
| `rs_ratio` | OHLCV | 5 bars **< t0** | ✅ | ✅ | ✅ | pre-event semivariance |
| `signed_vol_ratio`, `net_signed_vol` | OHLCV | last 5 bars **incl. t0** | ✅ | ✅ | ✅ | **verified leak-free** (corr 1.000 w/ pre-event recompute); dominated by t0 spike bar |
| `vpin_proxy` | OHLCV | last 5 bars **incl. t0** | ✅ | ✅ | ✅ | **verified leak-free**; \|signed_vol\| |
| `spread_bp` | OHLCV | t0 bar | ✅ | ✅ | ✅ | at-t0 estimate |
| `pre_spike_run_len` | OHLCV | bars **< t0** | ✅ | ✅ | ✅ | pre-event |
| `adj_zscore_tod`, `spy_zscore_t0` | OHLCV | at t0 | ✅ | ✅ | ✅ | time-of-day / market context |
| `is_macro_driven` | derived | from `spy_zscore_t0` | ✅ | ✅ | ✅ | flag |
| ticker news (`ticker_news`) | Polygon news API | **< t0** *(after fix)* | ⚠️ | ✅ | (via agent) | **WAS a leak** — old window `[t0−3h, t0+lookahead]` pulled 279 post-t0 articles into 197/797 events; fixed to end at t0. Rebuild: strict `published_utc < t0` + contemporaneous-catalyst design. |
| macro news (SPY/QQQ) | Polygon news API | **< t0** *(after fix)* | ⚠️ | ✅ | (via agent) | same window fix |
| agent outputs (labels/conf/scores) | LLM | from above inputs | ✅ | — | ✅ | only valid if all inputs above are valid |
| **`cluster_len`** | OHLCV | bars **≥ t0** | ❌ | removed | removed | **LEAK — removed** (overlapped label window) |
| **`forward_return_60m`** (LABEL) | OHLCV | t0 → t0+60min | ❌ | never | label only | outcome; must NEVER enter inputs |

**Status:** all *input* fields are available at t0 (verified). The only historical leaks were `cluster_len` (removed) and the news window (fixed). The label is strictly post-t0.

**Rebuild additions to slot into this table (with the same columns):** catalyst type, catalyst release timestamp + source (EDGAR/PR-wire), EPS surprise (point-in-time estimate), guidance/transcript text — each must be timestamped and confirmed `≤ decision time` before it can enter a prompt.
