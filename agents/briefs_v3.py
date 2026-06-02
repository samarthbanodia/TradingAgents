"""
V3 brief construction.

Extends V2.1 with 7 new microstructure features:
1. Idiosyncratic residual  — stock return minus beta * SPY return
2. Realized semivariance   — RS_minus / total (directional exhaustion signal)
3. Net signed volume proxy — net buy/sell pressure over cluster bars
4. VPIN proxy              — volume-weighted order flow imbalance
5. Spread proxy            — EDGE-inspired bid-ask spread approximation
6. Pre-spike run length    — momentum persistence before the spike
7. Macro-driven flag       — SPY z-score at t0 > 1.5

V2.1 news and macro brief builders are unchanged — re-exported as V3.
Skeptic brief builder is REMOVED (no Skeptic in V3).
"""

import logging

import numpy as np
import pytz

from agents.briefs_v2 import (
    build_open_baselines,
    build_news_brief_v2,
    build_macro_brief_v2,
    build_technical_brief_v2,
    _is_opening_bell,
    _parse_t0,
    _to_et_str,
    _get_open_adjusted,
    _get_ohlcv_context,
    FORBIDDEN_FIELDS,
)

# Re-export unchanged V2 builders under V3 names
build_news_brief_v3  = build_news_brief_v2
build_macro_brief_v3 = build_macro_brief_v2

logger = logging.getLogger(__name__)
ET_TZ  = pytz.timezone("America/New_York")


# ── Bar extraction helpers ────────────────────────────────────────────────────

def _get_pre_event_bars(ticker, t0, ohlcv_df, n_bars=25):
    """Return up to n_bars bars ending at t0 (inclusive), sorted ascending."""
    if ohlcv_df is None or t0 is None:
        return None
    try:
        sub    = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("timestamp")
        before = sub[sub["timestamp"] <= t0]
        if len(before) < 2:
            return None
        return before.iloc[-n_bars:].reset_index(drop=True)
    except Exception as e:
        logger.debug(f"_get_pre_event_bars error for {ticker}: {e}")
        return None


def _get_cluster_bars(ticker, t0, cluster_len, ohlcv_df):
    """Return cluster_len bars starting at t0 (inclusive)."""
    if ohlcv_df is None or t0 is None or cluster_len < 1:
        return None
    try:
        sub       = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("timestamp")
        at_after  = sub[sub["timestamp"] >= t0]
        if at_after.empty:
            return None
        return at_after.iloc[:cluster_len].reset_index(drop=True)
    except Exception as e:
        logger.debug(f"_get_cluster_bars error for {ticker}: {e}")
        return None


# ── Feature computation functions ─────────────────────────────────────────────

def _compute_beta(stock_bars, spy_bars):
    """
    OLS beta of stock vs SPY from aligned 1-min returns.
    Returns 1.0 if insufficient data.
    """
    try:
        s_idx  = stock_bars.set_index("timestamp")
        sp_idx = spy_bars.set_index("timestamp")
        common = s_idx.index.intersection(sp_idx.index)
        if len(common) < 6:
            return 1.0
        s_rets  = s_idx.loc[common, "close"].pct_change().dropna()
        sp_rets = sp_idx.loc[common, "close"].pct_change().dropna()
        c2      = s_rets.index.intersection(sp_rets.index)
        if len(c2) < 4:
            return 1.0
        s_arr   = s_rets.loc[c2].values
        sp_arr  = sp_rets.loc[c2].values
        var_sp  = np.var(sp_arr)
        if var_sp < 1e-12:
            return 1.0
        beta = float(np.cov(s_arr, sp_arr)[0, 1] / var_sp)
        return round(max(0.0, min(4.0, beta)), 3)
    except Exception as e:
        logger.debug(f"_compute_beta error: {e}")
        return 1.0


def _compute_semivariance(bars):
    """
    Realized semivariance ratio from bar returns.

    RS_plus  = sum(r^2 for r in returns if r > 0)
    RS_minus = sum(r^2 for r in returns if r < 0)
    rs_ratio = RS_minus / (RS_plus + RS_minus)  ∈ [0, 1]

    rs_ratio → 0.5 : two-directional noise → overreaction → reversal signal
    rs_ratio → 1.0 : all downward variance → downward herding
    rs_ratio → 0.0 : all upward variance   → upward herding
    """
    result = {"rs_ratio": None}
    if bars is None or len(bars) < 2:
        return result
    try:
        rets     = bars["close"].pct_change().dropna().values
        if len(rets) == 0:
            return result
        rs_plus  = float(np.sum(rets[rets > 0] ** 2))
        rs_minus = float(np.sum(rets[rets < 0] ** 2))
        total    = rs_plus + rs_minus
        if total < 1e-14:
            return result
        result["rs_ratio"] = round(rs_minus / total, 4)
        return result
    except Exception as e:
        logger.debug(f"_compute_semivariance error: {e}")
        return result


def _compute_signed_volume(bars):
    """
    Net signed volume proxy over cluster bars.
    body_to_range = (close - open) / (high - low + 1e-9)  ∈ [-1, 1]
    signed_volume = volume * body_to_range

    Positive net in UP spike  → sustained buying    → continuation
    Negative net in UP spike  → selling absorbed    → reversal
    """
    result = {"net_signed_vol": None, "signed_vol_ratio": None}
    if bars is None or len(bars) == 0:
        return result
    try:
        btr        = (bars["close"].values - bars["open"].values) / (
                      bars["high"].values  - bars["low"].values + 1e-9)
        signed     = bars["volume"].values * btr
        net        = float(np.sum(signed))
        total_vol  = float(bars["volume"].sum())
        result["net_signed_vol"]   = round(net, 0)
        result["signed_vol_ratio"] = round(net / (total_vol + 1e-9), 4) if total_vol > 0 else None
        return result
    except Exception as e:
        logger.debug(f"_compute_signed_volume error: {e}")
        return result


def _compute_vpin_proxy(cluster_bars):
    """
    VPIN proxy = |net signed volume| / total volume  ∈ [0, 1].
    > 0.7: concentrated directional flow → informed → continuation
    < 0.3: balanced flow → noise/liquidity → reversal
    """
    if cluster_bars is None or len(cluster_bars) == 0:
        return None
    try:
        btr      = (cluster_bars["close"].values - cluster_bars["open"].values) / (
                    cluster_bars["high"].values   - cluster_bars["low"].values + 1e-9)
        signed   = cluster_bars["volume"].values * btr
        total    = float(cluster_bars["volume"].sum())
        if total < 1:
            return None
        return round(abs(float(np.sum(signed))) / total, 4)
    except Exception as e:
        logger.debug(f"_compute_vpin_proxy error: {e}")
        return None


def _compute_spread_proxy(t0_bar):
    """
    EDGE-inspired bid-ask spread proxy from a single OHLC bar.
    spread = sqrt(max(0, log(H/L)^2 - log(C/O)^2)) * 10000  [basis points]

    Wider spread at t0 → higher market-maker uncertainty → reversal more likely.
    """
    result = {"spread_bp": None}
    if t0_bar is None or len(t0_bar) == 0:
        return result
    try:
        row = t0_bar.iloc[0]
        H, L, O, C = float(row["high"]), float(row["low"]), float(row["open"]), float(row["close"])
        if L <= 0 or O <= 0 or H <= L:
            return result
        log_hl  = np.log(H / L)
        log_co  = abs(np.log(C / O)) if O > 0 and C > 0 else 0.0
        spread  = float(np.sqrt(max(0.0, log_hl ** 2 - log_co ** 2))) * 10000
        result["spread_bp"] = round(spread, 2)
        return result
    except Exception as e:
        logger.debug(f"_compute_spread_proxy error: {e}")
        return result


def _compute_pre_spike_run_length(ticker, t0, direction, ohlcv_df):
    """
    Count consecutive 1-min bars moving in the spike direction immediately before t0.

    Longer run → exhaustion → reversal signal (Markov analysis finding).
    0 = spike is abrupt (ambiguous).
    """
    bars = _get_pre_event_bars(ticker, t0, ohlcv_df, n_bars=10)
    if bars is None or len(bars) < 2:
        return 0
    try:
        # Pre-t0 bars only (exclude the t0 bar itself)
        pre  = bars.iloc[:-1]
        if len(pre) == 0:
            return 0
        dirs = np.sign(pre["close"].values - pre["open"].values)
        run  = 0
        for d in reversed(dirs):
            if d == direction or d == 0:
                run += 1
            else:
                break
        return min(run, 5)
    except Exception as e:
        logger.debug(f"_compute_pre_spike_run_length error: {e}")
        return 0


def _compute_spy_zscore_at_t0(t0, ohlcv_df):
    """
    SPY 1-bar return z-score at t0 vs its rolling 78-bar baseline.
    |z| > 1.5 → event is macro-driven (market-wide).
    """
    if ohlcv_df is None or t0 is None:
        return None
    try:
        spy_bars = _get_pre_event_bars("SPY", t0, ohlcv_df, n_bars=80)
        if spy_bars is None or len(spy_bars) < 10:
            return None
        rets = spy_bars["close"].pct_change().dropna()
        if len(rets) < 5:
            return None
        baseline = rets.iloc[:-1]
        mean_r   = float(baseline.mean())
        std_r    = float(baseline.std())
        if std_r < 1e-10:
            return None
        last_r = float(rets.iloc[-1])
        return round((last_r - mean_r) / std_r, 2)
    except Exception as e:
        logger.debug(f"_compute_spy_zscore_at_t0 error: {e}")
        return None


def _compute_tod_adjusted_zscore(row, ohlcv_df):
    """
    Time-of-day adjusted z-score.
    Divides the t0 bar's absolute return by the typical volatility for this
    ticker in the same 30-min time bucket.

    Higher → more anomalous for this time slot → stronger signal.
    """
    if ohlcv_df is None:
        return None
    try:
        ticker = row["ticker"]
        t0     = _parse_t0(row["t0_utc"])
        if t0 is None:
            return None
        t0_et  = t0.tz_convert(ET_TZ)
        bucket = t0_et.hour * 2 + (1 if t0_et.minute >= 30 else 0)

        sub = ohlcv_df[ohlcv_df["ticker"] == ticker].copy()
        if sub.empty:
            return None
        sub["ts_et"]  = sub["timestamp"].dt.tz_convert(ET_TZ)
        sub["bucket"] = sub["ts_et"].dt.hour * 2 + (sub["ts_et"].dt.minute >= 30).astype(int)

        bucket_rets = (
            sub[sub["bucket"] == bucket]["close"]
            .pct_change().dropna()
            .abs()
        )
        if len(bucket_rets) < 10:
            return None
        hist_vol = float(bucket_rets.std())
        if hist_vol < 1e-8:
            return None

        t0_bar = sub[sub["timestamp"] == t0]
        if t0_bar.empty:
            return None
        bar_ret = abs(float(t0_bar["close"].iloc[0]) / float(t0_bar["open"].iloc[0]) - 1)
        return round(bar_ret / hist_vol, 2)
    except Exception as e:
        logger.debug(f"_compute_tod_adjusted_zscore error: {e}")
        return None


# ── Master V3 feature computation ─────────────────────────────────────────────

def compute_v3_features(row, ohlcv_df, baselines):
    """
    Compute all V3 microstructure features for one event row.
    Returns a dict; any unavailable feature is None.
    """
    ticker      = row["ticker"]
    t0          = _parse_t0(row["t0_utc"])
    direction   = int(row.get("direction", 0))
    cluster_len = int(row.get("cluster_len", 1))

    _null = dict(
        beta=None, idio_resid_bp=None, rs_ratio=None,
        net_signed_vol=None, signed_vol_ratio=None,
        vpin_proxy=None, spread_bp=None,
        pre_spike_run_len=0, adj_zscore_tod=None,
        spy_zscore_t0=None, is_macro_driven=False,
    )

    if ohlcv_df is None or t0 is None:
        return _null

    # ── Raw bar windows ──
    stock_pre   = _get_pre_event_bars(ticker,  t0, ohlcv_df, n_bars=25)
    spy_pre     = _get_pre_event_bars("SPY",   t0, ohlcv_df, n_bars=25)
    # NOTE: cluster_bars (t0 onwards) are NOT used for ML features — they overlap
    # the 60-min label window and cause data leakage. VPIN/signed_vol now use
    # the 5 bars immediately before t0 (pre-event only).

    # t0 bar for spread proxy
    t0_bar = None
    if stock_pre is not None and len(stock_pre) > 0:
        if stock_pre.iloc[-1]["timestamp"] == t0:
            t0_bar = stock_pre.iloc[[-1]]

    features = {}

    # 1. Beta + idiosyncratic residual
    beta = 1.0
    if stock_pre is not None and spy_pre is not None:
        beta = _compute_beta(stock_pre, spy_pre)
    features["beta"] = beta

    spy_ctx    = _get_ohlcv_context(t0, ohlcv_df, "SPY")
    spy_15m_bp = spy_ctx.get("SPY_ret_15m_bp")
    stock_3bar_signed_bp = float(row["peak_abs_ret_3_bp"]) * direction
    if spy_15m_bp is not None:
        features["idio_resid_bp"] = round(stock_3bar_signed_bp - beta * spy_15m_bp, 2)
    else:
        features["idio_resid_bp"] = None

    # 2. Realized semivariance (5 pre-event bars before t0)
    sv_bars = stock_pre.iloc[-6:-1] if stock_pre is not None and len(stock_pre) >= 6 else stock_pre
    sv      = _compute_semivariance(sv_bars)
    features["rs_ratio"] = sv["rs_ratio"]

    # 3. Signed volume + VPIN — last 5 bars BEFORE t0 (leak-free)
    pre_vol_bars = stock_pre.iloc[-5:] if stock_pre is not None and len(stock_pre) >= 5 else stock_pre
    sv_out = _compute_signed_volume(pre_vol_bars)
    features["net_signed_vol"]   = sv_out["net_signed_vol"]
    features["signed_vol_ratio"] = sv_out["signed_vol_ratio"]
    features["vpin_proxy"]       = _compute_vpin_proxy(pre_vol_bars)

    # 4. Spread proxy at t0
    spread = _compute_spread_proxy(t0_bar)
    features["spread_bp"] = spread["spread_bp"]

    # 5. Pre-spike run length
    features["pre_spike_run_len"] = _compute_pre_spike_run_length(
        ticker, t0, direction, ohlcv_df
    )

    # 6. Time-of-day adjusted z-score
    features["adj_zscore_tod"] = _compute_tod_adjusted_zscore(row, ohlcv_df)

    # 7. SPY z-score + macro-driven flag
    spy_z = _compute_spy_zscore_at_t0(t0, ohlcv_df)
    features["spy_zscore_t0"]  = spy_z
    features["is_macro_driven"] = bool(spy_z is not None and abs(spy_z) > 1.5)

    return features


# ── Technical Brief V3 ────────────────────────────────────────────────────────

def build_technical_brief_v3(row, ohlcv_df, baselines, v3_features=None):
    """
    Technical brief for Microstructure Agent (V3).

    Builds the V2.1 base brief and appends the V3 microstructure features section.

    Returns: (brief_str: str, v3_features: dict)
    """
    if v3_features is None:
        v3_features = compute_v3_features(row, ohlcv_df, baselines)

    base = build_technical_brief_v2(row, ohlcv_df, baselines)
    base = base.replace("=== TECHNICAL BRIEF (V2.1) ===", "=== TECHNICAL BRIEF (V3) ===")

    direction   = int(row.get("direction", 0))
    is_up       = direction == 1

    lines = ["", "--- V3 Microstructure Features (NEW) ---"]

    # Idiosyncratic residual
    idio = v3_features.get("idio_resid_bp")
    beta = v3_features.get("beta", 1.0)
    if idio is not None:
        lines.append(f"  idio_residual_bp   : {idio:+.1f} bp  (beta={beta:.2f}; "
                     f"= stock_3bar_ret - beta*SPY_3bar_ret)")
        if is_up:
            if idio > 30:
                lines.append("  → large positive idio residual: stock moved FAR beyond market "
                             "→ idiosyncratic shock → REVERSAL more likely")
            elif idio < -20:
                lines.append("  → negative idio residual: market drove this move "
                             "→ macro-driven → CONTINUATION more likely")
            else:
                lines.append("  → small idio residual: mixed market vs. stock contribution")
        else:  # DOWN spike
            if idio < -30:
                lines.append("  → large negative idio residual (down spike): stock fell far more "
                             "than market → idiosyncratic panic → REVERSAL likely")
    else:
        lines.append("  idio_residual_bp   : [unavailable — SPY data missing]")

    # Realized semivariance ratio
    rs = v3_features.get("rs_ratio")
    if rs is not None:
        lines.append(f"  rs_ratio           : {rs:.3f}  (RS_minus / total; 0=all upward, 1=all downward)")
        if rs > 0.70:
            lines.append("  → high RS_ratio: mostly downward pre-event vol → downward herding "
                         "(DOWN spike: continuation; UP spike: noise context)")
        elif rs < 0.30:
            lines.append("  → low RS_ratio: mostly upward pre-event vol → upward herding "
                         "(UP spike: continuation signal)")
        else:
            lines.append("  → mid RS_ratio (~0.5): two-directional noise → overreaction → REVERSAL signal")
    else:
        lines.append("  rs_ratio           : [unavailable]")

    # Signed volume
    svr = v3_features.get("signed_vol_ratio")
    nsv = v3_features.get("net_signed_vol")
    if svr is not None:
        lines.append(f"  signed_vol_ratio   : {svr:+.3f}  (net buyer/seller pressure over cluster; -1=all sell, +1=all buy)")
        lines.append(f"  net_signed_vol     : {nsv:+,.0f} shares")
        if is_up:
            if svr > 0.30:
                lines.append("  → strong net buying in UP spike → informed buying → CONTINUATION")
            elif svr < -0.10:
                lines.append("  → net selling despite UP spike → buyers exhausted → REVERSAL")
        else:
            if svr < -0.30:
                lines.append("  → strong net selling in DOWN spike → informed selling → CONTINUATION")
            elif svr > 0.10:
                lines.append("  → net buying despite DOWN spike → sellers exhausted → REVERSAL")
    else:
        lines.append("  signed_vol_ratio   : [unavailable]")

    # VPIN proxy
    vpin = v3_features.get("vpin_proxy")
    if vpin is not None:
        lines.append(f"  vpin_proxy         : {vpin:.3f}  (order flow imbalance; 0=balanced, 1=one-sided)")
        if vpin > 0.70:
            lines.append("  → high VPIN: concentrated directional flow → informed actor → CONTINUATION")
        elif vpin < 0.30:
            lines.append("  → low VPIN: balanced two-sided flow → noise/liquidity event → REVERSAL")
    else:
        lines.append("  vpin_proxy         : [unavailable]")

    # Spread proxy
    spread_bp = v3_features.get("spread_bp")
    if spread_bp is not None:
        lines.append(f"  spread_proxy_bp    : {spread_bp:.1f} bp  (approx bid-ask width at t0)")
    else:
        lines.append("  spread_proxy_bp    : [unavailable]")

    # Pre-spike run length
    run_len = v3_features.get("pre_spike_run_len", 0)
    lines.append(f"  pre_spike_run_len  : {run_len} bars moving in spike direction before t0")
    if run_len >= 3:
        lines.append("  → long pre-spike run → exhaustion pattern → REVERSAL likely")
    elif run_len == 0:
        lines.append("  → abrupt spike with no prior run → ambiguous")

    # Macro-driven
    spy_z    = v3_features.get("spy_zscore_t0")
    is_macro = v3_features.get("is_macro_driven", False)
    if spy_z is not None:
        lines.append(f"  spy_zscore_t0      : {spy_z:.2f}  (|z|>1.5 = macro-driven event)")
        if is_macro:
            lines.append("  is_macro_driven    : YES → market-wide move → reduces reversal bias")
        else:
            lines.append("  is_macro_driven    : no  → idiosyncratic spike → reversal more likely")
    else:
        lines.append("  spy_zscore_t0      : [unavailable]")

    # Time-of-day adjusted z-score
    adj_z = v3_features.get("adj_zscore_tod")
    if adj_z is not None:
        lines.append(f"  adj_zscore_tod     : {adj_z:.2f}  (z vs typical vol for this ticker/time-of-day slot)")
        if adj_z > 5.0:
            lines.append("  → very unusual for this time slot → strong idiosyncratic signal")
    else:
        lines.append("  adj_zscore_tod     : [unavailable]")

    return base + "\n" + "\n".join(lines), v3_features
