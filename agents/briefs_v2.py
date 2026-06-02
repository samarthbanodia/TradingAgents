"""
V2.1 brief construction.

Reconstructed from briefs.py (V1) + briefs_v3.py import signatures + run_ip_v3.py call patterns.
Key changes vs V1:
  - _parse_t0 / _to_et_str use pytz-aware pandas Timestamps
  - _get_ohlcv_context signature changed to (t0, ohlcv_df, ticker)
  - build_news_brief_v2 returns (brief_str, has_pre_event_news) tuple
  - build_macro_brief_v2 / build_news_brief_v2 accept news_lookback_minutes
  - build_open_baselines + _get_open_adjusted added
  - _filter_pre_event_news / _parse_pub_time / _count_post_event added
"""

import json
import logging

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

ET_TZ = pytz.timezone("America/New_York")

FORBIDDEN_FIELDS = {"label_proxy", "forward_return_60m", "val_window_start", "val_window_end"}


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _parse_t0(t0_str):
    """Parse t0 string into UTC-aware pandas Timestamp."""
    if not t0_str:
        return None
    try:
        ts = pd.Timestamp(t0_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def _to_et_str(t0):
    """Convert UTC-aware Timestamp to ET display string."""
    try:
        if t0 is None:
            return ""
        t0_et = t0.tz_convert(ET_TZ)
        return t0_et.strftime("%Y-%m-%d %H:%M ET (%A)")
    except Exception:
        return str(t0)


def _is_opening_bell(t0_str):
    """Check if t0 is within 30 min of 9:30 ET."""
    try:
        t0 = _parse_t0(t0_str)
        if t0 is None:
            return False
        t0_et = t0.tz_convert(ET_TZ)
        market_open = t0_et.replace(hour=9, minute=30, second=0, microsecond=0)
        return abs((t0_et - market_open).total_seconds()) <= 1800
    except Exception:
        return False


# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def _get_ohlcv_context(t0, ohlcv_df, ticker):
    """
    Compute pre-event returns for a ticker around t0.
    t0 is a UTC-aware pandas Timestamp.
    Returns dict with keys like {ticker}_ret_15m_bp, etc.
    """
    if ohlcv_df is None or t0 is None:
        return {}
    try:
        sub = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("timestamp")
        sub = sub.set_index("timestamp")
        result = {}
        mask = sub.index <= t0
        n = mask.sum()

        if n >= 4:
            recent = sub[mask].iloc[-4:]
            ret = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 10000
            result[f"{ticker}_ret_15m_bp"] = round(float(ret), 1)

        if n >= 13:
            recent = sub[mask].iloc[-13:]
            ret = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 10000
            result[f"{ticker}_ret_60m_pre_bp"] = round(float(ret), 1)
            rets = recent["close"].pct_change().dropna()
            result[f"{ticker}_realized_vol_pre"] = round(float(rets.std()) * 100, 4)

        return result
    except Exception:
        return {}


def build_open_baselines(ohlcv_df):
    """
    Build per-ticker median volume / range baselines for open-adjusted metrics.
    Returns dict keyed by ticker.
    """
    baselines = {}
    if ohlcv_df is None or ohlcv_df.empty:
        return baselines
    try:
        for ticker, grp in ohlcv_df.groupby("ticker"):
            baselines[ticker] = {
                "median_volume": float(grp["volume"].median()),
                "median_range":  float((grp["high"] - grp["low"]).median()),
                "std_volume":    float(grp["volume"].std()),
            }
    except Exception as e:
        logger.debug(f"build_open_baselines error: {e}")
    return baselines


def _get_open_adjusted(row, ohlcv_df, baselines):
    """
    Return (adj_vol_mult, adj_range_mult) — spike bar metrics normalized
    by per-ticker historical medians.
    """
    try:
        ticker = row["ticker"]
        t0 = _parse_t0(row["t0_utc"])
        if t0 is None or ohlcv_df is None:
            return None, None
        sub = ohlcv_df[(ohlcv_df["ticker"] == ticker) & (ohlcv_df["timestamp"] == t0)]
        if sub.empty:
            return None, None
        bar      = sub.iloc[0]
        baseline = baselines.get(ticker, {})
        med_vol  = baseline.get("median_volume", 1) or 1
        med_rng  = baseline.get("median_range",  1) or 1
        adj_vol  = round(float(bar["volume"]) / med_vol, 2)
        adj_rng  = round((float(bar["high"]) - float(bar["low"])) / med_rng, 2)
        return adj_vol, adj_rng
    except Exception:
        return None, None


# ── News filtering helpers ────────────────────────────────────────────────────

def _parse_pub_time(pub_str):
    """Parse article published_utc string to UTC-aware Timestamp."""
    if not pub_str:
        return None
    try:
        ts = pd.Timestamp(pub_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def _filter_pre_event_news(articles, t0, lookback_minutes=180):
    """Return articles published in [t0 - lookback_minutes, t0)."""
    if not articles or t0 is None:
        return []
    cutoff_early = t0 - pd.Timedelta(minutes=lookback_minutes)
    pre = []
    for art in articles:
        pub = _parse_pub_time(art.get("published_utc", ""))
        if pub is not None and cutoff_early <= pub < t0:
            pre.append(art)
    return pre


def _count_post_event(articles, t0):
    """Count articles published at or after t0."""
    if not articles or t0 is None:
        return 0
    return sum(
        1 for art in articles
        if _parse_pub_time(art.get("published_utc", "")) is not None
        and _parse_pub_time(art.get("published_utc", "")) >= t0
    )


# ── Brief builders ────────────────────────────────────────────────────────────

def build_technical_brief_v2(row, ohlcv_df, baselines):
    """Build the V2.1 technical/microstructure brief."""
    t0_str  = row["t0_utc"]
    ticker  = row["ticker"]
    t0      = _parse_t0(t0_str)
    dir_int = int(row["direction"])
    direction = "UP (+1)" if dir_int == 1 else "DOWN (-1)"
    is_bell = _is_opening_bell(t0_str)

    adj_vol, adj_range = _get_open_adjusted(row, ohlcv_df, baselines)

    lines = [
        "=== TECHNICAL BRIEF (V2.1) ===",
        f"Ticker: {ticker}",
        f"Event time (UTC): {t0_str}",
        f"Event time (ET): {_to_et_str(t0)}",
        f"Direction: {direction}",
        f"Opening bell event: {'YES' if is_bell else 'no'}",
        "",
        "--- Spike Metrics ---",
        f"peak_abs_ret_3_bp  : {float(row['peak_abs_ret_3_bp']):.1f}",
        f"peak_z_ret_3       : {float(row['peak_z_ret_3']):.2f}",
        f"cluster_len        : {int(row['cluster_len'])} bars",
        "",
        "--- Liquidity Signals ---",
        f"peak_volume_mult   : {float(row['peak_volume_mult']):.1f}x",
        f"peak_range_mult    : {float(row['peak_range_mult']):.1f}x",
        f"peak_impact_mult   : {float(row['peak_impact_mult']):.4f}",
    ]

    if adj_vol is not None:
        lines.append(f"open_adj_vol_mult  : {adj_vol:.1f}x  (vs per-ticker median)")
    if adj_range is not None:
        lines.append(f"open_adj_range_mult: {adj_range:.1f}x  (vs per-ticker median)")

    # SPY / QQQ context
    if t0 is not None and ohlcv_df is not None:
        for idx_tk in ("SPY", "QQQ"):
            ctx = _get_ohlcv_context(t0, ohlcv_df, idx_tk)
            if ctx:
                lines.append("")
                lines.append(f"--- {idx_tk} Context ---")
                for k, v in ctx.items():
                    lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def build_news_brief_v2(row, packet, news_lookback_minutes=180):
    """
    Build the V2.1 news brief for the News Agent.
    Returns (brief_str, has_pre_event_news).
    """
    ticker = row["ticker"]
    t0_str = row["t0_utc"]
    t0     = _parse_t0(t0_str)
    news   = packet.get("news", {})
    all_articles = news.get("ticker_news", [])
    macro  = news.get("macro_news", {})

    # Filter to strictly pre-event articles
    pre_articles = _filter_pre_event_news(all_articles, t0, news_lookback_minutes)
    post_count   = _count_post_event(all_articles, t0)
    has_pre      = len(pre_articles) > 0

    lines = [
        "=== NEWS BRIEF (V2.1) ===",
        f"Ticker: {ticker}",
        f"Event time (UTC): {t0_str}",
        f"Event time (ET): {_to_et_str(t0)}",
        "",
    ]

    if not has_pre:
        lines.append("no_pre_event_news: true")
        lines.append(
            f"No ticker-specific news found in the {news_lookback_minutes}-minute window before this event."
        )
        if post_count > 0:
            lines.append(f"Note: {post_count} article(s) were published AFTER the event (not shown — forward-looking).")
    else:
        lines.append(f"--- Ticker News ({len(pre_articles[:6])} pre-event articles) ---")
        for i, art in enumerate(pre_articles[:6], 1):
            desc = (art.get("description") or "")[:200]
            lines.append(f"[{i}] {art.get('published_utc', '?')} | {art.get('publisher', {}).get('name', '?') if isinstance(art.get('publisher'), dict) else art.get('publisher', '?')}")
            lines.append(f"    Title: {art.get('title', '?')}")
            if desc:
                lines.append(f"    Summary: {desc}")
            lines.append("")
        if post_count > 0:
            lines.append(f"(Note: {post_count} article(s) published AFTER event — excluded to prevent look-ahead)")
            lines.append("")

    # Macro news
    macro_articles = []
    for idx_tk in ("SPY", "QQQ"):
        for art in macro.get(idx_tk, []):
            pub = _parse_pub_time(art.get("published_utc", ""))
            if pub is not None and pub < t0:
                macro_articles.append(art)

    if macro_articles:
        lines.append(f"--- Macro Headlines ({min(len(macro_articles), 4)}) ---")
        for art in macro_articles[:4]:
            lines.append(f"  {art.get('published_utc', '?')} | {art.get('title', '?')}")
        lines.append("")

    return "\n".join(lines), has_pre


def build_macro_brief_v2(row, packet, ohlcv_df, news_lookback_minutes=180):
    """Build the V2.1 macro/correlation brief for the Macro Agent."""
    t0_str = row["t0_utc"]
    ticker = row["ticker"]
    t0     = _parse_t0(t0_str)
    news   = packet.get("news", {})
    macro  = news.get("macro_news", {})

    lines = [
        "=== MACRO BRIEF (V2.1) ===",
        f"Event time (UTC): {t0_str}",
        f"Event time (ET): {_to_et_str(t0)}",
        f"Ticker in question: {ticker}",
        "",
    ]

    # SPY / QQQ price context
    if t0 is not None and ohlcv_df is not None:
        for idx_tk in ("SPY", "QQQ"):
            ctx = _get_ohlcv_context(t0, ohlcv_df, idx_tk)
            if ctx:
                lines.append(f"--- {idx_tk} Context ---")
                for k, v in ctx.items():
                    lines.append(f"  {k}: {v}")
                lines.append("")

    # Macro headlines (pre-event only)
    macro_articles = []
    for idx_tk in ("SPY", "QQQ"):
        for art in macro.get(idx_tk, []):
            pub = _parse_pub_time(art.get("published_utc", ""))
            if pub is not None and t0 is not None and pub < t0:
                macro_articles.append(art)

    if macro_articles:
        lines.append(f"--- Macro Headlines ({min(len(macro_articles), 4)}) ---")
        for art in macro_articles[:4]:
            desc = (art.get("description") or "")[:150]
            lines.append(f"  {art.get('published_utc', '?')} | {art.get('title', '?')}")
            if desc:
                lines.append(f"    {desc}")
        lines.append("")
    else:
        lines.append("No macro headlines available for this time window.")

    return "\n".join(lines)


def build_skeptic_input_v2(row, packet, micro_out, news_out, macro_out):
    """Build combined brief for the Skeptic (V2.1). Not used in V3."""
    ticker    = row["ticker"]
    t0_str    = row["t0_utc"]
    t0        = _parse_t0(t0_str)
    direction = "UP" if int(row["direction"]) == 1 else "DOWN"
    news      = packet.get("news", {})
    has_news  = bool(news.get("ticker_news")) and not news.get("no_news_found", False)

    lines = [
        "=== SKEPTIC INPUT (V2.1) ===",
        f"Ticker: {ticker} | Direction: {direction} | Time: {_to_et_str(t0)}",
        f"Spike: {float(row['peak_abs_ret_3_bp']):.0f}bp, z={float(row['peak_z_ret_3']):.1f}, "
        f"cluster={int(row['cluster_len'])} bars",
        f"Liquidity: vol_mult={float(row['peak_volume_mult']):.0f}x, "
        f"range_mult={float(row['peak_range_mult']):.1f}x",
        f"Opening bell: {'YES' if _is_opening_bell(t0_str) else 'no'}",
        f"Has ticker news: {'yes' if has_news else 'NO'}",
        "",
        f"Micro  → {micro_out.get('label','?')} (conf={micro_out.get('confidence',0):.2f})",
        f"News   → {news_out.get('label','?')}  (conf={news_out.get('confidence',0):.2f})",
        f"Macro  → {macro_out.get('label','?')} (conf={macro_out.get('confidence',0):.2f})",
    ]
    return "\n".join(lines)
