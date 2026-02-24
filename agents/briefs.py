"""Brief construction for each agent from event CSV row + news packet."""

import json
from datetime import datetime, timezone, timedelta

ET_OFFSET = timedelta(hours=-5)  # EST (simplified, no DST handling needed for display)

# Fields that must NEVER appear in any brief
FORBIDDEN_FIELDS = {"label_proxy", "forward_return_60m", "val_window_start", "val_window_end"}


def _utc_to_et(t0_str: str) -> str:
    """Convert UTC timestamp string to ET display string."""
    try:
        dt = datetime.fromisoformat(t0_str.replace("+00:00", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et = dt + ET_OFFSET
        return et.strftime("%Y-%m-%d %H:%M ET (%A)")
    except Exception:
        return t0_str


def _is_opening_bell(t0_str: str) -> bool:
    """Check if t0 is within 30 min of 9:30 ET."""
    try:
        dt = datetime.fromisoformat(t0_str.replace("+00:00", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et = dt + ET_OFFSET
        market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
        diff = abs((et - market_open).total_seconds())
        return diff <= 1800
    except Exception:
        return False


def _get_ohlcv_context(ticker: str, t0_str: str, ohlcv_df=None, ref_ticker: str = None):
    """Compute returns from OHLCV data for a given ticker around t0."""
    if ohlcv_df is None:
        return {}
    try:
        import pandas as pd
        t0 = pd.Timestamp(t0_str)
        tk = ref_ticker or ticker
        sub = ohlcv_df[ohlcv_df["ticker"] == tk].sort_values("timestamp")
        sub = sub.set_index("timestamp")

        result = {}
        # 15m return (3 bars back)
        mask_before = sub.index <= t0
        if mask_before.sum() >= 4:
            recent = sub[mask_before].iloc[-4:]
            ret_15m = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 10000
            result[f"{tk}_ret_15m_bp"] = round(ret_15m, 1)

        # 60m return (12 bars back)
        if mask_before.sum() >= 13:
            recent = sub[mask_before].iloc[-13:]
            ret_60m = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 10000
            result[f"{tk}_ret_60m_pre_bp"] = round(ret_60m, 1)

        # Realized vol (12 bars pre)
        if mask_before.sum() >= 13:
            recent = sub[mask_before].iloc[-13:]
            rets = recent["close"].pct_change().dropna()
            result[f"{tk}_realized_vol_pre"] = round(rets.std() * 100, 4)

        return result
    except Exception:
        return {}


def build_technical_brief(row: dict, ohlcv_df=None) -> str:
    """Build the technical/microstructure brief for the Microstructure Agent."""
    t0 = row["t0_utc"]
    ticker = row["ticker"]
    direction = "UP (+1)" if int(row["direction"]) == 1 else "DOWN (-1)"
    is_bell = _is_opening_bell(t0)

    lines = [
        "=== TECHNICAL BRIEF ===",
        f"Ticker: {ticker}",
        f"Event time (UTC): {t0}",
        f"Event time (ET): {_utc_to_et(t0)}",
        f"Direction: {direction}",
        f"Opening bell event: {'YES' if is_bell else 'no'}",
        "",
        "--- Spike Metrics ---",
        f"peak_abs_ret_3_bp: {float(row['peak_abs_ret_3_bp']):.1f}",
        f"peak_z_ret_3: {float(row['peak_z_ret_3']):.2f}",
        f"cluster_len: {int(row['cluster_len'])} bars",
        "",
        "--- Liquidity Signals ---",
        f"peak_volume_mult: {float(row['peak_volume_mult']):.1f}x",
        f"peak_range_mult: {float(row['peak_range_mult']):.1f}x",
        f"peak_impact_mult: {float(row['peak_impact_mult']):.4f}",
    ]

    # OHLCV-derived features
    ctx = {}
    if ohlcv_df is not None:
        ctx.update(_get_ohlcv_context(ticker, t0, ohlcv_df))
        ctx.update(_get_ohlcv_context(ticker, t0, ohlcv_df, ref_ticker="SPY"))
        ctx.update(_get_ohlcv_context(ticker, t0, ohlcv_df, ref_ticker="QQQ"))

    if ctx:
        lines.append("")
        lines.append("--- Market Context (from OHLCV) ---")
        for k, v in ctx.items():
            lines.append(f"{k}: {v}")

    return "\n".join(lines)


def build_news_brief(row: dict, packet: dict) -> str:
    """Build the news brief for the News/Fundamental Agent."""
    ticker = row["ticker"]
    t0 = row["t0_utc"]
    news = packet.get("news", {})
    ticker_articles = news.get("ticker_news", [])
    macro = news.get("macro_news", {})

    lines = [
        "=== NEWS BRIEF ===",
        f"Ticker: {ticker}",
        f"Event time (UTC): {t0}",
        f"Event time (ET): {_utc_to_et(t0)}",
        "",
    ]

    if not ticker_articles or news.get("no_news_found", False):
        lines.append("no_news_found: true")
        lines.append("No ticker-specific news articles were found in the 3-hour window around this event.")
    else:
        lines.append(f"--- Ticker News ({len(ticker_articles[:6])} articles) ---")
        for i, art in enumerate(ticker_articles[:6], 1):
            desc = (art.get("description") or "")[:200]
            lines.append(f"[{i}] {art.get('published_utc', '?')} | {art.get('publisher', '?')}")
            lines.append(f"    Title: {art.get('title', '?')}")
            if desc:
                lines.append(f"    Summary: {desc}")
            lines.append("")

    # Macro news
    macro_articles = []
    for idx_ticker in ("SPY", "QQQ"):
        for art in macro.get(idx_ticker, []):
            macro_articles.append(art)

    if macro_articles:
        lines.append(f"--- Macro Headlines ({len(macro_articles[:4])}) ---")
        for art in macro_articles[:4]:
            lines.append(f"  {art.get('published_utc', '?')} | {art.get('title', '?')}")
        lines.append("")

    return "\n".join(lines)


def build_macro_brief(row: dict, packet: dict, ohlcv_df=None) -> str:
    """Build the macro/correlation brief for the Macro Agent."""
    t0 = row["t0_utc"]
    ticker = row["ticker"]
    news = packet.get("news", {})
    macro = news.get("macro_news", {})

    lines = [
        "=== MACRO BRIEF ===",
        f"Event time (UTC): {t0}",
        f"Event time (ET): {_utc_to_et(t0)}",
        f"Ticker in question: {ticker}",
        "",
    ]

    # OHLCV-based SPY/QQQ returns
    if ohlcv_df is not None:
        for idx_tk in ("SPY", "QQQ"):
            ctx = _get_ohlcv_context(ticker, t0, ohlcv_df, ref_ticker=idx_tk)
            if ctx:
                lines.append(f"--- {idx_tk} Context ---")
                for k, v in ctx.items():
                    lines.append(f"  {k}: {v}")
                lines.append("")

    # Macro headlines
    macro_articles = []
    for idx_ticker in ("SPY", "QQQ"):
        for art in macro.get(idx_ticker, []):
            macro_articles.append(art)

    if macro_articles:
        lines.append(f"--- Macro Headlines ({len(macro_articles[:4])}) ---")
        for art in macro_articles[:4]:
            desc = (art.get("description") or "")[:150]
            lines.append(f"  {art.get('published_utc', '?')} | {art.get('title', '?')}")
            if desc:
                lines.append(f"    {desc}")
        lines.append("")
    else:
        lines.append("No macro headlines available for this time window.")

    return "\n".join(lines)


def build_combined_brief(row: dict, packet: dict) -> str:
    """Build a combined summary brief for the Skeptic."""
    ticker = row["ticker"]
    t0 = row["t0_utc"]
    direction = "UP" if int(row["direction"]) == 1 else "DOWN"
    news = packet.get("news", {})
    has_news = bool(news.get("ticker_news")) and not news.get("no_news_found", False)

    lines = [
        "=== COMBINED BRIEF (for Skeptic) ===",
        f"Ticker: {ticker} | Direction: {direction} | Time: {_utc_to_et(t0)}",
        f"Spike: {float(row['peak_abs_ret_3_bp']):.0f}bp, z={float(row['peak_z_ret_3']):.1f}, "
        f"cluster={int(row['cluster_len'])} bars",
        f"Liquidity: vol_mult={float(row['peak_volume_mult']):.0f}x, "
        f"range_mult={float(row['peak_range_mult']):.1f}x",
        f"Opening bell: {'YES' if _is_opening_bell(t0) else 'no'}",
        f"Has ticker news: {'yes' if has_news else 'NO — no news found'}",
    ]

    return "\n".join(lines)
