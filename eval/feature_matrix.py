"""
Feature matrix builder for V3 ML stage.

Loads V3b JSONL + events CSV and produces a flat numerical feature matrix
suitable for TabPFN / LightGBM.

Features (~36 total):
  Agent outputs  : label encodings, confidence, score differences, score ratio
  Ensemble       : vote margin, dir_weight, dir_prior
  V3 features    : idio_resid, rs_ratio, signed_vol_ratio*, vpin*, spread, run_len
  Event features : peak_z, peak_volume_mult, peak_range_mult  (cluster_len EXCLUDED — leaky)
  Context flags  : is_up_spike, has_pre_event_news, is_opening_bell, is_macro_driven

* signed_vol_ratio and vpin_proxy are recomputed from OHLCV pre-event bars (5 bars
  before t0) to avoid data leakage — the stored JSONL values used cluster_bars which
  overlap the 60-min label window.
"""

import csv
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_ENC = {"continuation": 0, "reversal": 1, "unclear": 2}


# ── Pre-event microstructure recompute (leak-free) ────────────────────────────

def _get_pre_bars(ticker, t0_utc_str, ohlcv_df, n_bars=5):
    """Last n_bars before t0 (t0 exclusive) for given ticker."""
    try:
        t0 = pd.Timestamp(t0_utc_str, tz="UTC") if not hasattr(pd.Timestamp(t0_utc_str), 'tzinfo') else pd.Timestamp(t0_utc_str)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        sub = ohlcv_df[ohlcv_df["ticker"] == ticker].sort_values("timestamp")
        before = sub[sub["timestamp"] < t0]
        if len(before) < 1:
            return None
        return before.iloc[-n_bars:].reset_index(drop=True)
    except Exception:
        return None


def _pre_event_signed_vol_ratio(ticker, t0_utc_str, ohlcv_df, n_bars=5):
    bars = _get_pre_bars(ticker, t0_utc_str, ohlcv_df, n_bars)
    if bars is None or len(bars) == 0:
        return np.nan
    btr = (bars["close"].values - bars["open"].values) / (
           bars["high"].values - bars["low"].values + 1e-9)
    signed = bars["volume"].values * btr
    total = float(bars["volume"].sum())
    if total < 1:
        return np.nan
    return float(np.sum(signed)) / total


def _pre_event_vpin(ticker, t0_utc_str, ohlcv_df, n_bars=5):
    bars = _get_pre_bars(ticker, t0_utc_str, ohlcv_df, n_bars)
    if bars is None or len(bars) == 0:
        return np.nan
    btr = (bars["close"].values - bars["open"].values) / (
           bars["high"].values - bars["low"].values + 1e-9)
    signed = bars["volume"].values * btr
    total = float(bars["volume"].sum())
    if total < 1:
        return np.nan
    return abs(float(np.sum(signed))) / total


def _safe_float(val, default=np.nan):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _label_to_int(lbl):
    return LABEL_ENC.get(lbl, 2)   # default unclear=2


def load_feature_matrix(jsonl_path, events_csv_path, include_unclear=False, ohlcv_path=None):
    """
    Load V3b outputs and build feature matrix.

    Parameters
    ----------
    jsonl_path : str
        Path to ip_outputs_v3.jsonl
    events_csv_path : str
        Path to selected_events.csv (for raw event features)
    include_unclear : bool
        If False (default), exclude events with true_label_proxy == 'unclear'
    ohlcv_path : str or None
        Path to OHLCV parquet. When provided, recomputes vpin_proxy and
        signed_vol_ratio from pre-event bars (5 bars before t0) to fix
        the cluster-bars data leakage. cluster_len is excluded regardless.

    Returns
    -------
    X : np.ndarray, shape (N, n_features)
    y : np.ndarray, shape (N,)  — 0=continuation, 1=reversal
    feature_names : list[str]
    records : list[dict]  — original records (for diagnostics)
    """
    # Load OHLCV for leak-free feature recompute
    ohlcv_df = None
    if ohlcv_path is not None:
        logger.info(f"Loading OHLCV from {ohlcv_path} for pre-event feature recompute")
        ohlcv_df = pd.read_parquet(ohlcv_path)
        if ohlcv_df["timestamp"].dt.tz is None:
            ohlcv_df["timestamp"] = ohlcv_df["timestamp"].dt.tz_localize("UTC")

    # Load events CSV for raw features
    events_by_id = {}
    with open(events_csv_path) as f:
        for row in csv.DictReader(f):
            events_by_id[row["event_id"]] = row

    # Load V3b jsonl
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "error" in r:
                    continue
                records.append(r)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(records)} valid records from {jsonl_path}")

    rows = []
    labels = []
    kept_records = []

    for r in records:
        true_lbl = r.get("true_label_proxy", "")
        if not true_lbl:
            continue
        if not include_unclear and true_lbl == "unclear":
            continue

        y_val = LABEL_ENC.get(true_lbl, 2)
        ev = events_by_id.get(r["event_id"], {})
        f3 = r.get("v3_features") or {}

        # ── Agent features ──────────────────────────────────────────────────
        micro_lbl  = _label_to_int(r.get("micro_label", ""))
        micro_conf = _safe_float(r.get("micro_conf", 0.5))
        micro_cont = _safe_float(r.get("micro_cont_score", 0))
        micro_rev  = _safe_float(r.get("micro_rev_score",  0))
        micro_diff = micro_cont - micro_rev   # positive → cont, negative → rev
        micro_total = micro_cont + micro_rev + 1e-6
        micro_ratio = micro_cont / micro_total

        news_lbl   = _label_to_int(r.get("news_label", ""))
        news_conf  = _safe_float(r.get("news_conf",  0.5))
        news_cont  = _safe_float(r.get("news_cont_score", 0))
        news_rev   = _safe_float(r.get("news_rev_score",  0))
        news_diff  = news_cont - news_rev
        news_total = news_cont + news_rev + 1e-6
        news_ratio = news_cont / news_total

        macro_lbl  = _label_to_int(r.get("macro_label", ""))
        macro_conf = _safe_float(r.get("macro_conf", 0.5))
        macro_cont = _safe_float(r.get("macro_cont_score", 0))
        macro_rev  = _safe_float(r.get("macro_rev_score",  0))
        macro_diff = macro_cont - macro_rev
        macro_total = macro_cont + macro_rev + 1e-6
        macro_ratio = macro_cont / macro_total

        # Agent entropy (disagreement between agents)
        agent_labels = [micro_lbl, news_lbl, macro_lbl]
        from collections import Counter
        cnt = Counter(agent_labels)
        unique = len(cnt)
        agent_agreement_enc = 0 if unique == 1 else (1 if unique == 2 else 2)

        # Combined score signals
        total_cont = micro_cont + news_cont + macro_cont
        total_rev  = micro_rev  + news_rev  + macro_rev
        combined_diff  = total_cont - total_rev
        combined_ratio = total_cont / (total_cont + total_rev + 1e-6)

        # ── Ensemble features ────────────────────────────────────────────────
        votes      = r.get("ensemble_votes", {})
        vote_cont  = _safe_float(votes.get("continuation", 0))
        vote_rev   = _safe_float(votes.get("reversal",     0))
        vote_margin = vote_cont - vote_rev
        dir_weight = _safe_float(r.get("dir_weight", 0.30))
        dir_prior  = _safe_float(r.get("dir_prior",  0.55))

        # ── V3 microstructure features ───────────────────────────────────────
        idio_resid   = _safe_float(f3.get("idio_resid_bp"))
        beta         = _safe_float(f3.get("beta", 1.0))
        rs_ratio     = _safe_float(f3.get("rs_ratio"))
        spread_bp    = _safe_float(f3.get("spread_bp"))
        run_len      = _safe_float(f3.get("pre_spike_run_len", 0))
        spy_z        = _safe_float(f3.get("spy_zscore_t0"))
        adj_z        = _safe_float(f3.get("adj_zscore_tod"))

        # signed_vol_ratio and vpin_proxy: recompute from pre-event bars if OHLCV
        # available, otherwise fall back to stored value (which may be leaky).
        ticker_r  = r.get("ticker", "")
        t0_utc_r  = r.get("t0_utc", "")
        if ohlcv_df is not None and ticker_r and t0_utc_r:
            svr  = _pre_event_signed_vol_ratio(ticker_r, t0_utc_r, ohlcv_df, n_bars=5)
            vpin = _pre_event_vpin(ticker_r, t0_utc_r, ohlcv_df, n_bars=5)
        else:
            # Leaky fallback — flagged in docstring
            svr  = _safe_float(f3.get("signed_vol_ratio"))
            vpin = _safe_float(f3.get("vpin_proxy"))

        # ── Context flags ────────────────────────────────────────────────────
        is_up        = float(r.get("is_up_spike", 0))
        has_news     = float(r.get("has_pre_event_news", 0))
        is_bell      = float(r.get("is_opening_bell", 0))
        is_macro     = float(r.get("is_macro_driven", 0))

        # ── Raw event features (from events CSV) ─────────────────────────────
        # cluster_len excluded: post-t0 data → leakage.
        # strength_score excluded: 20% of it is cluster_len (persistence term) → partial leak.
        peak_z       = _safe_float(ev.get("peak_z_ret_3"))
        peak_vol     = _safe_float(ev.get("peak_volume_mult"))
        peak_range   = _safe_float(ev.get("peak_range_mult"))

        feat_row = [
            # Agent labels (encoded)
            micro_lbl, news_lbl, macro_lbl,
            # Agent confidence
            micro_conf, news_conf, macro_conf,
            # Agent score differences (pos=cont, neg=rev)
            micro_diff, news_diff, macro_diff,
            # Agent score ratios (0=all_rev, 1=all_cont)
            micro_ratio, news_ratio, macro_ratio,
            # Combined signals
            combined_diff, combined_ratio,
            agent_agreement_enc,
            # Ensemble
            vote_cont, vote_rev, vote_margin,
            dir_weight, dir_prior,
            # V3 microstructure (vpin + signed_vol recomputed pre-event)
            idio_resid, rs_ratio, svr, vpin, spread_bp, run_len,
            spy_z, adj_z,
            # Context
            is_up, has_news, is_bell, is_macro,
            # Raw event (cluster_len and strength_score removed — both carry cluster_len)
            peak_z, peak_vol, peak_range,
        ]

        rows.append(feat_row)
        labels.append(y_val)
        kept_records.append(r)

    feature_names = [
        "micro_lbl", "news_lbl", "macro_lbl",
        "micro_conf", "news_conf", "macro_conf",
        "micro_diff", "news_diff", "macro_diff",
        "micro_ratio", "news_ratio", "macro_ratio",
        "combined_diff", "combined_ratio",
        "agent_agreement",
        "vote_cont", "vote_rev", "vote_margin",
        "dir_weight", "dir_prior",
        "idio_resid_bp", "rs_ratio", "signed_vol_ratio_pre", "vpin_proxy_pre", "spread_bp", "pre_spike_run_len",
        "spy_zscore_t0", "adj_zscore_tod",
        "is_up_spike", "has_pre_event_news", "is_opening_bell", "is_macro_driven",
        # cluster_len removed (post-t0 data leakage)
        # strength_score removed (20% of it is cluster_len via persistence term)
        "peak_z_ret_3", "peak_volume_mult", "peak_range_mult",
    ]

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)

    logger.info(f"Feature matrix: {X.shape[0]} events × {X.shape[1]} features")
    logger.info(f"Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    # NaN stats
    nan_counts = np.isnan(X).sum(axis=0)
    nan_feats = [(feature_names[i], int(nan_counts[i])) for i in range(len(feature_names)) if nan_counts[i] > 0]
    if nan_feats:
        logger.info(f"NaN features: {nan_feats}")

    return X, y, feature_names, kept_records


def impute_median(X):
    """Replace NaN with column medians (computed on non-NaN values)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            median = np.nanmedian(col)
            col[mask] = median if not np.isnan(median) else 0.0
            X[:, j] = col
    return X
