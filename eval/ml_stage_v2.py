"""
V3 ML Stage v2 — Temporal holdout + chronological CV + ablations + McNemar tests.

Replaces LOO-CV with honest temporal evaluation:
  1. Sort events by date → 80% train / 20% holdout (strictly later in time)
  2. 5-fold chronological CV on training set (comparable to LOO for larger N)
  3. Train final models on full train set
  4. Evaluate math-only / llm-only / full on same holdout → apples-to-apples ablation
  5. McNemar significance tests: full vs math-only, full vs llm-only
  6. Leakage null-model check: shuffled labels on train → expect ~50% CV
  7. Backtest PnL on holdout predictions with per-ticker transaction costs

Usage:
    python -m eval.ml_stage_v2
    python -m eval.ml_stage_v2 \\
        --jsonl out_agents_extended/ip_outputs_v3.jsonl \\
        --events_csv out_extended_final/selected_events.csv \\
        --ohlcv ohlcv_5min_extended.parquet \\
        --out_dir out_eval_extended
"""

import argparse
import json
import logging
import csv as csv_mod
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import chi2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Feature groups (mirrors ml_stage.py) ─────────────────────────────────────

AGENT_FEATURES = {
    "micro_lbl", "news_lbl", "macro_lbl",
    "micro_conf", "news_conf", "macro_conf",
    "micro_diff", "news_diff", "macro_diff",
    "micro_ratio", "news_ratio", "macro_ratio",
    "combined_diff", "combined_ratio", "agent_agreement",
    "vote_cont", "vote_rev", "vote_margin",
    "dir_weight", "dir_prior",
}
MATH_FEATURES = {
    "idio_resid_bp", "rs_ratio", "signed_vol_ratio_pre", "vpin_proxy_pre",
    "spread_bp", "pre_spike_run_len", "spy_zscore_t0", "adj_zscore_tod",
    "is_up_spike", "has_pre_event_news", "is_opening_bell", "is_macro_driven",
    "peak_z_ret_3", "peak_volume_mult", "peak_range_mult",
}

# Per-ticker round-trip transaction costs (basis points).
# Sources: Frazzini, Israel, Moskowitz (2018); broker spread surveys 2024-2025.
COSTS_BPS = {
    "AAPL": 12, "MSFT": 12, "SPY": 10, "QQQ": 10, "XLK": 10,
    "NVDA": 17, "TSLA": 18, "META": 15, "AMD": 17, "NFLX": 16,
    "PLTR": 28, "AMZN": 15, "GOOGL": 15, "JPM": 12, "COIN": 30,
}
DEFAULT_COST_BPS = 20  # fallback for unknown tickers

LABEL_DEC = {0: "continuation", 1: "reversal", 2: "unclear"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _accuracy(preds, labels):
    arr_p = np.asarray(preds)
    arr_l = np.asarray(labels)
    return float((arr_p == arr_l).mean()) if len(arr_l) else 0.0


def _feature_mask(feature_names, keep_set):
    return np.array([i for i, n in enumerate(feature_names) if n in keep_set])


def chrono_kfold(n, n_splits=5):
    """Yield (train_idx, val_idx) for chronologically ordered events."""
    indices = np.arange(n)
    fold_size = n // n_splits
    rem = n % n_splits
    folds, start = [], 0
    for k in range(n_splits):
        end = start + fold_size + (1 if k < rem else 0)
        folds.append(indices[start:end])
        start = end
    for k in range(n_splits):
        val   = folds[k]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        yield train, val


def mcnemar_p(y_true, preds_a, preds_b):
    """McNemar test (with continuity correction). Returns p-value."""
    a_ok = np.asarray(preds_a) == np.asarray(y_true)
    b_ok = np.asarray(preds_b) == np.asarray(y_true)
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    disc = n01 + n10
    if disc == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / disc if disc < 25 else (n01 - n10) ** 2 / disc
    return float(1 - chi2.cdf(stat, df=1))


# ── Classifier factories ──────────────────────────────────────────────────────

def _lgbm_clf():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=8, min_child_samples=5,
        learning_rate=0.05, n_estimators=100, verbose=-1, random_state=42,
    )


def _rf_clf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=4,
        class_weight="balanced", random_state=42,
    )


def _et_clf():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=4,
        class_weight="balanced", random_state=42,
    )


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_chrono_cv(X, y, clf_factory, n_splits=5, name="model"):
    """5-fold chronological CV. Returns per-fold accuracies."""
    accs = []
    for k, (tr_idx, va_idx) in enumerate(chrono_kfold(len(y), n_splits)):
        clf = clf_factory()
        clf.fit(X[tr_idx], y[tr_idx])
        acc = _accuracy(clf.predict(X[va_idx]), y[va_idx])
        accs.append(acc)
        log.info(f"    {name} fold {k+1}/{n_splits}: {acc:.3f}")
    return np.array(accs)


# ── Stacked ensemble predict ──────────────────────────────────────────────────

def stack_predict(models_and_proba_fns, X, n_classes=2):
    probas = np.zeros((len(X), n_classes))
    for clf in models_and_proba_fns:
        p = clf.predict_proba(X)
        probas[:, :p.shape[1]] += p
    probas /= len(models_and_proba_fns)
    return np.argmax(probas, axis=1), probas


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(records, preds, events_df_indexed, label_name="full"):
    """
    Compute per-trade PnL in basis points.

    For each event:
      trade_direction = spike_direction × prediction_direction
        (continuation → same as spike; reversal → opposite)
      gross_bps = forward_return_60m × 10000
      pnl = trade_direction × gross_bps − round_trip_cost_bps

    Parameters
    ----------
    records : list[dict]
        JSONL records for holdout events (sorted by time, same order as preds).
    preds : array-like of int
        Predicted labels (0=continuation, 1=reversal).
    events_df_indexed : pd.DataFrame
        events CSV indexed by event_id, must have 'forward_return_60m'.
    """
    pnl_list = []
    meta_list = []

    for record, pred in zip(records, preds):
        event_id = record.get("event_id", "")
        ticker   = record.get("ticker", "")
        direction = int(record.get("direction", 1))   # +1 up spike, -1 down spike

        # forward_return_60m from events CSV
        ev_row = events_df_indexed.get(event_id)
        if ev_row is None:
            continue
        try:
            fwd_ret_frac = float(ev_row.get("forward_return_60m", 0) or 0)
        except (TypeError, ValueError):
            continue

        # Prediction direction: cont=same as spike, rev=opposite
        pred_label = LABEL_DEC.get(int(pred), "unclear")
        if pred_label == "unclear":
            continue  # don't trade unclear predictions
        pred_dir = +1 if pred_label == "continuation" else -1

        trade_dir  = direction * pred_dir   # +1=long, -1=short
        gross_bps  = fwd_ret_frac * 10000   # stock's actual return in bps
        cost_bps   = COSTS_BPS.get(ticker, DEFAULT_COST_BPS)
        pnl_bps    = trade_dir * gross_bps - cost_bps

        pnl_list.append(pnl_bps)
        meta_list.append({
            "event_id":  event_id,
            "ticker":    ticker,
            "pred":      pred_label,
            "true":      LABEL_DEC.get(int(record.get("true_label_proxy_enc", 2)), "?"),
            "gross_bps": round(gross_bps, 1),
            "cost_bps":  cost_bps,
            "pnl_bps":   round(pnl_bps, 1),
        })

    if not pnl_list:
        return {"error": "no tradeable events in holdout"}

    pnl = np.array(pnl_list)
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd   = peak - cum

    # Events per year from the holdout record timestamps
    try:
        times = [pd.Timestamp(r["t0_utc"]) for r in records if r.get("t0_utc")]
        if len(times) >= 2:
            span_years = (max(times) - min(times)).days / 365.25
            events_per_year = len(pnl_list) / max(span_years, 0.1)
        else:
            events_per_year = 250
    except Exception:
        events_per_year = 250

    ann_factor = float(np.sqrt(events_per_year))
    sharpe     = float(pnl.mean() / (pnl.std() + 1e-9) * ann_factor)

    result = {
        "n_trades":           len(pnl),
        "mean_pnl_bps":       round(float(pnl.mean()),  2),
        "median_pnl_bps":     round(float(np.median(pnl)), 2),
        "std_pnl_bps":        round(float(pnl.std()),   2),
        "total_pnl_bps":      round(float(pnl.sum()),   2),
        "win_rate":           round(float((pnl > 0).mean()), 4),
        "annualized_sharpe":  round(sharpe, 3),
        "max_drawdown_bps":   round(float(dd.max()), 1),
        "events_per_year_est": round(events_per_year, 1),
        "pnl_per_trade":      [round(p, 1) for p in pnl_list],
        "trade_meta":         meta_list[:20],  # first 20 for inspection
    }

    log.info(f"\n{'─'*50}")
    log.info(f"BACKTEST [{label_name}]  n={len(pnl)} trades")
    log.info(f"  Mean PnL     : {result['mean_pnl_bps']:+.1f} bps/trade")
    log.info(f"  Median PnL   : {result['median_pnl_bps']:+.1f} bps/trade")
    log.info(f"  Win rate     : {result['win_rate']:.1%}")
    log.info(f"  Ann. Sharpe  : {result['annualized_sharpe']:+.3f}")
    log.info(f"  Max drawdown : {result['max_drawdown_bps']:.0f} bps")
    log.info(f"  Total PnL    : {result['total_pnl_bps']:+.0f} bps")
    viable = "✓ VIABLE" if result["mean_pnl_bps"] > 0 else "✗ NOT PROFITABLE"
    log.info(f"  Assessment   : {viable}")
    log.info(f"{'─'*50}")

    return result


# ── LGBM feature importance ───────────────────────────────────────────────────

def save_importance(feature_names, importances, path, top_n=20):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        order = np.argsort(importances)[::-1][:top_n]
        names = [feature_names[i] for i in reversed(order)]
        vals  = [importances[i]   for i in reversed(order)]
        colors = ["#4e79a7" if n in AGENT_FEATURES else "#f28e2b" for n in names]

        fig, ax = plt.subplots(figsize=(9, 0.45 * top_n + 1.2))
        ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Mean LGBM importance (temporal CV folds)")
        ax.set_title("Feature Importance — V3 ML Stage v2 (temporal holdout)", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[
            Patch(color="#4e79a7", label="LLM agent features"),
            Patch(color="#f28e2b", label="Math / microstructure features"),
        ], loc="lower right")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Feature importance plot → {path}")
    except Exception as e:
        log.warning(f"Could not save importance plot: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V3 ML Stage v2 — temporal holdout evaluation")
    parser.add_argument("--jsonl",        default="out_agents_v3b/ip_outputs_v3.jsonl")
    parser.add_argument("--events_csv",   default="out_final/selected_events.csv")
    parser.add_argument("--ohlcv",        default="ohlcv_5min.parquet")
    parser.add_argument("--out_dir",      default="out_agents_v3b")
    parser.add_argument("--holdout_frac", type=float, default=0.20,
                        help="Fraction of events (by time) held out for final test")
    parser.add_argument("--n_splits",     type=int,   default=5,
                        help="Chronological CV folds on training set")
    parser.add_argument("--include_unclear", action="store_true")
    parser.add_argument("--null_model_runs", type=int, default=10,
                        help="Number of shuffle runs for null model check")
    args = parser.parse_args()

    from eval.feature_matrix import load_feature_matrix, impute_median, LABEL_ENC

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_path = args.ohlcv if Path(args.ohlcv).exists() else None
    if ohlcv_path is None:
        log.warning(f"OHLCV not found at {args.ohlcv} — using stored (potentially leaky) VPIN values")

    # ── Load features ──────────────────────────────────────────────────────────
    log.info("Building feature matrix...")
    X_raw, y, feature_names, records = load_feature_matrix(
        args.jsonl, args.events_csv,
        include_unclear=args.include_unclear,
        ohlcv_path=ohlcv_path,
    )
    import numpy as _np
    X_raw[~_np.isfinite(X_raw)] = _np.nan   # replace inf/-inf with NaN before imputation
    X = impute_median(X_raw)
    n = len(y)
    log.info(f"Matrix: {n} events × {X.shape[1]} features | Labels: {dict(Counter(y.tolist()))}")

    # Load events CSV for forward_return_60m (backtest)
    events_by_id = {}
    with open(args.events_csv) as f:
        for row in csv_mod.DictReader(f):
            events_by_id[row["event_id"]] = row

    # ── Sort by timestamp (chronological) ─────────────────────────────────────
    log.info("Sorting by event timestamp...")
    timestamps = []
    for r in records:
        ts_str = r.get("t0_utc", "")
        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            ts = pd.Timestamp("2000-01-01", tz="UTC")
        timestamps.append(ts)

    sort_idx = np.argsort(timestamps)
    X        = X[sort_idx]
    y        = y[sort_idx]
    records  = [records[i] for i in sort_idx]
    timestamps = [timestamps[i] for i in sort_idx]

    log.info(f"Date range: {timestamps[0].date()} → {timestamps[-1].date()}")

    # ── Temporal 80/20 split ──────────────────────────────────────────────────
    split_point = int(n * (1 - args.holdout_frac))
    X_train,   X_holdout   = X[:split_point],    X[split_point:]
    y_train,   y_holdout   = y[:split_point],    y[split_point:]
    rec_train, rec_holdout = records[:split_point], records[split_point:]
    ts_train,  ts_holdout  = timestamps[:split_point], timestamps[split_point:]

    log.info(f"\nTemporal split ({int((1-args.holdout_frac)*100)}/{int(args.holdout_frac*100)}):")
    log.info(f"  Train  : {len(y_train)} events | {ts_train[0].date()} → {ts_train[-1].date()}")
    log.info(f"  Holdout: {len(y_holdout)} events | {ts_holdout[0].date()} → {ts_holdout[-1].date()}")
    log.info(f"  Train labels  : {dict(Counter(y_train.tolist()))}")
    log.info(f"  Holdout labels: {dict(Counter(y_holdout.tolist()))}")

    # ── Feature group masks ───────────────────────────────────────────────────
    math_idx   = _feature_mask(feature_names, MATH_FEATURES)
    agent_idx  = _feature_mask(feature_names, AGENT_FEATURES)
    full_idx   = np.arange(len(feature_names))

    configs = [
        ("math_only",  math_idx),
        ("llm_only",   agent_idx),
        ("full_stack", full_idx),
    ]

    # ── Baselines ─────────────────────────────────────────────────────────────
    dir_preds = [1 if r.get("direction", 0) == 1 else 0 for r in rec_holdout]
    ens_preds = [LABEL_ENC.get(r.get("ensemble_label", ""), 2) for r in rec_holdout]
    dir_acc   = _accuracy(dir_preds, y_holdout)
    ens_acc   = _accuracy(ens_preds, y_holdout)
    log.info(f"\nBaselines (holdout):")
    log.info(f"  Direction heuristic : {dir_acc:.3f}  ({dir_acc*100:.1f}%)")
    log.info(f"  V3b ensemble        : {ens_acc:.3f}  ({ens_acc*100:.1f}%)")

    # ── Null model check (on train set) ───────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"NULL MODEL CHECK (shuffled labels, {args.null_model_runs} runs)")
    log.info(f"{'='*55}")
    null_accs = []
    for run_i in range(args.null_model_runs):
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        fold_accs = []
        for tr_idx, va_idx in chrono_kfold(len(y_shuffled), args.n_splits):
            clf = _lgbm_clf()
            clf.fit(X_train[tr_idx], y_shuffled[tr_idx])
            fold_accs.append(_accuracy(clf.predict(X_train[va_idx]), y_shuffled[va_idx]))
        null_accs.append(np.mean(fold_accs))
    null_mean = np.mean(null_accs)
    null_std  = np.std(null_accs)
    log.info(f"  Null model CV acc: {null_mean:.3f} ± {null_std:.3f}")
    leak_flag = null_mean > 0.56
    log.info(f"  {'⚠ WARNING: null >56%, possible structural leak' if leak_flag else '✓ Null ~50% — no structural leakage detected'}")

    # ── CV on training set + holdout evaluation ───────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"CHRONOLOGICAL {args.n_splits}-FOLD CV (train set) + HOLDOUT EVALUATION")
    log.info(f"{'='*55}")

    cv_results  = {}
    holdout_preds = {}
    lgbm_importances = np.zeros(len(feature_names))

    for label, feat_idx in configs:
        X_tr   = X_train[:, feat_idx]
        X_ho   = X_holdout[:, feat_idx]
        fn_sub = [feature_names[i] for i in feat_idx]

        log.info(f"\n  Config: {label} ({len(feat_idx)} features)")

        # CV
        lgbm_cv = run_chrono_cv(X_tr, y_train, _lgbm_clf, args.n_splits, f"LGBM[{label}]")
        rf_cv   = run_chrono_cv(X_tr, y_train, _rf_clf,   args.n_splits, f"RF[{label}]")
        et_cv   = run_chrono_cv(X_tr, y_train, _et_clf,   args.n_splits, f"ET[{label}]")

        cv_results[label] = {
            "lgbm": {"mean": float(lgbm_cv.mean()), "std": float(lgbm_cv.std()), "folds": lgbm_cv.tolist()},
            "rf":   {"mean": float(rf_cv.mean()),   "std": float(rf_cv.std()),   "folds": rf_cv.tolist()},
            "et":   {"mean": float(et_cv.mean()),   "std": float(et_cv.std()),   "folds": et_cv.tolist()},
            "stack_mean": float((lgbm_cv + rf_cv + et_cv).mean() / 3),
        }
        log.info(f"    CV  LGBM : {lgbm_cv.mean():.3f} ± {lgbm_cv.std():.3f}")
        log.info(f"    CV  RF   : {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")
        log.info(f"    CV  ET   : {et_cv.mean():.3f} ± {et_cv.std():.3f}")

        # Train final models on full training set
        lgbm_final = _lgbm_clf();  lgbm_final.fit(X_tr, y_train)
        rf_final   = _rf_clf();    rf_final.fit(X_tr, y_train)
        et_final   = _et_clf();    et_final.fit(X_tr, y_train)

        # Holdout stacked prediction
        ho_preds, ho_probas = stack_predict([lgbm_final, rf_final, et_final], X_ho)
        ho_acc = _accuracy(ho_preds, y_holdout)
        cv_results[label]["holdout_acc"] = ho_acc
        holdout_preds[label] = ho_preds.tolist()
        # Embed true label encoding in records (used by backtest)
        for i, r in enumerate(rec_holdout):
            r["true_label_proxy_enc"] = LABEL_ENC.get(r.get("true_label_proxy", ""), 2)

        log.info(f"    Holdout stacked acc: {ho_acc:.3f}  ({ho_acc*100:.1f}%)")

        # Accumulate LGBM importance for full config
        if label == "full_stack":
            for i_fold, (tr_idx, _) in enumerate(chrono_kfold(len(y_train), args.n_splits)):
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(
                    objective="binary", num_leaves=8, min_child_samples=5,
                    learning_rate=0.05, n_estimators=100, verbose=-1, random_state=42,
                )
                clf.fit(X_tr[tr_idx], y_train[tr_idx])
                lgbm_importances[feat_idx] += clf.feature_importances_
            lgbm_importances /= args.n_splits

    # ── McNemar tests ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("McNemar SIGNIFICANCE TESTS (on holdout)")
    log.info(f"{'='*55}")
    y_ho = y_holdout.tolist()
    p_full_vs_math = mcnemar_p(y_ho, holdout_preds["full_stack"], holdout_preds["math_only"])
    p_full_vs_llm  = mcnemar_p(y_ho, holdout_preds["full_stack"], holdout_preds["llm_only"])
    p_llm_vs_math  = mcnemar_p(y_ho, holdout_preds["llm_only"],   holdout_preds["math_only"])

    log.info(f"  full vs math-only : p = {p_full_vs_math:.4f}  {'✓ sig' if p_full_vs_math < 0.05 else '— not sig (p≥0.05)'}")
    log.info(f"  full vs llm-only  : p = {p_full_vs_llm:.4f}  {'✓ sig' if p_full_vs_llm  < 0.05 else '— not sig (p≥0.05)'}")
    log.info(f"  llm vs math-only  : p = {p_llm_vs_math:.4f}  {'✓ sig' if p_llm_vs_math  < 0.05 else '— not sig (p≥0.05)'}")

    # ── Backtest PnL ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("BACKTEST PnL (full_stack predictions on holdout)")
    log.info(f"{'='*55}")
    backtest = run_backtest(rec_holdout, holdout_preds["full_stack"], events_by_id, label_name="full_stack")

    # ── Feature importance ────────────────────────────────────────────────────
    order = np.argsort(lgbm_importances)[::-1]
    log.info(f"\n{'─'*50}")
    log.info("Top 15 features (LightGBM, full config, CV-averaged):")
    for rank, idx in enumerate(order[:15], 1):
        bar = "█" * int(lgbm_importances[idx] / (lgbm_importances[order[0]] + 1e-9) * 20)
        log.info(f"  {rank:>2}. {feature_names[idx]:<30} {lgbm_importances[idx]:>6.2f}  {bar}")
    save_importance(feature_names, lgbm_importances, out_dir / "feature_importance_v2.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*55}")
    log.info(f"  N total        : {n}")
    log.info(f"  N train        : {len(y_train)}")
    log.info(f"  N holdout      : {len(y_holdout)}")
    log.info(f"  Date split     : {ts_train[-1].date()} / {ts_holdout[0].date()}")
    log.info(f"  Direction heur : {dir_acc:.3f} ({dir_acc*100:.1f}%) [holdout]")
    log.info(f"  V3b ensemble   : {ens_acc:.3f} ({ens_acc*100:.1f}%) [holdout]")
    for label, _ in configs:
        ha = cv_results[label]["holdout_acc"]
        log.info(f"  {label:<15}: {ha:.3f} ({ha*100:.1f}%) [holdout]")
    log.info(f"  Backtest Sharpe: {backtest.get('annualized_sharpe', 'N/A')}")
    log.info(f"  Null model acc : {null_mean:.3f} ± {null_std:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "meta": {
            "n_total": n,
            "n_train": len(y_train),
            "n_holdout": len(y_holdout),
            "holdout_frac": args.holdout_frac,
            "n_splits": args.n_splits,
            "train_date_range": [str(ts_train[0].date()),  str(ts_train[-1].date())],
            "holdout_date_range":[str(ts_holdout[0].date()), str(ts_holdout[-1].date())],
            "label_dist_train":  dict(Counter(y_train.tolist())),
            "label_dist_holdout":dict(Counter(y_holdout.tolist())),
        },
        "baselines": {
            "direction_heuristic_holdout": dir_acc,
            "v3b_ensemble_holdout":        ens_acc,
        },
        "null_model": {
            "mean": null_mean, "std": null_std,
            "all_runs": null_accs, "leak_flagged": leak_flag,
        },
        "cv_results": cv_results,
        "holdout_accuracy": {k: cv_results[k]["holdout_acc"] for k, _ in configs},
        "holdout_preds": holdout_preds,   # consumed by backtest_pnl.py
        "mcnemar": {
            "full_vs_math_p":   p_full_vs_math,
            "full_vs_llm_p":    p_full_vs_llm,
            "llm_vs_math_p":    p_llm_vs_math,
        },
        "backtest": backtest,
        "feature_names": feature_names,
        "lgbm_importance": lgbm_importances.tolist(),
    }

    out_path = out_dir / "ml_stage_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
