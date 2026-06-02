"""
V3 ML Stage — TabPFN + LightGBM on top of V3b agent outputs.

Approach:
  1. Build feature matrix from V3b JSONL (agent scores + microstructure + context)
  2. Run TabPFN v2 with Leave-One-Out CV (N=85, small dataset)
  3. Run LightGBM with Leave-One-Out CV (for feature importance)
  4. Stack ensemble: average TabPFN + LightGBM probability predictions
  5. Report accuracy vs V3b ensemble and direction heuristic
  6. Print feature importance ranking

Usage:
    python eval/ml_stage.py
    python eval/ml_stage.py --jsonl out_agents_v3b/ip_outputs_v3.jsonl
    python eval/ml_stage.py --include_unclear   # include unclear events (3-class)
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _accuracy(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    n = len(labels) if hasattr(labels, '__len__') else 0
    return round(correct / n, 4) if n else 0.0


def _clear_accuracy(preds, labels, true_labels_raw):
    """Accuracy excluding true 'unclear' events (label==2)."""
    clear_pairs = [(p, l) for p, l, tr in zip(preds, labels, true_labels_raw) if tr != "unclear"]
    if not clear_pairs:
        return None, 0
    correct = sum(p == l for p, l in clear_pairs)
    return round(correct / len(clear_pairs), 4), len(clear_pairs)


def run_loocv_model(X, y, clf_factory, name="model"):
    """
    Leave-One-Out cross-validation with any sklearn-compatible classifier.
    Returns predicted labels and predicted probabilities.
    """
    n      = len(y)
    n_cls  = len(np.unique(y))
    preds  = np.zeros(n, dtype=int)
    probas = np.zeros((n, n_cls))

    logger.info(f"  {name} LOO-CV over {n} samples...")
    for i in range(n):
        mask = np.arange(n) != i
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[[i]]

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[0]
        pred  = int(clf.classes_[np.argmax(proba)])
        preds[i] = pred
        for ci, cls in enumerate(clf.classes_):
            if int(cls) < n_cls:
                probas[i, int(cls)] = proba[ci]

        if (i + 1) % 20 == 0:
            logger.info(f"    LOO {i+1}/{n} done, running acc={_accuracy(preds[:i+1], y[:i+1]):.3f}")

    return preds, probas


def run_loocv_lgbm(X, y, feature_names):
    """
    Leave-One-Out cross-validation with LightGBM.
    Returns predicted labels, probabilities, and feature importance.
    """
    import lightgbm as lgb
    n = len(y)
    n_classes = len(np.unique(y))
    preds  = np.zeros(n, dtype=int)
    probas = np.zeros((n, n_classes))
    importances = np.zeros(len(feature_names))

    params = {
        "objective":     "binary" if n_classes == 2 else "multiclass",
        "num_class":     n_classes if n_classes > 2 else None,
        "num_leaves":    8,
        "min_child_samples": 5,
        "learning_rate": 0.05,
        "n_estimators":  100,
        "verbose":       -1,
        "random_state":  42,
    }
    if params["num_class"] is None:
        del params["num_class"]

    logger.info(f"  LightGBM LOO-CV over {n} samples...")
    for i in range(n):
        mask = np.arange(n) != i
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[[i]]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[0]
        pred  = int(np.argmax(proba))
        preds[i] = pred
        probas[i, :len(proba)] = proba
        importances += clf.feature_importances_

        if (i + 1) % 10 == 0:
            logger.info(f"    LOO {i+1}/{n} done, running acc={_accuracy(preds[:i+1], y[:i+1]):.3f}")

    importances /= n  # average importance across folds
    return preds, probas, importances


def run_stacked_ensemble(tabpfn_probas, lgbm_probas, y, w_tabpfn=0.60, w_lgbm=0.40):
    """Average TabPFN + LightGBM probabilities."""
    stacked = w_tabpfn * tabpfn_probas + w_lgbm * lgbm_probas
    preds   = np.argmax(stacked, axis=1).astype(int)
    return preds, stacked


def print_feature_importance(feature_names, importances, top_n=15):
    order = np.argsort(importances)[::-1]
    logger.info(f"\n{'─'*50}")
    logger.info(f"Top {top_n} features (LightGBM importance):")
    logger.info(f"{'─'*50}")
    for rank, idx in enumerate(order[:top_n], 1):
        bar = "█" * int(importances[idx] / importances[order[0]] * 20)
        logger.info(f"  {rank:>2}. {feature_names[idx]:<30} {importances[idx]:>6.2f}  {bar}")
    logger.info(f"{'─'*50}")


def save_importance_plot(feature_names, importances, out_path, top_n=20):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(importances)[::-1][:top_n]
        names = [feature_names[i] for i in reversed(order)]
        vals  = [importances[i]   for i in reversed(order)]

        # Colour-code by feature group
        AGENT_SET = {
            "micro_lbl","news_lbl","macro_lbl",
            "micro_conf","news_conf","macro_conf",
            "micro_diff","news_diff","macro_diff",
            "micro_ratio","news_ratio","macro_ratio",
            "combined_diff","combined_ratio","agent_agreement",
            "vote_cont","vote_rev","vote_margin","dir_weight","dir_prior",
        }
        colors = ["#4e79a7" if n in AGENT_SET else "#f28e2b" for n in names]

        fig, ax = plt.subplots(figsize=(9, 0.45 * top_n + 1.2))
        bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Mean LGBM importance (avg over LOO folds)", fontsize=10)
        ax.set_title("Feature Importance — V3 ML Stage (leak-fixed)", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.spines[["top","right"]].set_visible(False)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#4e79a7", label="LLM agent features"),
            Patch(color="#f28e2b", label="Math / microstructure features"),
        ], fontsize=9, loc="lower right")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved → {out_path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping importance plot")


def main():
    parser = argparse.ArgumentParser(description="V3 ML Stage — LightGBM + RandomForest + ExtraTrees")
    parser.add_argument("--jsonl",           default="out_agents_v3b/ip_outputs_v3.jsonl")
    parser.add_argument("--events_csv",      default="out_final/selected_events.csv")
    parser.add_argument("--ohlcv",           default="ohlcv_5min.parquet",
                        help="OHLCV parquet for pre-event VPIN/signed_vol recompute (leak fix)")
    parser.add_argument("--out_dir",         default="out_agents_v3b")
    parser.add_argument("--include_unclear", action="store_true")
    args = parser.parse_args()

    from eval.feature_matrix import load_feature_matrix, impute_median, LABEL_ENC
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

    ohlcv_path = args.ohlcv if Path(args.ohlcv).exists() else None
    if ohlcv_path is None:
        logger.warning(f"OHLCV not found at {args.ohlcv} — using stored (potentially leaky) VPIN/signed_vol values")

    # ── Load features ──
    logger.info("Building feature matrix...")
    X_raw, y, feature_names, records = load_feature_matrix(
        args.jsonl, args.events_csv,
        include_unclear=args.include_unclear,
        ohlcv_path=ohlcv_path,
    )
    X = impute_median(X_raw)

    n = len(y)
    logger.info(f"Matrix: {n} events × {X.shape[1]} features")
    logger.info(f"Label dist: {dict(Counter(y.tolist()))}")

    true_labels_raw = [r.get("true_label_proxy", "") for r in records]

    # ── Baselines ──
    direction_preds = [
        1 if r.get("direction", 0) == 1 else 0   # up→reversal=1, down→cont=0
        for r in records
    ]
    dir_acc     = _accuracy(direction_preds, y.tolist())
    v3ens_preds = [LABEL_ENC.get(r.get("ensemble_label", ""), 2) for r in records]
    v3ens_acc   = _accuracy(v3ens_preds, y.tolist())

    logger.info(f"\n{'='*55}")
    logger.info("BASELINES (on this subset):")
    logger.info(f"  Direction heuristic : {dir_acc:.4f}  ({dir_acc*100:.1f}%)")
    logger.info(f"  V3b ensemble        : {v3ens_acc:.4f}  ({v3ens_acc*100:.1f}%)")

    # ── Ablation: math-only vs agents-only vs full ──────────────────────────
    # Proves whether LLM agents add "Contextual Alpha" beyond pure microstructure.
    #
    # math_only  : microstructure + context flags + raw event features
    # agents_only: agent labels/scores/ratios + ensemble votes
    # full       : everything (same as main run below)
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
        "peak_z_ret_3", "peak_volume_mult", "peak_range_mult", "strength_score",
    }

    def _mask(names, keep):
        return np.array([i for i, n in enumerate(names) if n in keep])

    math_idx   = _mask(feature_names, MATH_FEATURES)
    agents_idx = _mask(feature_names, AGENT_FEATURES)
    full_idx   = np.arange(len(feature_names))

    logger.info(f"\n{'='*55}")
    logger.info("ABLATION STUDY (LightGBM LOO-CV)")
    logger.info(f"{'='*55}")

    ablation_results = {}
    for label, idx in [("math_only", math_idx), ("agents_only", agents_idx), ("full_stack", full_idx)]:
        X_sub = X[:, idx]
        fn_sub = [feature_names[i] for i in idx]
        preds_ab, _, imps_ab = run_loocv_lgbm(X_sub, y, fn_sub)
        acc_ab = _accuracy(preds_ab.tolist(), y.tolist())
        ablation_results[label] = {"acc": acc_ab, "importances": imps_ab, "feature_names": fn_sub}
        logger.info(f"  {label:<15} : {acc_ab:.4f}  ({acc_ab*100:.1f}%)  [{len(idx)} features]")

    math_acc   = ablation_results["math_only"]["acc"]
    agents_acc = ablation_results["agents_only"]["acc"]
    full_abl   = ablation_results["full_stack"]["acc"]

    agent_delta = full_abl - math_acc
    logger.info(f"\n  Agent contribution (full − math_only): {agent_delta:+.4f} ({agent_delta*100:+.1f}pp)")
    if agent_delta > 0.02:
        logger.info("  → LLM agents add meaningful Contextual Alpha ✓")
    elif agent_delta > 0:
        logger.info("  → LLM agents add marginal signal (within noise at N=85)")
    else:
        logger.info("  → LLM agents do NOT improve over math features alone ✗")
    logger.info(f"{'='*55}")

    # ── LightGBM LOO-CV ──
    logger.info("\nRunning LightGBM LOO-CV...")
    lgbm_preds, lgbm_probas, lgbm_importances = run_loocv_lgbm(X, y, feature_names)
    lgbm_acc = _accuracy(lgbm_preds.tolist(), y.tolist())
    l_clr, l_clr_n = _clear_accuracy(lgbm_preds.tolist(), y.tolist(), true_labels_raw)
    logger.info(f"  LightGBM  acc={lgbm_acc:.4f}  clear={l_clr} (n={l_clr_n})")
    logger.info(f"  LightGBM pred dist: {dict(Counter(lgbm_preds.tolist()))}")

    # ── RandomForest LOO-CV ──
    logger.info("\nRunning RandomForest LOO-CV...")
    rf_factory = lambda: RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=4,
        class_weight="balanced", random_state=42
    )
    rf_preds, rf_probas = run_loocv_model(X, y, rf_factory, name="RandomForest")
    rf_acc = _accuracy(rf_preds.tolist(), y.tolist())
    r_clr, r_clr_n = _clear_accuracy(rf_preds.tolist(), y.tolist(), true_labels_raw)
    logger.info(f"  RandomForest  acc={rf_acc:.4f}  clear={r_clr} (n={r_clr_n})")
    logger.info(f"  RandomForest pred dist: {dict(Counter(rf_preds.tolist()))}")

    # ── ExtraTrees LOO-CV ──
    logger.info("\nRunning ExtraTrees LOO-CV...")
    et_factory = lambda: ExtraTreesClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=4,
        class_weight="balanced", random_state=42
    )
    et_preds, et_probas = run_loocv_model(X, y, et_factory, name="ExtraTrees")
    et_acc = _accuracy(et_preds.tolist(), y.tolist())
    e_clr, e_clr_n = _clear_accuracy(et_preds.tolist(), y.tolist(), true_labels_raw)
    logger.info(f"  ExtraTrees  acc={et_acc:.4f}  clear={e_clr} (n={e_clr_n})")
    logger.info(f"  ExtraTrees pred dist: {dict(Counter(et_preds.tolist()))}")

    # ── Stacked ensemble (equal weight average) ──
    logger.info("\nBuilding stacked ensemble (LGBM + RF + ET)...")
    n_cls = lgbm_probas.shape[1]

    def _pad(proba, n_cls):
        p = np.zeros((len(proba), n_cls))
        p[:, :proba.shape[1]] = proba
        return p

    stacked_probas = (
        _pad(lgbm_probas, n_cls) * (1/3) +
        _pad(rf_probas,   n_cls) * (1/3) +
        _pad(et_probas,   n_cls) * (1/3)
    )
    stacked_preds = np.argmax(stacked_probas, axis=1).astype(int)
    stacked_acc   = _accuracy(stacked_preds.tolist(), y.tolist())
    s_clr, s_clr_n = _clear_accuracy(stacked_preds.tolist(), y.tolist(), true_labels_raw)
    logger.info(f"  Stacked (LGBM+RF+ET) acc={stacked_acc:.4f}  clear={s_clr} (n={s_clr_n})")
    logger.info(f"  Stacked pred dist: {dict(Counter(stacked_preds.tolist()))}")

    # ── LGBM + RF only (best 2) ──
    stack2_probas = _pad(lgbm_probas, n_cls) * 0.5 + _pad(rf_probas, n_cls) * 0.5
    stack2_preds  = np.argmax(stack2_probas, axis=1).astype(int)
    stack2_acc    = _accuracy(stack2_preds.tolist(), y.tolist())
    s2_clr, s2_clr_n = _clear_accuracy(stack2_preds.tolist(), y.tolist(), true_labels_raw)
    logger.info(f"  Stacked (LGBM+RF)    acc={stack2_acc:.4f}  clear={s2_clr} (n={s2_clr_n})")

    # ── Summary ──
    logger.info(f"\n{'='*55}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*55}")
    logger.info(f"  V2.1 full pipeline     : 0.163  (16.3%)")
    logger.info(f"  V1  full pipeline      : 0.352  (35.2%)")
    logger.info(f"  Direction heuristic    : {dir_acc:.4f}  ({dir_acc*100:.1f}%)")
    logger.info(f"  V3b ensemble           : {v3ens_acc:.4f}  ({v3ens_acc*100:.1f}%)")
    logger.info(f"  LightGBM LOO-CV        : {lgbm_acc:.4f}  ({lgbm_acc*100:.1f}%)  {'✓' if lgbm_acc > v3ens_acc else '✗'} beats V3b")
    logger.info(f"  RandomForest LOO-CV    : {rf_acc:.4f}  ({rf_acc*100:.1f}%)  {'✓' if rf_acc > v3ens_acc else '✗'} beats V3b")
    logger.info(f"  ExtraTrees LOO-CV      : {et_acc:.4f}  ({et_acc*100:.1f}%)  {'✓' if et_acc > v3ens_acc else '✗'} beats V3b")
    logger.info(f"  Stacked LGBM+RF+ET     : {stacked_acc:.4f}  ({stacked_acc*100:.1f}%)  {'✓' if stacked_acc > v3ens_acc else '✗'} beats V3b")
    logger.info(f"  Stacked LGBM+RF        : {stack2_acc:.4f}  ({stack2_acc*100:.1f}%)  {'✓' if stack2_acc > v3ens_acc else '✗'} beats V3b")
    logger.info(f"{'='*55}")

    # ── Feature importance ──
    print_feature_importance(feature_names, lgbm_importances)
    save_importance_plot(
        feature_names, lgbm_importances,
        out_path=str(Path(args.out_dir) / "feature_importance.png"),
    )

    # ── Save results ──
    best_acc = max(lgbm_acc, rf_acc, et_acc, stacked_acc, stack2_acc)
    results = {
        "n_events":    n,
        "label_dist":  dict(Counter(y.tolist())),
        "baselines": {
            "direction_heuristic": dir_acc,
            "v3b_ensemble":        v3ens_acc,
        },
        "ablation": {
            "math_only":   math_acc,
            "agents_only": agents_acc,
            "full_stack":  full_abl,
            "agent_delta_pp": round((full_abl - math_acc) * 100, 2),
        },
        "lgbm_loocv":     lgbm_acc,
        "rf_loocv":       rf_acc,
        "et_loocv":       et_acc,
        "stacked_3way":   stacked_acc,
        "stacked_lgbm_rf": stack2_acc,
        "best_acc":       best_acc,
        "feature_names":  feature_names,
        "lgbm_importance": lgbm_importances.tolist(),
    }

    out_path = Path(args.out_dir) / "ml_stage_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
