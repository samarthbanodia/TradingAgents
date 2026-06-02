"""
Leakage Falsification Audit.

Runs the full ML pipeline (LGBM LOO-CV on full dataset, or chrono-CV) on
randomly shuffled labels. If accuracy is materially above 50%, a structural
data leak exists.

Per Nikolopoulos (2026, arXiv:2604.15531): any pipeline that consistently
exceeds 53–55% accuracy on permuted labels has a structural leak.

Usage:
    python -m eval.falsification_audit
    python -m eval.falsification_audit \\
        --jsonl out_agents_extended/ip_outputs_v3.jsonl \\
        --events_csv out_extended_final/selected_events.csv \\
        --ohlcv ohlcv_5min_extended.parquet \\
        --out_dir out_eval_extended \\
        --n_runs 20
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def chrono_kfold(n, n_splits=5):
    indices = np.arange(n)
    fold_size = n // n_splits
    rem = n % n_splits
    folds, start = [], 0
    for k in range(n_splits):
        end = start + fold_size + (1 if k < rem else 0)
        folds.append(indices[start:end])
        start = end
    for k in range(n_splits):
        val = folds[k]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        yield train, val


def _accuracy(preds, labels):
    return float((np.asarray(preds) == np.asarray(labels)).mean())


def run_null_audit(X, y, n_runs=20, n_splits=5, seed=0):
    """
    Run n_runs permutations of y, compute 5-fold chrono-CV accuracy each time.
    Returns array of null-model accuracies.
    """
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    null_accs = []

    for run_i in range(n_runs):
        y_perm = y.copy()
        rng.shuffle(y_perm)

        fold_accs = []
        for tr_idx, va_idx in chrono_kfold(len(y_perm), n_splits):
            clf = lgb.LGBMClassifier(
                objective="binary", num_leaves=8, min_child_samples=5,
                learning_rate=0.05, n_estimators=100, verbose=-1, random_state=42,
            )
            clf.fit(X[tr_idx], y_perm[tr_idx])
            fold_accs.append(_accuracy(clf.predict(X[va_idx]), y_perm[va_idx]))

        run_acc = float(np.mean(fold_accs))
        null_accs.append(run_acc)
        log.info(f"  Run {run_i+1:>2}/{n_runs}: null CV acc = {run_acc:.3f}")

    return np.array(null_accs)


def main():
    parser = argparse.ArgumentParser(description="Leakage falsification audit — shuffled label test")
    parser.add_argument("--jsonl",        default="out_agents_v3b/ip_outputs_v3.jsonl")
    parser.add_argument("--events_csv",   default="out_final/selected_events.csv")
    parser.add_argument("--ohlcv",        default="ohlcv_5min.parquet")
    parser.add_argument("--out_dir",      default="out_agents_v3b")
    parser.add_argument("--n_runs",       type=int, default=20,
                        help="Number of label permutations to run")
    parser.add_argument("--n_splits",     type=int, default=5,
                        help="CV folds per permutation")
    parser.add_argument("--include_unclear", action="store_true")
    args = parser.parse_args()

    from eval.feature_matrix import load_feature_matrix, impute_median

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_path = args.ohlcv if Path(args.ohlcv).exists() else None

    log.info("Loading feature matrix...")
    X_raw, y, feature_names, records = load_feature_matrix(
        args.jsonl, args.events_csv,
        include_unclear=args.include_unclear,
        ohlcv_path=ohlcv_path,
    )
    X = impute_median(X_raw)
    n = len(y)
    log.info(f"Matrix: {n} events × {X.shape[1]} features | Labels: {dict(Counter(y.tolist()))}")

    # Real CV accuracy (reference point)
    import lightgbm as lgb
    log.info("\nComputing REAL label CV accuracy (reference)...")
    real_accs = []
    for tr_idx, va_idx in chrono_kfold(n, args.n_splits):
        clf = lgb.LGBMClassifier(
            objective="binary", num_leaves=8, min_child_samples=5,
            learning_rate=0.05, n_estimators=100, verbose=-1, random_state=42,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        real_accs.append(_accuracy(clf.predict(X[va_idx]), y[va_idx]))
    real_mean = float(np.mean(real_accs))
    log.info(f"  Real CV acc: {real_mean:.3f} ± {np.std(real_accs):.3f}")

    # Null model permutations
    log.info(f"\nRunning {args.n_runs} label permutations ({args.n_splits}-fold CV each)...")
    null_accs = run_null_audit(X, y, n_runs=args.n_runs, n_splits=args.n_splits)

    null_mean = float(null_accs.mean())
    null_std  = float(null_accs.std())
    null_max  = float(null_accs.max())

    # Assess
    LEAK_THRESHOLD = 0.56  # >56% on shuffled labels → strong evidence of leak
    WARN_THRESHOLD = 0.53  # 53-56% → worth investigating

    if null_mean > LEAK_THRESHOLD:
        verdict = "FAIL — structural data leak detected"
        detail  = f"Null model acc {null_mean:.3f} > {LEAK_THRESHOLD} threshold"
    elif null_mean > WARN_THRESHOLD:
        verdict = "WARN — marginal inflation above 50%, investigate"
        detail  = f"Null model acc {null_mean:.3f} > {WARN_THRESHOLD} warning threshold"
    else:
        verdict = "PASS — no structural leakage detected"
        detail  = f"Null model acc {null_mean:.3f} ≈ 50%, consistent with no leak"

    log.info(f"\n{'='*55}")
    log.info("FALSIFICATION AUDIT RESULTS")
    log.info(f"{'='*55}")
    log.info(f"  Real CV accuracy   : {real_mean:.3f} ± {float(np.std(real_accs)):.3f}")
    log.info(f"  Null model acc     : {null_mean:.3f} ± {null_std:.3f} (mean over {args.n_runs} permutations)")
    log.info(f"  Null model max     : {null_max:.3f}")
    log.info(f"  Info ratio         : {real_mean / (null_mean + 1e-9):.2f}x")
    log.info(f"  Verdict            : {verdict}")
    log.info(f"  Detail             : {detail}")
    log.info(f"{'='*55}")

    results = {
        "real_cv_acc": real_mean,
        "real_cv_std": float(np.std(real_accs)),
        "real_cv_folds": real_accs,
        "null_mean":  null_mean,
        "null_std":   null_std,
        "null_max":   null_max,
        "null_all":   null_accs.tolist(),
        "n_runs":     args.n_runs,
        "n_splits":   args.n_splits,
        "verdict":    verdict,
        "detail":     detail,
        "leak_threshold": LEAK_THRESHOLD,
        "warn_threshold": WARN_THRESHOLD,
    }

    out_path = out_dir / "falsification_audit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
