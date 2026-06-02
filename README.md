# TradingAgents — Intraday Price-Spike Continuation/Reversal Prediction

A research pipeline that mines intraday price spikes from 5-minute equity bars, has a panel of
LLM agents reason about each spike from news + microstructure context, and trains a second-stage
ML classifier to predict whether the spike **continues** or **reverses** over the next 60 minutes.

> **Honest bottom line (read this first).** After fixing a serious data-leak and scaling to
> N=800 events with a proper temporal holdout, **the LLM agents add no statistically significant
> signal** (all McNemar p > 0.46) and **a trivial "direction heuristic" baseline (65.3%) beats
> every ML variant on the out-of-sample holdout.** The backtest is **not profitable**
> (Sharpe −0.37). This repo is published as a rigorous **negative result** with full methodology,
> not a working trading strategy.

---

## Table of Contents
1. [Problem statement](#problem-statement)
2. [Pipeline overview](#pipeline-overview)
3. [Timeline of everything tried](#timeline-of-everything-tried)
4. [Results](#results)
5. [The data-leak crisis](#the-data-leak-crisis)
6. [Extended N=800 study](#extended-n800-study)
7. [Honest conclusions](#honest-conclusions)
8. [Repository layout](#repository-layout)
9. [How to run](#how-to-run)
10. [Research backing](#research-backing)

---

## Problem statement

Given a sudden intraday price spike (a cluster of 5-min bars with abnormal return *z*-score and
volume), classify the next 60 minutes as one of:

- **continuation** — price keeps moving in the spike direction
- **reversal** — price retraces against the spike
- **unclear** — neither dominates (label used during curation; the ML target is binary)

The hypothesis under test: *do LLM agents reasoning over news + order-flow context add predictive
signal beyond cheap microstructure features and a momentum baseline?*

---

## Pipeline overview

```
OHLCV (5-min)  ──►  Event Miner  ──►  Curation/Selection  ──►  News Packets
                                                                    │
                                                                    ▼
                                              V3 Agent Panel (3 LLM calls/event)
                                              Micro · News · Macro  →  weighted-vote ensemble
                                                                    │
                                                                    ▼
                                   Feature Matrix (microstructure + agent outputs + context)
                                                                    │
                                                                    ▼
                              ML Stage (LightGBM + RandomForest + ExtraTrees, stacked)
                                                                    │
                                                                    ▼
                       Evaluation: temporal CV · falsification audit · McNemar · backtest
```

---

## Timeline of everything tried

### V1 — Skeptic + Judge multi-agent debate
A multi-round debate: a proposing agent, a Skeptic, and an LLM Judge. **35.2% accuracy** — worse
than chance on the 3-way task. Diagnosed as **reversal-biased** and suffering the known
multi-agent-debate failure modes (sycophancy, degeneration-of-thought). Output in `out_agents/`.

### V2.1 — Revised debate
Reworked prompts and brief structure. Accuracy *collapsed to 16.3%* — a clear failure. Abandoned.
Output in `out_agents_v2/`. (Note: `agents/briefs_v2.py` was lost from git and later
reconstructed from `briefs.py` + the v3 import signatures.)

### V3 — Direct-label ensemble (the redesign)
Dropped the Skeptic, the Judge, and the debate loop entirely. Replaced with **3 direct-label
agents** that each independently output `continuation`/`reversal` + a confidence score:

- **Micro** — order-flow / microstructure reasoning (WHO→WHOM→WHAT causal schema)
- **News** — pre-event news-packet reasoning
- **Macro** — market-wide / SPY-context reasoning

Combined by a **research-backed weighted-vote ensemble** (`eval/ensemble.py`) with an asymmetric
reversal prior (UP spikes get a higher reversal prior; DOWN spikes scaled by 0.82× for leverage
asymmetry), time-of-day conditioning, and a macro-driven flag. **V3b ensemble: 49.0% (3-way),
57.6% on clear events.** Output in `out_agents_v3b/`.

Seven new microstructure features were added per event (`agents/briefs_v3.py`):
`idio_residual`, `rs_ratio`, `signed_vol_ratio`, `vpin_proxy`, `spread_bp`,
`pre_spike_run_len`, `is_macro_driven`.

### ML stage on V3 outputs
A second-stage classifier (LightGBM + RandomForest + ExtraTrees, equal-weight probability
stacking) on a 35-feature matrix combining agent outputs, microstructure, and context flags.
**Initially reported 82.4% (LOO-CV, N=85) — this was wrong (see leak crisis below).**
Clean, leak-fixed result: **67.1%**.

### Extended study — N=800 with temporal holdout
Scaled OHLCV to 15 tickers × 2 years (1.25M bars), mined 3,372 raw events, selected 800, ran the
full V3 agent panel + ML stack with a proper **date-based holdout** and **purged-style temporal
CV**, plus a falsification audit, McNemar significance tests, and a transaction-cost backtest.
This is where the honest negative result emerged.

---

## Results

### N=85 (clean, leak-fixed, LOO-CV)

| Strategy | Accuracy |
|---|---|
| V2.1 pipeline | 16.3% |
| V1 pipeline | 35.2% |
| Always reversal | 49.4% |
| Always continuation | 50.6% |
| Direction heuristic | 56.5% |
| V3b ensemble | 57.6% |
| Math-only ML (LGBM) | 57.6% |
| Agents-only ML (LGBM) | 61.2% |
| Full LGBM LOO-CV | 62.4% |
| Full RandomForest | 65.9% |
| **Stacked LGBM+RF+ET** | **67.1%** |

LLM agent contribution at N=85: **+4.7pp** (full vs math-only). This apparent contribution
**did not survive** scaling and significance testing (see below).

### N=800 (honest, out-of-sample holdout, N=144 Jan–Apr 2026)

| Strategy | Holdout accuracy |
|---|---|
| **Direction heuristic** | **65.3%** ← best |
| Math-only stack | 59.0% |
| LLM-only stack | 56.9% |
| Full stack (LGBM+RF+ET) | 56.2% |
| V3b ensemble | 35.4% |

The full stack **overfits**: 66.7% CV → 56.2% holdout (a 10.5pp drop). The trivial momentum
baseline is the hardest thing to beat and nothing beats it out-of-sample.

---

## The data-leak crisis

An external reviewer found that the headline **82.4% (N=85)** result was inflated by a **+15.3pp
look-ahead leak**.

**Root cause:** `_get_cluster_bars()` used `timestamp >= t0` (inclusive), pulling **post-event
bars** into three of the top features (`vpin_proxy`, `signed_vol_ratio`, `cluster_len`) —
contaminating them with outcome-period information.

**Fix** (`eval/feature_matrix.py`):
- `vpin_proxy_pre` and `signed_vol_ratio_pre` recomputed from the **last 5 bars strictly before t0**
- `cluster_len` removed from the ML features entirely
- `strength_score` removed (it embedded ~20% `cluster_len` weight)

**Result:** 82.4% → **67.1%**. The 67.1% figure is the honest one; **82.4% is invalidated.**

Three residual disclosures kept on record:
1. `cluster_len` was shown to agents in their brief → indirect contamination of agent outputs
   (cannot be retroactively undone without re-running all LLM calls).
2. The N=85 events were partly selected for having *clear* outcomes → sample-selection bias.
3. `build_open_baselines()` normalizes over the full dataset (standard practice, mild look-ahead).

---

## Extended N=800 study

**Dataset:** 800 events, 15 tickers, 2024-05-28 → 2026-04-30. Labels: 490 continuation / 230
reversal / 80 unclear (68% continuation base rate). Median spike magnitude 137.7 bps.
Train: 576 (→ 2026-01-05) · Holdout: 144 (2026-01-06 →).

**Cross-validation (5-fold on train):** Full stack 66.7% ±5.1% · LLM-only 65.1% ±2.6% ·
Math-only 63.2% ±5.3%.

**McNemar significance tests** (do the variants actually differ?):

| Comparison | p-value |
|---|---|
| Full vs Math | 0.465 |
| Full vs LLM | 0.853 |
| LLM vs Math | 0.674 |

All non-significant. **The LLM features add no statistically significant signal**, and math-only
actually beats the full stack on the holdout.

**Falsification audit** (full pipeline on *shuffled* labels): real CV 65.1% ±1.3% vs null
(shuffled) 64.9% ±1.1% — gap of **+0.2pp**. The audit's "FAIL" verdict here is a **class-imbalance
false alarm**, not a real leak: with a 68% continuation base rate, any model that leans
majority-class scores ~65% regardless of the labels. The near-zero real-vs-null gap confirms the
feature set carries **almost no genuine signal**.

**Backtest** (full stack, 144 holdout trades, per-ticker round-trip costs 10–30 bps):

| Metric | Value |
|---|---|
| Win rate | 56.3% |
| Mean PnL | −4.05 bps/trade |
| Median PnL | +37.1 bps/trade |
| Annualized Sharpe | **−0.367** |
| Max drawdown | 2,155 bps |

A positive win rate but losses larger than wins in magnitude → negative expectancy.
**Not viable.**

---

## Honest conclusions

1. **The direction (momentum) heuristic is the hardest baseline to beat** — 65.3% on holdout,
   essentially equal to predicting the majority class.
2. **ML models overfit** — a 10.5pp CV→holdout gap; patterns learned on 2024–2025 don't generalize
   to the 2026 regime.
3. **No measurable LLM contribution** — every McNemar p > 0.46; adding LLM features slightly *hurts*
   versus math-only out-of-sample.
4. **Backtest is not profitable** — Sharpe −0.37 despite a 56% win rate.
5. **Class imbalance dominates** — the 68% continuation rate is the real "signal" models latch onto.

This is presented as a **publishable negative result** with rigorous methodology: temporal
holdout, falsification audit, significance testing, and realistic transaction costs. The apparent
gains at small N (the +4.7pp LLM lift, the 82.4% headline) dissolve under leak-fixing,
scaling, and significance testing — a textbook illustration of why all three matter.

---

## Repository layout

```
TradingAgents/
├── fetch_ohlcv.py / fetch_ohlcv_extended.py   # OHLCV ingestion (Polygon.io)
├── mine_events.py / mine_events_strict.py     # Spike event miner
├── curate_events.py / select_best_events.py   # Curation + selection
├── build_news_packets.py                      # Pre-event news packets
├── run_ip_debate.py                           # V1 debate pipeline
├── run_ip_v3.py                               # V3 direct-label agent pipeline
├── run_extended_pipeline.py                   # End-to-end N=800 driver
├── agents/
│   ├── briefs_v2.py / briefs_v3.py            # Brief builders + microstructure features
│   └── prompts_v3.py                          # Direct-label agent prompts
├── eval/
│   ├── ensemble.py                            # Weighted-vote ensemble
│   ├── feature_matrix.py                      # Feature matrix (leak-fixed)
│   ├── ml_stage.py / ml_stage_v2.py           # ML stack + CV + ablation
│   ├── falsification_audit.py                 # Null-model (shuffled-label) test
│   ├── constants_sweep.py                     # Ensemble constant stability
│   └── backtest_pnl.py                        # Transaction-cost backtest
├── out_agents_v3b/                            # N=85 V3 outputs + ML results
├── out_agents_extended/                       # N=800 agent outputs
├── out_eval_extended/                         # N=800 ML / audit / backtest results
├── out_extended_strict/ / _final/ / _news/    # N=800 mined events, selection, news
├── ohlcv_5min_extended.parquet                # 1.25M 5-min bars (26 MB)
├── v3_rebuild/                                # Clean V3 snapshot (for review)
├── report.tex / report.pdf                    # LaTeX write-up
└── STUDY_GUIDE.md                             # Plain-language guide
```

---

## How to run

Requires a `.env` with a Polygon.io API key and an LLM provider key (the `.env` is **not** committed).

```bash
# 1. Ingest OHLCV
python fetch_ohlcv_extended.py

# 2. Mine + select events
python mine_events_strict.py
python select_best_events.py

# 3. News packets
python build_news_packets.py

# 4. V3 agent panel (3 LLM calls/event)
python run_ip_v3.py

# 5. Full extended pipeline + evaluation
python run_extended_pipeline.py

# Individual eval modules
python -m eval.ml_stage_v2
python -m eval.falsification_audit
python -m eval.backtest_pnl
```

---

## Research backing

V3 design drew on 30+ papers, key among them:

- **Drop Skeptic+Judge** — MAD sycophancy / degeneration-of-thought (arXiv:2502.08788, 2305.19118, 2509.23055)
- **Confidence-weighted voting** — ReConcile (ACL 2024, arXiv:2309.13007)
- **Intraday reversal asymmetry** (arXiv:2501.16772)
- **Evidence scoring vs anchor bias** — RETuning (arXiv:2510.21604)
- **Causal schema for the Micro agent** — WHO→WHOM→WHAT (arXiv:2512.17923)
- **Microstructure features** — VPIN (Easley/de Prado/O'Hara), idiosyncratic residual
  (Brogaard et al. SSRN 4731947), EDGE spread (JFE 2024), semivariance (Liu et al. 2023)

---

*Author: Arinjay Nigam, IIT Bombay. Pipeline status and detailed results also in `STUDY_GUIDE.md`
and `report.pdf`.*
