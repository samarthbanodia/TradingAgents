# TradingAgents — Full Project Context (for external review)

This document states **what was built, what was run, and what was observed**, as
factually as possible. Numbers and methods are reported as-is. Interpretations by
the current assistant are explicitly labeled `[ASSISTANT HYPOTHESIS — scrutinize]`
and should be treated as claims to verify, not facts. The reviewer is encouraged to
distrust them and re-derive conclusions independently. **Extra scrutiny is requested
on Part 1 (the original pipeline, built before the current assistant), because the
team suspects an overlooked flaw there may be responsible for the poor results.**

Authors: Arinjay Nigam, Samarth Banodia (IIT Bombay, undergraduates). Target venue:
ICAIF (or an ICAIF/NeurIPS/ACL-style finance-AI workshop). Goal: a *novel, positive*
result presentable at a conference.

---

# PART 1 — THE ORIGINAL PROJECT (built before the current assistant)

## 1.1 Task
Given a sudden intraday price **spike** in a 5-minute window during regular trading
hours, predict whether the price will **continue** in the spike direction or
**reverse**, over the **next 60 minutes**. Three classes: continuation / reversal /
unclear. Framed as: continuation = spike driven by real information; reversal =
spike was mechanical (panic, stop-cascade, liquidity).

## 1.2 Data
- **Prices:** Polygon.io 5-minute OHLCV bars. Original report: 11 tickers (AAPL,
  MSFT, NVDA, TSLA, META, AMD, NFLX, PLTR, SPY, QQQ, XLK), Oct 2025–Jan 2026,
  126,719 rows. Later extended to 15 tickers (added COIN, GOOGL, JPM, AMZN, removed
  none; final 15), ~2 years (2024-05 → 2026-04), ~1.25M bars.
- **News:** Polygon news API, queried per event within a **±window around t0**
  (originally `[t0 − 3h, t0 + lookahead]`). Per-event JSON packets.
- **Label proxy:** 60-min forward return from t0. If > +0.3% in the spike direction
  → continuation; if < −0.3% opposite → reversal; else unclear. Labels never shown
  to agents.

## 1.3 Event mining (mine_events_strict.py)
- **Stage 1 trigger:** a 5-min bar with return z-score ≥ 3, OR volume ≥ 4× typical,
  OR range ≥ 3× typical.
- **Stage 2 confirm:** stricter thresholds (z ≥ 4, absolute move ≥ 60 bp) plus a
  persistence check (6-bar return ≥ 50% of the 3-bar peak).
- `t0` = timestamp of the last bar of the "event window"; `close_t0` = that bar's
  close (used as the entry/reference price). Forward return measured from `close_t0`.
- **Selection (select_best_events.py):** per-ticker quotas (8–15 events for stocks,
  3–8 for ETFs), cap "unclear" at 15%, 60-min temporal de-duplication per ticker.
- Counts (original): 451 candidate events → 100 selected → 85 "clear" (non-unclear).
  Extended: ~3,372 raw → 800 selected (490 continuation / 230 reversal / 80 unclear).

## 1.4 Agent architectures (history)
- **V1 / V2.1 — multi-agent debate (5 LLM calls/event).** Framed each spike as a mix
  of Information (I) vs Panic (P). Three specialists (Microstructure, News, Macro)
  produced I/P scores; a **Skeptic** critiqued; agents revised; a **Judge** issued
  the verdict. V1 = **35.2%** (3-way). V2.1 = **16.3%** (the Judge predicted
  continuation 95/98 times; documented as "degeneration-of-thought").
- **V3 — direct-label ensemble (3 LLM calls/event).** Dropped Skeptic & Judge. Three
  agents each output continuation/reversal + confidence, using an evidence-scoring
  format (enumerate evidence points 0–20 for each label before committing). "Unclear"
  forbidden per-agent. Model routing: **Micro = GPT-4o-mini, News = Claude Haiku,
  Macro = DeepSeek-Chat** (provider diversity for decorrelated errors).
  - **Micro agent** input: spike stats + 7 microstructure features (below), reasons
    via a WHO→WHOM→WHAT causal schema.
  - **News agent** input: up to 5 pre-event articles (strictly before t0) + timestamps.
  - **Macro agent** input: SPY/QQQ 15-min & 60-min pre-event returns + up to 4 macro
    headlines.

## 1.5 The 7 microstructure features (V3)
`idio_resid_bp` (stock 3-bar return − β·SPY return; large ⇒ idiosyncratic
overreaction), `rs_ratio` (downward semivariance / total, pre-event), `signed_vol_ratio`
(net buy/sell pressure, pre-event), `vpin_proxy` (order-flow imbalance, pre-event),
`spread_bp` (estimated bid-ask spread at t0), `pre_spike_run_len` (consecutive bars
in spike direction before t0), `is_macro_driven` (|SPY z| > 1.5 at t0).

## 1.6 Ensemble (eval/ensemble.py)
Weighted vote: `votes[ℓ] = w_micro·c_micro + w_news·c_news + w_macro·c_macro +
w_dir·p_dir`, with `w = {0.35, 0.20, 0.15, 0.22–0.38}`. A **direction heuristic**
acts as a 4th voter: **up-spike → reversal, down-spike → continuation** (justified by
intraday mean-reversion literature). `w_dir` varies by time-of-day. Asymmetric
**reversal prior** `p_dir`: 0.62 for z > 3 (large spikes revert more), −15% if
macro-driven, −18% for down-spikes. If `max(votes) < 0.35` → "unclear".

## 1.7 ML stage (eval/ml_stage_v2.py)
V3 agent outputs (labels, confidences, score ratios) + the 7 features + event stats
→ a 35-feature matrix. Classifiers: **LightGBM** (num_leaves=8, min_child_samples=5,
lr=0.05), **RandomForest** & **ExtraTrees** (max_depth=4, min_samples_leaf=4,
class_weight=balanced). Stacked = equal-weight probability average. Original N=85
used **Leave-One-Out CV**; extended N=800 used a **chronological 80/20 temporal
holdout** + 5-fold chronological CV on the train portion.

## 1.8 The data leak (found by an external reviewer, in the original work)
`vpin_proxy`, `signed_vol_ratio`, and `cluster_len` were computed from cluster bars
starting **at or after t0** (`timestamp >= t0`), overlapping the 60-min label window.
**Fix:** recompute `vpin`/`signed_vol` from the 5 bars strictly before t0; remove
`cluster_len` from the ML matrix. **Impact: stacked accuracy 82.4% → 67.1% (N=85).**

## 1.9 Results — original N=85 (leak-fixed, LOO-CV, "clear" subset)
| Strategy | Accuracy |
|---|---|
| V2.1 pipeline | 9.6% |
| Always reversal | 49.4% |
| Always continuation | 50.6% |
| Direction heuristic | 56.5% |
| V3b ensemble | 57.6% |
| Math-only ML (no agents) | 57.6% |
| Agents-only ML (no microstructure) | 61.2% |
| Full LightGBM | 62.4% |
| Full RandomForest | 65.9% |
| **Stacked LGBM+RF+ET** | **67.1%** |

Reported ablation: agents add **+4.7pp** over math-only (61.2 vs 57.6). Constants
stability sweep: 9/14 ensemble constants had ~0 effect at ±10% perturbation.

## 1.10 Results — extended N=800 (temporal holdout, Jan–Apr 2026 = 144 events)
- 5-fold chronological CV (train): full stack **66.7% ±5.1%**, LLM-only 65.1%,
  math-only 63.2%.
- **Holdout:** direction heuristic **65.3%** (best); math-only 59.0%; LLM-only 56.9%;
  full stack **56.2%** (CV→holdout drop of 10.5pp); V3b ensemble 35.4%.
- **McNemar** (full vs math p=0.465; full vs LLM p=0.853; LLM vs math p=0.674) — none
  significant.
- **Falsification audit** (full pipeline on shuffled labels): real CV 65.1% vs null
  64.9% — gap +0.2pp.
- **Backtest** (full stack, 144 holdout trades, per-ticker round-trip costs 10–30 bps):
  win rate 56.3%, mean PnL −4.05 bps/trade, median +37.1 bps, **annualized Sharpe
  −0.367**, max drawdown 2,155 bps.
- Base rate of the extended set: **68% continuation** (among clear events).

---

# PART 2 — WORK DONE WITH THE CURRENT ASSISTANT (factual log)

All re-analysis below reused the existing 800-event outputs + OHLCV unless noted.
Scripts in `analysis/`, `pathc/`, `pilot/`.

## 2.1 Reproduction + horizon sweep
- Reproduced the stored 60-min labels exactly from OHLCV (800/800 match).
- Re-labeled all events at horizons {30m, 60m, 120m, EOD, next-open, +1d, +2d}. On
  the holdout, the V3 ensemble accuracy was **below the majority baseline at every
  horizon** (lift −0.30 at 30m shrinking to −0.11 at +2d); per-agent accuracy of the
  News agent rose from 0.35 (30m) to 0.50 (+2d).

## 2.2 Reversal-bias diagnosis + debias ("v4")
- Observation: V3 agents predicted "reversal" 59–87% of the time (News agent 87%),
  while the data is ~68% continuation. The V3 **prompts** stated base rates "UP spike
  → ~58% reverse, DOWN → ~45% reverse" and broke ties toward reversal; the technical
  **brief** appended editorial hints like "→ REVERSAL signal" to raw feature lines.
- Measured actual base rates: **UP spikes 38% reverse / 62% continue; DOWN spikes
  27% reverse / 73% continue.**
- Rewrote prompts (correct base rates, symmetric scoring, tie→continuation, instruction
  to ignore brief hints). Re-ran all 800 (~$2.30). **Result:** News-agent reversal
  rate 87%→9%, ensemble 72%→25%; 60-min accuracy 37.5%→56.5% (full set), holdout
  35.4%→53.1%. **Still below the ~65% baseline at every horizon and coverage.**

## 2.3 Selective prediction (rigorous)
- Ranked events by agent evidence (ensemble vote margin); validation→test protocol;
  coverage-risk curves vs always-continue and a microstructure-only selector. **At
  every coverage the agents were at/below the baselines.** Unanimous-agreement subset
  (43% coverage): ensemble 64.5% vs always-continue 66.1% on the same events.

## 2.4 Pivot to earnings drift (interday) — "Path C"
- Hypothesis tested: predict whether the post-earnings move continues (drift) or
  fades over ~5 days. Built earnings events via yfinance.
- **Signal check (corrected, market-adjusted, 667 events, 48 tickers):** abnormal
  drift continuation rate ~50–54% (95% CIs include 0.50 at every horizon); EPS
  surprise sign → drift direction ~47–48%. By cap tier (5-day): mega 0.66, large 0.62,
  mid 0.49, small 0.51 (wide CIs). `[ASSISTANT HYPOTHESIS — scrutinize]` direction is
  ~efficient on liquid names.

## 2.5 News-data inspection (manual)
- For high-overreaction events, compared our news packets to what actually happened:
  several major catalysts were **missing** from the packets (e.g., Coinbase +24% on
  S&P 500 inclusion → 0 articles in packet; Palantir earnings beat → 0 articles).
- `[OBSERVATION]` For at-t0 catalysts (earnings, index changes) the catalyst arrives
  *at* t0 and *causes* the spike, so a pre-event news window is empty by construction.

## 2.6 Data-integrity audit (no LLM)
- **News look-ahead leak (real, fixed):** `build_news_packets.py` used
  `end = t0 + lookahead`, pulling **279 articles published ≥ t0 into 197/797 events
  (25%)** — including the catalyst article itself (e.g., the AMD–OpenAI deal article
  published 10 min after t0 sat in the packet). Fixed to end strictly at t0.
- **Feature-leak false alarm (corrected):** an initial check flagged
  `signed_vol`/`vpin` as leaked (corr 0.38 to a recompute), but that used a
  strictly-before-t0 window; with the **t0-inclusive** pre-event window the stored
  pipeline actually uses, **corr = 1.000 → not leaked.** The original cluster-bar leak
  *was* genuinely fixed.
- **A real, leak-free signal:** idiosyncratic overreaction (`idio_resid`) predicts
  intraday reversal — top tercile **55%** reversal vs **25%** bottom, bootstrap CI
  excludes 0, tercile cuts fixed on train. (Pre-event order-flow alone, excluding the
  t0 bar, is weak/insignificant.)
- Built: a decision-time feature table, a pre-registration template, automated
  leakage tests (feature pre-event PASS; news<t0 documents the fixed leak;
  shuffled-label null PASS), spike-audit plots, and a data-source lit-scan.

## 2.7 Clean catalyst-anchored pilot (no LLM, baseline-first)
- Built leak-free, market-adjusted earnings events on a heterogeneous universe;
  catalyst-strength proxy = |EPS surprise| (and surprise-reaction alignment).
- **Small pilot (382 events, 33 tickers):** among high-overreaction events, STRONG
  catalyst reversal 40.6% vs WEAK 50.0% — the hypothesized direction, but CIs overlap
  (not significant).
- **Scaled (1,423 events, 105 tickers, ~460 distinct dates; reversal base 0.476):**
  the interaction **does not hold and flips sign** — by surprise magnitude gap
  (weak−strong) = **−0.061** [−0.15, +0.03] n.s.; by alignment **−0.077** [−0.17,
  +0.01] n.s.; holdout **−0.116** [−0.32, +0.08] n.s. The small-pilot signal did not
  survive scaling.

## 2.8 Open question status (as of now, unbiased)
Tested cleanly and found **null/insignificant**: 60-min direction; longer-horizon
direction; selective/abstention; earnings-drift direction; catalyst-strength ×
overreaction interaction; news (presence and |surprise| materiality) overall.
Found **real and leak-free**: idiosyncratic-overreaction → intraday reversal (a known
short-term-reversal effect; a single number; no LLM needed).
**Not yet tested:** (a) whether catalyst strength predicts the *magnitude* of the
move (vs direction); (b) an LLM-derived (rich-text) measure of catalyst strength as
opposed to the |surprise| proxy; (c) the supervisor's controlled news-swap experiment
(valid vs shuffled vs wrong-ticker vs future news), which needs LLM runs.

---

# PART 3 — RESOURCES AVAILABLE
- 800-event V3 (biased) and V4 (debiased) agent outputs (per-agent labels,
  confidences, evidence scores, ensemble) + 1.25M 5-min OHLCV bars (15 tickers) +
  800 news packets.
- Free data: yfinance (prices, earnings dates + surprises, any ticker), SEC EDGAR
  (filings, primary timestamps), Polygon (the team's key: prices + news).
- LLM budget: modest (single-digit to low-tens of dollars). Open-weight or closed
  models both feasible. The team is undergraduate; engineering effort is available but
  bounded.
- Full reproducible code + audit harness + pre-registration in the repo.

# PART 4 — CONSTRAINTS / WANTS
- Needs a **novel** contribution (the architecture "multi-agent LLM trading" and the
  task "LLMs for earnings/volatility" are both already published — e.g. arXiv
  2412.20138 "TradingAgents"; ECC Analyzer; FNSPID; "Debate Only When Necessary").
- Wants a **positive** result, ideally tradeable or at least a measurable signal,
  presentable at conference level. The team values the multi-agent **debate** idea.
- Rigorous evaluation is a strength and is expected by the venue (temporal holdout,
  significance, cost-aware backtest, leakage audit).
