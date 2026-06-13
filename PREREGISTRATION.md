# Pilot Pre-Registration (rebuild)

Fixed **before** running the rebuild pilot, so results can't be cherry-picked after
the fact. Any deviation must be logged as a dated amendment below.

> Status: DRAFT — confirm the bracketed `[…]` choices with the team/supervisor before locking.

## 1. Research question & hypotheses
- **RQ:** Do LLM-read catalysts add information *beyond* cheap quantitative baselines for predicting what happens after an event-driven spike?
- **H1 (primary):** A *catalyst-backed* overreaction continues while a *catalyst-less* one reverts — i.e. an LLM catalyst judgment **conditions** the `idio_resid` overreaction signal. Tested as an interaction effect.
- **H2 (secondary):** LLM-extracted catalyst features add incremental **information coefficient** over the quant baseline for the post-event *reaction magnitude*.
- **H0:** the LLM adds no signal beyond the quant baselines (the result if both fail).

## 2. Population & event inclusion (fixed)
- **Universe:** [heterogeneous, ≥ N tickers across ≥ 4 sectors, incl. some mid/smaller-cap; point-in-time membership; survivorship-aware]. NOT the 15-name tech basket.
- **Event type:** discrete catalyst events [earnings; index inclusion; M&A/deal; guidance/8-K].
- **Inclusion:** spike defined by [z-score ≥ z*, volume ≥ k×] within [window]; minimum price/liquidity filter [≥ $X ADV]; exclude halted/illiquid bars.
- **De-duplication / independence:** one event per (ticker, catalyst); report **effective N** (cluster on date).
- **Opening-bell:** [cap opening-bell events at ≤ X% OR stratify and report separately].

## 3. Decision time, catalyst window, label horizon (fixed)
- **Decision time t0:** the first *tradeable* moment after the catalyst is public [e.g. next regular-session open after an AMC release].
- **Catalyst window:** read catalyst content with `release_timestamp ≤ t0` only; primary source timestamps (EDGAR acceptance / PR-wire), no post-hoc explainer articles.
- **Feature window:** strictly ≤ t0 (verified by `analysis/leakage_tests.py`).
- **Label:** [market-adjusted] forward return over [H = 5 trading days]; continuation/fade by sign with [θ] dead-zone; magnitude label = |abnormal move|. Computed strictly post-t0.

## 4. Baselines (must beat these, at equal coverage)
1. Always-continue / majority class.
2. Momentum / direction heuristic.
3. **Same-ticker historical reaction size** (for magnitude).
4. **`idio_resid` overreaction-only** (quant, no LLM) — the key control.
5. EPS-surprise-sign (where applicable).
6. Pre-event realized vol / spike magnitude.
7. [If feasible] options-implied vol (the market's own uncertainty forecast).

## 5. Metrics (pre-specified)
- Primary: **interaction effect** (H1) — reversal rate, catalyst-backed vs catalyst-less, within high-overreaction events, with bootstrap 95% CI on the gap.
- **Information coefficient** (Spearman, predicted vs realized abnormal move) for H2.
- Selective: accuracy/Sharpe–coverage (AURC).
- Economic (secondary): net-of-cost Sharpe, deflated Sharpe, turnover.
- Calibration: Brier / ECE.
- All numbers reported with bootstrap CIs; comparisons with McNemar + Diebold–Mariano.

## 6. Success condition (declared in advance)
- **H1 supported** iff the catalyst×overreaction interaction gap is ≥ [Δ] pp with the 95% CI excluding 0 on the **frozen holdout**, AND the LLM-conditioned rule beats the `idio_resid`-only control at equal coverage.
- **H2 supported** iff incremental IC ≥ [0.02] over the quant baseline, CI excluding 0.
- Otherwise → reported as a **negative/methodology result** (no spin).

## 7. Frozen temporal holdout (locked)
- Final chronological slice = [dates], **post all model training cutoffs**, untouched during any prompt/feature/model tuning. Tuning happens only on train/validation. Holdout is scored **once**.

## 8. Models & leakage control
- Primary: [open-weight model, documented training cutoff] so test events are provably post-cutoff; benchmark [one closed frontier model] on a subset.
- Report cutoffs; consider entity/date masking; run the shuffled-label null and the `published_utc < t0` test as gates before any result is believed.

## 9. Multiple-testing discipline
- Pre-list every comparison; correct for multiplicity (deflated Sharpe / Bonferroni where relevant). No post-hoc threshold/horizon/universe shopping — any change after seeing results is an amendment, logged below.

## Amendments (dated)
- _(none yet)_
