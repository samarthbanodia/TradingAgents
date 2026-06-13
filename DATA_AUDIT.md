# Data-Layer Integrity Audit — to complete BEFORE the rebuild

**Principle:** the data layer is the highest-risk part of this project. It has broken twice (the look-ahead leak; the empty-pre-event-news problem). Until we can prove the model receives *valid information at the correct time*, no result — positive or negative — is trustworthy. This document is the checklist we must clear and document before any pilot.

Legend: ⭐ = raised by supervisor · ➕ = added by us (think beyond the obvious).

---

## 1. Temporal / look-ahead leakage

- ⭐ **Decision time (t0).** Define one precise, defensible timestamp per event. Everything strictly before t0 is allowed as input; everything at/after is outcome. For after-hours catalysts (earnings at 16:01 ET), the decision time is the next *tradeable* moment (e.g., next regular open), not the 16:01 print.
- ⭐ **Feature window strictly < t0; label window strictly > t0.** (This is what the original leak violated — features used post-t0 bars.)
- ➕ **Timezone + DST correctness.** A single ET↔UTC or DST error shifts a pre-event window into the future. Assert every timestamp's tz explicitly; test events across a DST boundary.
- ➕ **Corporate-action / split adjustment.** `yfinance auto_adjust` restates history; a split inside a feature or label window corrupts returns. Decide adjusted-vs-raw once, document it, and verify no split/dividend falls inside any window unhandled.
- ➕ **Point-in-time vs restated data (subtle leak).** Earnings *estimates*, *surprise %*, fundamentals, and even index membership are **backfilled/restated**. The "surprise" yfinance shows today may not be what was known at t0. Index "as of today" membership includes names *added because they did well* → survivorship/look-ahead. Use point-in-time sources or flag the limitation.
- ➕ **Label↔feature price-series consistency.** Features and the label return must come from the same (adjusted) series, same exchange, same bar convention.

## 2. LLM-specific leakage (unique to us, easy to miss) ➕

- ➕ **Pretraining look-ahead.** The LLM has *memorized* events within its training cutoff. Testing on pre-cutoff dates lets it "recall" outcomes — a silent, severe leak that inflates many published LLM-trading results. **Mitigations:** (a) document each model's cutoff; (b) treat any result on pre-cutoff events as contaminated; (c) keep the *real* test set strictly post-cutoff; (d) consider entity-anonymization (mask ticker/company) to blunt recall.
- ➕ **Prompt-field leakage.** Audit *every* field in every prompt for outcome information (the original briefs leaked `cluster_len`). The brief must contain nothing computed from at/after t0.
- ➕ **Memory/recall via context.** If the model can infer the date+ticker, it may fill in the outcome from memory even without explicit leakage. Test with date/entity masked.

## 3. Catalyst / news validity

- ⭐ **Primary release timestamp + source.** Use the authoritative time: SEC EDGAR filing *acceptance* timestamp, the PR-wire (BusinessWire/GlobeNewswire) timestamp, the index provider's announcement time — **not** a news aggregator's later republish time.
- ⭐ **No post-hoc explainers.** Articles written *after* the move that explain it must never enter an earlier decision window. Filter strictly by original publication time < decision-relevant window.
- ➕ **Completeness / coverage gaps.** The broken-news problem was missing catalysts. For each event, verify we captured the *actual* catalyst (cross-check against EDGAR/PR), and log coverage rate. A missing catalyst is as bad as a leaked one.
- ➕ **De-duplication + wrong-entity contamination.** Same story republished N times shouldn't count as "lots of news"; multi-ticker tagged articles and same-day different-ticker news must be handled deliberately.
- ➕ **Catalyst-at-t0 vs drift.** For earnings/deals the catalyst *is* the spike (arrives at t0). The valid design reads the catalyst *content* and predicts the move *after* — the input is the catalyst, the label is the forward window. Make this explicit so we never read "pre-event news" that structurally can't exist.

## 4. Event / universe construction ➕

- ➕ **Survivorship bias.** yfinance only lists *currently existing* tickers; delisted/acquired/bankrupt names are gone. Document the bias; ideally use a point-in-time universe.
- ➕ **Point-in-time index membership.** If we pick "S&P 100," use membership *as of each event date*, not today's list (else look-ahead).
- ⭐ **Reproducible inclusion criteria.** Spike/event thresholds fixed and documented, not tuned on outcomes.
- ➕ **Tradeability / halts.** Stocks are often *halted* at a catalyst; if you can't trade at the decision time, the event is invalid for a tradeable claim. Filter halted/illiquid/penny names; record halt status.
- ➕ **Event independence.** Log clustering (our events clumped on market-wide days). Report effective N, not raw N.

## 5. Price / OHLCV hygiene ➕

- Bad ticks / zero-range bars (we saw `range_mult=0`); RTH vs pre/post-market bar inclusion; consolidated vs primary-exchange; gapped/halted bars; entry price must be an actually-tradeable price (not the exact spike close).

## 6. Reproducibility & provenance ➕

- ➕ **Frozen raw snapshots.** APIs drift (restatements, newly-indexed articles). Snapshot every raw response with its retrieval timestamp so the pipeline is byte-reproducible.
- ➕ **Data-source defensibility.** yfinance/Polygon are convenient but scraper/vendor-dependent and not point-in-time; finance venues expect CRSP/Compustat/I-B-E-S-grade or at least EDGAR primary sources. Decide what's defensible for ICAIF and state limitations. (Action: short lit-scan of what comparable LLM-finance papers actually use.)
- Version every dataset + code commit behind each result.

---

## 7. Audit deliverables (what we produce)

1. ⭐ **Decision-time feature table** — one row per feature/catalyst field: `name | source | timestamp basis | available at t0? (Y/N) | in prompt? | in model?`. Anything "N" at t0 is a leak.
2. ⭐ **Manual 20–30 event audit** — for each: event time, spike plot, catalyst source + timestamp, exactly what the agent sees, and the label window. Catches broken assumptions early.
3. ⭐ **Pre-registration** — fixed *before* the pilot: event-inclusion criteria, catalyst window, label horizon, baselines, metrics, and the success condition. No post-hoc cherry-picking.
4. ⭐ **Frozen temporal holdout** — a final chronological slice (post-model-cutoff) never touched during any prompt/feature/model tuning.
5. ➕ **Leakage unit-tests** — automated asserts: no feature/prompt field has a timestamp ≥ t0; label uses only > t0; shuffled-label null ≈ base rate; a "future-news" injection *should* boost performance (if it doesn't, our timing filter is wrong — and if real news doesn't beat shuffled, we have no signal).

---

## 8. Should we explore open-source models? — Yes, primarily for *data integrity*, not vibes ➕

The strongest argument is **leakage control**: with an open checkpoint we know the *exact* training cutoff, so we can *guarantee* test events are post-cutoff — impossible to verify with a closed API whose cutoff is undisclosed and fluid. Plus: reproducibility (a frozen open model + open data is far more defensible to reviewers), cost at scale, and the option to anonymize/fine-tune. Trade-off: weaker raw reasoning than frontier closed models. **Plan:** use a strong open model (e.g., Qwen-2.5 / Llama-3.x / DeepSeek class) as the documented, reproducible workhorse, and benchmark a closed frontier model on a subset to bound the reasoning-quality gap. Report both.

---

## 9. Sequence

Audit (this doc) → manual 20–30 event review → decision-time feature table + leakage unit-tests → pre-registration + frozen holdout defined → *only then* build the catalyst-anchored pipeline and run the pilot.

---

## 10. Issues register (what the static audit actually found)

Run for free on the existing N=797 set (`analysis/static_audit.py`, `verify_orderflow.py`, `audit_pipeline.py`, `foundation_tests.py`). Status: ✅ fixed · 🔧 fix in rebuild · 🔎 verified-OK · ⚠️ to-control.

| # | Issue | Status | Detail |
|---|---|---|---|
| 1 | **News look-ahead leak** | ✅ **fixed** | `build_news_packets.py` used `end = t0 + lookahead`, pulling **279 post-t0 articles into 197/797 events (25%)** — e.g. the AMD–OpenAI article (the catalyst) published 10 min after t0 sat in the packet. Window now ends at t0; rebuild adds a strict `published_utc < t0` post-filter. |
| 2 | **Feature leak (signed_vol/vpin)** | 🔎 **false alarm** | My first sweep flagged corr 0.38, but that used a *strictly-before-t0* window. With the **t0-inclusive** pre-event window `briefs_v3.py` actually uses, **corr = 1.000** → no leak. The original cluster-bar leak was genuinely fixed. (Correction logged for honesty.) |
| 3 | **"signed_vol → reversal +33pp"** | 🔎 reinterpreted | Real, *not* a leak — but it's largely the **overreaction / spike-bar** effect (redundant with `idio_resid`), not independent *pre-event* order-flow. Pre-event-only order flow is weak (+0.10, n.s.). The clean, independent signal is **`idio_resid` overreaction → reversal** (+0.30, CI excludes 0). |
| 4 | **News completeness** | 🔧 rebuild | The real catalysts (COIN S&P inclusion, PLTR earnings) were **missing entirely** — for at-t0 catalysts the event *is* the news, so a pre-event window is empty. Rebuild reads the catalyst contemporaneously and predicts the forward window. |
| 5 | **Opening-bell concentration** | 🔧 rebuild | 73% of events are at the open (a noisy auction regime). Curate/control or stratify by time-of-day. |
| 6 | **Homogeneous mega-cap universe** | 🔧 rebuild | 15 correlated tech names; agents lose on all 15 (structural, no single bad ticker). Expand to a heterogeneous, less-efficient universe. |
| 7 | **Label threshold too harsh?** | 🔎 verified-OK | Median 60-min move = **1.69%**, only 10% sit near the 0.3% threshold — labels are *not* noise. Not the problem. |
| 8 | **Unclear/ambiguous events** | 🔧 rebuild | ~10% labelled "unclear" and silently dropped. Decide explicitly (model abstention, or report). |
| 9 | **Event independence** | ⚠️ document | 797 events over 357 days, top-10 days = 10%. Report **effective N**, not raw N; cluster-robust significance. |
| 10 | **Decisive-move weakness** | ⚠️ note | Agents are *worst* on the biggest moves (48.5% vs 64% base) — the trades that matter most. |
| 11 | **LLM pretraining look-ahead** | ⚠️ control | Test set strictly post-model-cutoff; document cutoffs; consider entity masking; open model gives a known cutoff. |
| 12 | **Survivorship / point-in-time** | ⚠️ control | yfinance lists only surviving tickers; estimates/index membership are restated. Use point-in-time or flag the limitation. |

**Headline:** the audit caught **one real leak (news, now fixed)** and **corrected one false alarm of our own (features)** — which is exactly what a rigorous audit should do, and it cost nothing.
