# TradingAgents → ICAIF: Experimental & Paper Plan

**Author target:** Arinjay Nigam, Samarth Banodia (IIT Bombay)
**Venue:** ICAIF (ACM Intl. Conf. on AI in Finance) — finance-literate reviewers, expects sound methodology + realistic, cost-aware backtests.
**Constraint:** Needs a *largely positive* result (satisfactory, not SOTA). Modest new-LLM budget. Reuse existing N=800 outputs where possible.

---

## 1. Diagnosis — why the current framing cannot be published as-is

1. **Your stated novelty is untested, not refuted — but it cannot be claimed as-is.** Debate architectures scored V1 = 35.2% and V2.1 = 16.3%; the *no-debate* V3 ensemble scored 57.6%. BUT the V1→V3 jump changed ~5 things at once (I/P scores → direct labels, added evidence-scoring, added 7 microstructure features, changed model routing, *and* dropped debate), and the V1/V2 failures had their own named confounds (a directionally-biased LLM Judge; a Skeptic round causing degeneration-of-thought). So debate was **never isolated** — we cannot conclude it helps *or* hurts. The claim "debate → more accurate" is currently **unsupported**, which is different from false. DTD is the controlled test that V1/V2 never provided.
2. **The intraday 60-min task is hostile to LLM reasoning.** A 5-min-bar, 60-min horizon is the microstructure/HFT regime: near-efficient, order-flow-dominated, 10–30 bps round-trip costs. You correctly note you are *not* an HFT. Text reasoning has little exploitable edge here.
3. **Accuracy is the wrong metric and the bar is brutal.** 68% continuation base rate → momentum/majority ≈ 65.3% on holdout, and all McNemar p > 0.46. Unconditional accuracy will never produce a "positive" story on this task.

**Conclusion:** Do not chase unconditional accuracy or unconditional PnL. Both are lost causes here and any "win" would be overfitting — exactly what ICAIF reviewers desk-reject.

---

## 2. The reframed thesis (the honest, positive version of your novelty)

> **Disagreement-Triggered Debate (DTD): agents debate *only where they disagree*, and trade *only where they agree*.**

This single idea rescues everything you care about and converts the negatives into contributions:

- **Debate gets a clean, controlled test for the first time.** V1/V2 ran debate on *every* event, with a biased LLM judge and a Skeptic round prone to degeneration-of-thought (Liang et al. 2024) — and confounded with several other changes. We do NOT claim debate is inherently bad. DTD isolates it: debate is applied *only* on the hard (disagreement) subset, with a surviving dissent and no biased judge, and measured against the same events pre-debate. The outcome ("targeted debate helps / doesn't") is an empirical result either way — and a more honest contribution than V1/V2 could support.
- **Agent consensus → a high-precision, tradeable subset (selective prediction).** "The system knows when it knows." Trade the consensus events; abstain on the rest. This is a positive, finance-meaningful result even when the unconditional model ties momentum.
- **Multi-agent structure is finally *used*, not decorative.** Disagreement is an uncertainty signal; debate is a resolution mechanism. Both are things multi-agent systems can genuinely do, unlike "predict returns better than quants."

**Three claims the paper will try to establish (each independently testable):**

- **C1 (Selective prediction).** On the agent-consensus subset, the system achieves materially higher precision and **positive risk-adjusted return**, and it beats an *equal-coverage* momentum/confidence baseline. ← primary positive result.
- **C2 (Targeted debate helps where applied).** Running a bounded debate round on the *disagreement* subset improves accuracy and/or calibration *on that subset*, where indiscriminate debate (V1/V2) hurt. ← mechanism contribution.
- **C3 (Adaptive system dominates at equal cost).** The combined DTD system beats both always-debate (V1/V2) and never-debate (V3) at matched LLM-call budget. ← systems contribution.

If C1 holds, you have a paper. C2/C3 strengthen it. If even C1 fails after the cheap diagnostics, fall back to §6.

---

## 3. Workstreams (ordered; cheap → expensive)

### WS0 — Repro harness & honest plumbing (no LLM spend)
- One script that loads `out_agents_extended/ip_outputs_v3.jsonl` + `ohlcv_5min_extended.parquet`, recomputes the temporal train/holdout split, and reproduces the published 56.2% / 65.3% numbers. Lock this as ground truth before changing anything.
- Centralize the **outcome labeler** as a function `label(event, horizon)` so every experiment shares identical, leak-free logic (forward return strictly from t0+ε).

### WS1 — Horizon sweep (no LLM spend) — *explanatory axis*
- Re-label all 800 events at horizons **{30m, 60m, 120m, EOD-close, next-open, +1d, +2d}** using existing OHLCV.
- For each horizon, recompute: base rate, momentum baseline, math-only stack, agents-only stack, full stack, and the **LLM contribution (full − math)** with bootstrap CIs + McNemar.
- **Hypothesis:** microstructure dominates at ≤60m; the LLM/news contribution (if any) grows at longer horizons as order-flow noise washes out. Produces the figure "where does reasoning live?" — and tells you which horizon to run C1/C2/C3 at.

### WS2 — Selective prediction / C1 (no LLM spend)
- Define an **agreement/disagreement** score per event from existing per-agent labels & confidences:
  - `n_agree` ∈ {unanimous, 2-1, ...}; confidence dispersion; ensemble margin (`ensemble_votes` gap).
- Build **accuracy–coverage** and **Sharpe–coverage** curves: sort events by confidence/agreement, sweep the coverage threshold, plot performance on the covered subset.
- **Critical baseline (the credibility test):** at every coverage level, compare agent-consensus selection against **momentum-confidence selection at the same coverage** (e.g., select the same fraction by |z-score| or |spike magnitude|). The positive claim is *only* valid if agent selection beats momentum selection at equal coverage, with significance — otherwise consensus is just picking easy high-base-rate events.
- Report on the traded subset: precision, hit rate, mean/median PnL, Sharpe, turnover, capacity, max drawdown — all net of the existing per-ticker transaction costs.

### WS3 — Calibration (no LLM spend) — *supporting*
- Brier score and ECE / reliability diagrams for ensemble vs each agent vs math-only.
- Confidence-sized backtest: position ∝ calibrated confidence, vs equal-weight. Does sizing by agent confidence improve Sharpe? (Honest secondary exhibit.)

### WS4 — Targeted debate / C2 + C3 (MODEST LLM spend — the only paid step)
- Re-run a **bounded 2-round debate** *only on the disagreement subset* (~470 events, or a stratified sample of ~150–200 to start). Design to avoid degeneration-of-thought:
  - Roles can't be overridden by a single confident agent; force a written dissent that survives to aggregation.
  - **No LLM judge** — resolve by calibrated confidence-weighted aggregation (ReConcile-style), not a sycophantic judge.
- Compare on the disagreement subset: pre-debate ensemble vs post-debate. Then assemble the full **DTD** system (consensus→ensemble, disagreement→debate) and compare to always-debate (V1/V2 numbers you already have) and never-debate (V3) at **matched call budget**.
- Budget note: this is the only new spend. Start with a stratified sample to confirm the effect before paying for the full subset.

### WS5 — Significance, ablations, robustness (no LLM spend)
- McNemar **and** Diebold–Mariano on accuracy; bootstrap CIs on every PnL/Sharpe number; deflated Sharpe (Bailey–López de Prado) to account for multiple-testing.
- Re-run the falsification (shuffled-label) audit on the *selective* strategy.
- Per-regime (2024–25 vs 2026), per-ticker, up vs down spike breakdowns.

---

## 4. Metric suite ("make it quantitative")

| Layer | Metrics |
|---|---|
| Classification | Accuracy, **balanced accuracy / F1** (base-rate-robust), precision on traded subset |
| Ranking | Information coefficient (Spearman of predicted prob vs realized fwd return), AUC |
| Calibration | Brier, ECE, reliability diagram |
| Selective | Accuracy–coverage & Sharpe–coverage curves, risk–coverage AURC |
| Economic | Net PnL, Sharpe, Sortino, max drawdown, turnover, capacity, hit rate — all after costs |
| Significance | McNemar, Diebold–Mariano, bootstrap CIs, **deflated Sharpe** |
| Sanity | Permutation/falsification audit on the final strategy |

**Headline number candidates (positive, defensible):** "On the X% of events where agents reach consensus, the strategy earns Sharpe S > 0 net of costs, beating equal-coverage momentum by Δ (p < 0.05)" and/or "Targeted debate improves balanced accuracy on hard events by Δpp where indiscriminate debate lost Ypp."

---

## 5. Baselines that make ICAIF reviewers trust you

1. Majority class / always-continuation.
2. Momentum (direction) heuristic — your hardest baseline.
3. **Equal-coverage momentum** (the key one for C1).
4. Math-only ML stack (microstructure features, no LLM).
5. Single-LLM (no multi-agent) and self-consistency (majority vote, no debate).
6. Always-debate (V1/V2) and never-debate (V3) at matched cost (for C3).

---

## 6. Risk register & fallbacks (be honest with ourselves early)

- **If WS2 shows agent-consensus does *not* beat equal-coverage momentum:** the multi-agent selective story is dead. Fallback → lead with WS1 horizon result if a longer-horizon LLM contribution survives significance.
- **If no horizon shows a significant LLM contribution either:** the honest move is the **methodology / negative-result paper** (rigorous protocol + "when does multi-agent help?"), which is publishable at an ICAIF *workshop* even if not the main track. Decide this only after WS1+WS2, which are nearly free.
- **If WS4 debate doesn't help on hard events:** keep C1 as the paper (selective prediction stands alone); drop C2/C3 to a discussion of why debate fails even when targeted.

**Decision gate:** run WS0→WS3 first (all free). Only spend LLM budget on WS4 if C1 or the horizon result is already positive and significant.

---

## 7. Paper structure (ICAIF, ~8 pages)

1. Intro — informed-vs-uninformed spike question; multi-agent reasoning as an uncertainty/abstention tool, not a return oracle.
2. Related work — MAD failure modes, ReConcile, VPIN/microstructure, intraday reversal, selective prediction.
3. Data & event mining (you have this).
4. Method — agent panel → **DTD** (consensus-trade / disagreement-debate).
5. Evaluation protocol — temporal holdout, leak audit (your case study is a *strength* — keep it), significance, cost-aware backtest.
6. Results — C1 selective prediction (primary), horizon analysis, C2/C3 debate.
7. Limitations & negative findings (the unconditional null result stays — honesty is your moat).
8. Conclusion.

**The data-leak case study is an asset, not a liability** — frame it as "why our protocol catches what others miss."

---

## 8. Immediate next actions

1. WS0 repro harness — confirm we reproduce 56.2%/65.3%.
2. WS1 horizon sweep — first real figure, ~1 script, no spend.
3. WS2 selective-prediction curves with the equal-coverage momentum baseline — the make-or-break test for C1.
4. **Gate:** review WS1+WS2. If positive → fund WS4 debate on a stratified disagreement sample. If flat → convene on the negative-result fallback.

*All of WS0–WS3 + WS5 are free re-analysis of data already in this repo. Only WS4 costs money.*

---

## 9. GATE FINDINGS (WS0–WS2 run; analysis/ scripts)

**WS0 — passed.** `analysis/harness.py` reproduces the stored 60-min labels exactly (800/800, 0 missing). Labeler is faithful; horizon sweep is trustworthy.

**WS1 — horizon hypothesis directionally confirmed, but agents never beat the baseline.** On holdout, the V3 ensemble accuracy is 0.35–0.44 at every horizon vs a 0.55–0.69 majority baseline. BUT the gap shrinks monotonically with horizon (ensemble lift −0.30 at 30m → −0.11 at T+2D) and the news agent climbs 0.35→0.50, macro 0.45→0.49. Signal is least-bad away from the HFT regime, as predicted — it just never crosses zero.

**WS2 — the selective-prediction rescue (C1) FAILS in the current outputs.** Selecting on agent confidence/agreement makes accuracy *worse*, not better: on the unanimous-agreement subset (60m) the ensemble is right only **28.8%** of the time vs 67.3% for the trivial rule on the same events. The agents are **most confidently wrong when they agree.** `agent_minus_dir` is negative at every coverage level.

**Root cause (airtight).** The agents bet reversal 59–87% of the time (news = 87%) while reality is **68% continuation**. They are *anti-aligned* with the base rate. This is a hardcoded reversal prior (I/P "panic" framing, `dir_prior=0.62`, up→reversal voter, "large spikes revert"), **not reasoning.** The current negative result is therefore **confounded by a broken prior** — it is not yet a fair test of "can LLM agents help."

**Revised gate decision.** Post-hoc recalibration won't fix it (the llm-only stack already remaps the existing outputs and still only hits 56.9% < 65.3%). The fix must be at the *reasoning level*: **re-run debiased agents** (drop the reversal prior; let them predict freely) on a modest sample, evaluated at **longer horizons (T+1D/T+2D)** where text reasoning has a fair shot. Outcome is decisive either way:
  - debiased agents beat the base rate out-of-sample → **positive ICAIF result**;
  - they still don't → a **clean, unconfounded negative result** (publishable, but not the "positive" the team wants).

This replaces WS4-first. The next paid step is the **debias re-run**, not the debate.
