# TradingAgents — Progress Briefing for Supervisors

**Prepared by:** Arinjay Nigam, Samarth Banodia (IIT Bombay)
**Context:** This brings you up to date since the last report you saw (*"LLM-Augmented Intraday Price Spike Classification"*). Everything below happened after that report. Target venue: **ICAIF**.

---

## 1. Where we were (the report you last saw)

**Task.** Given a sudden intraday price spike in a 5-minute window, predict whether the move **continues** or **reverses** over the next 60 minutes.

**Method.** A multi-agent LLM panel — a **Microstructure**, **News**, and **Macro** agent — each reasons over its own inputs and outputs a continuation/reversal label; the labels are combined by a weighted-vote ensemble, then a small ML stage (LightGBM + RandomForest + ExtraTrees) makes the final call. (Earlier "debate" versions V1/V2 with a Skeptic + Judge were dropped after they underperformed.)

**Headline results in that report.** On N=85 events we found and fixed a data leak (82.4% → **67.1%** honest accuracy) and reported a **+4.7pp** apparent contribution from the LLM agents over a numbers-only baseline.

### Why that was *not* conference-ready
- **Tiny dataset (N=85)** over a short 3-month window — fragile, regime-specific.
- **Sample-selection bias** — events were partly chosen for having clear outcomes.
- **No out-of-sample test and no significance testing** — the +4.7pp could easily be noise.
- **The novelty was unproven.** The project's intended novelty is *multi-agent debate*, but the debate had been *removed* in the version that worked; "debate improves accuracy" was never cleanly demonstrated.

**So our plan was:** scale the data, validate out-of-sample, add statistical rigor, and pin down a defensible novelty.

---

## 2. What we did first — scale to N=800 with proper rigor

We expanded to **800 events, 15 tickers, ~2 years (2024–2026)**, and added: a strict **temporal holdout** (train on the past, test on the future), **significance tests** (McNemar), a **falsification audit** (shuffled-label null model), and a **transaction-cost backtest**.

### Result: the positive finding did not survive
- The **+4.7pp LLM contribution vanished** out-of-sample.
- A **trivial momentum baseline (65.3%)** beat every ML variant on the holdout.
- The full model **overfit** (66.7% in cross-validation → 56.2% on the holdout).
- **All significance tests were non-significant** (p > 0.46). The backtest was **unprofitable** (Sharpe −0.37).

In short: at honest scale, with honest testing, **the agents added no measurable value.** This is the single most important thing to communicate — the report's headline did not replicate.

---

## 3. How we diagnosed it — and a real bug we found

We did not stop at "it doesn't work." We asked *why*. We found the agents were predicting **"reversal" 59–87% of the time**, while reality is **~68% continuation** — they were betting the wrong way systematically.

**The cause was a bug in our own prompts.** The prompts told the agents *"up-spikes reverse ~58% of the time"* when our own data says up-spikes **continue 62%** of the time (and down-spikes continue 73%). The prompts also broke ties toward reversal and the data briefs literally injected "→ REVERSAL signal" hints. So the negative result was **partly confounded by a self-inflicted bias** — not yet a fair test of the idea.

### We fixed it (the "v4 debias") and re-ran all 800 events (~$2.30)
- The bias broke as intended: the News agent's reversal rate dropped **87% → 9%**; accuracy jumped **37.5% → 56.5%**.
- **But even debiased, the agents still lost to the baseline** at every horizon (60 min through 2 days) and at every confidence level. Trading only when all three agents agree gave **64.5%**, versus **66.1%** for simply always saying "continuation" on the same events.

**What this means:** with the bug removed, this is now a **clean negative result**, not a confounded one. The agents genuinely add no exploitable signal on this task.

### The two root causes — neither fixable by a bigger model or better prompts
1. **Market efficiency.** 60-minute moves on mega-cap stocks are too efficient to predict — this is the high-frequency regime, and we are not a high-frequency shop.
2. **No information to reason over.** Only 435/800 events had *any* pre-event news, mostly generic. The News agent had nothing to read, so it defaulted to a constant. A smarter model cannot extract signal from an absent article.

**Conclusion on intraday: it is finished as a route to a positive result.** It remains valuable as a rigorous case study (the leak, the bias bug, the honest null).

---

## 4. Where we pivoted — interday, event-driven (post-earnings drift)

Our own data hinted that the signal is least-bad at **longer horizons** (the agents' news/macro reasoning matters more as microstructure noise washes out). The natural move is to a task where **reading text actually matters**: **post-earnings drift** — after a company reports earnings, does the initial price move **continue (drift)** or **fade** over the next ~5 days?

Why this fits the project better: earnings give the agents **rich, dated, readable text** (the surprise, guidance, analyst tone); it leaves the high-frequency regime; and the **debate finally has a real job** ("is this a high-quality beat that will drift, or a hollow one that will fade?").

### What we tested, and the honest result
We built the data layer for free (yfinance): **91 earnings events, 12 stocks**, with EPS surprises. An initial signal check looked promising (drift continued ~60% over 3–5 days; the surprise predicted drift ~57%). **But a careful review found this is not yet a valid test:**
- **N=91 is far too small** — roughly half the events needed to detect a realistic effect with statistical confidence. Any result would land in "promising but not significant" — the exact trap we just escaped.
- **A labeling bug** measured raw returns instead of market-adjusted returns, which **inflates "continuation"** in a generally rising market.
- **The 12 stocks are mega-caps** — precisely where post-earnings drift is most arbitraged away and weakest.
- **The agents were still fed numbers, not real text** — so they had no advantage over ordinary statistical models.

**So Path C as currently built is the N=85 trap in new clothes. We have *not* spent money running agents on it, deliberately.**

---

## 5. The reframe that genuinely has upside

Across all of this, we have been testing the agents as an **oracle** ("can they predict direction better than a baseline?"). On efficient markets that is nearly impossible. The frame where LLMs are known to win is as a **feature-extractor**:

> *Do features extracted by LLMs from real earnings text add **incremental predictive signal** (information coefficient) on top of standard price/fundamental features?*

This changes the success bar from "beat 65% accuracy" (effectively impossible on an efficient task) to "add a small but real, tradeable ranking signal" (plausible). The **debate stays central** — as the mechanism that extracts and refines those text features and resolves the hard, disagreement cases.

**Important novelty context for you:** the broad ideas are already published — there is a paper literally named *TradingAgents* (multi-agent LLM trading), LLMs-for-earnings-drift exists, and "debate only on hard cases" exists. So our contribution **cannot** be the architecture or the task; it must be a **specific combination + genuine rigor + a concrete finding** (e.g., our prior-misspecification bug is itself a novel cautionary result). This sharpens, but does not block, a publishable angle.

---

## 6. The decision — two honest options

| | **Option A — Go for a positive result** | **Option B — Methodology / negative-result paper** |
|---|---|---|
| **What it is** | Rebuild the interday/earnings approach *properly*: ≥500 events **including small/mid-cap** stocks (where drift survives), feed agents **real point-in-time text** (earnings-call transcripts, guidance, 8-Ks), measure **information coefficient** with proper power analysis and confidence intervals, then layer in the **debate**. | Write up what we have already rigorously proven: the data-leak case study, the prompt-bias case study, the horizon analysis, and the selective-prediction null — as a careful evaluation-methodology paper. |
| **Cost** | New data collection + meaningful time; **~$20–40** of new LLM spend. | Mostly **writing** — the experiments are already done and in the repo. |
| **Probable outcome** | A **real but not guaranteed** shot at a positive, debate-included result. Markets are hard; it may still show the agents add little. | **Reliably publishable** (workshop tier), honest and rigorous — but a **negative headline**, which is not the "we beat the benchmark" story we'd prefer. |
| **Risk** | Could spend the effort and still land on a null. | Low risk; lower ceiling. |

### Our recommendation
Treat **B as the floor and A as the upside.** Before committing any money to Option A, we run **one free, corrected check**: fix the labeling bug and honestly test — on a larger, more realistic set of stocks, with proper statistics — whether post-earnings drift is even predictable. 
- If **yes** → build Option A for real (real text + debate, enough events to detect a signal). 
- If **no** → take Option B without having wasted resources.

This keeps the **debate** — the part of the project we care about and the part least covered by prior work — central, while making sure we only invest where there is genuine signal to debate over.

---

## 7. One-paragraph summary (if you read nothing else)

The report's positive result did not survive proper scaling and significance testing — and we proved this cleanly, including finding and fixing a real prompt-bias bug along the way. The intraday task is fundamentally unsuited to LLM reasoning (too efficient, too little text). We have pivoted toward earnings-driven, multi-day prediction, where reading text genuinely matters and the debate mechanism has a real role — but our first attempt there is under-powered and needs rebuilding with more, smaller-cap events and real earnings text. We now face a clear fork: **(A)** invest modest time and money for a real but uncertain shot at a positive, debate-centred result, or **(B)** write up the rigorous negative findings we already have for a safe but lower-ceiling publication. We recommend running a free signal check first, then defaulting to B with A as the upside — and we'd like your steer on which way to lean.
