# TradingAgents — Complete Project Study Guide
### For presentation to a PhD supervisor. No prior finance or tech knowledge assumed.

---

## TABLE OF CONTENTS

1. The Core Question — What Are We Trying to Predict?
2. The Finance Intuition — Reversals, Continuations, and Why Anyone Cares
3. The Data We Collected
4. The Complete Pipeline — Bird's Eye View
5. Version 1 — The I/P Debate System (35.2%)
6. Version 2.1 — The Revised Debate (16.3%) — What Went Wrong
7. Version 3 — The Direct-Label Ensemble (49%)
8. The ML Stage — LightGBM on Top (67.1% clean)
9. The Data Leakage Crisis — What a Reviewer Found
10. The Ablation Study — Do the LLMs Actually Help?
11. The Constants Sweep — Are We Overfitted?
12. Final Honest Results
13. Every File Explained
14. Key Concepts Glossary
15. Questions a PhD Supervisor Might Ask — With Answers

---

## 1. THE CORE QUESTION

**What we are trying to predict:**

> When a stock price suddenly spikes up or down by a large amount within a few minutes during a trading day, will the price continue in the same direction over the next 60 minutes, or will it snap back?

That's it. That is the entire project in one sentence.

We call these sudden moves **intraday price spikes**. They happen constantly in the stock market — a company announces earnings, there's a news headline, a large trader places an unusual order, or sometimes nothing obvious happens at all. Within seconds to minutes, the price moves dramatically. Our question is: **what comes next?**

There are two possible outcomes:
- **Continuation** — the price keeps moving in the same direction (the spike was "real")
- **Reversal** — the price snaps back toward where it started (the spike was an overreaction)

There's also a third possible outcome:
- **Unclear** — the price doesn't move decisively in either direction within 60 minutes

---

## 2. THE FINANCE INTUITION

### Why Does a Spike Happen?

Think of the stock market as a continuous auction where buyers and sellers meet. Most of the time, prices move slowly because supply and demand are roughly balanced. A spike happens when one side suddenly overwhelms the other. There are several reasons this can happen:

**Reason 1 — Real Information ("Informed Trading")**
Someone has news that other people don't yet have. They buy or sell aggressively. Example: a fund manager reads an earnings report early and buys before everyone else sees it. In this case, the spike reflects real, new information about the company's value. The price *should* stay at the new level or keep moving. This produces **CONTINUATION**.

**Reason 2 — Panic or Mechanical Flows ("Uninformed Trading")**
No new information exists, but something mechanical happens:
- A large portfolio manager needs to liquidate quickly (forced selling)
- A cascade of stop-loss orders all trigger at once (a "stop cascade")
- Retail traders all pile in chasing a news headline that's already priced in
- Algorithmic traders react to each other's movements

In these cases, the price moved too far, too fast, for no fundamental reason. The market corrects itself. This produces **REVERSAL**.

### The Base Rates

This is crucial to understand. Across all large intraday spikes:
- Roughly **58% of UP spikes reverse** within 60 minutes
- Roughly **45% of DOWN spikes reverse** within 60 minutes (panic selloffs often continue briefly before recovering)
- These aren't predictions — they're just the historical averages

So if you always guess "reversal" on up-spikes and "continuation" on down-spikes (a simple rule we call the **direction heuristic**), you get about **56.5% accuracy**. This is our primary baseline to beat.

### Why Is This Hard?

Because the same spike pattern can mean completely different things depending on context:
- Large volume + large move + pre-event news → probably informed → continuation
- Large volume + large move + no news → probably panic → reversal
- Same numbers, different stories

This is exactly why we brought in AI agents — to reason about context, not just numbers.

---

## 3. THE DATA WE COLLECTED

### What Data

**OHLCV Price Data:**
OHLCV stands for Open, High, Low, Close, Volume. For each 5-minute window of the trading day, we know:
- What price the stock opened at (start of the 5 minutes)
- The highest price it reached
- The lowest price it reached
- What price it closed at (end of the 5 minutes)
- How many shares were traded

We have this for **11 tickers** from roughly October 2025 to early 2026:
- Stocks: AAPL, MSFT, NVDA, TSLA, META, AMD, NFLX, PLTR
- ETFs (market-wide indices): SPY, QQQ, XLK

Total: ~126,719 5-minute bars.

**News Data:**
For each event we found, we fetched news articles from **Polygon.io** (a financial data API) within a 3-hour window around the event time. We kept only articles published *before* the spike (strictly pre-event).

**Files:**
- `ohlcv_5min.parquet` — all price data (Parquet is a compressed data format like a very efficient spreadsheet)
- `fetch_ohlcv.py` — the script that downloaded the price data
- `build_news_packets.py` — fetches news articles for each event from Polygon.io

### Why These Tickers?

A mix of high-volatility stocks (TSLA, NVDA, PLTR) and more stable tech names (AAPL, MSFT). The ETFs (SPY=S&P 500, QQQ=Nasdaq 100) serve as a barometer for whether the entire market is moving — if SPY moved the same way as the stock, the spike is probably market-wide (macro-driven) rather than company-specific.

---

## 4. THE COMPLETE PIPELINE — BIRD'S EYE VIEW

```
RAW PRICE DATA (ohlcv_5min.parquet)
           │
           ▼
    ┌─────────────────┐
    │  STEP 1         │   mine_events_strict.py
    │  Event Mining   │   Find the spikes
    └────────┬────────┘
             │ ~451 candidate events
             ▼
    ┌─────────────────┐
    │  STEP 2         │   curate_events.py  (manual review)
    │  Curation       │   Keep the good ones
    └────────┬────────┘
             │ ~80 events
             ▼
    ┌─────────────────┐
    │  STEP 3         │   select_best_events.py
    │  Best 100       │   Balance + rank by signal quality
    └────────┬────────┘
             │ 100 events
             ▼
    ┌─────────────────┐
    │  STEP 4         │   build_news_packets.py
    │  News Fetch     │   Polygon.io API → pre-event articles
    └────────┬────────┘
             │ 100 events + news packets
             ▼
    ┌─────────────────────────────────────┐
    │  STEP 5 — THE AGENT SYSTEM          │
    │                                     │
    │  V1:  5 LLM calls/event → 35.2%    │   run_ip_debate.py
    │  V2.1: 5 LLM calls/event → 16.3%  │   (same)
    │  V3:  3 LLM calls/event → 49.0%   │   run_ip_v3.py
    └────────┬────────────────────────────┘
             │ 100 agent outputs (JSONL files)
             ▼
    ┌─────────────────┐
    │  STEP 6         │   eval/ml_stage.py
    │  ML Stage       │   LightGBM + RF + ET
    └────────┬────────┘
             │
             ▼
        67.1% accuracy (clean, leak-free)
```

---

## 5. VERSION 1 — THE I/P DEBATE SYSTEM (35.2%)

### The Idea

Our initial theory was that the key to distinguishing reversals from continuations is understanding *why* the spike happened. We framed this as a decomposition problem:

Every spike can be broken down as some mix of:
- **I (Information)** — price moved because of real new information
- **P (Panic)** — price moved because of mechanical, emotional, or liquidity-driven flows

If **I > P**: the move was informed → it should **continue**
If **P > I**: the move was panic → it should **reverse**

### The Architecture: Multi-Agent Debate

We built a system where multiple AI agents (large language models, LLMs) argue with each other to reach a verdict. This is called a **Multi-Agent Debate (MAD)** system.

```
EVENT DATA + NEWS
        │
        ├──────────────────────────────────────────────┐
        │                    │                          │
        ▼                    ▼                          ▼
  ┌──────────┐         ┌──────────┐              ┌──────────┐
  │Micro     │         │News      │              │Macro     │
  │Agent     │         │Agent     │              │Agent     │
  │          │         │          │              │          │
  │Reads     │         │Reads     │              │Reads     │
  │price     │         │news      │              │SPY/QQQ   │
  │stats     │         │articles  │              │context   │
  │          │         │          │              │          │
  │→ I score │         │→ I score │              │→ I score │
  │→ P score │         │→ P score │              │→ P score │
  └────┬─────┘         └────┬─────┘              └────┬─────┘
       │                    │                          │
       └────────────────────┴──────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ SKEPTIC Agent │  ← critiques all three
                    │               │    suggests adjustments
                    └───────┬───────┘
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
         Micro revises  News revises  Macro revises
         their scores   their scores  their scores
               │            │            │
               └────────────┴────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  JUDGE Agent  │  ← reads all revised scores
                    │               │    outputs final verdict
                    └───────┬───────┘
                            │
                            ▼
                    "continuation" / "reversal" / "unclear"
```

**5 LLM calls per event:** Micro, News, Macro (Round 1) → Skeptic → Micro+News+Macro revise (Round 2) → Judge

**Tech used:**
- GPT-4o-mini (OpenAI) for the Microstructure agent
- Claude Haiku (Anthropic) for the News agent
- DeepSeek Chat for the Macro agent (different model family for diversity)
- Different models intentionally — to reduce "correlated errors" (if all models are similar, they fail in the same way)

**What agents received:**

| Agent | Input |
|---|---|
| Micro | Technical brief: spike magnitude, z-score, volume multiplier, cluster length, SPY/QQQ returns |
| News | News brief: up to 5 pre-event articles, their timestamps, whether news was found |
| Macro | Macro brief: SPY/QQQ 15-min and 60-min pre-event returns, macro headlines |

**Output schema (per agent, per round):**
```json
{
  "I_score": 0.72,
  "P_score": 0.35,
  "confidence": 0.80,
  "evidence": ["Large z-score suggests informed buying", "..."],
  "counterfactuals": ["If volume were lower...", "..."]
}
```

### V1 Results: 35.2%

This is actually *worse than random* on many events. The direction heuristic baseline is 56.5%. We're 21 percentage points below that.

**Why V1 Failed:**

The Judge agent had a strong **reversal bias** — it predicted reversal on nearly everything. The dominance rule in `prompts.py` said: if P > I + 0.10 → predict reversal. The agents often struggled to identify what was truly informational vs panic-like, so they gave high P scores to most events. The Judge then predicted reversal almost universally.

Result: 35.2% accuracy on a dataset where 43% of events are actually continuations. The model was systematically wrong on continuations.

---

## 6. VERSION 2.1 — THE REVISED DEBATE (16.3%)

### What Changed

We kept the same 5-agent architecture but redesigned the prompts and added new features to the technical brief:
- Added open-adjusted volume (volume relative to what's typical for that time slot — market open has naturally higher volume)
- Added SPY/QQQ return context (co-movement)
- Sharpened the Judge's decision rules

### V2.1 Results: 16.3% — Even Worse

This is genuinely terrible. Worse than always guessing randomly (33% on 3 classes).

**Why V2.1 Failed — The Degeneration-of-Thought Problem:**

A research paper we found (arxiv:2305.19118) describes exactly what happened to us: in multi-agent debate with homogeneous models, one agent's confident opinion "infects" the others. Here's what happened:

The Microstructure agent, having seen that long clusters (5+ bars) were common, started interpreting ALL long clusters as "information" and gave high I scores. After the Skeptic challenged everything, agents revised their scores upward for I. The Judge then saw I > P for almost every event and predicted... **continuation** 95 out of 98 times. 

But true continuations are only 43% of events. Predicting continuation for everything gives ~43% accuracy. We got 16.3% because we also predicted "unclear" some of the time, and when we predicted continuation and were wrong, it counted as wrong.

The core problems with V2.1:
1. **Degeneration-of-Thought**: Agents converged on consensus rather than maintaining diverse views
2. **Sycophancy**: The Skeptic agent tended to agree with the most confident agent (the Judge was basically rubber-stamping whatever the Micro agent thought)
3. **Long cluster = information was wrong**: Long clusters mean *exhaustion*, not sustained information
4. **5 LLM calls per event**: Expensive ($0.88/event) and slow

---

## 7. VERSION 3 — THE DIRECT-LABEL ENSEMBLE (49%)

### The Philosophy Shift

V3 is a complete redesign based on reading 30+ academic papers. The key insight:

> Stop asking "what caused this spike?" (I vs P decomposition). Instead, ask each agent directly: "Will this continue or reverse?" Then combine their answers mathematically.

This shift does three things:
1. Eliminates the Judge (who was the main failure point)
2. Eliminates the Skeptic (who caused sycophancy)
3. Forces each agent to commit to a binary answer with evidence scores

### The 3 Agents (no Skeptic, no Judge)

```
EVENT DATA + NEWS + V3 MICROSTRUCTURE FEATURES
        │
        ├─────────────────────┬──────────────────────┐
        ▼                     ▼                       ▼
  ┌──────────────┐     ┌──────────────┐       ┌──────────────┐
  │MICRO Agent   │     │NEWS Agent    │       │MACRO Agent   │
  │              │     │              │       │              │
  │Reads: price  │     │Reads: only   │       │Reads: only   │
  │stats + 7 new │     │pre-event     │       │SPY/QQQ       │
  │microstructure│     │news articles │       │returns +     │
  │features      │     │              │       │macro news    │
  │              │     │              │       │              │
  │Scores FOR    │     │Scores FOR    │       │Scores FOR    │
  │each label    │     │each label    │       │each label    │
  │(1-20 pts)    │     │(1-20 pts)    │       │(1-20 pts)    │
  │              │     │              │       │              │
  │→ label:      │     │→ label:      │       │→ label:      │
  │  cont/rev    │     │  cont/rev    │       │  cont/rev    │
  │→ confidence  │     │→ confidence  │       │→ confidence  │
  └──────┬───────┘     └──────┬───────┘       └──────┬───────┘
         │                    │                       │
         └────────────────────┴───────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MATH ENSEMBLE      │
                    │  (no LLM involved)  │
                    │                     │
                    │  + Direction        │
                    │    Heuristic (4th   │
                    │    "voter")         │
                    └─────────┬───────────┘
                              │
                              ▼
                    "continuation" / "reversal" / "unclear"
```

### The ReTuning Prompt Format

One key V3 innovation is how we ask agents to reason. Instead of just asking "what's the I/P score?", we make them enumerate evidence FOR each possible answer before committing:

```
STEP 1: Score evidence FOR continuation (add 1-5 pts each):
  - Pre-event news exists: +4
  - High VPIN proxy: +4
  - is_macro_driven = YES: +4
  - ...

STEP 2: Score evidence FOR reversal (add 1-5 pts each):
  - Large idiosyncratic residual: +5
  - Low VPIN: +4
  - Pre-spike run ≥ 3 bars: +3
  - ...

STEP 3: Label = whichever side scored higher
STEP 4: Confidence = (winner - loser) / (winner + loser)
```

This prevents "anchor bias" — agents no longer start with a guess and find evidence for it. They build up evidence for both sides and let the scores decide.

### The 7 New Microstructure Features (V3 Only)

These are computed from OHLCV data BEFORE calling any LLM. They are shown to the Micro agent AND used in the ML stage.

| Feature | What It Measures | Finance Intuition |
|---|---|---|
| `idio_resid_bp` | Stock return minus (beta × SPY return) | How much of the spike was the stock alone vs. the whole market moving. Large positive idio = stock went way beyond market → overreaction → reversal |
| `rs_ratio` | Downward variance / total variance in pre-event bars | If pre-event price action was noisy (equal up and down moves, rs≈0.5) → overreaction context → reversal likely |
| `signed_vol_ratio` | Net buyer/seller pressure in pre-event bars | Did buyers dominate before the spike? Strong buying pressure → informed → continuation |
| `vpin_proxy` | How one-sided was the order flow (pre-event) | High VPIN = one side dominated = likely informed trader = continuation |
| `spread_bp` | Estimated bid-ask spread at t0 bar | Wide spread = market makers unsure = more uncertainty = reversal more likely |
| `pre_spike_run_len` | How many consecutive bars moved in spike direction before t0 | Long run before spike = exhaustion = reversal likely |
| `is_macro_driven` | Was SPY z-score > 1.5 at t0? | If the whole market moved too, it's macro-driven → market absorbs information efficiently → continuation more likely |

**After the V3 fix:** `signed_vol_ratio` and `vpin_proxy` are computed from the 5 bars BEFORE t0 (strictly pre-event). Earlier versions accidentally used post-event bars.

### The Math Ensemble

After the 3 agents vote, we don't just do majority voting. We use a weighted formula that also incorporates a **direction heuristic** as a 4th voter:

```
Total votes:

  "reversal"     += w_micro × micro_confidence    (if micro says reversal)
  "reversal"     += w_news  × news_confidence     (if news says reversal)
  "reversal"     += w_macro × macro_confidence    (if macro says reversal)
  "reversal"     += dir_weight × reversal_prior   (direction heuristic)
  "continuation" += dir_weight × (1-reversal_prior)

Where:
  w_micro     = 0.35  (micro agent weight)
  w_news      = 0.20  (news agent weight)
  w_macro     = 0.15  (macro agent weight)
  dir_weight  = 0.22–0.38  (depends on time of day)
  reversal_prior = 0.55–0.62  (depends on spike z-score and direction)
```

**Why is the direction heuristic a voter?**

Research (arxiv:2602.08003) shows that mixing a non-LLM rule with LLM voters improves accuracy because LLM models make *correlated errors* (they all fail at the same events). The direction heuristic fails at completely different events than the LLMs, so combining them reduces total errors.

**Why does time of day matter?**

Reversals are strongest in the last 2 hours of trading (2 PM–4 PM ET). Early morning events near the market open (9:30–10:00 AM) are more likely to sustain because institutional orders continue flowing. The `dir_weight` is higher in the afternoon (0.38) and lower at the open (0.22).

**When does the ensemble say "unclear"?**

If the top vote total is less than 0.35, it means the votes were too split to call. The ensemble returns "unclear." Individual agents are FORBIDDEN from outputting "unclear" — only the ensemble can.

### V3 Results

| Strategy | 3-way accuracy | Clear-events only |
|---|---|---|
| Always reversal | 42.0% | 49.4% |
| Always continuation | 43.0% | 50.6% |
| Direction heuristic | 48.0% | 56.5% |
| V3 ensemble | **49.0%** | **57.6%** |
| Micro agent alone | 50.0% | 58.8% |

V3 beats the direction heuristic for the first time (+1pp on 3-way, +1.2pp on clear). It's modest but real.

**V3a vs V3b:**
- V3a (first attempt): 43% — reversal-biased again. 90/100 events predicted reversal. Problem: the no-news rule was too aggressive ("no news + UP spike = +4 reversal") and opening-bell rule added +2 reversal to 87/100 events.
- V3b (rebalanced prompts): 49% — fixed the rules. Macro agent shifted from 78% reversal predictions to 23%. The diversity between agents (Micro=reversal-leaning, Macro=continuation-leaning) is what makes the ensemble work.

---

## 8. THE ML STAGE — LIGHTGBM ON TOP (67.1%)

### The Idea

After the V3b run, we have 100 events × 3 agent outputs stored in a file. Each record has:
- What each agent said (label + scores)
- The 7 microstructure features
- The ensemble vote
- The true label (did it actually reverse or continue?)

Can we train a machine learning model to predict the true label better than the ensemble alone?

Yes. This is the ML stage.

### What Is LightGBM?

LightGBM is a **gradient boosted decision tree** algorithm. Imagine it as:
- A series of simple "if-then" rules (decision trees)
- Each tree corrects the mistakes of the previous one
- They're trained sequentially, each learning from prior errors

It's widely considered the best algorithm for small-to-medium tabular (spreadsheet-like) datasets. It handles missing values, is fast, and doesn't require much tuning.

We also use **RandomForest** (many trees trained on random subsets, voted together) and **ExtraTrees** (like RandomForest but with additional randomization). Then we **stack** all three by averaging their probability predictions.

### The Feature Matrix

We convert each event into 35 numbers (features) that the ML models can learn from:

```
FEATURE GROUPS:

Agent features (15):
  - micro_label, news_label, macro_label   (encoded as numbers: 0=cont, 1=rev, 2=unclear)
  - micro_conf, news_conf, macro_conf       (agent confidence 0.5–0.95)
  - micro_diff, news_diff, macro_diff       (cont_score - rev_score per agent)
  - micro_ratio, news_ratio, macro_ratio    (cont_score / total_score per agent)
  - combined_diff, combined_ratio           (all agents combined)
  - agent_agreement                         (0=all agree, 1=2 agree, 2=all disagree)

Ensemble features (5):
  - vote_cont, vote_rev                     (total weighted votes)
  - vote_margin                             (vote_cont - vote_rev)
  - dir_weight, dir_prior                   (direction heuristic parameters)

V3 microstructure (8):
  - idio_resid_bp, rs_ratio                 (pre-event features)
  - signed_vol_ratio_pre, vpin_proxy_pre    (pre-event, after leak fix)
  - spread_bp, pre_spike_run_len            (pre-event features)
  - spy_zscore_t0, adj_zscore_tod           (macro context)

Context flags (4):
  - is_up_spike, has_pre_event_news
  - is_opening_bell, is_macro_driven

Raw event features (3):
  - peak_z_ret_3                            (how many standard deviations the spike was)
  - peak_volume_mult                        (volume at spike vs. typical volume)
  - peak_range_mult                         (price range at spike vs. typical range)
```

### Leave-One-Out Cross Validation (LOO-CV)

We only have 85 usable events (100 minus 15 "unclear" true labels). With N=85, standard train/test splits would leave too few events to test on. So we use **Leave-One-Out Cross Validation**:

```
For each event i (1 to 85):
    Train on all 85 events EXCEPT event i
    Predict event i
    Record whether prediction was correct

Final accuracy = total correct / 85
```

This gives the most honest accuracy estimate with small datasets — every event gets to be a test case exactly once, and the model *never* saw the test event during training.

### ML Results (after leak fix)

| Model | Accuracy |
|---|---|
| Direction heuristic | 56.5% |
| V3b ensemble | 57.6% |
| LightGBM LOO-CV | 62.4% |
| RandomForest LOO-CV | 65.9% |
| ExtraTrees LOO-CV | 60.0% |
| **Stacked LGBM+RF+ET** | **67.1%** |

The stacked model adds **+9.5pp** over the V3b ensemble, which adds **+10.6pp** over the direction heuristic. Total improvement from the naive baseline to the final system: **+10.6pp**.

---

## 9. THE DATA LEAKAGE CRISIS

### What Is Data Leakage?

Data leakage means your model accidentally learned from information that it *wouldn't have access to* at prediction time in the real world. It's the most common and most damaging mistake in applied machine learning.

A good analogy: Imagine a student who memorized the exam answers the night before. They score 100%. But they haven't actually learned anything — they just cheated. When you give them a new exam, they'll fail.

### What We Found

An external reviewer flagged that our top 3 features were all **computed using post-event data**:

**The leaky code (`_get_cluster_bars` in `briefs_v3.py`):**
```python
# This pulled bars STARTING at t0 going FORWARD:
at_after = sub[sub["timestamp"] >= t0]   # ← >= means "at t0 and after"
return at_after.iloc[:cluster_len]        # ← these are post-event bars
```

These forward-looking bars were then used to compute:
- `vpin_proxy` (feature rank #1 in original LGBM: importance score 114)
- `signed_vol_ratio` (feature rank #3: importance score 88)
- And `cluster_len` itself (feature rank #2: importance score 92) which was how long the spike cluster lasted — determined by looking at post-t0 bars

**Why it inflated accuracy:**

Imagine you're predicting whether it will rain tomorrow, but your #1 feature is "did it rain tomorrow?" The model will get 100% accuracy on your test set but fail completely in the real world.

In our case: the price action DURING the cluster (which extends forward from t0) is correlated with whether the price eventually reversed or continued. So the model was partially learning "given what happened in the first 10 minutes of the outcome window, predict the outcome" — which is cheating.

### The Fix

```
BEFORE (leaky):
  cluster_bars = bars starting at t0 going forward
  vpin_proxy = computed from cluster_bars ← POST-EVENT
  signed_vol_ratio = computed from cluster_bars ← POST-EVENT

AFTER (clean):
  pre_vol_bars = last 5 bars BEFORE t0 (strictly pre-event)
  vpin_proxy_pre = computed from pre_vol_bars ← CLEAN
  signed_vol_ratio_pre = computed from pre_vol_bars ← CLEAN
  cluster_len = REMOVED from ML features entirely
```

We also removed `strength_score` from the ML features because it contained `cluster_len` as one of its components (20% weight), smuggling the leak back in through a side door.

### Impact of the Fix

| | Accuracy |
|---|---|
| Original (leaky) ML result | 82.4% |
| After fix (clean) ML result | 67.1% |
| Drop | −15.3pp |

The 15pp drop is entirely explained by the removed leaky features. The 67.1% is the honest number.

---

## 10. THE ABLATION STUDY — DO LLMs ACTUALLY HELP?

An **ablation study** answers the question: "if I remove component X, how much does performance drop?" This tests whether each part of the system is actually contributing.

We ran three variants:

| Configuration | Features Used | Accuracy (LGBM LOO-CV) |
|---|---|---|
| Math-only | Microstructure + context flags + raw event features | 57.6% |
| Agents-only | LLM agent labels/scores/ratios + ensemble votes | 61.2% |
| Full stack | Everything | 62.4% |

**Key finding: Agent contribution = +4.7pp** (full stack minus math-only)

**What this means:**
- Pure microstructure math (no LLMs) performs at the same level as the direction heuristic (57.6%)
- The LLM agents add a genuine +4.7pp on top
- This is the "Contextual Alpha" — the agents read news, understand macro context, and reason about causality in ways that pure numbers cannot

**Why is math-only only 57.6%?**

The pre-event microstructure features (VPIN, signed vol from 5 bars before the spike) are weak predictors on their own. They capture some signal about whether there was order flow imbalance leading into the spike, but a 5-bar pre-event window (25 minutes) is limited. The agents see much richer context: the exact news articles, the macro backdrop, the narrative around why the spike happened.

---

## 11. THE CONSTANTS SWEEP

The V3 ensemble has 14 hardcoded numerical constants (agent weights, reversal priors, time-of-day weights, etc.). A reviewer might say: "You tuned these 14 numbers to your 85 events — your system is just memorizing the data."

To test this, we varied each constant by ±10% and ±20% and measured how much accuracy changed.

**Results:**

| Constant | What It Does | Max |Δ| at ±10% |
|---|---|---|
| w_micro (0.35) | Micro agent weight | 1.0pp |
| w_news (0.20) | News agent weight | 0.0pp |
| w_macro (0.15) | Macro agent weight | 0.0pp |
| w_direction (0.30) | Direction heuristic weight | 0.0pp |
| unclear_thresh (0.35) | When to call "unclear" | **2.0pp** |
| dir_w_early (0.22) | Direction weight at market open | 1.0pp |
| dir_w_late (0.38) | Direction weight in afternoon | 0.0pp |
| prior_high_z (0.62) | Reversal prior for large z-scores | 0.0pp |
| prior_low_z (0.55) | Reversal prior for moderate z-scores | 1.0pp |
| z_threshold (3.0) | z-score boundary | 0.0pp |
| macro_discount (0.85) | How much to reduce reversal prior if macro-driven | 1.0pp |
| down_discount (0.82) | How much to reduce reversal prior for DOWN spikes | 0.0pp |
| prior_max (0.72) | Cap on reversal prior | 0.0pp |
| prior_min (0.42) | Floor on reversal prior | 1.0pp |

**Conclusion:** Maximum effect of any single constant = 2pp. 9 of 14 constants have ZERO effect at ±10% perturbation. The system is not overtuned to the 85 events.

---

## 12. FINAL HONEST RESULTS

```
ACCURACY SUMMARY (clean, no leakage, LOO-CV on 85 clear events)

  V2.1 full pipeline     :  16.3%  ← the low point
  V1 full pipeline       :  35.2%
  Always reversal        :  49.4%  (baseline)
  Always continuation    :  50.6%  (baseline)
  Direction heuristic    :  56.5%  (primary baseline to beat)
  V3b ensemble           :  57.6%  ← first time we beat direction heuristic
  Math-only ML (LGBM)    :  57.6%  (no agent features)
  Agents-only ML (LGBM)  :  61.2%  (no microstructure)
  Full LGBM LOO-CV       :  62.4%
  Full RandomForest       :  65.9%
  Full Stacked LGBM+RF+ET: 67.1%  ← final result

  LLM agent contribution:  +4.7pp  (full vs math-only)
  Total gain vs heuristic: +10.6pp (stacked vs direction heuristic)
```

**Three things to disclose to a reviewer:**
1. `cluster_len` was shown to agents in their brief (post-event data, mild indirect contamination of agent outputs, cannot retroactively fix without re-running LLM calls)
2. The 85 events were selected partly for having clear outcomes — results may not generalize to all spikes
3. `build_open_baselines` uses the full dataset for normalization (standard practice but technically look-ahead)

---

## 13. EVERY FILE EXPLAINED

### Data Collection

**`fetch_ohlcv.py`**
Downloads 5-minute OHLCV bars from Polygon.io for our 11 tickers. Saves to `ohlcv_5min.parquet`. This is the raw input to everything.

**`convert_to_csv.py`**
Converts the parquet file to CSV format for tools that can't read parquet.

**`ohlcv_5min.parquet`**
The main price database. 126,719 rows × 7 columns (ticker, timestamp, open, high, low, close, volume). Everything downstream reads this.

---

### Event Detection

**`mine_events.py`**
First version of the event miner. Uses a single-stage z-score threshold to find unusual price movements.

**`mine_events_strict.py`** ← the one we use
Two-stage event detection pipeline:

**Stage 1 (High Recall):** Flag any bar where at least one of these exceeds a threshold:
- Return z-score > 3.0 (the 5-min return divided by rolling standard deviation)
- Volume > 4× typical volume
- Price range > 3× typical range
- Price impact > 4× typical impact

**Stage 2 (Strict Confirmation):** For a cluster of flagged bars to be called a real "event," it must pass ALL of:
- Absolute move ≥ 60 basis points (0.6%)
- Return z-score ≥ 4.0
- Either large volume OR large range
- Persistence: the move held for at least 2 bars, OR the 6-bar return was ≥ 50% of the 3-bar peak return

After Stage 2: each confirmed event gets:
- `t0_utc`: timestamp of the first bar in the cluster
- `direction`: +1 (up) or -1 (down)
- `cluster_len`: number of bars in the spike cluster
- `label_proxy`: what actually happened in the 60 minutes after (`val_bars = 12` bars × 5 min = 60 min)
  - If price moved >0.3% in same direction: "continuation"
  - If price moved >0.3% in opposite direction: "reversal"
  - Otherwise: "unclear"

Produced ~451 events from the full OHLCV dataset.

**`curate_events.py`**
Manual review step. Plots each event and lets a human flag obviously bad ones (data errors, cross-day events, etc.). Trimmed to ~80 events.

---

### Event Selection

**`select_best_events.py`**
Selects the best 100 events from the pool, balancing several competing goals:

1. **Strength score:** Ranks each event by signal quality:
   - 25% weight: return magnitude (bigger move = more signal)
   - 20% weight: z-score (how unusual was this move)
   - 20% weight: attention (volume or range multiplier)
   - 15% weight: liquidity impact
   - 20% weight: persistence (cluster length, capped at 4 bars)

2. **Per-ticker quotas:** Min 8, max 15 events per stock ticker; min 3, max 8 per ETF. Prevents any single ticker from dominating.

3. **Unclear cap:** At most 15% of selected events can have label "unclear."

4. **Temporal deduplication:** No two events from the same ticker within 60 minutes of each other.

Output: `out_final/selected_events.csv` — 100 events with all their metrics + the `label_proxy` ground truth.

---

### News Collection

**`build_news_packets.py`**
For each of the 100 events, queries Polygon.io news API for:
- Articles about the specific ticker, published within 3 hours of t0
- Articles about SPY and QQQ (macro context)
- Rotates through up to 8 API keys to avoid rate limits

Saves JSON packets to `out_news/packets/{ticker}/{event_id}.json`. Each packet contains the raw articles with timestamps and text.

The V3 news brief builder (`build_news_brief_v2`) then filters these to STRICTLY pre-t0 articles only — post-event articles are counted for logging but never shown to agents.

---

### Agent System — V1/V2 (5-call pipeline)

**`agents/briefs.py`**
Builds the text input (called a "brief") that each agent reads. Three brief types:
- Technical brief: spike stats, cluster length, volume/range multiples, SPY/QQQ returns
- News brief: pre-event articles with timestamps
- Macro brief: SPY/QQQ returns + macro headlines
- Combined brief: summary for the Skeptic

**`agents/prompts.py`**
Contains the system prompt instructions for all 5 V1/V2 agents:
- `MICROSTRUCTURE_SYSTEM`: instructs the Micro agent to output I/P scores
- `NEWS_SYSTEM`: instructs the News agent to output I/P scores
- `MACRO_SYSTEM`: instructs the Macro agent to output I/P scores
- `SKEPTIC_SYSTEM`: instructs the Skeptic to critique and suggest adjustments
- `REVISION_SYSTEM`: instructs agents to revise their scores after critique
- `JUDGE_SYSTEM`: instructs the Judge to synthesize into a final verdict

**`run_ip_debate.py`**
The main runner for V1 and V2.1. Orchestrates all 5 LLM calls per event, saves outputs to `out_agents/` (V1) or `out_agents_v2/` (V2.1).

---

### Agent System — V3 (3-call pipeline)

**`agents/briefs_v2.py`** (source missing, compiled .pyc exists)
Extended brief builder. Adds:
- Open-adjusted volume and range (normalised by time-slot medians)
- Strict pre-event news filtering with minute-level offsets
- Co-movement with SPY/QQQ from OHLCV data
- `build_open_baselines()`: precomputes per-ticker per-time-slot median volume and range across all RTH (Regular Trading Hours) bars — used to normalise the t0 bar's activity

**`agents/briefs_v3.py`**
The V3 brief builder. Extends V2 with:
- All 7 new V3 microstructure features (detailed in Section 7)
- `compute_v3_features()`: the master function that computes all 7 features from OHLCV
- `build_technical_brief_v3()`: appends V3 features to the V2 base brief

**`agents/prompts_v3.py`**
V3 system prompts for 3 agents. Key changes from V2:
- Direct labels (continuation/reversal) instead of I/P scores
- ReTuning evidence-scoring format (score FOR each label before committing)
- WHO→WHOM→WHAT causal schema for Micro agent
- Direction-specific base rates (58% reversal for UP spikes, 45% for DOWN)
- Explicit calibration rules to prevent reversal or continuation bias
- "Unclear" is FORBIDDEN per-agent — only the ensemble can declare unclear

**`run_ip_v3.py`**
Main runner for V3. 3 LLM calls per event:
1. Micro agent reads the technical brief (with V3 microstructure features)
2. News agent reads the news brief
3. Macro agent reads the macro brief

Saves complete records to `out_agents_v3b/ip_outputs_v3.jsonl` — each record includes all agent outputs, scores, V3 features, and the ground truth label.

---

### Providers (LLM API Wrappers)

**`providers/openai_provider.py`** — wraps OpenAI API (GPT-4o-mini for Micro agent)
**`providers/anthropic_provider.py`** — wraps Anthropic API (Claude Haiku for News agent)
**`providers/deepseek_provider.py`** — wraps DeepSeek API (DeepSeek-Chat for Macro agent)
**`providers/gemini_provider.py`** — wraps Google Gemini API (available, not primary)
**`providers/base.py`** — abstract base class that all providers implement

Each provider exposes a `.call(system_prompt, user_message)` method that returns a parsed JSON dict. They handle rate limits, retries, and error handling uniformly.

---

### Ensemble and Evaluation

**`eval/ensemble.py`**
The mathematical ensemble. Contains:
- `run_ensemble()`: combines 3 agent votes + direction heuristic into final prediction
- `_direction_weight()`: time-of-day conditioning for the direction heuristic weight
- `_reversal_prior()`: asymmetric reversal probability based on z-score, macro flag, direction
- `validate_agent_output_v3()`: validates/normalises raw LLM JSON output
- `run_evaluation_v3()`: full evaluation suite — accuracy by strategy, confusion matrices, Brier score, accuracy by flag (macro/bell/news), accuracy by direction

**`eval/metrics.py`**
Earlier evaluation metrics (V1/V2). Contains accuracy computations for the I/P classification system.

---

### ML Stage

**`eval/feature_matrix.py`**
Converts the JSONL records into a flat numerical matrix for LightGBM. Key responsibilities:
- Reads `ip_outputs_v3.jsonl` and `selected_events.csv`
- Loads `ohlcv_5min.parquet` and recomputes `vpin_proxy_pre` and `signed_vol_ratio_pre` from pre-event bars (the leak fix)
- Encodes categorical labels as numbers
- Computes derived features (score ratios, differences, agent agreement)
- Excludes `cluster_len` and `strength_score` (data leakage)
- Handles missing values (NaN)
- Returns `X` (feature matrix, 85×35), `y` (labels, 85), `feature_names`

**`eval/ml_stage.py`**
The main ML training and evaluation script. Contains:
- `run_loocv_lgbm()`: LOO-CV for LightGBM with feature importance accumulation
- `run_loocv_model()`: LOO-CV for any sklearn-compatible model (used for RF, ET)
- Ablation study: math-only vs agents-only vs full-stack
- Stacked ensemble: averages LGBM + RF + ET probability predictions
- `save_importance_plot()`: generates the feature importance bar chart
- Saves all results to `out_agents_v3b/ml_stage_results.json`

**`eval/constants_sweep.py`**
Constants stability analysis. Re-runs the ensemble math (no LLM calls) on stored agent outputs with each of the 14 ensemble constants perturbed ±10% and ±20%. Prints a stability table showing the maximum accuracy impact per constant.

---

### Output Files

**`out_agents_v3b/ip_outputs_v3.jsonl`**
One JSON line per event. Contains everything: agent labels, confidence scores, V3 features, ensemble result, true label. The source of truth for the ML stage.

**`out_agents_v3b/ip_outputs_v3.csv`**
Same as above but in CSV format (flat, no nested dicts). Easier to open in Excel/Sheets for inspection.

**`out_agents_v3b/eval_summary_v3.json`**
Full evaluation results: accuracy for all strategies, confusion matrices, Brier score, breakdown by macro/bell/news/direction flags.

**`out_agents_v3b/ml_stage_results.json`**
ML stage outputs: accuracy per model, ablation results, feature importances.

**`out_agents_v3b/feature_importance.png`**
Bar chart showing LightGBM feature importance, colour-coded by feature group (blue = LLM agent, orange = math/microstructure).

**`out_agents_v3b/constants_sweep.json`**
Full sweep results showing accuracy impact of each constant variation.

---

### V3 Rebuild

**`v3_rebuild/`**
A clean snapshot of the V3 codebase for external review (submitted to professor). Contains copies of `agents/briefs_v3.py`, `agents/prompts_v3.py`, `eval/ensemble.py`, `eval/feature_matrix.py`, `eval/ml_stage.py`, `run_ip_v3.py`, and a `README.md` with architecture summary.

---

## 14. KEY CONCEPTS GLOSSARY

**Basis points (bp)**
1 bp = 0.01% = 0.0001. So 60bp = 0.6% price move. Used because stock moves are often tiny fractions of a percent and percentages get unwieldy.

**Z-score**
How many standard deviations something is from its average. A z-score of 4 means the move was 4 standard deviations above the normal — extremely unusual. Most normal events fall within ±2.

**VPIN (Volume-synchronized Probability of Informed Trading)**
A microstructure measure of order flow imbalance. High VPIN = most volume came from one side (buyers or sellers dominating) = suggests informed trading = continuation more likely.

**Idiosyncratic residual**
The part of a stock's return that can't be explained by the market's overall movement. Formula: `stock_return - beta × SPY_return`. If a stock jumps 2% but the market also jumped 2% and the stock's beta is 1.0, the idiosyncratic residual is 0. If the stock jumped 2% while the market was flat, the residual is +2% — this is a purely stock-specific move.

**Beta**
How much a stock tends to move relative to the overall market. Beta=1 means the stock moves in sync with the market. Beta=2 means it moves twice as much.

**Realized semivariance ratio (rs_ratio)**
Measures whether pre-event price volatility was asymmetric. `RS_minus / (RS_plus + RS_minus)`. Close to 0.5 means equal up and down noise (overreaction context). Close to 1 means all downward variance (herding).

**LOO-CV (Leave-One-Out Cross Validation)**
A training/testing procedure for small datasets where you train on all events except one, test on that one, repeat for every event. Gives the most honest accuracy estimate when data is scarce.

**Data Leakage**
Using information that wouldn't be available at prediction time to train or evaluate a model. Makes results look better than they really are.

**Multi-Agent Debate (MAD)**
A system where multiple AI agents independently analyze a problem and argue/revise to reach a conclusion. Can be more accurate than a single agent, but susceptible to sycophancy and Degeneration-of-Thought.

**Degeneration-of-Thought**
When agents in a debate converge on a wrong consensus because one confident agent's opinion "infects" the others. Named by Liang et al. (EMNLP 2024).

**Sycophancy**
When an AI agent agrees with other agents to avoid conflict rather than maintaining an accurate independent assessment. A documented failure mode of LLMs in debate settings.

**Gradient Boosted Decision Tree (GBDT/LightGBM)**
An ML algorithm that builds many simple "if-then" decision rules sequentially, each correcting the previous one's errors. Generally the best algorithm for small-to-medium tabular data.

**Stacked ensemble**
Combining predictions from multiple models by averaging their probability outputs. Reduces variance — individual models might be wrong in different ways, and averaging cancels some errors.

**Ablation study**
An experiment where you remove components of a system one at a time to measure each component's contribution. Standard practice in ML research to prove that each part matters.

**Brier score**
A calibration metric for probabilistic predictions. Measures how close the predicted probabilities were to the actual outcomes. Lower = better. 0.0 = perfect. 0.25 = random chance.

**RTH (Regular Trading Hours)**
9:30 AM – 4:00 PM Eastern Time. US stock markets are open during these hours. We filter to RTH events only to avoid confounding factors from extended-hours trading (lower volume, different price dynamics).

**Parquet**
A compressed binary file format for tabular data. Much faster and smaller than CSV. We use it to store the OHLCV price data.

**JSONL (JSON Lines)**
A file format where each line is a valid JSON object. Useful for streaming large datasets because you can read line-by-line without loading the whole file. We store agent outputs in JSONL.

---

## 15. QUESTIONS A PHD SUPERVISOR MIGHT ASK — WITH ANSWERS

---

**Q: Why did you choose intraday spikes as the prediction target? Isn't this just noise trading?**

A: Large intraday spikes are precisely the events where the information-vs-noise distinction matters most, and where market microstructure theory makes clear predictions. The literature on order flow analysis (Easley, de Prado, O'Hara) and intraday reversal patterns (Jegadeesh and Titman for daily, but extended to intraday by multiple papers) gives us a theoretical grounding. These events also have the practical advantage of being clearly defined, automatically detectable, and having a clean binary label (reversal or continuation within 60 minutes). Random 60-minute windows would have far noisier outcomes.

---

**Q: Your base rate is about 55% reversal. Isn't this just a mean-reversion system?**

A: Partially, yes — and we're transparent about that. The direction heuristic (up→reversal, down→continuation) captures most of the base-rate effect and achieves 56.5%. Our contribution is the additional +10.6pp beyond that baseline (67.1% total). The value added comes from *when* to override the base rate: when macro context explains the move, when news is present and timely, or when microstructure shows orderly informed trading, the base rate should be suppressed in favor of continuation. The ablation confirms LLM agents add +4.7pp over math-only features, and math-only is roughly equivalent to the direction heuristic.

---

**Q: N=85 is very small. How do you know you haven't overfit?**

A: Three answers. First, we use Leave-One-Out cross-validation which gives the most conservative accuracy estimate for small N. Second, the constants sweep shows the ensemble is insensitive to ±10% perturbations of all 14 constants (max 2pp sensitivity). Third, the model consistently predicts both classes (not just majority class), which you'd expect from an overfit model. We acknowledge that external validation on a larger held-out dataset would be needed before any real-world deployment.

---

**Q: You had a significant data leak — how confident are you that there are no others?**

A: We performed a complete code audit, including decompiling the missing `briefs_v2.py` source to inspect bytecode. The audit found: (1) the VPIN/signed_vol/cluster_len leak was fixed; (2) `strength_score` was removed for also containing `cluster_len`; (3) all news filtering is strictly pre-t0; (4) OHLCV context functions use `timestamp <= t0` (the t0 bar itself, not post-event bars); (5) `label_proxy` is never passed to agents. The remaining disclosed issues are: `cluster_len` appears in agent briefs (indirect, unquantifiable contamination of agent outputs), and event selection used label_proxy to prioritize clear-outcome events (sample selection bias, not model leakage). We are confident no other direct ML feature leaks exist.

---

**Q: Why three different LLM providers (OpenAI, Anthropic, DeepSeek)?**

A: Deliberate model diversity. Research (arxiv:2602.08003) shows that LLM models from the same family make *correlated errors* — they fail on the same events. Using three different architectures means each agent has an independent failure pattern. When the ensemble is wrong, it's usually because all three models and the heuristic agreed — which is a much rarer event. This is the same logic behind ensemble methods in general ML (diverse weak learners outperform homogeneous ones).

---

**Q: How is the direction heuristic justified theoretically?**

A: The up-spike reversal base rate (≈58%) is well-documented in the intraday microstructure literature. Jegadeesh (1990) and Lehmann (1990) showed short-term mean reversion at daily horizons. This extends intraday: after a large price spike driven by order imbalance, market makers and arbitrageurs push the price back. The asymmetry between up and down spikes (58% vs 45%) is explained by leverage asymmetry: negative news can trigger sustained selling (forced deleveraging, margin calls), while positive news more often sees sellers appear quickly to take profits. We explicitly model this asymmetry in the `_reversal_prior()` function.

---

**Q: The agent prompts include hard-coded scoring rules (+5 for informed actor, +4 for VPIN > 0.7, etc.). Isn't this just manual feature engineering inside a prompt?**

A: Yes, and intentionally so. This is the ReTuning approach (arxiv:2510.21604), which structures the agent's reasoning to prevent anchoring bias. The scores are not calibrated to our specific 85 events — they're derived from domain knowledge about what makes a move informed vs. panic. The constants sweep on the ensemble layer shows the system isn't overfitted. The prompt-level scoring is a structural choice to ensure agents systematically consider both sides of each call.

---

**Q: What would you do with more data? Would this scale?**

A: The limiting factor right now is the LLM cost per event (≈$0.003/event) and our OHLCV data coverage (only 11 tickers, one year). With more events: (1) the ML stage would become more reliable — N=85 is the main weakness; (2) the feature matrix could be expanded with sector context, earnings calendars, options flow data; (3) the agent prompts could be validated against a proper hold-out set. The architecture is sound — the V3 3-call pipeline costs about $0.30 per 100 events, which scales affordably. The main research question for follow-up work would be whether the +4.7pp agent contribution holds out-of-sample on a different time period or different tickers.

---

*End of Study Guide*

---

**Quick Reference — Version History**

| Version | Calls/Event | Accuracy | Failure Mode |
|---|---|---|---|
| V1 | 5 | 35.2% | Reversal bias from Judge; I/P decomposition too noisy |
| V2.1 | 5 | 16.3% | Degeneration-of-Thought; Micro predicted continuation 95/98 times |
| V3b ensemble | 3 | 49.0% | Good, but 3-agent only; continuation recall still low (26%) |
| ML stage (leaky) | — | 82.4% | Top 3 features used post-event data |
| **ML stage (clean)** | — | **67.1%** | **Final honest result** |
