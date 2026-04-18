# V3 Rebuild — Direct-Label Ensemble + ML Stack

This folder contains the full V3 redesign of the intraday price-spike prediction pipeline.
V3 replaces the V1 Skeptic+Judge multi-agent debate (35.2% accuracy) with a direct-label
weighted-vote ensemble and a second-stage ML classifier.

---

## Results Summary

| System                        | Accuracy (3-way) | Accuracy (clear events only) |
|-------------------------------|-----------------|------------------------------|
| V1 Skeptic+Judge debate       | 35.2%           | —                            |
| Direction heuristic (baseline)| 48.0%           | 56.5%                        |
| V3 ensemble (this repo)       | 49.0%           | **57.6%**                    |
| ML stage on V3 outputs        | —               | **82.4%** (LOO-CV, n=85)     |

Evaluated on 100 intraday events (43 continuation / 42 reversal / 15 unclear).

---

## What Changed from V1

**Dropped:**
- Skeptic agent
- LLM Judge
- Multi-round debate loop

**Added:**
- 3 direct-label agents (Micro, News, Macro) each output `continuation` or `reversal` + confidence scores
- 7 new microstructure features computed per event (see `agents/briefs_v3.py`)
- Research-backed weighted vote ensemble with asymmetric reversal prior (see `eval/ensemble.py`)
- Second-stage ML classifier (LightGBM + RF + ET stacked, see `eval/ml_stage.py`)

---

## File Structure

```
v3_rebuild/
├── agents/
│   ├── briefs_v3.py        # Brief builders + 7 microstructure features
│   └── prompts_v3.py       # Direct-label prompts (RETuning evidence scoring)
├── eval/
│   ├── ensemble.py         # Weighted vote ensemble + evaluation
│   ├── feature_matrix.py   # 37-feature matrix builder for ML stage
│   └── ml_stage.py         # LGBM + RF + ET LOO-CV + stacked ensemble
├── out/
│   ├── ip_outputs_v3.jsonl      # Per-event agent outputs (100 events)
│   ├── ip_outputs_v3.csv        # Same, CSV format
│   ├── eval_summary_v3.json     # Full evaluation breakdown by strategy
│   └── ml_stage_results.json   # ML stage accuracy + feature importance
├── run_ip_v3.py            # Main pipeline entrypoint
└── README.md
```

> **Note:** `briefs_v3.py` imports from `agents/briefs_v2.py` (not included here) for shared
> helpers. `run_ip_v3.py` and the eval modules also depend on the root-level `providers/` and
> `data/` directories. This folder is a snapshot of the V3 code — run from the project root.

---

## V3 Microstructure Features (`agents/briefs_v3.py`)

Seven new features added on top of V2.1 brief data:

| Feature | Description | Paper basis |
|---------|-------------|-------------|
| `idio_residual` | Stock return minus beta × SPY return at t0 | Brogaard et al. (SSRN 4731947, 2024) |
| `rs_ratio` | RS_minus / (RS_plus + RS_minus) — downside semivariance ratio | Liu et al. (J. Empirical Finance, 2023) |
| `signed_vol_ratio` | Net signed volume (buy/sell pressure) over cluster bars | VPIN literature (Easley, de Prado, O'Hara) |
| `vpin_proxy` | \|net signed vol\| / total vol — order flow imbalance | Easley et al. |
| `spread_bp` | EDGE bid-ask spread proxy in basis points | JFE 2024 |
| `pre_spike_run_len` | Consecutive bars in spike direction before t0 | arXiv:2601.04959 |
| `is_macro_driven` | 1 if SPY z-score at t0 > 1.5 (market-wide event) | arXiv:2408.03594 |

---

## Ensemble Design (`eval/ensemble.py`)

- **3 LLM agents** vote: Micro (0.35), News (0.20), Macro (0.15)
- **Direction heuristic** as 4th voter (weight 0.22–0.38, time-of-day conditioned)
- **Asymmetric reversal prior:** UP spikes get 60–62% reversal prior for z > 3; DOWN spikes reduced by 0.82× (leverage asymmetry)
- **Macro-driven flag** reduces reversal prior (macro moves absorb info efficiently → continuation)
- **Unclear** only declared at ensemble level when top vote < 0.35; agents always output a binary label

---

## ML Stage (`eval/ml_stage.py` + `eval/feature_matrix.py`)

**Input:** 37 features built from V3 agent outputs + microstructure

| Feature group | Count | Examples |
|---|---|---|
| Agent features | 15 | micro/news/macro label, conf, score diff/ratio, agreement |
| Ensemble features | 5 | vote_cont, vote_rev, vote_margin, dir_weight, dir_prior |
| V3 microstructure | 8 | vpin_proxy, signed_vol_ratio, idio_resid_bp, rs_ratio, ... |
| Context flags | 4 | is_up_spike, has_pre_event_news, is_opening_bell, is_macro_driven |
| Raw event features | 5 | peak_z_ret_3, cluster_len, peak_volume_mult, peak_range_mult, strength_score |

**Classifiers:** LightGBM, RandomForest, ExtraTrees — equal-weight probability stacking

**Validation:** Leave-One-Out CV (each prediction trained on the other 84 events — no data leakage)

**Top features by LightGBM importance:**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | vpin_proxy | 114 |
| 2 | cluster_len | 92 |
| 3 | signed_vol_ratio | 88 |
| 4 | peak_range_mult | 38 |
| 5 | idio_resid_bp | 35 |
| 6 | news_ratio | 34 |
| 7 | peak_z_ret_3 | 31 |

**Key finding:** Microstructure features (VPIN, cluster_len, signed_vol) dominate. LLM agent
labels have low individual importance; their score ratios (news_ratio, score diff) matter more
than their discrete label outputs.

---

## Running

From the project root:

```bash
# Run V3 agent pipeline (writes to out_agents_v3b/)
python run_ip_v3.py

# Run ML stage on existing outputs
python -m eval.ml_stage
python -m eval.ml_stage --jsonl out_agents_v3b/ip_outputs_v3.jsonl
```

---

## Research Backing

Key papers informing V3 design:

- MAD sycophancy / degeneration-of-thought → drop Skeptic+Judge (arXiv:2502.08788, 2305.19118, 2509.23055)
- ReConcile confidence-weighted voting (ACL 2024, arXiv:2309.13007)
- Intraday reversal asymmetry (arXiv:2501.16772)
- RETuning evidence scoring to prevent anchor bias (arXiv:2510.21604)
- WHO→WHOM→WHAT causal schema for Micro agent (arXiv:2512.17923)
- OFI/VAR macro-continuation (arXiv:2508.06788)
