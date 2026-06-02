---
name: v2_pipeline_status
description: V2.1 I-P debate pipeline — design decisions, V1 baseline analysis, and run instructions
type: project
---

V2.1 pipeline built (2026-03-13). All code written, awaiting .env keys to run.

**Why:** V1 full pipeline scored only 35.2% — WORSE than always predicting reversal (45.1%). Root causes identified and fixed.

**Root cause of V1 failure:**
- 87% of events are opening-bell (09:30 ET). Raw peak_volume_mult averaged 497x for these events. V1 agents saw 497x volume → called panic → predicted reversal. But open-adjusted volume averaged only 1.34x (completely normal). 85/87 opening-bell events had adj_vol < 3x.
- Post-event news was included in briefs (backward causality).
- Round-2 re-reasoning degraded Micro R1 from 45.1% to 35.2%.
- SPY/QQQ co-move was null (pyarrow missing in V1 run).

**V1 baseline table (N=91):**
- Micro R1 alone: 45.1% (= always-reversal; dataset is 45.1% reversal)
- Direction heuristic (up=reversal, down=cont): 48.3% — best simple baseline
- V1 full pipeline: 35.2% — worst of all strategies

**V2.1 fixes:**
1. Open-adjusted volume: per-ticker per-slot median from OHLCV
2. Pre-event news only (published_utc < t0 strictly)
3. SPY/QQQ co-move restored
4. No Round-2 re-reasoning (5 calls vs 8)
5. Direction prior (weak +0.03 to P for up-spikes when uncertain)
6. R1 agreement level → Judge confidence modifier

**Files:** run_ip_v2.py, agents/briefs_v2.py, agents/prompts_v2.py, eval/metrics_v2.py
**Output:** out_agents_v2/ip_outputs_v2.jsonl, ip_outputs_v2.csv, eval_summary_v2.json

**Run commands:**
- Smoke test: python run_ip_v2.py --limit 3
- Full run: python run_ip_v2.py
- Resume: python run_ip_v2.py --resume

**Cost:** ~$1.46 for 100 events (5 calls/event, vs $1.85 for V1's 8 calls)

**How to apply:** .env file needed with OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY. Models: gpt-4o-mini (micro), claude-haiku-4-5-20251001 (news+skeptic), deepseek-chat (macro), gpt-4o (judge).
