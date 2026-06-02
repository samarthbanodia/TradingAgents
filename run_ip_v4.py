"""
V4 runner — re-runs the V3 agent pipeline with the DEBIASED v4 prompts.

It reuses ALL of run_ip_v3's machinery (providers, brief building, ensemble,
parsing, cost logging, resume) and only swaps the three system prompts. Output
goes to a new directory so the original v3 outputs are never touched.

Usage (defaults target the 800-event extended dataset):
    python run_ip_v4.py                 # all 800 events -> out_agents_v4a/
    python run_ip_v4.py --limit 50      # quick pilot on first 50 events
    python run_ip_v4.py --resume        # skip already-done events

Requires a .env with OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY
(same keys the original v3 run used). Cost ≈ $0.00287/event (≈ $2.30 for all 800).
"""
import sys
import run_ip_v3
from agents import prompts_v4

# Swap the system prompts in run_ip_v3's namespace (functions read these globals
# at call time, so reassigning here takes effect for the whole run).
run_ip_v3.MICROSTRUCTURE_SYSTEM_V3 = prompts_v4.MICROSTRUCTURE_SYSTEM_V4
run_ip_v3.NEWS_SYSTEM_V3 = prompts_v4.NEWS_SYSTEM_V4
run_ip_v3.MACRO_SYSTEM_V3 = prompts_v4.MACRO_SYSTEM_V4

# Default to the extended (N=800) dataset paths unless the user overrides them.
_DEFAULTS = {
    "--events_csv": "out_extended_final/selected_events.csv",
    "--packets_dir": "out_extended_news/packets",
    "--ohlcv_path": "ohlcv_5min_extended.parquet",
    "--out_dir": "out_agents_v4a",
}
for flag, val in _DEFAULTS.items():
    if flag not in sys.argv:
        sys.argv += [flag, val]

if __name__ == "__main__":
    print("[v4] DEBIASED prompts active — output -> out_agents_v4a/")
    run_ip_v3.main()
