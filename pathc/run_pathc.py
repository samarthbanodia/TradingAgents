"""
Path C step 4 — run the 3-agent earnings-drift panel.  (NEEDS API KEYS — run on the keyed machine.)

Reads out_pathc/briefs.jsonl, calls Fundamentals / News / Reaction agents, combines
them with a confidence-weighted vote (continuation prior ~0.60), and writes
out_pathc/agent_outputs.jsonl.

Reuses the existing provider wrappers + .env loading. Single key works (all three
agents fall back to whatever provider you have), exactly like run_ip_v4.

  python pathc/run_pathc.py --limit 20     # pilot
  python pathc/run_pathc.py                # all 91
"""
import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from providers import OpenAIProvider, AnthropicProvider, DeepSeekProvider
from pathc.prompts_pathc import FUNDAMENTALS_SYSTEM, NEWS_SYSTEM, REACTION_SYSTEM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIEFS = os.path.join(HERE, "out_pathc", "briefs.jsonl")
OUT = os.path.join(HERE, "out_pathc", "agent_outputs.jsonl")
CONT_PRIOR = 0.60  # measured drift base rate


def parse(raw):
    """Normalize an agent response. providers.call() already returns a parsed
    dict; fall back to lenient string extraction for raw text responses."""
    if isinstance(raw, dict):
        o = raw
    else:
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            o = json.loads(m.group(0)) if m else {}
        except Exception:
            o = {}
    lab = str(o.get("label", "")).lower()
    lab = "continuation" if "cont" in lab else "reversal" if "rev" in lab else "continuation"
    try:
        conf = float(o.get("confidence", 0.55))
    except Exception:
        conf = 0.55
    conf = min(0.95, max(0.50, conf))
    return {"label": lab, "confidence": conf,
            "continuation_score": o.get("continuation_score"),
            "reversal_score": o.get("reversal_score"),
            "key_reason": o.get("key_reason", "")}


def ensemble(f, n, r):
    """Confidence-weighted vote + continuation prior."""
    cont = CONT_PRIOR
    rev = 1 - CONT_PRIOR
    w = {"f": 0.40, "n": 0.30, "r": 0.30}
    for key, a in (("f", f), ("n", n), ("r", r)):
        if a["label"] == "continuation":
            cont += w[key] * a["confidence"]
        else:
            rev += w[key] * a["confidence"]
    label = "continuation" if cont >= rev else "reversal"
    margin = abs(cont - rev) / (cont + rev)
    return label, round(margin, 4), {"continuation": round(cont, 4), "reversal": round(rev, 4)}


def init_providers(args):
    load_dotenv()
    ok, ak, dk = os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY"), os.getenv("DEEPSEEK_API_KEY")
    if not (ok or ak or dk):
        sys.exit("No API keys in .env (OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY)")

    def pick(pref):
        for kind in pref:
            if kind == "openai" and ok:
                return OpenAIProvider(args.openai_model, ok, args.temperature, args.max_tokens)
            if kind == "anthropic" and ak:
                return AnthropicProvider(args.anthropic_model, ak, args.temperature, args.max_tokens)
            if kind == "deepseek" and dk:
                return DeepSeekProvider(args.deepseek_model, dk, args.temperature, args.max_tokens)
        return None
    return {
        "fundamentals": pick(["openai", "anthropic", "deepseek"]),
        "news": pick(["anthropic", "openai", "deepseek"]),
        "reaction": pick(["deepseek", "openai", "anthropic"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--anthropic_model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--deepseek_model", default="deepseek-chat")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    prov = init_providers(args)
    briefs = [json.loads(l) for l in open(BRIEFS)]
    if args.limit:
        briefs = briefs[:args.limit]

    with open(OUT, "w") as out:
        for i, b in enumerate(briefs):
            fb = b["fundamentals_brief"] + "\n\nSURPRISE: " + str(b.get("surprise_pct"))
            nb = b.get("news_brief", "") + f"\n(reaction was {b['reaction_ret']:+.2%})"
            rb = b["reaction_brief"]
            f = parse(prov["fundamentals"].call(FUNDAMENTALS_SYSTEM, fb))
            n = parse(prov["news"].call(NEWS_SYSTEM, nb))
            r = parse(prov["reaction"].call(REACTION_SYSTEM, rb))
            label, margin, votes = ensemble(f, n, r)
            rec = {**{k: b[k] for k in ("event_id", "ticker", "announce_date",
                                        "reaction_date", "reaction_dir", "reaction_ret",
                                        "surprise_pct", "true_label", "drift_5d")},
                   "fundamentals_label": f["label"], "fundamentals_conf": f["confidence"],
                   "news_label": n["label"], "news_conf": n["confidence"],
                   "reaction_label": r["label"], "reaction_conf": r["confidence"],
                   "ensemble_label": label, "ensemble_margin": margin, "ensemble_votes": votes}
            out.write(json.dumps(rec) + "\n")
            print(f"[{i+1}/{len(briefs)}] {b['event_id']:22} -> {label:12} "
                  f"(F={f['label'][:4]} N={n['label'][:4]} R={r['label'][:4]}) true={b['true_label']}")
    print(f"\nwrote {len(briefs)} -> {OUT}")


if __name__ == "__main__":
    main()
