"""
Path C step 3 — build per-event agent briefs (free data: yfinance daily OHLCV).

Produces out_pathc/briefs.jsonl, one record per event with:
  event_id, ticker, label fields (carried through), and
  fundamentals_brief / reaction_brief  (text the agents will read).

The NEWS brief is added on the run machine (point-in-time Polygon headlines around
the earnings date — reuse build_news_packets.py logic). Here we leave a stub so the
pipeline runs even without news.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(HERE, "out_pathc", "earnings_events.csv")
OUT = os.path.join(HERE, "out_pathc", "briefs.jsonl")
PRIMARY_HORIZON = 5  # label_5d is the target


def main():
    import json
    ev = pd.read_csv(EVENTS)
    tickers = sorted(ev["ticker"].unique())
    start = (pd.to_datetime(ev["announce_date"]).min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(ev["announce_date"]).max() + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False, group_by="ticker")

    def bars(tk):
        d = px[tk].dropna().copy()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        return d

    cache = {tk: bars(tk) for tk in tickers}

    n = 0
    with open(OUT, "w") as f:
        for _, e in ev.iterrows():
            tk = e["ticker"]
            d = cache[tk]
            rdate = pd.Timestamp(e["reaction_date"])
            if rdate not in d.index:
                continue
            R = d.index.get_loc(rdate)
            cR = d["Close"].iloc[R]
            volR = d["Volume"].iloc[R]
            # context windows (all strictly up to and including reaction day)
            avg_vol = d["Volume"].iloc[max(0, R - 20):R].mean()
            pre20 = d["Close"].iloc[R - 20] if R >= 20 else d["Close"].iloc[0]
            trend_20d = cR / pre20 - 1.0
            hi_52 = d["High"].iloc[max(0, R - 252):R + 1].max()
            lo_52 = d["Low"].iloc[max(0, R - 252):R + 1].min()
            dist_hi = cR / hi_52 - 1.0
            # historical typical |earnings reaction| for this ticker (other events)
            other = ev[(ev["ticker"] == tk) & (ev["event_id"] != e["event_id"])]
            typ_react = other["reaction_ret"].abs().mean() if len(other) else np.nan
            react = e["reaction_ret"]; surp = e["surprise_pct"]
            rdir = "UP" if e["reaction_dir"] == 1 else "DOWN"
            surp_sign = "beat" if (pd.notna(surp) and surp > 0) else "miss" if pd.notna(surp) else "n/a"
            contradicts = (pd.notna(surp) and np.sign(surp) != e["reaction_dir"])

            fundamentals_brief = (
                f"EARNINGS FUNDAMENTALS — {tk} (reaction {e['reaction_date']})\n"
                f"  EPS surprise: {surp:+.1f}%  ({surp_sign})\n" if pd.notna(surp) else
                f"EARNINGS FUNDAMENTALS — {tk} (reaction {e['reaction_date']})\n  EPS surprise: n/a\n"
            )
            fundamentals_brief += (
                f"  Earnings-day reaction: {react:+.2%} ({rdir})\n"
                f"  Reaction vs surprise: {'CONTRADICTS surprise sign (guidance/expectations driven)' if contradicts else 'matches surprise sign'}\n"
                f"  Typical |reaction| for {tk}: {typ_react:.2%}\n" if pd.notna(typ_react) else ""
            )

            reaction_brief = (
                f"PRICE REACTION — {tk} (reaction {e['reaction_date']})\n"
                f"  Earnings-day move: {react:+.2%} ({rdir}) on volume {volR/avg_vol:.1f}x 20d-avg\n"
                f"  Move vs typical earnings move: {react/typ_react:+.1f}x\n" if pd.notna(typ_react) and typ_react>0 else ""
            )
            reaction_brief += (
                f"  20-day trend into earnings: {trend_20d:+.2%}\n"
                f"  Distance from 52-week high: {dist_hi:+.2%}\n"
                f"  Trend {'ALIGNED with' if np.sign(trend_20d)==e['reaction_dir'] else 'AGAINST'} reaction direction\n"
            )

            rec = {
                "event_id": e["event_id"], "ticker": tk,
                "announce_date": e["announce_date"], "reaction_date": e["reaction_date"],
                "reaction_dir": int(e["reaction_dir"]), "reaction_ret": float(react),
                "surprise_pct": (None if pd.isna(surp) else float(surp)),
                "true_label": e[f"label_{PRIMARY_HORIZON}d"],
                "drift_5d": (None if pd.isna(e["drift_5d"]) else float(e["drift_5d"])),
                "fundamentals_brief": fundamentals_brief,
                "reaction_brief": reaction_brief,
                "news_brief": "[news brief added on run machine via Polygon]",
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} briefs -> {OUT}")


if __name__ == "__main__":
    main()
