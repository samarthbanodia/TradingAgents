"""
Scaled quantitative interaction test (still NO LLM, free). Decides whether the
catalyst x overreaction interaction is real before any LLM spend.

~110 tickers across caps/sectors. Leak-free, market-adjusted, chronological holdout.
Two notions of 'catalyst strength':
  (a) magnitude  = |EPS surprise|
  (b) alignment  = does the surprise SIGN justify the reaction direction? (confirmed)
For each, among HIGH-overreaction events: reversal rate strong vs weak, with a
bootstrap 95% CI ON THE GAP (the actual significance test).
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import yfinance as yf

UNIVERSE = list(dict.fromkeys([
 "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AMD","TSLA","NFLX","CRM","ORCL","ADBE","AVGO","INTC","QCOM",
 "CSCO","TXN","NOW","PANW","SNOW","CRWD","DDOG","NET","ZS","PLTR","COIN",
 "JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW","V","MA","PYPL","SOFI","HOOD","UPST","AFRM",
 "JNJ","UNH","PFE","MRK","ABBV","LLY","TMO","ABT","BMY","GILD","AMGN","CVS","MRNA",
 "WMT","HD","PG","KO","PEP","MCD","NKE","SBUX","TGT","LOW","COST","DIS","CMG","ETSY","RBLX","DKNG","ROKU","CHWY","BMBL",
 "CAT","BA","GE","HON","UPS","XOM","CVX","COP","SLB","FCX","NUE","DE","LMT","RTX",
 "PLUG","RUN","ENPH","FSLR","RIVN","LCID","ASAN","GTLB","DOCN","FUBO","BYND","SMCI","MARA","RIOT","CVNA","U","PATH","AI"]))
H = 5; SURP_CAP = 50.0; HOLDOUT_DATE = pd.Timestamp("2025-10-01")


def gap_ci(strong_rev, weak_rev, n=4000, seed=0):
    """bootstrap CI on (weak reversal - strong reversal); H1 => positive, CI excludes 0."""
    rng = np.random.default_rng(seed); s = np.asarray(strong_rev,float); w = np.asarray(weak_rev,float)
    if len(s) < 5 or len(w) < 5: return (np.nan, np.nan, np.nan)
    d = [w[rng.integers(0,len(w),len(w))].mean() - s[rng.integers(0,len(s),len(s))].mean() for _ in range(n)]
    return (w.mean()-s.mean(), np.percentile(d,2.5), np.percentile(d,97.5))


def main():
    LO, HI = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-05-01")
    cal = []
    for t in UNIVERSE:
        try: df = yf.Ticker(t).get_earnings_dates(limit=40)
        except Exception: continue
        if df is None or df.empty: continue
        df = df.reset_index()
        for _, r in df.iterrows():
            dt = r["Earnings Date"]
            if pd.Timestamp(LO,tz=dt.tz) <= dt <= pd.Timestamp(HI,tz=dt.tz):
                cal.append({"ticker":t,"dt":dt,"surprise":r.get("Surprise(%)"),
                            "timing":"AMC" if dt.time()>=pd.Timestamp("16:00").time() else "BMO" if dt.time()<=pd.Timestamp("09:30").time() else "INTRA"})
    cal = pd.DataFrame(cal); cal["date"]=cal["dt"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    px = yf.download(UNIVERSE+["SPY"], start="2022-10-01", end="2026-06-01", interval="1d", auto_adjust=True, progress=False, group_by="ticker")
    def C(t):
        try: s=px[t]["Close"].dropna(); s.index=pd.to_datetime(s.index).tz_localize(None).normalize(); return s
        except Exception: return None
    spy=C("SPY")
    rows=[]
    for _,e in cal.iterrows():
        s=C(e["ticker"])
        if s is None or pd.isna(e["surprise"]): continue
        idx=s.index; R= idx.searchsorted(e["date"]) if e["timing"]=="BMO" else idx.searchsorted(e["date"],side="right")
        if R<1 or R+H>=len(idx): continue
        ar=(s.iloc[R]/s.iloc[R-1]-1)-((spy.loc[idx[R]]/spy.loc[idx[R-1]]-1) if idx[R] in spy.index and idx[R-1] in spy.index else 0)
        ad=(s.iloc[R+H]/s.iloc[R]-1)-((spy.loc[idx[R+H]]/spy.loc[idx[R]]-1) if idx[R+H] in spy.index and idx[R] in spy.index else 0)
        rdir=1 if ar>0 else -1
        rows.append({"ticker":e["ticker"],"date":e["date"],"over":abs(ar),
                     "mag":min(abs(e["surprise"]),SURP_CAP),
                     "confirmed":int(np.sign(e["surprise"])==rdir),
                     "rev":int(ad*rdir<0)})
    df=pd.DataFrame(rows)
    eff_n = df.groupby("date").ngroups
    print(f"events={len(df)} | tickers={df.ticker.nunique()} | distinct dates(eff.N)={eff_n} | reversal base={df.rev.mean():.3f}")
    hi=df[df.over>=df.over.quantile(2/3)]
    print(f"high-overreaction subset n={len(hi)}\n")

    def test(sub, mask_strong, name):
        s=sub[mask_strong]; w=sub[~mask_strong]
        g,lo,h=gap_ci(s.rev.values,w.rev.values)
        sig="*** SIGNIFICANT (CI excludes 0)" if (lo>0) else ("borderline" if lo>-0.03 else "n.s.")
        print(f"{name}")
        print(f"   strong: rev={s.rev.mean():.3f} (n={len(s)}) | weak: rev={w.rev.mean():.3f} (n={len(w)}) | gap(weak-strong)={g:+.3f} CI=[{lo:+.3f},{h:+.3f}]  {sig}")

    print("=== INTERACTION TESTS (H1: strong catalyst -> continues; weak -> reverts) ===")
    test(hi, hi.mag>hi.mag.median(), "(a) by surprise MAGNITUDE (|surp|>median):")
    test(hi, hi.confirmed==1,        "(b) by surprise ALIGNMENT (surprise justifies the move):")
    print("\n=== robustness: same alignment test at full sample + holdout ===")
    test(df, df.confirmed==1, "(b-all events) alignment, all overreaction levels:")
    ho=df[df.date>=HOLDOUT_DATE]; hoi=ho[ho.over>=df.over.quantile(2/3)]
    test(hoi, hoi.confirmed==1, "(b-holdout) alignment, high-overreaction, OUT-OF-SAMPLE:")


if __name__ == "__main__":
    main()
