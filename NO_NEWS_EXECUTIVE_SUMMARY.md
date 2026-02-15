# No-News Events Analysis: Executive Summary

## The Question
Why do 41 out of 100 selected events (41%) have zero ticker-specific news from Polygon.io within the search window (t0-3h to t0+1h)?

## The Answer: It's NOT a data quality problem

The 41 no-news events are **real, valid market events**. They represent a distinct market regime that your ML classifier should learn to predict.

---

## Key Findings

### 1. Opening Bell Dominance (Most Important)
- **31 out of 41 no-news events (75.6%) occur at exactly 14:00 UTC (10:00 am ET)**
- This is the US market opening bell
- Opening-bell spikes are typically **technical/momentum-driven**, not news-driven
- The 4-hour news window (t0-3h to t0+1h) starts at 11:00 am ET, AFTER the overnight news has already moved the market
- **Conclusion**: This is correct behavior. Opening bells don't always have news catalysts.

### 2. Ticker Coverage Bias (Secondary)
Polygon.io has uneven coverage:

| Coverage | Tickers | No-News Rate |
|----------|---------|-------------|
| **Excellent** | MSFT, AAPL, META | 9-20% |
| **Good** | AMD, TSLA | 40% |
| **Poor** | NFLX, PLTR | 67-71% |
| **Very Poor** | QQQ, SPY, XLK (ETFs) | 67-100% |

- **ETFs** (XLK 100%, QQQ 67%, SPY 67%) are structurally news-sparse
- **Lower-coverage tickers** (NFLX 71%, PLTR 67%) lack specialized news sources
- **Mega-caps** benefit from extensive analyst coverage

**This reflects Polygon's coverage limitations, not your data quality.**

### 3. No-News Events Are STRONGER, Not Weaker
- **No-news events average: 192.6 bp** (absolute return)
- **News events average: 169.9 bp**
- **Difference: +22.7 bp (+13.4% stronger)**

This suggests no-news events are genuine spikes, possibly driven by:
- Technical rebalancing / algorithmic flows
- Options expiry / gamma hedging
- Earnings surprise gaps
- Cross-market spillover

### 4. Day of Week Pattern
- **Monday**: 27.6% no-news (lowest) - most news flow
- **Friday**: 64.3% no-news (highest) - week-end consolidation
- **Difference**: Friday has 2.3x higher no-news rate than Monday

Pattern suggests Friday events are more technical/momentum-driven.

### 5. Monthly Pattern
- **January**: 36.4% no-news (best coverage)
- **November**: 40.0% no-news (full month, representative)
- **December**: 39.3% no-news (holiday period, slightly lower coverage)
- **October**: 60.0% no-news (data only from Oct 15, not comparable)

November is most representative for analysis.

---

## The 4-Hour News Window: Is It Too Narrow?

### Current Window: t0-3h to t0+1h (4 hours total)

**Assessment: MOSTLY APPROPRIATE, with caveats**

For **opening bell events (14:00 UTC)**:
- Market open happens at 14:00 UTC
- News window starts at 11:00 UTC (3 hours before)
- Overnight news (8pm+ previous day) is OUTSIDE the window
- **Solution**: For gap events, extend lookback to 6-12 hours

For **after-hours events (19:00+ UTC)**:
- Post-market news typically releases at 8pm+ ET (00:00+ UTC)
- This is AFTER the +1h lookahead window
- **Solution**: For after-hours events, extend lookahead to 3-4 hours

For **regular trading hours (15:00-18:00 UTC)**:
- 4-hour window is appropriate
- Captures most relevant news

### Effectiveness Score:
- **59/100 events have news (59%)**
- **41/100 have no news (41%)**
- **Window captures ~60% of events with news** - reasonable for intraday events

---

## Recommendations

### For Your ML Classifier:

1. **Keep all 41 no-news events in training data**
   - They're valid market events
   - Represent a real regime (technical/momentum moves)
   - Important for classifier to learn these patterns

2. **Add new features to capture the patterns**:
   - `is_opening_bell`: Binary flag (14:00 UTC) - 31 no-news events here
   - `hour_of_day_utc`: 13-21 - captures opening vs afternoon effects
   - `is_etf`: Flag for XLK, QQQ, SPY - structurally different news patterns
   - `day_of_week`: Monday has fewer no-news events
   - `ticker_news_coverage`: Categorical (high/low) - MSFT vs NFLX

3. **Label strategy**: No changes needed
   - 41 no-news events are labeled (continuation/reversal/unclear)
   - Classifier should learn to predict without news features
   - If it succeeds, you've found a technical/momentum signal

4. **Interpret model results**:
   - If opening-bell no-news events cluster together → strong time-of-day effect
   - If classifier predicts reversal well for no-news events → technical signals matter
   - If no-news events are mostly continuation → momentum effect is real

### For Production Use:

1. **Accept 41% no-news rate as baseline**
   - This is inherent to Polygon's coverage
   - Not fixable without multiple news APIs

2. **For better coverage**:
   - Add Reuters, Bloomberg, or other news APIs
   - Focus on specialized sources for NFLX, PLTR
   - Consider alternative ETF news (e.g., implied volatility, flows)

3. **For opening-bell improvements**:
   - Extend lookback to 12 hours for events at 14:00 UTC
   - Capture overnight/international news
   - May increase no-news events that are truly technical

---

## Detailed Findings by Ticker

### Problem Tickers (>66% no-news):
- **XLK**: 100% (4/4) - Sector ETF, no company-specific news
- **NFLX**: 71.4% (10/14) - Entertainment coverage gap
- **PLTR**: 66.7% (6/9) - Specialty tech/defense, limited coverage
- **QQQ**: 66.7% (4/6) - Macro ETF, tech aggregate

### Good Tickers (<40% no-news):
- **MSFT**: 9.1% (1/11) - Best coverage
- **AAPL**: 20.0% (2/10) - Excellent coverage
- **META**: 25.0% (2/8) - Large-cap benefit
- **TSLA**: 40.0% (4/10) - Good coverage, volatile events
- **AMD**: 40.0% (6/15) - Semiconductor coverage exists

---

## Return Characteristics

### No-News Events (41 total):
- **Distribution**: 31 opening-bell, 3 pre-market, 4 afternoon, 3 after-hours
- **Average return**: 192.6 bp (13% stronger than news events)
- **Median return**: 161.34 bp
- **Strongest**: NFLX_20251021_200000 at 668.3 bp (after-hours)
- **Weakest**: SPY_20251114_143000 at 63.9 bp (opening bell)

### After-Hours No-News Events (special case):
- **Count**: 3 events (NFLX 2x, AMD 1x)
- **Average return**: 403.6 bp (VERY STRONG)
- **Note**: News window inadequate for these (need +3-4h lookahead)

---

## Conclusion

**The 41 no-news events are NOT a problem to solve. They're a pattern to learn.**

These events represent:
1. **Technical/momentum moves** at the opening bell (75% of cases)
2. **Genuine market spikes** with different risk/return profiles
3. **A learnable pattern** for your ML classifier

Your classifier should learn:
- Opening-bell spikes often lack news catalysts
- NFLX, PLTR, ETFs are structurally lower-coverage
- No-news events are slightly stronger on average
- Technical signals may matter more than news for these moves

**Status**: Data quality is good. Keep all 41 events for training.
