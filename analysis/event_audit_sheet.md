# Per-Event Audit Sheet (sample)

For each: decision time, spike path, catalyst+timestamp, what the agent saw, label window.

## AMD_20251006_105000  (AMD)
- decision time t0 (UTC): 2025-10-06 10:50:00+00:00   dir: UP
- spike: idio_resid=+2353bp, vol_mult=55.77, range_mult=111.77
- catalyst packet: 2 articles (1 AT/AFTER t0 = LEAK if >0)
    [2025-10-06T08:32] Meet the Unstoppable Semiconductor Stock Crushing Nvidia, AMD, and Broadcom Righ
    [2025-10-06T11:00] AMD and OpenAI Announce Strategic Partnership to Deploy 6 Gigawatts of AMD GPUs <-- POST-t0 LEAK
- label window: t0 -> t0+60min | fwd_ret=+4.584% | LABEL=continuation
- agent saw: micro/news/macro features above + 2 articles; predicted continuation

## PLTR_20250203_210500  (PLTR)
- decision time t0 (UTC): 2025-02-03 21:05:00+00:00   dir: UP
- spike: idio_resid=+1519bp, vol_mult=129.57, range_mult=56.93
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+8.737% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## PLTR_20240805_200000  (PLTR)
- decision time t0 (UTC): 2024-08-05 20:00:00+00:00   dir: UP
- spike: idio_resid=+1451bp, vol_mult=9.96, range_mult=4.9
- catalyst packet: 1 articles (1 AT/AFTER t0 = LEAK if >0)
    [2024-08-05T20:14] Why Palantir Technologies, Super Micro Computer, and Other Artificial Intelligen <-- POST-t0 LEAK
- label window: t0 -> t0+60min | fwd_ret=+12.721% | LABEL=continuation
- agent saw: micro/news/macro features above + 1 articles; predicted reversal

## NFLX_20260226_211500  (NFLX)
- decision time t0 (UTC): 2026-02-26 21:15:00+00:00   dir: UP
- spike: idio_resid=+1158bp, vol_mult=3.27, range_mult=19.39
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-1.626% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted reversal

## NFLX_20250121_205500  (NFLX)
- decision time t0 (UTC): 2025-01-21 20:55:00+00:00   dir: UP
- spike: idio_resid=+1045bp, vol_mult=29.05, range_mult=2.07
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+12.943% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted reversal

## COIN_20250731_195500  (COIN)
- decision time t0 (UTC): 2025-07-31 19:55:00+00:00   dir: DOWN
- spike: idio_resid=-943bp, vol_mult=51.86, range_mult=2.87
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-6.251% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## META_20250730_200000  (META)
- decision time t0 (UTC): 2025-07-30 20:00:00+00:00   dir: UP
- spike: idio_resid=+911bp, vol_mult=16.44, range_mult=6.82
- catalyst packet: 1 articles (0 AT/AFTER t0 = LEAK if >0)
    [2025-07-30T18:02] Meta Earnings Preview: Can Social Media Giant Justify Massive Bet on AI?
- label window: t0 -> t0+60min | fwd_ret=+8.979% | LABEL=continuation
- agent saw: micro/news/macro features above + 1 articles; predicted continuation

## COIN_20250512_211500  (COIN)
- decision time t0 (UTC): 2025-05-12 21:15:00+00:00   dir: UP
- spike: idio_resid=+866bp, vol_mult=34.4, range_mult=21.31
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+2.700% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## SPY_20241218_190500  (SPY)
- decision time t0 (UTC): 2024-12-18 19:05:00+00:00   dir: DOWN
- spike: idio_resid=+0bp, vol_mult=33.68, range_mult=9.02
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-0.892% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## SPY_20260323_110500  (SPY)
- decision time t0 (UTC): 2026-03-23 11:05:00+00:00   dir: UP
- spike: idio_resid=+2bp, vol_mult=37.88, range_mult=70.12
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-0.991% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## XLK_20240730_200000  (XLK)
- decision time t0 (UTC): 2024-07-30 20:00:00+00:00   dir: DOWN
- spike: idio_resid=+5bp, vol_mult=1.14, range_mult=19.59
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+1.476% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## XLK_20260323_110500  (XLK)
- decision time t0 (UTC): 2026-03-23 11:05:00+00:00   dir: UP
- spike: idio_resid=-5bp, vol_mult=0.9, range_mult=39.75
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-1.027% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## QQQ_20250207_150000  (QQQ)
- decision time t0 (UTC): 2025-02-07 15:00:00+00:00   dir: DOWN
- spike: idio_resid=-18bp, vol_mult=36.57, range_mult=11.39
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-0.467% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## SPY_20250430_133500  (SPY)
- decision time t0 (UTC): 2025-04-30 13:35:00+00:00   dir: DOWN
- spike: idio_resid=-21bp, vol_mult=28.25, range_mult=8.3
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+0.349% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## AMD_20241213_142500  (AMD)
- decision time t0 (UTC): 2024-12-13 14:25:00+00:00   dir: DOWN
- spike: idio_resid=-306bp, vol_mult=3.18, range_mult=3.61
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-2.779% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted reversal

## AMD_20250620_132500  (AMD)
- decision time t0 (UTC): 2025-06-20 13:25:00+00:00   dir: UP
- spike: idio_resid=+266bp, vol_mult=8.67, range_mult=5.94
- catalyst packet: 2 articles (1 AT/AFTER t0 = LEAK if >0)
    [2025-06-20T12:21] Why Is Berkshire Hathaway Hoarding Cash?
    [2025-06-20T13:41] Congress’s May Stock Trades: What They Know That You Don’t <-- POST-t0 LEAK
- label window: t0 -> t0+60min | fwd_ret=+0.872% | LABEL=continuation
- agent saw: micro/news/macro features above + 2 articles; predicted continuation

## MSFT_20251103_140000  (MSFT)
- decision time t0 (UTC): 2025-11-03 14:00:00+00:00   dir: DOWN
- spike: idio_resid=-139bp, vol_mult=7.55, range_mult=19.98
- catalyst packet: 4 articles (0 AT/AFTER t0 = LEAK if >0)
    [2025-11-03T11:15] The S&P 500 Is Sounding a Familiar Alarm. Here's Why You Should Buy and Hold Sto
    [2025-11-03T12:20] AI's Power Problem
    [2025-11-03T12:36] Dollar Returns, Central Banks Signal Caution, Capital Seeks Safety
- label window: t0 -> t0+60min | fwd_ret=+0.824% | LABEL=reversal
- agent saw: micro/news/macro features above + 4 articles; predicted reversal

## AMZN_20250805_133000  (AMZN)
- decision time t0 (UTC): 2025-08-05 13:30:00+00:00   dir: UP
- spike: idio_resid=+131bp, vol_mult=170.03, range_mult=9.54
- catalyst packet: 6 articles (2 AT/AFTER t0 = LEAK if >0)
    [2025-08-05T11:19] I Can't Lie, I'm Excited About Amazon Stock After Its Recent Earnings Report. He
    [2025-08-05T12:30] Neural Tech Drives the Growth of the AI Wearables Market
    [2025-08-05T12:30] If I Could Only Buy and Hold a Single Stock, This Would Be It
- label window: t0 -> t0+60min | fwd_ret=-0.315% | LABEL=reversal
- agent saw: micro/news/macro features above + 6 articles; predicted continuation

## AAPL_20241111_142500  (AAPL)
- decision time t0 (UTC): 2024-11-11 14:25:00+00:00   dir: DOWN
- spike: idio_resid=-96bp, vol_mult=4.35, range_mult=3.83
- catalyst packet: 1 articles (1 AT/AFTER t0 = LEAK if >0)
    [2024-11-11T14:45] Apple Stock Investors Are Excited About Its Prospects Following New Product Rele <-- POST-t0 LEAK
- label window: t0 -> t0+60min | fwd_ret=-0.984% | LABEL=continuation
- agent saw: micro/news/macro features above + 1 articles; predicted continuation

## AMD_20251008_133000  (AMD)
- decision time t0 (UTC): 2025-10-08 13:30:00+00:00   dir: DOWN
- spike: idio_resid=-324bp, vol_mult=117.99, range_mult=8.51
- catalyst packet: 4 articles (0 AT/AFTER t0 = LEAK if >0)
    [2025-10-08T10:39] Intel To Reportedly Unveil Details Of Upcoming Laptop Chip This Week
    [2025-10-08T11:15] Is Nvidia Stock a Buy After Its Blockbuster Deal With OpenAI?
    [2025-10-08T11:32] Meet the Low-Cost Vanguard ETF That Has 20% of Its Holdings in Nvidia, Broadcom,
- label window: t0 -> t0+60min | fwd_ret=+4.032% | LABEL=reversal
- agent saw: micro/news/macro features above + 4 articles; predicted reversal

## JPM_20240910_132500  (JPM)
- decision time t0 (UTC): 2024-09-10 13:25:00+00:00   dir: DOWN
- spike: idio_resid=-356bp, vol_mult=0.11, range_mult=2.75
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=-5.856% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## GOOGL_20251117_141000  (GOOGL)
- decision time t0 (UTC): 2025-11-17 14:10:00+00:00   dir: DOWN
- spike: idio_resid=-272bp, vol_mult=7.25, range_mult=6.74
- catalyst packet: 4 articles (1 AT/AFTER t0 = LEAK if >0)
    [2025-11-17T11:32] 2 Quantum Computing Stocks That Could Make You a Millionaire
    [2025-11-17T13:08] Alphabet Valuation Re-Rated as Berkshire Shifts Toward High-Growth Tech
    [2025-11-17T13:36] Stock Market Today: Dow, Nasdaq Futures Rise As Investors Brace For Employment D
- label window: t0 -> t0+60min | fwd_ret=+2.313% | LABEL=reversal
- agent saw: micro/news/macro features above + 4 articles; predicted reversal

## TSLA_20241122_153500  (TSLA)
- decision time t0 (UTC): 2024-11-22 15:35:00+00:00   dir: UP
- spike: idio_resid=+245bp, vol_mult=59.43, range_mult=8.8
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+1.715% | LABEL=continuation
- agent saw: micro/news/macro features above + 0 articles; predicted continuation

## COIN_20250402_132500  (COIN)
- decision time t0 (UTC): 2025-04-02 13:25:00+00:00   dir: DOWN
- spike: idio_resid=-455bp, vol_mult=0.53, range_mult=2.9
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+4.396% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted reversal

## NFLX_20260304_142500  (NFLX)
- decision time t0 (UTC): 2026-03-04 14:25:00+00:00   dir: DOWN
- spike: idio_resid=-195bp, vol_mult=0.31, range_mult=2.91
- catalyst packet: 0 articles (0 AT/AFTER t0 = LEAK if >0)
- label window: t0 -> t0+60min | fwd_ret=+2.245% | LABEL=reversal
- agent saw: micro/news/macro features above + 0 articles; predicted unclear

## MSFT_20260128_205500  (MSFT)
- decision time t0 (UTC): 2026-01-28 20:55:00+00:00   dir: UP
- spike: idio_resid=+710bp, vol_mult=94.27, range_mult=5.51
- catalyst packet: 4 articles (2 AT/AFTER t0 = LEAK if >0)
    [2026-01-28T18:03] Microsoft Earnings: High Capex Signals Confidence — Now the Numbers Must Deliver
    [2026-01-28T20:15] Only 2 "Magnificent Seven" Stocks Outperformed the S&P 500 in 2025. Are They Bot
    [2026-01-28T21:03] Lantz Financial Buys $5 Million of Invesco BulletShares 2027 Corporate Bond ETF <-- POST-t0 LEAK
- label window: t0 -> t0+60min | fwd_ret=-4.741% | LABEL=reversal
- agent saw: micro/news/macro features above + 4 articles; predicted reversal
