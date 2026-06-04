# Data Dictionary

This data dictionary describes the unified dataset used by the Ternary Market Direction Classifier. It is built by merging BTC-USD market data, derived technical indicators, target labels, macroeconomic series from FRED, external crypto/macro series, and cross-asset closes from Yahoo Finance. Rolling-window features use daily periods; missing values are expected at the start of each series and where source frequencies are weekly or irregular.

| Column | Type | Definition / Calculation | Source / Notes |
| --- | --- | --- | --- |
| date | object | Calendar date of the observation. | Merge key from all sources; stored as string in the unified file and should be parsed to datetime for modeling. |
| Close | float64 | Adjusted close price of BTC-USD. | Yahoo Finance `BTC-USD` with `auto_adjust=True`. |
| High | float64 | Adjusted daily high price of BTC-USD. | Yahoo Finance. |
| Low | float64 | Adjusted daily low price of BTC-USD. | Yahoo Finance. |
| Open | float64 | Adjusted daily open price of BTC-USD. | Yahoo Finance. |
| Volume | float64 | Daily trading volume reported for BTC-USD. | Yahoo Finance; unitt is provider-defined and not rescaled. |
| Scaled_SMA20 | float64 | $Close - SMA_{20}(Close)$. | 20-day simple moving average; positive values mean price above SMA. |
| Scaled_SMA50 | float64 | $Close - SMA_{50}(Close)$. | 50-day simple moving average. |
| Scaled_EMA20 | float64 | $Close - EMA_{20}(Close)$. | EMA computed with `period=20`, `adjust=False`. |
| Scaled_EMA50 | float64 | $Close - EMA_{50}(Close)$. | EMA computed with `period=50`, `adjust=False`. |
| Scaled_HMA20 | float64 | $Close - HMA_{20}(Close)$. | Hull MA using WMA components: $HMA = WMA(2\cdot WMA_{10} - WMA_{20}, \sqrt{20})$. |
| Scaled_HMA50 | float64 | $Close - HMA_{50}(Close)$. | Hull MA using WMA components with $\sqrt{50}$. |
| Momentum_20p | float64 | $Close - Close_{t-20}$. | 20-day momentum in price units. |
| Momentum_50p | float64 | $Close - Close_{t-50}$. | 50-day momentum in price units. |
| RSI | float64 | $100 - 100 / (1 + RS)$ with $RS = AvgGain_{14} / AvgLoss_{14}$. | Uses simple rolling means of gains/losses from close-to-close changes. |
| Stoch_K | float64 | $100\cdot (Close - LL_{14}) / (HH_{14} - LL_{14})$. | %K with 14-day highs/lows; `smooth_k=1` (no extra smoothing). |
| Stoch_D | float64 | 3-day SMA of `Stoch_K`. | `smooth_d=3`. |
| WilliamsR | float64 | $-100\cdot (HH_{21} - Close) / (HH_{21} - LL_{21})$. | 21-day Williams %R. |
| Norm_ATR | float64 | $ATR_{14} / Close$. | ATR uses true range: max of high-low, abs(high-prev close), abs(low-prev close). |
| Scaled_Upper_Bollinger | float64 | $Close - (SMA_{21} + 2\cdot STD_{21})$. | Upper Bollinger band scaled by close. |
| Scaled_Lower_Bollinger | float64 | $Close - (SMA_{21} - 2\cdot STD_{21})$. | Lower Bollinger band scaled by close. |
| Scaled_Upper_Keltner | float64 | $Close - (EMA_{21} + 2\cdot ATR_{21})$. | Keltner upper channel scaled by close. |
| Scaled_Lower_Keltner | float64 | $Close - (EMA_{21} - 2\cdot ATR_{21})$. | Keltner lower channel scaled by close. |
| OBV | float64 | Cumulative sum of $sign(\Delta Close) \cdot Volume$. | On-Balance Volume. |
| Anchored_VWAP | float64 | $\sum TP\cdot Vol / \sum Vol$ since anchor. | Typical price $TP=(High+Low+Close)/3$; anchor at first row (`anchor_index=0`). |
| ILV | float64 | $\ln(High/Low)$. | Intraday logarithmic volatility. |
| Relative_Volume_Category | object | Discrete category from volume ratio quantiles. | Compute $Volume / MA_{30}(Volume)$; labels: Very Low (<=Q25), Low (Q25-Q50), High (Q50-Q75), Very High (>Q75), else No Data. |
| Day_Type | object | Accumulation/Distribution/Neutral day type. | Accumulation if Close > Open and Volume > MA30; Distribution if Close < Open and Volume > MA30; else Neutral. |
| Weekly_Breakout | object | Weekly range breakout classification. | Compare Close to prior week (5-day) high/low: Bullish Breakout, Bearish Breakout, else Inside Range. |
| Unnamed: 0_x | float64 | Legacy index column from a prior CSV merge. | Artifact; safe to drop. |
| Future_Return | float64 | $(Close_{t+1} - Close_t) / Close_t$. | 1-day forward return; rounded to 4 decimals; last row is dropped. |
| Target | object | Ternary label of future return. | Bullish if `Future_Return` > 0.01, Bearish if < -0.01, else Neutral. |
| Unnamed: 0_y | float64 | Legacy index column from a prior CSV merge. | Artifact; safe to drop. |
| T5YIE | float64 | 5-Year Breakeven Inflation Rate. | FRED series `T5YIE`. |
| T10Y2Y | float64 | 10Y Treasury constant maturity minus 2Y Treasury constant maturity. | FRED series `T10Y2Y`. |
| T10Y3M | float64 | 10Y Treasury constant maturity minus 3M Treasury constant maturity. | FRED series `T10Y3M`. |
| BAA10Y | float64 | Moody's Seasoned Baa yield relative to 10Y Treasury. | FRED series `BAA10Y`. |
| EFFR | float64 | Effective Federal Funds Rate. | FRED series `EFFR`. |
| High_Yield_Spread | float64 | ICE BofA US High Yield Index option-adjusted spread. | Derived from FRED/GitHub series `BAMLH0A0HYM2`; daily values with some weekly gaps. |
| Corporate_Bond_Spread | float64 | ICE BofA 7-10Y US Corporate Bond Index effective yield. | Derived from FRED/GitHub series `BAMLC4A0C710YEY`. |
| National_Conditions_Index | float64 | National Financial Conditions Index. | FRED series `NFCI` (weekly); fewer observations. |
| Financial_Stress_Index | float64 | St. Louis Fed Financial Stress Index. | FRED series `STLFSI4` (weekly); fewer observations. |
| CRYPTOCAP_BTC.D, 1D | float64 | Close value for TradingView symbol `CRYPTOCAP_BTC.D`. | BTC dominance index; daily close from external CSV. |
| CRYPTOCAP_TOTAL, 1D | float64 | Close value for TradingView symbol `CRYPTOCAP_TOTAL`. | Total crypto market cap; daily close from external CSV. |
| CRYPTOCAP_TOTAL2, 1D | float64 | Close value for TradingView symbol `CRYPTOCAP_TOTAL2`. | Total market cap excluding BTC; daily close from external CSV. |
| CRYPTOCAP_TOTALES, 1D | int64 | Close value for TradingView symbol `CRYPTOCAP_TOTALES`. | Total market cap excluding BTC and ETH; daily close from external CSV. |
| CRYPTOCAP_USDT.D, 1D | float64 | Close value for TradingView symbol `CRYPTOCAP_USDT.D`. | USDT dominance index; daily close from external CSV. |
| FRED_WM2NS, 1D | float64 | M2 Money Stock, not seasonally adjusted. | TradingView/FRED symbol `WM2NS`; daily close from external CSV. |
| INDEX_BDI, 1D | float64 | Baltic Dry Index level. | TradingView symbol `INDEX_BDI`; daily close from external CSV. |
| Unnamed: 0 | float64 | Legacy index column from a prior CSV merge. | Artifact; safe to drop. |
| VIX_close | float64 | Close price of ^VIX. | Yahoo Finance; CBOE Volatility Index. |
| GOLD_close | float64 | Close price of GC=F. | Yahoo Finance; Gold futures. |
| OIL_close | float64 | Close price of CL=F. | Yahoo Finance; Crude oil futures. |
| TLT_close | float64 | Close price of TLT. | Yahoo Finance; iShares 20+ Year Treasury Bond ETF. |
| RSP_close | float64 | Close price of RSP. | Yahoo Finance; Invesco S&P 500 Equal Weight ETF. |
| TNX_close | float64 | Close price of ^TNX. | Yahoo Finance; 10-year Treasury yield index. |
| IWM_close | float64 | Close price of IWM. | Yahoo Finance; iShares Russell 2000 ETF. |
| UUP_close | float64 | Close price of UUP. | Yahoo Finance; Invesco DB US Dollar Index Bullish Fund. |
| MSTR_close | float64 | Close price of MSTR. | Yahoo Finance; Strategy (MicroStrategy) equity. |
| SPY_close | float64 | Close price of SPY. | Yahoo Finance; S&P 500 ETF. |

