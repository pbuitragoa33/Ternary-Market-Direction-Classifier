# Insights from EDA

This report summarizes the insights gained from the EDA Analysis of the unified dataset. The EDA process involved exploring the data, identifying patterns, understanding the relationships between different features and how they behaved over time.


## Completeness and Missing Values

Among all the features, we can classify them into 3 categories based on the reason of incompleteness:

1. **Lagged Features**: These features are derived from the original ones, so thet are incomplete because they are calculated based on a previous number of values, like Moving Averages, RSI, etc. In conclusion, the numer of missing values depends on the number of lags used to calcuate them, that's why the project focused on features with a maximum of 50 lags.

2. **Features with different frequency**: Some features, like the economic indicators or the data extracted via FRED API, are not available on a daily basis, so they have missing values because usually they get relesed on a weekly basis. Due to the join method used to merge the different datasets, these features have missing values on the days they are not released.

3. **Crypto related features**: Normally, the crypto market is open 24/7, so the features related to this market have more rows comparted to other features. The idea is to process teh data in a weekly-market structure (weeks of 5 days) rather than impute the weekend missing values of the other features.


## Kurtosis and Skewness

**Reference**: 

-  Skew ≈ 0 y Kurtosis ≈ 0 → Normal Distribution
-  |Skew| > 1 → Asymmetric Distribution
-  Kurtosis > 3 → Heavy Tails 

Instead of going through all the features one by one, it's more useful to group them based on their values bearing in mind the referencde above. For example, we can group the features into 7 categories:

1. **Stress Indicators**: Kurtosis in [12, 33] range and Skewness in [2, 4.5] range. Basically, these features are flat/steady for most of the time, but they explode durting stress periods like crisis, bear markets or liquidity crunches.

2. **Volatility Bands and Defensive Assets**: Kurtosis in [2.7, 8.8] range and Skewness in [1, 2.6] range. It's a classic pattern of calm regime with episodes of expansion duting stredd periodds.

3. **Price and Moving Averages**: Kurtosis in [0.8, 6.0] range and Skewness in [0.3, 1.2] range. Typical distributions of bullish time periods, with a right skewness and a moderate kurtosis (leptokurtic distributions).

4. **Oscillators**: Kurtosis in [-1.4, -0.4] range and Skewness near 0. These features are bounded between 0 and 100, so they have a platykurtic distribution with a negative kurtosis. The skewness is near 0 because they tend to oscillate around a central value.

5. **Macro variables and regime switches**: These are platykurtic features with negarive kurtosis and low skewness. Multi-regime behavior with strong/slow trends.

6. **Approximately Normal**: These feature have a kurtosis near 3 and a skewness near 0, so they have an approximately normal distribution. 

**Note**: `Unnamed: 0_x`, `Unnamed: 0_y`, `Unnamed: 0` will be dropped in the future. 

| **Group** | **Variables** |
|-------|------------|
| Group 1 | `Financial_Stress_Index`, `ILV`, `VIX_close`, `Volume` |
| Group 2 | `Scaled_Upper_Bollinger`, `Scaled_Lower_Bollinger`, `Scaled_Upper_Keltner`, `Scaled_Lower_Keltner`, `Norm_ATR`, `MSTR_close`, `GOLD_close`, `High_Yield_Spread`, `INDEX_BDI`, `National_Conditions_Index`, `BAA10Y` |
| Group 3 | `Close`, `High`, `Low`, `Open`, `Scaled_SMA20`, `Scaled_SMA50`, `Scaled_EMA20`, `Scaled_EMA50`, `Momentum_20p`, `Momentum_50p` |
| Group 4 | `RSI`, `Stoch_K`, `Stoch_D`, `WilliamsR` |
| Group 5 | `T10Y2Y`, `T10Y3M`, `EFFR`, `OBV`, `Anchored_VWAP`, `FRED_WM2NS`, `CRYPTOCAP_*`, `TNX_close`, `RSP_close`, `IWM_close`, `UUP_close`, `SPY_close`, `T5YIE`, `TLT_close` |
| Group 6 | `OIL_close`, `Corporate_Bond_Spread`, `Future_Return` |


## Kolmogorov-Smirnov Test Results

**Reference**: 

- p-value > 0.05 → Fail to reject the null hypothesis (data follows a normal distribution)
- p-value ≤ 0.05 → Reject the null hypothesis (data does not follow a normal distribution)

1. **Quasi-Lognormal Features**: Features with a p-value near 0.05 such as `Financial_Stress_Index`, `National_Conditions_Index`, `ILV`, `VIX_close`, `High_Yield_Spread`, `OIL_close`. These features have a distribution that is close to lognormal, which is common for financial variables that cannot take negative values and have a long right tail. Could be transformed using a log transformation to make them more normally distributed.

2. **Rejected but less extreme**: Low p-values but not extremely tiny, like `RSI`, `Future_Return`, `T5YIE`, `T10Y2Y`. 

3. **Absolutely Rejected of 4 distributions**: Where most of the variables belong, with extremely low p-values.

4. **Oscillators**: These features are bounded between 0 and 100, so they do not follow a normal distribution, neither lognormal, exponential nor uniform. They have a platykurtic distribution with a negative kurtosis and a skewness near 0 because they tend to oscillate around a central value.


## About Discretization of Numerical Features (Binning)

Another analysis performed was to discretize the numerical features into 10 bins, here is the insights gained from this process in some of them:

- **Bitcoin and Strategy (MSTR - MicroStrategy)**: More than 50% of the observation of `Close` (BTC Close) and `MSTR_close` are in the first bin, which means that most of the time theu are in a low price regime. Bitcoin has been the 50.6% of the time between USD $ 86 and USD $ 12.664, and just 3.5% of the time above USD $112.000. But seeing that in a holistic way, Bitcoin has skyrocketed since its creation, so it's normal that most of the time it has been in the lower price bins.

- **The Yield Curve screamed "Recession"**: The spread between the 10-year and 2-year Treasury yield (`T10Y2Y`), that is the most watched and trusted recession indicator has been in negative territory (inverted) a bit more than 15% of the time, but the recession has not been triggered yet.

- **VIX and the High Yield Spread**: The VIX index stayed 57.41% of the time in complacent zones (between 9 and 24). Just 0.58% of the time it has been above 45 (real panic/extreme stress). The High Yield Spread has been in the lowest bin (between 2.58 and 4.25) more than 58% of the time, which is a sign of a low stress period, but it has been above 8.38 (high credit stress) just 1.8% of the time.
These 2 features express that the cost of the risk has been steady-low during most of the time. This divergence (inverted yield curve + calm market) is one of the biggest puzzles of the current market encironment.

- **Anchored VWAP**: This feature has a kind of trimodal distribution, with 3 categories that are clearly demarcated: 25% between USD $ 194 and USD $ 5.290, 26% between USD $ 5.290 and USD $ 10.336 and 25% between USD $ 25.473 and USD $ 30.519. Between those categories, there are valleys with very low activity. The VWAP has long periods of acummulation/distribution and then it has explosive moves that take it to another level.

- **RSI and WilliamsR**: These confirm the bullish skew of the BTC market. The RSI spent about 53% of the time between 42 and 71, and just 0.68% below 14 (extreme oversold conditions). The WilliamsR spent 17% between -10 and 0 (overbought conditions) and 6.22% between -100 and -90 (oversold conditions). These features confirm that the BTC market has been in a bullish regime during most of the time, with long periods of acummulation and distribution, and then explosive movess to the upside. The oversold conditions have been very rare, which is a sign of a strong bullish trend.

In a nutshell, the dataset captures a period of 'anomalous expansion': macro recessionary signals were active (with the yield curve inverted roughly 15% of the time), yet failed to transmit to risk assets (VIX and credit spreads remained subdued approximately 58% of the time), while BTC and MSTR surged to unprecedented heights only during the final stretches of the dataset. This essentially reflects the post-COVID era of abundant liquidity, where monetary policy distorted traditional macroeconomic relationships.


## About Categorical Features

There are 4 categorical features in the dataset, however, there are some important notes that have to be addressed:

- `Day_Type`: This confirms the bullish skew. Despite the fact thaat the "Neutral" days are the most common ones (54.7%), the "Accumulation" days are more frequent (24.7%) than the "Distribution" days (20.6%). The market spends more tome building position rather than liquidating the, which is a sign of a bullish regime.

**Note**: 
* Accumulation: Close > Open & Volume > Last 30 day average Volume
* Distribution: Close < Open & Volume > Last 30 day average Volume
* Neutral: The rest of the cases

- `Weekly_Breakout`: The market consolidates most of the time. The results are astonishing anf forceful → Inside Range (75.7%), Bullish Breakout (15.0%), Bearish Breakout (9.3%). The explosive movements has a bullish bias.

**Note**: 
* Close > High of the previous week → Bullish Breakout
* Close < Low of the previous week → Bearish Breakout
* The rest of the cases → Inside Range

- `Target`: The target variable reflects a bit of the bullish skew, with 33% of the time in the "Bullish" category, 27.4% in the "Bearish" category and 39.6% in the "Neutral" category. The market has been more bullish than bearish, but the most common state is the neutral one.

**Note**: Future_Return is the return of the next day, so it is a forward-looking variable. The target variable is created based on the Future_Return, with the following rules:

* Future_Return > 1.0% → Bullish
* Future_Return < -1.0% → Bearish
* The rest of the cases → Neutral (between -1.0% and 1.0%)


## Target Variable vs Categorical Features

1. **Relative_Volume_Category & Target**: The pattern is recognizable → As relative volume increases, Neutral falls and Bullish rises. In the "Very High" category, the Bullish class suppasses the Neutral class, which is a sign that the volume confirms the movement, specifically, the bullish ones. In contrast, the volume won't confirm the price falls. 

2. **Day_Type & Target**: It's not very telling.

3. **Weekly_Breakout & Target**: Here is a key-counterintuitive insight → the "Bullish Breakouts" are followed of bullish outcomes and with more frequency tan the "Bearish Breakouts". Basically, the "Bearish Breakouts" are not really berakots, they are more like False Breakouts (fakeouts) that are followed by a quick recovery and a bullish outcome. In a bullish regime, the bearish breakouts does not follow through, they get bought back and the market goes up.

**Note**: The Bearish class in the target is the less common across the dataset, so it's going to be hard for the model to predict it. In addition, these bullish environment and artifial-maken bull markets in the center of that, will be a challenge to find those bearish periods hidden between that bull sentiment.


## Correlation Analysis

Both the Pearson correlation and the Spearman correlation show a strong positive correlation between many features → Multicollinearity is going to be a problem for the model, so it's going to be necessary to perform feature selection or dimensionality reduction techniques to mitigate that issue or scaled some feature to make them more comparable. Also, changing the close price to returns could be a good idea to mitigate the problem. 


## MI, RF, SHAP and Autocorrelation (Ljung-Box) Analysis

* **MI**: Indicates data leakage with the variables `Unnamed: 0_x`, `Unnamed: 0_y`, `Unnamed: 0`, which are artifacts from the merging process. These should be dropped to prevent data leakage. Not much else to say about the MI results, as they are not very informative due to the presence of these artifacts.

* **RF and SHAP**: The forest doesn't prioritize the artifacts that produce sata leakake. Therefore, other columns like `Norm_ATR`, ``Scaled_Upper_Bollinger`, `Scaled_Lower_Bollinger`, `Scaled_Upper_Keltner`, `Scaled_Lower_Keltner`, `RSI`, `Stoch_K`, `Stoch_D`, `WilliamsR`, `OBV` and `ILV` were weighted as the most impportant features. The model found that in order to predict the direction of the market, it starts to predict if a movement could happen. This reinforces the idea of adding a feature based on a Hidden Markov Model (HMM) to predict the regime of the market (bullish, bearish or neutral) and use that as an input for the model.

* **Autocorrelation (Ljung-Box)**: Values near 1.0 are normal because we are talking about financial time series. The autocorrelation is high despite of the fact that the return are unpredictable (autocorrelation doesn't mean predictability). Respect to the Ljung-Box test, all the p-values are 0.0, that means that we reject the null hypothesis, in other words, there is no autocorrelation until lag 20. This is a expected result.