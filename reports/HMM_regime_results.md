# Analysis of HMM Regime Results

This report presents the analysis of the results obtained from both the PCA-based and the ALL-Features HMM regime models. The focus is on understanding the differences in the regimes identified by each model and their implications for the underlying data. For both approaches, the number of regimes was set to 3, and the models were trained on the same dataset to ensure comparability.


## PCA-Based HMM Regime Results

This approach utilizes Princpial Component Analysis to reduce the dimensionality of the data (data_consolidated in silver) before aplying the HMM. The results indicate that the PCA-based HMM barely identified the regimes, a proof of this is that the average duration of each regime varies between less than 1 day to 2 days, which is not a significant duration for a regime. This suggests that the PCA-based HMM may not be capturing the underlying structure of the data effectively, possibly due to the loss of information during dimensionality reduction. Therefore, this column won't be used for the classificator training, as it doesn't provide meaningful insights into the regimes present in the data.


## ALL-Features HMM Regime Results

This approach applies the HMM directly to the full set of features without dimensionality reduction, but off course, the corresponding processing steps were applied to the data. The 3 states identified by the HMM can be interpreted as follows:

- **State 0 as Volatility:Expansion_Regime**: This state or regime is characterized by a high volatility and a market in expansion. It is associated with more turbulent conditions and also with higher dispersion in the data with an average duration of 4 days. This regime may indicate periods of market growth accompanied by increased uncertainty. Most frequent in days of distribution, with very high volume and very positive/negative return. This is the less frequent regime and is bidirectional, meaning that it can be associated with both positive and negative returns.

- **State 1 as Consolidation_Regime**: This state or regime is characterized by a low volatility and a market in consolidation. It is associated with more stable conditions and also with lower dispersion in the data with an average duration of 4-5 days. This regime may indicate periods of market stability accompanied by reduced uncertainty. Most frequent in days of distribution, neutral days and price bouncing inside the range, with moderate volume and mixed positive/negative return. This is the most frequent regime and is unidirectional, meaning that it can be associated with either scarce returns, very close to zero, or with small positive returns.

- **State 2 as Bullish_Regime**: Corresponding with the "absolute" bullish regime, because it is characterized by positve returns and upward movements. Here is where the bullish breakouts are located, where the target is "Bullish" and coincides with accumulation days (buy-side pressure) with very high volume and very positive return. This regime may indicate periods of market growth accompanied by increased optimism. It is the second most frequent regime and is unidirectional, meaning that it can be associated with positive returns. And finally, it has an average duration of 2 days, which is shorter than the other two regimes, indicating that bullish conditions may be more transient compared to the other regimes.