# 3. Derived Indicators 

# In this file, the idea is to do the following:

#  - Load the raw target data from S3 bucket
#  - Build a set of derived inidcators based on the BTC-USD data
#  - Upload the derived indicators to S3 bucket (silver folder)


# Libraries 

import pandas as pd
import numpy as np
import boto3
import os
from dotenv import load_dotenv
import tempfile


# Constants and important definitions

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_KEY_RAW = "raw/btc_target_data.csv"


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get BTC-USD data from S3 

def get_target_data_s3(bucket_name = S3_BUCKET_NAME, 
                       access_key_id = S3_ACCESS_KEY_ID, 
                       secret_access_key = S3_SECRET_ACCESS_KEY, 
                       s3_key = S3_KEY_RAW):
    
    # Initialize the client
    
    s3_client = boto3.client(
        "s3",
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key
    )

    # Download the file and convert it to a pandas Dataframe

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    try:
        s3_client.download_file(bucket_name, s3_key, temp_file_path)

        df = pd.read_csv(temp_file_path)

        # Process the dataframe:

        # Remove the first 2 rows
        # Reset tge index
        # Rename the first column to "Date"
        # Set the "Date" column as index

        df = df.iloc[2:].reset_index(drop = True)
        df = df.rename(columns = {df.columns[0]: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df = df.apply(pd.to_numeric, errors = 'coerce')

    finally:

        # Remove the temporary file

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)

    return df


# ------------------------------------------------------------------------------------------------
# Build Derived Indicators
# ------------------------------------------------------------------------------------------------

# 1. Scaled Simple Moving Average (Scaled SMA) 

# Scaled Simple Moving Average (Close - SMA)

def scaled_SMA(df, period):

    sma = df['Close'].rolling(period).mean()
    scaled_sma = df['Close'] - sma

    return scaled_sma

# 2. Scaled Exponential Moving Average (Scaled EMA) 

# Scaled Exponential Moving Average (Close - EMA)

def scaled_EMA(df, period):

    ema = df['Close'].ewm(span = period, adjust = False).mean()
    scaled_ema = df['Close'] - ema

    return scaled_ema

# 3. Scaled Hull Moving Average (Scaled HMA) 

# Scaled Hull Moving Average (Close - HMA)

# First must be calculated the WMA, but inside the HMA function

def scaled_HMA(df, period):

    if period < 2:

        return pd.Series(index = df.index, dtype = float)

    # WMA

    def WMA_component(series, length):

        if length < 1:

            return pd.Series(index = series.index, dtype = float)

        weights = np.arange(1, length + 1)
        result = series.rolling(window = length)
        result = result.apply(lambda x: np.dot(x, weights) / weights.sum(), raw = True)

        return result
    
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))

    wma1 = WMA_component(df['Close'], half)
    wma2 = WMA_component(df['Close'], period)

    hma = WMA_component(2 * wma1 - wma2, sqrt_period)
    scaled_hma = df['Close'] - hma

    return scaled_hma

# 4. Momentum

# Momentum Indicator

def momentum(df, period):

    momtm = df['Close'] - df['Close'].shift(period)

    return momtm

# 5. Relative Strength Index (RSI)

# RSI (Relative Strength Index)

def rsi(df, period):

    delta = df['Close'].diff()
    
    gain = delta.clip(lower = 0)
    loss = - delta.clip(upper = 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value

# 6. Stochastic Oscillator (%K and %D)

# Stochastic Oscillator (%K and %D)

def stochastic(df, period, smooth_k = 1, smooth_d = 3):

    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()

    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['Close'] - low_min) / denom

    k_smooth = k.rolling(smooth_k).mean()
    d_smooth = k_smooth.rolling(smooth_d).mean()

    return k_smooth, d_smooth

# 7. Williams %R

# Williams %R

def williams_r(df, period):

    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()

    denom = (high_max - low_min).replace(0, np.nan)

    wr = - 100 * (high_max - df['Close']) / denom

    return wr

# 8. Normalized Average True Range (NATR)

# Normalized ATR (ATR / Close)

def normalized_atr(df, period):

    if period < 1:

        raise ValueError("period must be >= 1")

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
    atr = tr.rolling(period).mean()

    norm_atr = atr / df['Close']

    return norm_atr

# 9. Scaled Bollinger Bands

# Scaled Bollinger Bands (with 2 standard deviations)

def scaled_bb(df, period, num_std = 2):

    sma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    scaled_upper = df['Close'] - upper
    scaled_lower = df['Close'] - lower
    
    return scaled_upper, scaled_lower

# 10. Scaled Keltner Channels

# Scaled Keltner Channels

def scaled_keltner(df, period, atr_mult = 2):

    if period < 1:

        raise ValueError("period must be >= 1")

    ema = df['Close'].ewm(span = period, adjust = False).mean()

    # ATR

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
    atr = tr.rolling(period).mean()

    upper = ema + (atr_mult * atr)
    lower = ema - (atr_mult * atr)

    scaled_upper = df['Close'] - upper
    scaled_lower = df['Close'] - lower

    return scaled_upper, scaled_lower

# 11. On-Balance Volume (OBV)

# On-Balance Volume

def obv(df):

    direction = np.sign(df['Close'].diff()).fillna(0)

    dir_vol = (direction * df['Volume']).cumsum()

    return dir_vol

# 12. Anchored Volume Weighted Average Price (Anchored VWAP)

# Anchored VWAP 

def anchored_vwap(df, anchor_index = 0):

    if anchor_index < 0 or anchor_index >= len(df):

        raise IndexError("anchor_index out of bounds")

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).cumsum() - (typical_price * df['Volume']).cumsum().iloc[anchor_index]
    cum_vol = df['Volume'].cumsum() - df['Volume'].cumsum().iloc[anchor_index]

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    
    return vwap

# 13. Intraday Logarithmic Volatility

# Intraday Logarithmic Volatility

def ilv(df):

    dlog = np.log(df['High'] / df['Low'])

    return dlog

# 14. Relative Volume with quantile (with MA  of 30 periods)

def relative_volume(df, period = 30):

    # Relative proportion of the actual volume to the average volume of the last 30 periods

    volume_ratio = df['Volume'] / df['Volume'].rolling(window = period).mean()

    # Quantile calculation for the relative volume

    q25 = volume_ratio.quantile(0.25)
    q50 = volume_ratio.quantile(0.50)   # Q50 = Median
    q75 = volume_ratio.quantile(0.75)

    # Definition of the condiotions

    vol_conditions = [
        (volume_ratio <= q25),
        (volume_ratio > q25) & (volume_ratio <= q50),
        (volume_ratio > q50) & (volume_ratio <= q75),
        (volume_ratio > q75)
    ]

    # Definition of the labels

    vol_labels = ["Very Low", "Low", "High", "Very High"]

    # Column creation with the conditions and labels

    volume_category = np.select(vol_conditions, vol_labels, default = "No Data")

    # Return the dataframe with the new column

    return pd.Series(volume_category, index = df.index)

# 15. Type of Day (Accumulation/Distribution/Neutral)

def day_type(df, period = 30):

    # Volume MA 30

    volume_ma = df['Volume'].rolling(window = period).mean()

    # Conditions for the type of day

    day_conditions = [
        (df['Close'] > df['Open']) & (df['Volume'] > volume_ma),
        (df['Close'] < df['Open']) & (df['Volume'] > volume_ma)
    ]

    # Labels for the type of day

    day_labels = ["Accumulation", "Distribution"]

    # Column creation with the conditions and labels

    day_type_series = np.select(day_conditions, day_labels, default = "Neutral")

    # Return the dataframe with the new column

    return pd.Series(day_type_series, index = df.index)

# 16. Weekly Breakout

def weekly_breakout(df, period_week = 5):

    # Max and Min of the last 5 days (1 week)

    df['Prev_Week_High'] = df['High'].shift(1).rolling(window = period_week).max()
    df['Prev_Week_Low'] = df['Low'].shift(1).rolling(window = period_week).min()

    # Conditions for the brakout

    breakout_conditions = [
        (df['Close'] > df['Prev_Week_High']),
        (df['Close'] < df['Prev_Week_Low'])
    ]

    # Labels fot the breakout

    breakout_labels = ["Bullish Breakout", "Bearish Breakout"]

    # Column creation with the conditions and labels

    df['Weekly_Breakout'] = np.select(breakout_conditions, breakout_labels, default = "Inside Range")

    # Return the dataframe with the new column

    return df['Weekly_Breakout']


# ------------------------------------------------------------------------------------------------
# Upload data to S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to upload the dataframe with the indicators to S3 bucket

def upload_indicators_s3(df, 
                         bucket_name = S3_BUCKET_NAME, 
                         access_key_id = S3_ACCESS_KEY_ID, 
                         secret_access_key = S3_SECRET_ACCESS_KEY, 
                         s3_key = "silver/btc_indicators_data.csv"):

    # Initialize the client

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key
    )

    # Save the dataframe to a temporary CSV file

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

        df.to_csv(temp_file_path)

        # Upload the file to S3 bucket

        s3_client.upload_file(temp_file_path, bucket_name, s3_key)



# Execution

if __name__ == "__main__":

    # Get the target data from S3 bucket

    df = get_target_data_s3()

    print("--" * 20)
    print("Target data retrieved from S3 bucket")
    print("--" * 20)

    # Apply the functions to build the derived indicators and add them to the dataframe

    # Add the indicators to dataframe

    def add_indicators(df, 
                    period_sma = 50,
                    period_sma2 = 200,
                    period_ema = 50,
                    period_ema2 = 200,
                    period_hma = 50,
                    period_hma2 = 200,
                    period_momentum = 20,
                    period_momentum2 = 100,
                    period_rsi = 14,
                    period_stochastic = 14,
                    period_williamsR = 21,
                    period_atr = 14,
                    period_bb = 21,
                    period_keltner = 21,
                    ):
        

        # SSMA50 and SSMA200

        df['Scaled_SMA50'] = scaled_SMA(df, period = period_sma)
        df['Scaled_SMA200'] = scaled_SMA(df, period = period_sma2)

        # SEMA50 and SEMA200

        df['Scaled_EMA50'] = scaled_EMA(df, period = period_ema)
        df['Scaled_EMA200'] = scaled_EMA(df, period = period_ema2)

        # SHMA50 and SHMA200

        df['Scaled_HMA50'] = scaled_HMA(df, period = period_hma)
        df['Scaled_HMA200'] = scaled_HMA(df, period = period_hma2)

        # Momentum

        df['Momentum_20p'] = momentum(df, period = period_momentum)
        df['Momentum_100p'] = momentum(df, period = period_momentum2)

        # RSI

        df['RSI'] = rsi(df, period = period_rsi)

        # Stochastic (%K and %D)

        k, d = stochastic(df, period = period_stochastic)
        df['Stoch_K'] = k
        df['Stoch_D'] = d

        # Williams %R

        df['WilliamsR'] = williams_r(df, period = period_williamsR)

        # NATR

        df['Norm_ATR'] = normalized_atr(df, period = period_atr)

        # Scaled Bollinger BAnds

        s_upper, s_lower = scaled_bb(df, period_bb)
        df['Scaled_Upper_Bollinger'] = s_upper
        df['Scaled_Lower_Bollinger'] = s_lower

        # Scaled Keltner Channels

        s_upper, s_lower = scaled_keltner(df, period_keltner)
        df['Scaled_Upper_Keltner'] = s_upper
        df['Scaled_Lower_Keltner'] = s_lower

        # OBV

        df['OBV'] = obv(df)

        # Anchored VWAP

        df['Anchored_VWAP'] = anchored_vwap(df)

        # Intraddy Logarithmic Volatility

        df['ILV'] = ilv(df)

        # Relative Volume with quantiles

        df['Relative_Volume_Category'] = relative_volume(df)

        # Type of Day (Accumulation/Distribution/Neutral)

        df['Day_Type'] = day_type(df)

        # Weekly Breakout

        df['Weekly_Breakout'] = weekly_breakout(df)

        # Drop intermediate columns 

        df.drop(columns = ["Volume_Ratio", "Volume_MA30", "Prev_Week_High", "Prev_Week_Low"], inplace = True, errors = "ignore")

        # Return the dataframe with the new indicators

        return df


    # Add indicators --> Invoke the fuction

    btc_df = add_indicators(df)

    print("--" * 20)
    print(btc_df.shape)
    print("--" * 20)

    # Upload the dataframe with the indicators to S3 bucket

    upload_indicators_s3(btc_df)

    print(f"Data with indicators uploaded to S3 bucket '{S3_BUCKET_NAME}' in silver/btc_indicators_data.csv")
    print("Pipeline completed")