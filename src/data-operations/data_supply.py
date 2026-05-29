# 1. Data Supply 

# This module aims the following:

#   - Extracting data from Yahoo Finance API
#   - Extracting data from FRED API 
#   - Upload data to S3 bucket

# Note: No manipulatior nor transformation will be done heree, the data will be stored in its raw format


# Libraries

import yfinance as yf
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
import tempfile
import re
import fredapi
import time
import urllib.error


# Constants and important definitions

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
FRED_API_KEY = os.getenv("FRED_API_KEY")

START_DATE = "2015-03-07"
END_DATE = "2026-01-15"
FREQUENCY = "1d"

M2_CONCAT_START_DATE = "2025-12-01"
M2_CONCAT_END_DATE = "2026-01-15"


# ------------------------------------------------------------------------------------------------
# Yahoo Finance Data
# ------------------------------------------------------------------------------------------------

# Function to get tickers data from YFinance

def get_data_yf(asset, start = START_DATE, end = END_DATE, frequency = FREQUENCY) -> pd.DataFrame:

    """
    Get data from Yahoo Finance with certain parameters and return a Pandas DataFrame
    """ 

    # Get data

    data = yf.download(tickers = asset, start = start, end = end, interval = frequency, auto_adjust = True)

    if data.empty:

        raise ValueError(f"No data returned for asset: {asset}")

    data = pd.DataFrame(data.reset_index())

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = ["_".join([str(c) for c in col if c]).lower() for col in data.columns]

    else:

        data.columns = [str(col).lower() for col in data.columns]

    if "date" not in data.columns:

        date_aliases = [col for col in data.columns if col in ("datetime", "date_", "index")]

        if not date_aliases:

            date_aliases = [col for col in data.columns if str(col).startswith("date")]

        if date_aliases:

            data = data.rename(columns = {date_aliases[0]: "date"})

        else:

            raise KeyError("No date column found after normalization")

    return data


# ------------------------------------------------------------------------------------------------
# FRED API Data
# ------------------------------------------------------------------------------------------------

# Initialize FRED client

fred = fredapi.Fred(api_key = FRED_API_KEY)


# Function to get tickers from FRED

def get_data_fred(serie_id, start = START_DATE, end = END_DATE, retries = 3, backoff_seconds = 2) -> pd.DataFrame:
    
    """
    Get data from FRED with certain parameters and return a Pandas DataFrame
    """

    last_error = None

    # Retry mechanism for handling API errors (FRED can be unstabble)

    for attempt in range(1, retries + 1):

        try:

            data = fred.get_series(serie_id, observation_start = start, observation_end = end)

            break

        except urllib.error.HTTPError as exc:

            last_error = exc

        except Exception as exc:

            last_error = exc

        if attempt < retries:

            time.sleep(2 * attempt)

    else:

        raise last_error

    if data.empty:

        raise ValueError(f"No data returned for serie: {serie_id}")
    
    data = data.to_frame().reset_index()

    data.columns = ["date", serie_id]

    return data


# Additional fuction for normalizing asset names 

def _normalize_asset_key(asset: str) -> str:

    """
    Normalize asset names into safe column suffixes (ascii, lowercase, underscore).
    """

    key = re.sub(r"[^0-9a-zA-Z]+", "_", asset).strip("_").lower()

    return key



# Execution

if __name__ == "__main__":

    # From Yahoo Finance:

    # - Bitcoin (BTC-USD) → Main feature of the project
    # - Volatility Index (^VIX)
    # - Gold Futures (GC=F)
    # - Crude Oil Futures (CL=F)
    # - iShares 20+ Year Treasury Bond ETF (TLT)
    # - Invesco S&P 500 Equal Weight ETF (RSP)
    # - 10-Year Treasury Note (^TNX)
    # - iShares Russell 2000 ETF (IWM)
    # - ETF Invesco DB US Dollar Index Bullish Fund (UUP)
    # - Strategy (MSTR) → a.k.a. MicroStrategy 
    # - Ethereum (ETH-USD)


    # Let's bring them in 2 batches

    assets_batch1 = ["^VIX", "GC=F", "CL=F", "TLT", "RSP"]
    assets_batch2 = ["^TNX", "IWM", "UUP", "MSTR", "ETH-USD", "BTC-USD"]

    data_yf_list = []

    for asset in assets_batch1 + assets_batch2:

        df = get_data_yf(asset)

        asset_key = _normalize_asset_key(asset)

        df = df.rename(columns = {col: f"{col}_{asset_key}" for col in df.columns if col != "date"})

        data_yf_list.append(df)

    # Join the dataframes together on date

    data_yf = data_yf_list[0]

    for df in data_yf_list[1:]:

        data_yf = data_yf.merge(df, on = "date", how = "outer")


    # From FRED:

    # - ICE BofA US High Yield Index Option-Adjusted Spread (BAMLH0A0HYM2)
    # - ICE BofA 7-10 Year US Corporate Bond Index Effective Yield (BAMLC4A0C710YEY)
    # - National Financial Conditions Index (NFCI)  -- weekly
    # - St. Louis Fed Financial Stress Index (STLFSI4)  -- weekly
    # - 5 Year Breakeven Inflation Rate(T5YIE)
    # - 10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity (T10Y2Y)
    # - 10 Year Treasury Constant Maturity Minus 3 Month Treasury Constant Maturity (T10Y3M)
    # - Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10 Year Treasuty Constant Maturity (BAA10Y)
    # - Effective Federal Funds Rate (EFFR)

    indicators = ['BAMLH0A0HYM2', 'BAMLC4A0C710YEY', 'NFCI', 'STLFSI4', 'T5YIE', 'T10Y2Y', 'T10Y3M', 'BAA10Y', 'EFFR']


    # Get ans save the features from FRED

    data_fred_list = []

    for serie in indicators:

        print(f"Downloading FRED series: {serie}")

        data_fred_list.append(get_data_fred(serie))

        time.sleep(5)

    data_fred = data_fred_list[0]

    for df in data_fred_list[1:]:

        data_fred = data_fred.merge(df, on = "date", how = "outer")


    # Get M2 Money Stock (M2SL) data from FRED 

    data_m2 = get_data_fred("M2SL", start = M2_CONCAT_START_DATE, end = M2_CONCAT_END_DATE)


    # Load all the data to S3 bucket ("raw" folder)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    # Files to be uploaded

    files_to_upload = [
        {"data": data_yf, "filename": "raw/data_yf.csv"},
        {"data": data_fred, "filename": "raw/data_fred.csv"},
        {"data": data_m2, "filename": "raw/data_m2.csv"}
    ]

    # Save the dataframes to CSV files in a temporary directory and upload to S3

    for item in files_to_upload:

        df = item["data"]

        s3_key = item["filename"]

        with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

            temp_file_path = tmp.name

            df.to_csv(temp_file_path, index = False)

            s3_client.upload_file(temp_file_path, BUCKET_NAME, s3_key)

            print(f"Uploaded: s3://{BUCKET_NAME}/{s3_key}")

            print("--" * 20)

            print("Done, data uploaded to S3 successfully")

            # Remove the temporary file

            os.remove(temp_file_path)