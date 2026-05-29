# 2. Data Target Supply → BTC-USD

# This module aims the following:

#   - Extracting data from Yahoo Finance API
#   - Upload data to S3 bucket (raw folder)


# Libraries

import yfinance as yf
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
import tempfile


# Constants and important definitions

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

START_DATE = "2015-03-07"
END_DATE = "2026-01-15"
FREQUENCY = "1d"
TARGET_ASSET = "BTC-USD"


# ------------------------------------------------------------------------------------------------
# Yahoo Finance Data
# ------------------------------------------------------------------------------------------------

# Function to get BTC-USD data from YFinance

def get_target_data_yf(asset = TARGET_ASSET, start = START_DATE, end = END_DATE, frequency = FREQUENCY) -> pd.DataFrame:

    """
    Get data of the target asset from Yahoo Finance with certain parameters and return a Pandas DataFrame
    """ 

    # Get data

    btc_data = yf.download(tickers = asset, start = start, end = end, interval = frequency, auto_adjust = True)

    if btc_data.empty:

        raise ValueError(f"No data returned for asset: {asset}")
    
    if isinstance(btc_data.columns, pd.MultiIndex):

        btc_data.columns = btc_data.columns.get_level_values(0)
    
    data = pd.DataFrame(btc_data.reset_index())

    return data



# Execution

if __name__ == "__main__":

    # Get data 

    btc_data = get_target_data_yf()

    # Load the S3 client

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    # Save the data to a temporary CSV file

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp_file:

        temp_file_path = tmp_file.name

        btc_data.to_csv(temp_file_path, index = False)

        s3_client.upload_file(temp_file_path, BUCKET_NAME, "raw/btc_target_data.csv")

        print(f"Data uploaded to S3 bucket '{BUCKET_NAME}' in 'raw/btc_target_data.csv'")

    # Remove the temporary file

    os.remove(temp_file_path)
