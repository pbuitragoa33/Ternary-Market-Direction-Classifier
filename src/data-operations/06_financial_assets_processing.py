# 6. Data of financial assets processing

# In this file, the idea is to do the following:

#  - Load 1 file from the S3 bucket (in raw folder)
#  - Transform and process the assets data (features from Yahoo Finance)
#  - Upload the transformed dataframe back to the S3 bucket (in silver folder)


# Libraries 

import pandas as pd
import numpy as np
import boto3
import os
from dotenv import load_dotenv
import tempfile
import re


# Important definitions

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_KEY_RAW = "raw/data_yf.csv"


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get financial assets data from S3 bucket

def get_fin_assets_to_transform(bucket_name = S3_BUCKET_NAME,
                              access_key_id = S3_ACCESS_KEY_ID,
                              secret_access_key = S3_SECRET_ACCESS_KEY,
                              key = S3_KEY_RAW):
    
    # Create the S3 client

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key
    )

    # Download the file and convert it to a pandas Dataframe

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    try:

        s3_client.download_file(bucket_name, key, temp_file_path)

        df = pd.read_csv(temp_file_path)

        print(f"Downloaded data from S3 bucket: s3://{bucket_name}/{key}")
        print("--" * 20)
        print(f"Data shape: {df.shape}")

    except Exception as e:

        print(f"Error downloading file from S3: {e}")

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)

    return df


# ------------------------------------------------------------------------------------------------
# Data transformation and processing
# ------------------------------------------------------------------------------------------------

# Function to transform and drop the unnecessaty columns from the financial assets data

def transform_fin_assets_data(df):

    # Only keep "date" and the "close" column for each asset

    asset_label_map = {
        "^vix": "VIX",
        "gc=f": "GOLD",
        "cl=f": "OIL",
        "tlt": "TLT",
        "rsp": "RSP",
        "^tnx": "TNX",
        "iwm": "IWM",
        "uup": "UUP",
        "mstr": "MSTR",
        "eth-usd": "ETH",
    }

    close_columns = [
        col for col in df.columns if isinstance(col, str) and col.startswith("('close',")
    ]

    rename_map = {}

    for col in close_columns:

        match = re.search(r"\('close',\s*'([^']+)'\)", col)

        if not match:

            continue

        asset_key = match.group(1)
        label = asset_label_map.get(asset_key, asset_key.upper())
        rename_map[col] = f"{label}_close"

    keep_columns = ["date"] + list(rename_map.keys())

    df = df[keep_columns].rename(columns = rename_map)

    return df


# ------------------------------------------------------------------------------------------------
# Upload data to S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to upload the dataframe with the target variable to S3 bucket

def upload_fin_assets_to_s3(df, 
                      bucket_name = S3_BUCKET_NAME, 
                      access_key_id = S3_ACCESS_KEY_ID, 
                      secret_access_key = S3_SECRET_ACCESS_KEY, 
                      s3_key = "silver/financial_assets.csv"):

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

    # Get data from S3 bucket

    print("Downloading data from S3 bucket")

    df = get_fin_assets_to_transform()

    # Transform the data

    df_fin_transformed = transform_fin_assets_data(df)

    # Upload to S3 bucket (silver folder)

    print("Uploading transformed data to S3 bucket")

    upload_fin_assets_to_s3(df_fin_transformed)

    print("Process completed successfully")