# 4. External Data Unification

# In this file, the idea is to do the following:

#  - Load 6 files from the S3 bucket (in raw folder)
#  - Merge them into a single dataframe (consolidation)
#  - Save the consolidated dataframe back to the S3 bucket (in silver folder)

# The files to be loaded are:

#    - CRYPTOCAP_BTC.D, 1D.csv
#    - CRYPTOCAP_TOTAL, 1D.csv
#    - CRYPTOCAP_TOTAL2, 1D.csv
#    - CRYPTOCAP_TOTALES, 1D.csv
#    - CRYPTOCAP_USDT.D, 1D.csv
#    - FRED_WM2NS, 1D.csv
#    - INDEX_BDI, 1D.csv

# Consolidate them in external_data_unified.csv


# Libraries 

import pandas as pd
import numpy as np
import boto3
import os
from dotenv import load_dotenv
import tempfile


# Important definitions

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

START_DATE = "2015-03-07"
END_DATE = "2026-01-15"

files_in_raw = [
    "raw/CRYPTOCAP_BTC.D, 1D.csv",
    "raw/CRYPTOCAP_TOTAL, 1D.csv",
    "raw/CRYPTOCAP_TOTAL2, 1D.csv",
    "raw/CRYPTOCAP_TOTALES, 1D.csv",
    "raw/CRYPTOCAP_USDT.D, 1D.csv",
    "raw/FRED_WM2NS, 1D.csv",
    "raw/INDEX_BDI, 1D.csv"
]


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get BTC-USD data from S3 

def get_target_data_s3(s3_key, 
                       bucket_name = S3_BUCKET_NAME, 
                       access_key_id = S3_ACCESS_KEY_ID, 
                       secret_access_key = S3_SECRET_ACCESS_KEY):
        
    # Initialize the client
    
    s3_client = boto3.client(
        "s3",
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key
    )

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    try:

        s3_client.download_file(bucket_name, s3_key, temp_file_path)

        df = pd.read_csv(temp_file_path)

        return df 
    
    except Exception as e:

        print(f"Error at downloading {s3_key}: {e}")

        return None
    
    finally:

        # Remove the temporary file after reading it

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)



# Execution 

if __name__ == "__main__":

    # Dictinary to store the dataframes

    dfs_found = {}

    # Loop to download each file and store it in the dictionary

    for file_path in files_in_raw:

        file_key = os.path.basename(file_path).replace(".csv", "")

        print(f"Downloading {file_path}")

        df_downloaded = get_target_data_s3(s3_key = file_path)

        if df_downloaded is not None:

            dfs_found[file_key] = df_downloaded

    # Merge the dataframes into a single one (consolidation)

    # Start with the first dataframe as the base

    consolidated_df = None

    for key, df in dfs_found.items():

        # Ensure required columns exist (only "time" and "close" are needed for consolidation)

        if "time" not in df.columns or "close" not in df.columns:

            print(f"Skipping {key}: missing 'time' or 'close' column")

            continue

        # Keep only time and close columns

        df = df[["time", "close"]].copy()

        # Ensure the time column is in datetime format

        df["time"] = pd.to_datetime(df["time"])

        # Filter the dataframe by the specified date range

        df_filtered = df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)]

        df_filtered = df_filtered.rename(columns = {"close": key})

        # If consolidated_df is None, initialize it with the first dataframe

        if consolidated_df is None:

            consolidated_df = df_filtered

        else:

            # Merge the current dataframe with the consolidated one on the "time" column

            consolidated_df = pd.merge(consolidated_df, 
                                       df_filtered[["time", key]], 
                                       on = "time", 
                                       how = "outer")
            
    # Save the consolidated dataframe back to the S3 bucket (in silver folder)

    if consolidated_df is None or consolidated_df.empty:

        print("No dataframes were consolidated. Skipping upload.")
        
        raise SystemExit(1)

    consolidated_df = consolidated_df.sort_values("time")

    # Save the consolidated dataframe to a temporary CSV file

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    consolidated_df.to_csv(temp_file_path, index = False)

    # Upload the temporary file to S3

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = S3_ACCESS_KEY_ID,
        aws_secret_access_key = S3_SECRET_ACCESS_KEY
    )

    try:

        s3_client.upload_file(temp_file_path, S3_BUCKET_NAME, "silver/external_data_unified.csv")

        print("Consolidated data uploaded successfully to S3.")

    except Exception as e:

        print(f"Error at uploading consolidated data to S3: {e}")

    finally:

        # Remove the temporary file after uploading it

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)