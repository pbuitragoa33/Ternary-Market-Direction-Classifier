# 8. External Data Unification

# In this file, the idea is to do the following:

#  - Load 5 files from the S3 bucket (in silver folder)
#  - Merge them into a single dataframe (consolidation)
#  - Save the consolidated dataframe back to the S3 bucket (in silver folder)

# The files to be loaded are:

#    - btc_indicators_data.csv
#    - btc_target_column.csv
#    - data_fred_transformed.csv
#    - external_data_unified.csv
#    - financial_assets.csv

# Consolidate them in data_unified.csv


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


# Files to download

files_in_silver = [
    "silver/btc_indicators_data.csv",
    "silver/btc_target_column.csv",
    "silver/data_fred_transformed.csv",
    "silver/external_data_unified.csv",
    "silver/financial_assets.csv"
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


# ------------------------------------------------------------------------------------------------
# Merge thge dataframes into a single one
# ------------------------------------------------------------------------------------------------

# Function merge

def merge_dfs(dfs_dict):

    # Possible merge keys becaise they are different

    possible_merge_keys = ["Date", "date", "time"]

    # Normalize date column name to "date" before merging

    def _normalize_date_column(df):

        for key in possible_merge_keys:

            if key in df.columns and key != "date":

                return df.rename(columns = {key: "date"})

        return df

    # Start with the first dataframe

    merged_df = None

    for key, df in dfs_dict.items():

        df = _normalize_date_column(df)

        if merged_df is None:

            merged_df = df

        else:

            # Find the common merge key

            merge_key = None

            if "date" in merged_df.columns and "date" in df.columns:

                merge_key = "date"

            if merge_key is not None:

                merged_df = pd.merge(merged_df, df, on = merge_key, how = "outer")

            else:

                print(f"No common merge key. Skipping merge for {key}.")

    return merged_df


# ------------------------------------------------------------------------------------------------
# Transform and upload the merged data to S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to upload the merged data to S3 bucket

def transform_upload_merged_data(df_merged):

    # 1. Rename the date column to "date" if it exists

    if "Date" in df_merged.columns:

        df_merged.rename(columns = {"Date": "date"}, inplace = True)

    elif "time" in df_merged.columns:

        df_merged.rename(columns = {"time": "date"}, inplace = True)

    # 2. Convert the date column to datetime format

    df_merged["date"] = pd.to_datetime(df_merged["date"])

    # 3. Sort the data by date

    df_merged = df_merged.sort_values("date").reset_index(drop = True)

    # 4. Save the merged dataframe back to the S3 bucket (in silver folder)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = S3_ACCESS_KEY_ID,
        aws_secret_access_key = S3_SECRET_ACCESS_KEY
    )

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

        df_merged.to_csv(temp_file_path, index = False)

        s3_client.upload_file(temp_file_path, S3_BUCKET_NAME, "silver/data_unified.csv")

    if os.path.exists(temp_file_path):

        os.remove(temp_file_path)


    

# Execution 

if __name__ == "__main__":

    # Dictinary to store the dataframes

    dfs_found = {}

    # Loop to download each file and store it in the dictionary

    for file_path in files_in_silver:

        file_key = os.path.basename(file_path).replace(".csv", "")

        print(f"Downloading {file_path}")

        df_downloaded = get_target_data_s3(s3_key = file_path)

        if df_downloaded is not None:

            dfs_found[file_key] = df_downloaded

    # Merge the dataframes into a single one (consolidation)

    print("Merging the datafrsmes")

    df_merged = merge_dfs(dfs_found)

    # Transform and upload the merged data to S3 bucket

    print("Transforming and uploading the merged data to S3 bucket")

    transform_upload_merged_data(df_merged)

    print("Data unification completed successfully in S3 bucket as silver/data_unified.csv")