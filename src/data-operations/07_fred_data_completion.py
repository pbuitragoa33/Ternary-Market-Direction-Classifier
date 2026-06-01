# 7. Data of economic indicators processing

# In this file, the idea is to do the following:

#  - Load 1 file from the S3 bucket (in raw folder)
#  - Transform and process the economic indicators data(features from FRED)
#  - Bring data from GitHub Repo to fill/correct some missing values in the Fred data
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
S3_KEY_RAW = "raw/data_fred.csv"

START_DATE = "2015-03-07"
END_DATE = "2026-01-15"
KEEP_FRED_START_DATE = "2025-11-24"
KEEP_FRED_END_DATE = "2026-01-15"

# From GitHub

URL_HIGH_YIELD = "https://raw.githubusercontent.com/pbuitragoa33/Multi-Model-Financial-Time-Series-Forecasting/main/src/data/raw/High_Yield_raw_data.csv"
URL_CORPORATE_BONDS = "https://raw.githubusercontent.com/pbuitragoa33/Multi-Model-Financial-Time-Series-Forecasting/main/src/data/raw/Corporate_Bond_710_raw_data.csv"
URL_NAT_CONDITIONS = "https://raw.githubusercontent.com/pbuitragoa33/Multi-Model-Financial-Time-Series-Forecasting/main/src/data/raw/NFCI_fin_condition_raw_data.csv"
URL_FINANCIAL_STRESS = "https://raw.githubusercontent.com/pbuitragoa33/Multi-Model-Financial-Time-Series-Forecasting/main/src/data/raw/STLFSI4_Stress_raw_data.csv"


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get economic indicators data from S3 bucket

def get_economic_indicators_to_transform(bucket_name = S3_BUCKET_NAME,
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

        df_fred = pd.read_csv(temp_file_path)

        print(f"Downloaded data from S3 bucket: s3://{bucket_name}/{key}")
        print("--" * 20)
        print(f"Data shape: {df_fred.shape}")

    except Exception as e:

        print(f"Error downloading file from S3: {e}")

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)

    return df_fred


# ------------------------------------------------------------------------------------------------
# Download data from my GitHub Repo
# ------------------------------------------------------------------------------------------------

# Function to get economic indicators data from GitHub Repo

def get_economic_data_from_github(url_high_yield = URL_HIGH_YIELD,
                                 url_corporate_bonds = URL_CORPORATE_BONDS,
                                 url_nat_conditions = URL_NAT_CONDITIONS,
                                 url_financial_stress = URL_FINANCIAL_STRESS):
    
    # Read the data from the GitHub URLs

    df_high_yield = pd.read_csv(url_high_yield)
    df_corporate_bonds = pd.read_csv(url_corporate_bonds)
    df_nat_conditions = pd.read_csv(url_nat_conditions)
    df_financial_stress = pd.read_csv(url_financial_stress)

    return df_high_yield, df_corporate_bonds, df_nat_conditions, df_financial_stress


# ------------------------------------------------------------------------------------------------
# Clean and prepare the data from GitHub Repo (recommended by AI)
# ------------------------------------------------------------------------------------------------

def _clean_github_df(df, value_col_original, value_col_renamed):

    df = df.copy()

    unnamed_cols = [col for col in df.columns if str(col).lower().startswith("unnamed")]

    if unnamed_cols:

        df = df.drop(columns = unnamed_cols)

    date_cols = [col for col in df.columns if str(col).lower() == "date"]

    if date_cols and date_cols[0] != "date":

        df = df.rename(columns = {date_cols[0]: "date"})

    df = df.rename(columns = {value_col_original: value_col_renamed})

    return df[["date", value_col_renamed]]


# ------------------------------------------------------------------------------------------------
# Process and transform the data (Replace)
# ------------------------------------------------------------------------------------------------

# Function to process and transform

def process_and_transform_data(df_fred, 
                               df_high_yield, 
                               df_corporate_bonds, 
                               df_nat_conditions, 
                               df_financial_stress):
    
    # The idea is simple:

    # 1. Normalize date columns and rename indicators from GitHub data

    date_cols = [col for col in df_fred.columns if str(col).lower() == "date"]

    if date_cols and date_cols[0] != "date":

        df_fred = df_fred.rename(columns = {date_cols[0]: "date"})

    df_high_yield = _clean_github_df(df_high_yield, "BAMLH0A0HYM2", "High_Yield_Spread")
    df_corporate_bonds = _clean_github_df(df_corporate_bonds, "BAMLC4A0C710YEY", "Corporate_Bond_Spread")
    df_nat_conditions = _clean_github_df(df_nat_conditions, "NFCI", "National_Conditions_Index")
    df_financial_stress = _clean_github_df(df_financial_stress, "STLFSI4", "Financial_Stress_Index")

    # 2. Merge the data from GitHub Repo with the FRED data (on "date" column)

    df_merged = df_fred.merge(df_high_yield, on = "date", how = "outer")
    df_merged = df_merged.merge(df_corporate_bonds, on = "date", how = "outer")
    df_merged = df_merged.merge(df_nat_conditions, on = "date", how = "outer")
    df_merged = df_merged.merge(df_financial_stress, on = "date", how = "outer")

    # 3. Replace the columns in Fred dataa with the corresponding columns fom GitHub

    df_merged["date"] = pd.to_datetime(df_merged["date"])
    keep_mask = (df_merged["date"] >= KEEP_FRED_START_DATE) & (df_merged["date"] <= KEEP_FRED_END_DATE)

    df_merged["High_Yield_Spread"] = np.where(
        keep_mask,
        df_merged["BAMLH0A0HYM2"],
        df_merged["High_Yield_Spread"]
    )
    df_merged["Corporate_Bond_Spread"] = np.where(
        keep_mask,
        df_merged["BAMLC4A0C710YEY"],
        df_merged["Corporate_Bond_Spread"]
    )
    df_merged["National_Conditions_Index"] = np.where(
        keep_mask,
        df_merged["NFCI"],
        df_merged["National_Conditions_Index"]
    )
    df_merged["Financial_Stress_Index"] = np.where(
        keep_mask,
        df_merged["STLFSI4"],
        df_merged["Financial_Stress_Index"]
    )

    df_merged.drop(columns = ["BAMLH0A0HYM2", "BAMLC4A0C710YEY", "NFCI", "STLFSI4"], inplace = True)

    # 4. Sort the data by date and filter by the desired date range

    df_merged = df_merged.sort_values("date")
    df_merged = df_merged[(df_merged["date"] >= START_DATE) & (df_merged["date"] <= END_DATE)].reset_index(drop = True)

    return df_merged


# ------------------------------------------------------------------------------------------------
# Upload the transformed data to S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to upload the transformed data to S3 bucket

def upload_transformed_data(df,
                            bucket_name = S3_BUCKET_NAME,
                            access_key_id = S3_ACCESS_KEY_ID,
                            secret_access_key = S3_SECRET_ACCESS_KEY,
                            s3_key = "silver/data_fred_transformed.csv"):

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

    if os.path.exists(temp_file_path):

        os.remove(temp_file_path)



# Execution

if __name__ == "__main__":

    # Get data from S3 raw folder

    print("Downloading data from S3 bucket")

    df_fred = get_economic_indicators_to_transform()

    # Get data from GitHub Repo

    print("Downloading data from GitHub Repo")

    df_high_yield, df_corporate_bonds, df_nat_conditions, df_financial_stress = get_economic_data_from_github()

    # Process and transform the data

    df_transformed = process_and_transform_data(df_fred, df_high_yield, df_corporate_bonds, df_nat_conditions, df_financial_stress)

    print("--" * 20)
    print(df_transformed.columns)
    print(f"Transformed data shape: {df_transformed.shape}")
    print("--" * 20)
    
    # Upload the transformed data to S3 bucket (silver folder)

    print("Uploading transformed data to S3 bucket")

    upload_transformed_data(df_transformed)

    print(f"Data uploaded to S3 bucket {S3_BUCKET_NAME} with key: silver/data_fred_transformed.csv")