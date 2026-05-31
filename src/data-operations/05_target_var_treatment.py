# 5. Target Variable Treatment

# In this file, the idea is to do the following:

#  - Load 1 file from the S3 bucket (in raw folder)
#  - Transform the target variable (Discretization)
#  - Upload the transformed dataframe back to the S3 bucket (in silver folder)


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
S3_KEY_RAW = "raw/btc_target_data.csv"

# Threshold for discretixation (1% change returns)

THRESHOLD = 0.01  

# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get BTC-USD data from S3 bucket

def get_btc_data_to_transform(bucket_name = S3_BUCKET_NAME,
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

        # Rename the colum "index" to "Date"

        df.rename(columns = {"index": "Date"}, inplace = True)

    except Exception as e:

        print(f"Error downloading file from S3: {e}")

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)

    return df


# ------------------------------------------------------------------------------------------------
# Target Treatment (Discretization)
# ------------------------------------------------------------------------------------------------  

# Function to discretize the variable

def discretize_target_variable(df, threshold = THRESHOLD):

    # Calculate future return 1 day ahead

    df["Future_Return"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]

    # Ternary discretization

    df["Target"] = np.where(
        df["Future_Return"] > threshold, "Bullish",
        np.where(
            df["Future_Return"] < -threshold, "Bearish",
            "Neutral"
        )
    )

    # Drop the null values cfeated by shift

    df = df.dropna(subset = ["Future_Return"])

    # 2 decimals for the future return

    df["Future_Return"] = df["Future_Return"].round(4)

    # Drop the other columns except "Date", "Target" and "Future_Return" (FR for analysis purposes)

    df = df[["Date", "Future_Return", "Target"]]

    return df


# ------------------------------------------------------------------------------------------------
# Upload data to S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to upload the dataframe with the target variable to S3 bucket

def upload_target_s3(df, 
                      bucket_name = S3_BUCKET_NAME, 
                      access_key_id = S3_ACCESS_KEY_ID, 
                      secret_access_key = S3_SECRET_ACCESS_KEY, 
                      s3_key = "silver/btc_target_column.csv"):

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

    # Get the data from S3 bucket

    print("Downloading data from S3 bucket")

    df = get_btc_data_to_transform()

    # Discretize the target variable

    df_transformed = discretize_target_variable(df)

    # Upload the transformed dataframe back to S3 bucket

    print("Uploading transformed data to S3 bucket")

    upload_target_s3(df_transformed)

    print(f"Data uploaded to S3 bucket {S3_BUCKET_NAME} with key: silver/btc_target_column.csv")