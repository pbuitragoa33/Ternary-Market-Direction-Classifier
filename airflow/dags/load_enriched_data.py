# Module for loading the unified data to be processed

# This script aims to:

#   - Load the data_enriched.csv located in S3 bucket in silver folder
#   - Return a pandas dataframe with the data
#   - Bring a reusable function to be used anywhere in the project


# Libraries

import pandas as pd
import boto3
from dotenv import load_dotenv
import os

# Important definitions

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_KEY_SILVER = "silver/data_enriched.csv"


# ------------------------------------------------------------------------------------------------
# Function to load the unified data
# ------------------------------------------------------------------------------------------------

def load_data() -> pd.DataFrame:

    # Create the S3 client

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = S3_ACCESS_KEY_ID,
        aws_secret_access_key = S3_SECRET_ACCESS_KEY
    )

    # Download the file and convert it to a pandas Dataframe

    try:

        response = s3_client.get_object(Bucket = S3_BUCKET_NAME, Key = S3_KEY_SILVER)

        df = pd.read_csv(response["Body"])

        df = pd.DataFrame(df)

    except Exception as e:

        print(f"Error downloading file from S3: {e}")

        df = pd.DataFrame()

    return df


# Main to test the function

if __name__ == "__main__":

    df = load_data()