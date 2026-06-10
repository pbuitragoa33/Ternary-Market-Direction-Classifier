# 11. Orchestration with Airflow DAGs

# This script aims to:

#  - Orchestrate the previous defined pipeline
#  - Source Verifying → Pipeline Execution → Output Validation
#  - Upload the transformed datasets back to the S3 bucket (in gold folder) 


# Libraries

from datetime import datetime, timedelta
import os
import boto3
import pandas as pd
from airflow.decorators import dag, task
from processing_pipeline import (
    S3_BUCKET_NAME,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    S3_KEY_SILVER,
    TARGET_MAPPING,
    execute_pipeline
)


# ------------------------------------------------------------------------------------------------
# DAG Configuration
# ------------------------------------------------------------------------------------------------

# Default arguments for the DAG

default_args = {
    "owner": "ternary-market-direction-classifier",
    "retries": 2,
    "retry-delay": timedelta(minutes = 5)
}


# ------------------------------------------------------------------------------------------------
# DAG Definition
# ------------------------------------------------------------------------------------------------

@dag(
    dag_id = "tmdc_pipeline_processing_dag",
    description = "Silver → Transformations → Temporal Split → Encoding and Scaling → Gold + artifacts",
    schedule = None,
    start_date = datetime(2026, 6, 10),
    catchup = False,
    default_args = default_args,
    tags = ["tmdc", "data-processing", "pipeline", "ML"] 
)

def tmdc_ml_preprocessing():

    # Task 1: Source Verification

    @task

    def source_verification() -> dict:

        s3 = boto3.client("s3",
                            aws_access_key_id = S3_ACCESS_KEY_ID,
                            aws_secret_access_key = S3_SECRET_ACCESS_KEY)


        response = s3.head_object(Bucket = S3_BUCKET_NAME, Key = S3_KEY_SILVER)

        metadata = {
            "key": S3_KEY_SILVER,
            "size_bytes": response["ContentLength"],
            "last_modified": response["LastModified"].isoformat()
        }

        print(f"Source verified: s3://{S3_BUCKET_NAME}/{S3_KEY_SILVER} "
                f"({metadata['size_bytes']} bytes, modified {metadata['last_modified']})")
        
        return metadata
    
    # Task 2: Pipeline Execution

    @task

    def preprocessing(source_metadata: dict) -> dict:

        metadata = execute_pipeline()

        return metadata
    
    # Task 3: Output Validation (gold)

    @task

    def output_validation(metadata: dict) -> None:

        s3 = boto3.client("s3",
                          aws_access_key_id = S3_ACCESS_KEY_ID,
                          aws_secret_access_key = S3_SECRET_ACCESS_KEY)


        # All assets must be present in the gold folder

        for key in metadata["keys_gold"] + [metadata["key_artifacts"]]:

            s3.head_object(Bucket = S3_BUCKET_NAME, Key = key)

            print(f"Output validated: s3://{S3_BUCKET_NAME}/{key} exists and is accessible")

        # Sanity checks about X_train

        path_X_train = f"s3://{S3_BUCKET_NAME}/gold/X_train.csv"

        X_train = pd.read_csv(path_X_train, index_col = 0)
 
        assert X_train.shape[0] == metadata["train_rows"], f"Rows without match: {X_train.shape[0]} vs {metadata['train_rows']}"
 
        assert X_train.shape[1] == metadata["n_features"], f"Features with mismatch: {X_train.shape[1]} vs {metadata['n_features']}"
 
        assert not X_train.isna().any().any(), f"X_train contains missing values"

        # Target vairable must contain 3 classes

        y_train = pd.read_csv(f"s3://{S3_BUCKET_NAME}/gold/y_train.csv", index_col = 0)

        real_classes = set(y_train.iloc[:, 0].unique())
        expected_classes = set(TARGET_MAPPING.values())

        assert real_classes == expected_classes, f"Target classes mismatch: {real_classes} vs {expected_classes}"

        # Balance report

        print(f"Class distribution in train: {metadata['balance_train']}")
        print(f"Splits → Train: {metadata['train_rows']} rows, Validation: {metadata['validation_rows']} rows, Test: {metadata['test_rows']} rows")


    # Orchestration

    source_metadata = source_verification()
    metadata = preprocessing(source_metadata)
    output_validation(metadata)


# ------------------------------------------------------------------------------------------------
# DAG Instantiation
# ------------------------------------------------------------------------------------------------

tmdc_ml_preprocessing()