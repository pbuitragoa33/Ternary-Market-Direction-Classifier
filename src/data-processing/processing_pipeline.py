# 10. Processing Pipeline script to clean and transform the project data

# In this file, the idea is to do the following:

#  - Load 1 file (data_enriched.csv) from the S3 bucket (in silver folder)
#  - Transform and process the data to be ready for modeling
#  - Prepare the script to be executed in an Airflow DAG (to be created in the next step)
#  - Upload the transformed datasets back to the S3 bucket (in gold folder)


# Libraries

import os
import re
import tempfile
import boto3
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
import sys
from pathlib import Path

# Solve the import from the src folder

project_root = Path(__file__).resolve().parent.parent.parent

if str(project_root) not in sys.path:

    sys.path.insert(0, str(project_root))

from configs.load_enriched_data import load_data


# ------------------------------------------------------------------------------------------------
# Configurations and Constants
# ------------------------------------------------------------------------------------------------

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
 
S3_KEY_SILVER = "silver/data_enriched.csv"
S3_PREFIX_GOLD = "gold/"
S3_KEY_ARTIFACTS = "artifacts/processing_artifacts.joblib"

ARTIFACTS_LOCAL_PATH = os.getenv("ARTIFACTS_LOCAL_PATH")


# Temporal cutoffs to split the data (80/10/10)

TRAIN_END_DATE = "2023-12-31"
VALIDATION_END_DATE = "2024-12-31"

# train: date <= TRAIN_END_DATE
# validation: TRAIN_END_DATE < date <= VALIDATION_END_DATE
# test: date > VALIDATION_END_DATE


# Target variable mapping

TARGET_COL = "Target"
TARGET_MAPPING = {
    "Bearish": 0,
    "Neutral": 1,
    "Bullish": 2
}


# ------------------------------------------------------------------------------------------------
# Variable Definitions
# ------------------------------------------------------------------------------------------------

# 1. Columns to drop (irrelevant and produce leakage)

columns_to_drop = ["Unnamed: 0_x", "Unnamed: 0_y", "Unnamed: 0", "Future_Return", "HMM_Regime_PCA"]


# 2. Groups of transformation (defined in 09_feature_HMM_regime.py)

# Columns to divide by "Close" (to "normalize" them)

columns_to_divide_by_close = ["Scaled_Upper_Bollinger", "Scaled_Lower_Bollinger",
                          "Scaled_Upper_Keltner", "Scaled_Lower_Keltner",
                          "Scaled_SMA20", "Scaled_SMA50", "Scaled_EMA20",
                          "Scaled_EMA50", "Scaled_HMA20", "Scaled_HMA50"]

# Columns to divide by "Close" lagged

columns_momentum_20 = ["Momentum_20p"]
columns_momentum_50 = ["Momentum_50p"]

# Columns to log-return transformartion

columns_log_return = ["Close", "High", "Low", "Open",
                   "VIX_close", "GOLD_close", "OIL_close", "TLT_close", "RSP_close",
                   "IWM_close", "UUP_close", "MSTR_close", "SPY_close",
                   "CRYPTOCAP_BTC.D, 1D", "CRYPTOCAP_TOTAL, 1D", "CRYPTOCAP_TOTAL2, 1D",
                   "CRYPTOCAP_TOTALES, 1D", "CRYPTOCAP_USDT.D, 1D", "INDEX_BDI, 1D"]

# Columns to pecentage change transformation

columns_pct_change = ["FRED_WM2NS, 1D"]

# Columns to differenciate

columns_to_diff = ["TNX_close", "T5YIE", "EFFR", "Corporate_Bond_Spread", "OBV"]

# Columns to divide by its MA30 (just "Volume")

columns_relative_volume = ["Volume"]


# 3. Categorical columns

# Ordinal columns

ordinal_cat_columns = ["Relative_Volume_Category"]

# Ordinal mapping for "Relative_Volume_Category" (categories)

ordinal_cat_categories = [["Very Low", "Low", "High", "Very High"]]

# One-hot encoded columns

one_hot_cat_columns = ["Day_Type", "Weekly_Breakout", "HMM_Regime_All"]



# ------------------------------------------------------------------------------------------------
#                                         Transformation Classes
# ------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------
# 1. Class to drop columns
# ------------------------------------------------------------------------------------------------

class ColumnDropper(BaseEstimator, TransformerMixin):

    # Class constructor

    def __init__(self, cols_to_drop):

        self.cols_to_drop = cols_to_drop

    # Fit method

    def fit(self, X, y = None):

        return self
    
    # Transform method

    def transform(self, X):

        return X.drop(columns = self.cols_to_drop, errors = "ignore")
    

# ------------------------------------------------------------------------------------------------
# 2. Class for temporal imputations
# ------------------------------------------------------------------------------------------------

# ffill: More natural imputation method fot time time series because it's the last observation

class TemporalImputer(BaseEstimator, TransformerMixin):

    # Class constructor

    def __init__(self, values_as_nan = ("No Data",)):

        self.values_as_nan = values_as_nan

    # Fit method

    def fit(self, X, y = None):

        return self
    
    # Transform method

    def transform(self, X):

        X = X.copy()

        # Categorical: "No Data" → NaN → ffill

        cat_cols = X.select_dtypes(exclude = [np.number]).columns

        for col in cat_cols:

            X[col] = X[col].replace(list(self.values_as_nan), np.nan)

            X[col] = X[col].ffill()

        # Numerical: ffill

        num_cols = X.select_dtypes(include = [np.number]).columns

        for col in num_cols:

            X[col] = X[col].ffill()

        return X


# ------------------------------------------------------------------------------------------------
# 3. Class for Seasonal and Various Transformations
# ------------------------------------------------------------------------------------------------

# Everything that needs "Close" columns has to be done before the transformations required

class VariousTransformations(BaseEstimator, TransformerMixin):

    # Class construvtor

    def __init__(self,
                 columns_to_divide_by_close = None,
                 columns_momentum_20 = None,
                 columns_momentum_50 = None,
                 columns_log_return = None,
                 columns_pct_change = None,
                 columns_to_diff = None,
                 columns_relative_volume = None,
                 relative_volume_window = 30):

        self.columns_to_divide_by_close = columns_to_divide_by_close or []
        self.columns_momentum_20 = columns_momentum_20 or []
        self.columns_momentum_50 = columns_momentum_50 or []
        self.columns_log_return = columns_log_return or []
        self.columns_pct_change = columns_pct_change or []
        self.columns_to_diff = columns_to_diff or []
        self.columns_relative_volume = columns_relative_volume or []
        self.relative_volume_window = relative_volume_window

    # Fit method

    def fit(self, X, y = None):

        return self
    
    # Transform method

    def transform(self, X):

        X = X.copy()

        # 1. Columns to divide by "Close"

        for col in self.columns_to_divide_by_close:

            X[col] = X[col] / X["Close"]

        # 2. Momentum columns (divide by "Close" lagged)

        for col in self.columns_momentum_20:

            X[col] = X[col] / X["Close"].shift(20)

        for col in self.columns_momentum_50:

            X[col] = X[col] / X["Close"].shift(50)

        # 3. Particular case of "Anchored_VWAP"

        if "Anchored_VWAP" in X.columns:

            X["Anchored_VWAP_Distance"] = X["Close"] / X["Anchored_VWAP"] - 1

            X = X.drop(columns = ["Anchored_VWAP"])

        # 4. Log-return transformation

        for col in self.columns_log_return:

            if col in X.columns:

                X[col] = np.log(X[col] / X[col].shift(1))

                X = X.rename(columns = {col: f"{col}_log_return"})

        # 5. Percentage change transformation

        for col in self.columns_pct_change:

            if col in X.columns:

                X[col] = X[col].pct_change()

                X = X.rename(columns = {col: f"{col}_pct_change"})

        # 6. Differencing transformation

        for col in self.columns_to_diff:

            if col in X.columns:

                X[col] = X[col].diff()

                X = X.rename(columns = {col: f"{col}_diff"})

        # 7. Particular case of "Relative_Volume"

        for col in self.columns_relative_volume:

            if col in X.columns:

                X[col] = X[col] / X[col].rolling(self.relative_volume_window).mean()

                X = X.rename(columns = {col: f"{col}_relative"})

        return X
    

# ------------------------------------------------------------------------------------------------
# 4. Class for Ordinal Encoding
# ------------------------------------------------------------------------------------------------

# Ordianl Enconding with explicit categories

class OrdinalEncoding(BaseEstimator, TransformerMixin):

    # Class constructor

    def __init__(self, cols = ordinal_cat_columns, categories = ordinal_cat_categories):

        self.cols = cols
        self.categories = categories
        self.encoder_ = None

    # Fit method

    def fit(self, X, y = None):

        self.encoder_ = OrdinalEncoder(categories = self.categories,
                                      handle_unknown = "use_encoded_value",
                                      unknown_value = -1)
        
        self.encoder_.fit(X[self.cols])

        return self
    
    # Transform method

    def transform(self, X):

        X = X.copy()

        X[self.cols] = self.encoder_.transform(X[self.cols])

        return X
    

# ------------------------------------------------------------------------------------------------
# 5. Class for OneHot Encoding
# ------------------------------------------------------------------------------------------------

class OneHotEncoding(BaseEstimator, TransformerMixin):

    # Class constructor

    def __init__(self, cols = one_hot_cat_columns):

        self.cols = cols
        self.encoder_ = None
        self.dummy_cols = None

    # Fit method

    def fit(self, X, y = None):

        self.encoder_ = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)

        self.encoder_.fit(X[self.cols])

        self.dummy_cols = self.encoder_.get_feature_names_out(self.cols)

        return self
    
    # Transform method

    def transform(self, X):

        encoded = self.encoder_.transform(X[self.cols])

        encoded_df = pd.DataFrame(encoded, columns = self.dummy_cols, index = X.index)

        X = X.drop(columns = self.cols)

        X = pd.concat([X, encoded_df], axis = 1)

        return X


# ------------------------------------------------------------------------------------------------
# 6. Class for Robust Scaling
# ------------------------------------------------------------------------------------------------

class RobustScaling(BaseEstimator, TransformerMixin):

    # Class constructor

    def __init__(self, prefixes_to_exclude = None, cols_to_exclude = None):

        self.prefixes_to_exclude = prefixes_to_exclude if prefixes_to_exclude is not None else [f"{c}_" for c in one_hot_cat_columns]
        self.cols_to_exclude = cols_to_exclude if cols_to_exclude is not None else list(ordinal_cat_columns)
        self.scaler_ = None
        self.continuous_cols_ = None

    # Select continuous columns

    def select_continuous_columns(self, X):

        num_cols = X.select_dtypes(include = [np.number]).columns.tolist()

        continuous = [
            c for c in num_cols
            if c not in self.cols_to_exclude
            and not any(c.startswith(p) for p in self.prefixes_to_exclude)
        ]

        return continuous
    
    # Fit method

    def fit(self, X, y = None):

        self.continuous_cols_ = self.select_continuous_columns(X)

        self.scaler_ = RobustScaler()

        self.scaler_.fit(X[self.continuous_cols_])

        return self
    
    # Transform method

    def transform(self, X):

        X = X.copy()

        X[self.continuous_cols_] = self.scaler_.transform(X[self.continuous_cols_])

        return X
    

# ------------------------------------------------------------------------------------------------
#                                             Pipelines
# ------------------------------------------------------------------------------------------------

# Pipeline BASE (stateless) → to be apllied before the split

pipeline_base = Pipeline(steps = [
    ("column_dropper", ColumnDropper(cols_to_drop = columns_to_drop)),
    ("temporal_imputer", TemporalImputer()),
    ("various_transformations", VariousTransformations())
])

# Pipeline ML (fit just in train, transform in validation and test)

def build_pipeline_ML():

    return Pipeline(steps = [
        ("ordinal_encoding", OrdinalEncoding()),
        ("one_hot_encoding", OneHotEncoding()),
        ("robust_scaling", RobustScaling())
    ])


# ------------------------------------------------------------------------------------------------
#                                       Support Functions
# ------------------------------------------------------------------------------------------------

# S3 Client

def s3_client():
 
    return boto3.client("s3",
                        aws_access_key_id = S3_ACCESS_KEY_ID,
                        aws_secret_access_key = S3_SECRET_ACCESS_KEY)


# Load file from S3

def load_processed_data_to_S3(df, key, bucket_name = S3_BUCKET_NAME):

    # Create the S3 client

    client = s3_client()

    # Save the dataframe to a temporary CSV file

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    try:

        df.to_csv(temp_file_path, index = True)

        # Upload the file to S3

        client.upload_file(temp_file_path, bucket_name, key)

        print(f"Uploaded updated data to S3 bucket: s3://{bucket_name}/{key}")

    except Exception as e:

        print(f"Error uploading file to S3: {e}")

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)


# Temporal Split (train/val/test on date cutoffs)

def temporal_split(df, train_end = TRAIN_END_DATE, validation_end = VALIDATION_END_DATE):

    df_train = df.loc[:train_end]
    df_validation = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days = 1):validation_end]
    df_test = df.loc[pd.Timestamp(validation_end) + pd.Timedelta(days = 1):]

    print(f"    - Train: {df_train.shape[0]} filas ({df_train.index.min().date()} → {df_train.index.max().date()})")
    print(f"    - Val:   {df_validation.shape[0]} filas ({df_validation.index.min().date()} → {df_validation.index.max().date()})")
    print(f"    - Test:  {df_test.shape[0]} filas ({df_test.index.min().date()} → {df_test.index.max().date()})")

    return df_train, df_validation, df_test


# Data Division (X/y split) → separate the features from the target variablee

def split_X_y(df, target_col = TARGET_COL, target_map = TARGET_MAPPING):

    X = df.drop(columns = [target_col])

    y = df[target_col].map(target_map)

    if y.isna().any():

        rare_values = df.loc[y.isna(), target_col].unique()

        raise ValueError(f"Target column contains values not in the mapping: {rare_values}")
    
    y = y.astype(int)

    return X, y


# ------------------------------------------------------------------------------------------------
# Cleanup the column names of the training data
# ------------------------------------------------------------------------------------------------

def cleanup_columns(df):

    clean = {col: re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_") for col in df.columns}

    df = df.rename(columns = clean)

    # Inc ase of duplicates after the clean, raise an error

    if df.columns.duplicated().any():

        raise ValueError(f"Column name collision after sanitizing: {df.columns[df.columns.duplicated()].tolist()}")

    return df

# ------------------------------------------------------------------------------------------------
#                                       Pipeline Execution
# ------------------------------------------------------------------------------------------------

def execute_pipeline(key_entry = S3_KEY_SILVER,
                     prefix_output = S3_PREFIX_GOLD,
                     key_artifacts = S3_KEY_ARTIFACTS):
    
    # Main function, importable from the Airflow DAG

    # 1. Load the enriched data from S3

    df = load_data()

    # 2. Index the dataframe by "date"

    df["date"] = pd.to_datetime(df["date"])

    df = df.set_index("date").sort_index()

    # 3. Apply the BASE pipeline

    df = pipeline_base.fit_transform(df)

    # 4. Drop of missing values generated by the shift/diff/rolling operations

    rows_before = len(df)

    df = df.dropna()

    print(f"Dropped {rows_before - len(df)} rows with NA values. Remaining rows: {len(df)}")

    # 5. Cleanup the column names

    df = cleanup_columns(df)

    # 6. Temporal split (train/validation/test)

    df_train, df_validation, df_test = temporal_split(df)

    # 7. Apply the variable split (X/y) for each set

    X_train, y_train = split_X_y(df_train)
    X_validation, y_validation = split_X_y(df_validation)
    X_test, y_test = split_X_y(df_test)

    # 8. Apply the ML pipeline

    pipeline_ML = build_pipeline_ML()

    pipeline_ML.fit(X_train)
    X_train = pipeline_ML.transform(X_train)
    X_validation = pipeline_ML.transform(X_validation)
    X_test = pipeline_ML.transform(X_test)

    # 9. Save the transformed data back to S3 (in gold folder)

    load_processed_data_to_S3(X_train, key = f"{prefix_output}X_train.csv")
    load_processed_data_to_S3(y_train.to_frame(), key = f"{prefix_output}y_train.csv")
    load_processed_data_to_S3(X_validation, key = f"{prefix_output}X_validation.csv")
    load_processed_data_to_S3(y_validation.to_frame(), key = f"{prefix_output}y_validation.csv")
    load_processed_data_to_S3(X_test, key = f"{prefix_output}X_test.csv")
    load_processed_data_to_S3(y_test.to_frame(), key = f"{prefix_output}y_test.csv")

    # 10. Save the artifacts (local and S3)

    artifacts = {
        "pipeline_ML": pipeline_ML,
        "target_mapping": TARGET_MAPPING,
        "target_mapping_inverse": {v: k for k, v in TARGET_MAPPING.items()},
        "final_columns": list(X_train.columns),
        "cutoffs": {"train_end": TRAIN_END_DATE, "validation_end": VALIDATION_END_DATE},
        "continuous_columns_scaled": pipeline_ML.named_steps["robust_scaling"].continuous_cols_
    }

    local_artifacts_path = os.path.join(ARTIFACTS_LOCAL_PATH, "processing_artifacts.joblib")

    joblib.dump(artifacts, local_artifacts_path)

    s3_client().upload_file(local_artifacts_path, S3_BUCKET_NAME, key_artifacts)

    print(f"Saved processing artifacts to S3 bucket: s3://{S3_BUCKET_NAME}/{key_artifacts}")

    # Metadata (suited for XCom and Airflow)

    metadata = {
        "rows_train": int(len(X_train)),
        "rows_validation": int(len(X_validation)),
        "rows_test": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
        "balance_train": y_train.value_counts(normalize = True).round(3).to_dict(),
        "keys_gold": [f"{prefix_output}{n}.csv" for n in ["X_train", "y_train", "X_validation", "y_validation", "X_test", "y_test"]],
        "key_artifacts": key_artifacts
    }

    print("X_train columns:", X_train.columns.tolist())

    print("Processing pipeline executed successfully")

    return metadata



# Main Execution

if __name__ == "__main__":

    load_dotenv()

    execute_pipeline()