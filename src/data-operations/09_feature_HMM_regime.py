# 9. Add HMM regime as a feature to the consolidated dataframe

# In this file, the idea is to do the following:

#  - Load the consolidated dataframe from the S3 bucket (in silver folder)
#  - Pipeline (temporal) the data to ensure it is in the correct format for the HMM model
#  - Execute the Hidden Markov Model to identify market regimes based on the close price BTC
#  - Run one HMM with PCA-reduced features
#  - Run one HMM with all features (without PCA)
#  - Add the identified regimes as a new feature to the consolidated dataframe
#  - Load the updated dataframe (original with new HMM regime feature) to the S3 bucket (silver folder)


# Libraries 

import pandas as pd
import numpy as np
import boto3
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from dotenv import load_dotenv
import tempfile


# Important definitions

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_KEY_SILVER = "silver/data_enriched.csv"


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get consolidated data from S3 bucket

def get_consolidated_data(bucket_name = S3_BUCKET_NAME,
                          access_key_id = S3_ACCESS_KEY_ID,
                          secret_access_key = S3_SECRET_ACCESS_KEY,
                          key = S3_KEY_SILVER):
    
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
# Pipeline of processing data for HMM model
# ------------------------------------------------------------------------------------------------

# Function 1: Cut the dataframe & Drop invalid columns & Set the date as index

def preprocess_data_for_HMM(df, start_date = "2015-03-09", end_date = "2026-01-14", 
                            index_column = "date", invalid_colums = []):
    
    # Copy the dataframe to avoid modifying the original one

    df_process = df.copy()

    # Index the dataframe by date and sort it

    df_process = df_process.set_index(index_column)
    df_process = df_process.sort_index()

    # Filter teh dataframe by the specified date range

    df_process = df_process.loc[start_date:end_date]

    # Drop the invalid columns

    df_process = df_process.drop(columns = invalid_colums)

    return df_process


# Function 2: Feature Selection for HMM 

def selected_features_for_HMM(df, features_to_select = []):

    # All left columns after dropping the invalid ones except the date

    df_selected = df[features_to_select] 

    return df_selected


# Function 3: Imputation of missing values with forward fill metohd

def impute_num_ffill(df):

    # Separate numerical

    num_cols = df.select_dtypes(include = ["float64", "int64"]).columns

    # Imputation for numerical columns with forward fill

    for col in num_cols:

        df[col] = df[col].ffill()

    # Return the dataframe with imputed values

    return df


# Function 4: Transform some features (if needed) for HMM

def transform_features_for_HMM(df):

    # 1. DIVIDE BY CLOSE: Columns to divide by the close price (BTC Close indeed) to get relative values

    cols_to_divide_by_close = ["Scaled_Upper_Bollinger", "Scaled_Lower_Bollinger", "Scaled_Upper_Keltner", 
                               "Scaled_Lower_Keltner", "Scaled_SMA20", "Scaled_SMA50", "Scaled_EMA20",
                               "Scaled_EMA50", "Scaled_HMA20", "Scaled_HMA50"]
    
    for col in cols_to_divide_by_close:

        df[col] = df[col] / df["Close"]

    # 2. DIVIDE BY CLOSE WITH 20/50 LAGS: Columns to divide by the close price with 20/50 lags to get relative values

    cols_to_divide_by_close_20lags = ["Momentum_20p"]
    cols_to_divide_by_close_50lags = ["Momentum_50p"]

    for col in cols_to_divide_by_close_20lags:

        df[col] = df[col] / df["Close"].shift(20)

    for col in cols_to_divide_by_close_50lags:

        df[col] = df[col] / df["Close"].shift(50)

    # 3. LOG-RETURNS: Columns to log return

    cols_to_log_return = ["Close", "High", "Low", "Open", 
                         "VIX_close", "GOLD_close", "OIL_close", "TLT_close", "RSP_close",
                         "IWM_close", "UUP_close", "MSTR_close", "SPY_close",
                         "CRYPTOCAP_BTC.D, 1D","CRYPTOCAP_TOTAL, 1D","CRYPTOCAP_TOTAL2, 1D",
                         "CRYPTOCAP_TOTALES, 1D","CRYPTOCAP_USDT.D, 1D","INDEX_BDI, 1D"]

    # Apply log-return transformation to these columns and rename them with a "_log_return" suffix

    for col in cols_to_log_return:

        df[col] = np.log(df[col] / df[col].shift(1))

        df = df.rename(columns = {col: f"{col}_log_return"})

    # 4. PCT-CHANGE: Columns to percentage changee

    cols_to_pct_change = ["FRED_WM2NS, 1D"]

    # Apply percentage change transformation to these columns and rename them with a "_pct_change" suffix

    for col in cols_to_pct_change:

        df[col] = df[col].pct_change()

        df = df.rename(columns = {col: f"{col}_pct_change"})

    # 5. DIFF: Columns to differenciate 

    cols_to_diff = ["TNX_close", "T5YIE", "EFFR", "Corporate_Bond_Spread", "OBV"]

    # Apply differencing transformation to these columns and rename them with a "_diff" suffix

    for col in cols_to_diff:

        df[col] = df[col].diff()

        df = df.rename(columns = {col: f"{col}_diff"})

    # 6. LOG-TRANSFORMATION: Columns to log-transform

    cols_to_log_transform = ["Volume"]

    # Apply log-transformation to these columns and rename them with a "_log" suffix

    for col in cols_to_log_transform:

        df[col] = np.log1p(df[col])

        df = df.rename(columns = {col: f"{col}_log"})


    # Return the dataframe with transformed features

    return df


# Function 5: Train and Test split for HMM
 
def temporal_split_for_HMM(df, train_end = 0.8, validation_end = 0.9):

    # Length of the datafrme

    n = len(df)

    # Cut-off indices for train, validation, and test sets (80/10/10 split)
     
    train_end_idx = int(n * train_end) 
    validation_end_idx = int(n * validation_end) 
    
    # Split the dataframe into train, validation, and test sets

    df_train = df.iloc[:train_end_idx]
    df_validation = df.iloc[train_end_idx:validation_end_idx]
    df_test = df.iloc[validation_end_idx:]

    print(f"    - Train set shape: {df_train.shape[0]}")
    print(f"    - Validation set shape: {df_validation.shape[0]}")
    print(f"    - Test set shape: {df_test.shape[0]}")

    return df_train, df_validation, df_test
                             

# Function 6: Fit One-Hot Encoder on the training set for the categorical features (if any) and return the encoder

def fit_onehot(train_df, cols):

    encoder = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)

    encoder.fit(train_df[cols])

    return encoder



# Function 7: Apply the fitted OHE to validation and test sets and return the transformed dataframes

def apply_onehot(df, cols, encoder):

    encoded = encoder.transform(df[cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns = encoder.get_feature_names_out(cols),
        index = df.index
    )

    df = df.drop(columns = cols)

    df = pd.concat([df, encoded_df], axis = 1)

    return df


# Function 8: Fit Ordinal Encoder on the training set for the categorical features (if any) and return the encoder

def fit_ordinal(train_df, cols, categories):

    encoder = OrdinalEncoder(categories = categories)

    encoder.fit(train_df[cols])

    return encoder


# Function 9: Apply the fitted Ordinal Encoder to validation and test sets and return the transformed dataframes

def apply_ordinal(df, cols, encoder):

    df = df.copy()

    df[cols] = encoder.transform(df[cols])

    return df


# Function 10: Fit and apply Robust Scaler on sets

def scale_data(train_df, val_df, test_df, num_cols):

    scaler_RS = RobustScaler()

    train_scaled = scaler_RS.fit_transform(train_df[num_cols])
    val_scaled = scaler_RS.transform(val_df[num_cols])
    test_scaled = scaler_RS.transform(test_df[num_cols])

    train_df[num_cols] = train_scaled
    val_df[num_cols] = val_scaled
    test_df[num_cols] = test_scaled

    return train_df, val_df, test_df, scaler_RS


# ------------------------------------------------------------------------------------------------
# Dimensionality Reduction with PCA
# ------------------------------------------------------------------------------------------------

# Exucution

if __name__ == "__main__":

    # 1. Load the consolidated dataframe from the S3 bucket (in silver folder)

    df_consolidated = get_consolidated_data()

    # 2. PIPELINE 

    # 2.1. Function 1

    cols_to_drop = ["Unnamed: 0_x", "Unnamed: 0_y", "Future_Return"]

    df = preprocess_data_for_HMM(df_consolidated, invalid_colums = cols_to_drop)

    # 2.2. Function 2

    cols_to_select = [col for col in df.columns if col != "date" and col != "Target"]

    df = selected_features_for_HMM(df, features_to_select = cols_to_select)

    # 2.3. Function 3

    df = impute_num_ffill(df)

    # 2.4. Function 4

    df = transform_features_for_HMM(df)

    # 2.5. Function 5

    df = df.dropna()

    df_train, df_validation, df_test = temporal_split_for_HMM(df)

    # Note: The numerical features are imputed (with missing values with ffill)
    # the next step aftthe split, would be to impute the categorical features but 
    # the previous dropna removed all. Also the 3 categorucal features only have 3
    # missing values before all of thois, so it is not a big deal to drop them.

    # 2.6. Function 6

    one_hot_cols = ["Day_Type", "Weekly_Breakout"]

    encoder_onehot = fit_onehot(df_train, one_hot_cols)

    # 2.7. Function 7

    df_train_coded = apply_onehot(df_train, one_hot_cols, encoder_onehot)
    df_validation_coded = apply_onehot(df_validation, one_hot_cols, encoder_onehot)
    df_test_coded = apply_onehot(df_test, one_hot_cols, encoder_onehot)

    # 2.8. Function 8

    ordinal_cols = ["Relative_Volume_Category"]

    ordinal_categories = [["Very Low", "Low", "High", "Very High"]]

    encoder_ordinal = fit_ordinal(df_train, ordinal_cols, ordinal_categories)

    # 2.9. Function 9

    df_train_ordinal_coded = apply_ordinal(df_train_coded, ordinal_cols, encoder_ordinal)
    df_validation_ordinal_coded = apply_ordinal(df_validation_coded, ordinal_cols, encoder_ordinal)
    df_test_ordinal_coded = apply_ordinal(df_test_coded, ordinal_cols, encoder_ordinal)

    # 2.10. Function 10

    num_cols = df_train_ordinal_coded.select_dtypes(include = ["float64", "int64"]).columns

    df_train_scaled, df_validation_scaled, df_test_scaled, scaler_RS = scale_data(df_train_ordinal_coded,
                                                                                 df_validation_ordinal_coded,
                                                                                 df_test_ordinal_coded,
                                                                                 num_cols)

    # 3. Hidden Markov Model 


