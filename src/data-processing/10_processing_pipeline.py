# 10. Processing Pipeline script to clean and transform the project data

# In this file, the idea is to do the following:

#  - Load 1 file (data_enriched.csv) from the S3 bucket (in silver folder)
#  - Transform and process the data to be ready for modeling
#  - Prepare the script to be executed in an Airflow DAG (to be created in the next step)
#  - Upload the transformed dataframe back to the S3 bucket (in gold folder)


# Libraries

import os
import tempfile
import boto3
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

# ATENTION: ME FALTA CAMBIAR LOS NOMRBES DE LAS CATEGPRIAS DE "HMM_Regime_All" (los 3 estados)

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

        self.columns_to_divide_by_close = columns_to_divide_by_close
        self.columns_momentum_20 = columns_momentum_20
        self.columns_momentum_50 = columns_momentum_50
        self.columns_log_return = columns_log_return
        self.columns_pct_change = columns_pct_change
        self.columns_to_diff = columns_to_diff
        self.columns_relative_volume = columns_relative_volume
        self.relative_volume_window = relative_volume_window

    # Fit method