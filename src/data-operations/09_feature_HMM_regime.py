# 9. Add HMM regime as a feature to the consolidated dataframe

# In this file, the idea is to do the following:

#  - Load the consolidated dataframe from the S3 bucket (in silver folder)
#  - Pipeline (temporal) the data to ensure it is in the correct format for the HMM model
#  - Execute the Hidden Markov Model to identify market regimes (Gaussian HMM)
#  - Run one HMM with PCA-reduced features
#  - Run one HMM with all features (without PCA)
#  - Add the identified regimes as a new feature to the consolidated dataframe
#  - Persist the fitted objects (scaler, PCA, HMMs) for later inference
#  - Load the updated dataframe (original with new HMM regime feature) to the S3 bucket (silver folder)


# Libraries

import pandas as pd
import numpy as np
import boto3
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM
from dotenv import load_dotenv
import tempfile


# Important definitions

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_KEY_SILVER_UNIFIED = "silver/data_unified.csv"
S3_KEY_SILVER_ENRICHED_NEW = "silver/data_enriched.csv"

S3_KEY_ARTIFACTS = "models/hmm_artifacts.joblib"

FT_ENGINEERING_PATH = os.getenv("FT_ENGINEERING_CONTENT_PATH")


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket
# ------------------------------------------------------------------------------------------------

# Function to get consolidated data from S3 bucket

def get_consolidated_data(bucket_name = S3_BUCKET_NAME,
                          access_key_id = S3_ACCESS_KEY_ID,
                          secret_access_key = S3_SECRET_ACCESS_KEY,
                          key = S3_KEY_SILVER_UNIFIED):

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

    # Parse the date column to datetime (so the index is a real Datetime, not a string)

    df_process[index_column] = pd.to_datetime(df_process[index_column])

    # Index the dataframe by date and sort it

    df_process = df_process.set_index(index_column)
    df_process = df_process.sort_index()

    # Filter the dataframe by the specified date range

    df_process = df_process.loc[start_date:end_date]

    # Drop the invalid columns

    df_process = df_process.drop(columns = invalid_colums)

    return df_process


# Function 2: Feature Selection for HMM

def selected_features_for_HMM(df, features_to_select = []):

    # All left columns after dropping the invalid ones except the date.

    df_selected = df[features_to_select].copy()

    return df_selected


# Function 3: Imputation of missing values with forward fill method

def impute_num_ffill(df):

    df = df.copy()

    # Separate numerical

    num_cols = df.select_dtypes(include = ["float64", "int64"]).columns

    # Imputation for numerical columns with forward fill (backward-looking, no leakage)

    for col in num_cols:

        df[col] = df[col].ffill()

    # Return the dataframe with imputed values

    return df


# Function 4: Transform some features (if needed) for HMM

def transform_features_for_HMM(df):

    df = df.copy()

    # 1. DIVIDE BY CLOSE: deviation/band features expressed in price units -> make them relative

    cols_to_divide_by_close = ["Scaled_Upper_Bollinger", "Scaled_Lower_Bollinger", "Scaled_Upper_Keltner",
                               "Scaled_Lower_Keltner", "Scaled_SMA20", "Scaled_SMA50", "Scaled_EMA20",
                               "Scaled_EMA50", "Scaled_HMA20", "Scaled_HMA50"]

    for col in cols_to_divide_by_close:

        df[col] = df[col] / df["Close"]

    # 2. DIVIDE BY CLOSE WITH 20/50 LAGS: turn price-unit momentum into a relative return

    cols_to_divide_by_close_20lags = ["Momentum_20p"]
    cols_to_divide_by_close_50lags = ["Momentum_50p"]

    for col in cols_to_divide_by_close_20lags:

        df[col] = df[col] / df["Close"].shift(20)

    for col in cols_to_divide_by_close_50lags:

        df[col] = df[col] / df["Close"].shift(50)

    # 3. ANCHORED VWAP -> DISTANCE: relative distance of price to its anchored VWAP

    df["Anchored_VWAP_Distance"] = df["Close"] / df["Anchored_VWAP"] - 1

    df = df.drop(columns = ["Anchored_VWAP"])

    # 4. LOG-RETURNS: price-like / cumulative-level series

    cols_to_log_return = ["Close", "High", "Low", "Open",
                          "VIX_close", "GOLD_close", "OIL_close", "TLT_close", "RSP_close",
                          "IWM_close", "UUP_close", "MSTR_close", "SPY_close",
                          "CRYPTOCAP_BTC.D, 1D", "CRYPTOCAP_TOTAL, 1D", "CRYPTOCAP_TOTAL2, 1D",
                          "CRYPTOCAP_TOTALES, 1D", "CRYPTOCAP_USDT.D, 1D", "INDEX_BDI, 1D"]

    for col in cols_to_log_return:

        df[col] = np.log(df[col] / df[col].shift(1))

        df = df.rename(columns = {col: f"{col}_log_return"})

    # 5. PCT-CHANGE: monotonically growing levels (M2 money supply)

    cols_to_pct_change = ["FRED_WM2NS, 1D"]

    for col in cols_to_pct_change:

        df[col] = df[col].pct_change()

        df = df.rename(columns = {col: f"{col}_pct_change"})

    # 6. DIFF: yields / policy rates / cumulative OBV 

    cols_to_diff = ["TNX_close", "T5YIE", "EFFR", "Corporate_Bond_Spread", "OBV"]

    for col in cols_to_diff:

        df[col] = df[col].diff()

        df = df.rename(columns = {col: f"{col}_diff"})

    # 7. RELATIVE VOLUME: Volume grows structurally over the years, so log alone keeps a trend. 
    # Dividing by its 30-day rolling mean detrends it 

    cols_to_relative_volume = ["Volume"]

    for col in cols_to_relative_volume:

        df[col] = df[col] / df[col].rolling(30).mean()

        df = df.rename(columns = {col: f"{col}_rel"})

    # Return the dataframe with transformed features

    return df


# Function 5: Train / Validation / Test split for HMM (temporal, no shuffling)

def temporal_split_for_HMM(df, train_end = 0.8, validation_end = 0.9):

    # Length of the dataframe

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


# Function 6: Fit One-Hot Encoder on the training set and return the encoder

def fit_onehot(train_df, cols):

    encoder = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)

    encoder.fit(train_df[cols])

    return encoder


# Function 7: Apply the fitted OHE to a dataframe and return the transformed dataframe

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


# Function 8: Fit Ordinal Encoder on the training set and return the encoder

def fit_ordinal(train_df, cols, categories):

    encoder = OrdinalEncoder(categories = categories)

    encoder.fit(train_df[cols])

    return encoder


# Function 9: Apply the fitted Ordinal Encoder and return the transformed dataframe

def apply_ordinal(df, cols, encoder):

    df = df.copy()

    df[cols] = encoder.transform(df[cols])

    return df


# Function 10: Fit and apply Robust Scaler on the sets.
# Note: num_cols should contain only the CONTINUOUS columns

def scale_data(train_df, val_df, test_df, num_cols):

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler_RS = RobustScaler()

    train_df[num_cols] = scaler_RS.fit_transform(train_df[num_cols])
    val_df[num_cols] = scaler_RS.transform(val_df[num_cols])
    test_df[num_cols] = scaler_RS.transform(test_df[num_cols])

    return train_df, val_df, test_df, scaler_RS


# ------------------------------------------------------------------------------------------------
# Dimensionality Reduction with PCA
# ------------------------------------------------------------------------------------------------

def tuning_apply_PCA(train_df, val_df, test_df,
                     var_threshold = 90, path_to_save_plot = FT_ENGINEERING_PATH):

    pca = PCA()

    # Fit PCA on the training set only

    pca.fit(train_df)

    # Cumulative explained variance ratio (in %)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100

    n_components = np.arange(1, len(cumulative_var) + 1)

    fig, ax = plt.subplots(figsize = (10, 6))

    ax.plot(n_components, cumulative_var, marker = "o", markersize = 2, color = "black",
            linewidth = 2, label = "Cumulative Explained Variance (%)")

    threshold_colors = {90: "navy", 95: "darkgreen", 99: "orangered"}

    for i in [90, 95, 99]:

        n_comp_optimal = np.searchsorted(cumulative_var, i) + 1

        ax.axhline(y = i, linestyle = "--", linewidth = 1, color = threshold_colors[i],
                   label = f"{i}% -> {n_comp_optimal} components")

        ax.axvline(x = n_comp_optimal, linestyle = "--", linewidth = 1, color = threshold_colors[i])

    ax.set_xlabel("Number of Principal Components", fontsize = 12)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize = 12)
    ax.set_title("Model with PCA: Explained Variance Analysis for HMM Regime Detection", fontsize = 14)
    ax.set_xlim(0, len(n_components))
    ax.set_ylim(0, 102)
    ax.legend()
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save_plot, "model_with_PCA_exp_var_analysis.png"))
    plt.show()

    num_opt_components = int(np.searchsorted(cumulative_var, var_threshold) + 1)

    print(f"Number of components to retain {var_threshold}% of variance: {num_opt_components}")

    # Apply PCA with the optimal number of components, keeping a DataFrame with the original index

    pca_opt = PCA(n_components = num_opt_components, random_state = 42)

    cols = [f"PC{i + 1}" for i in range(num_opt_components)]

    train_pca = pd.DataFrame(pca_opt.fit_transform(train_df), index = train_df.index, columns = cols)
    val_pca = pd.DataFrame(pca_opt.transform(val_df), index = val_df.index, columns = cols)
    test_pca = pd.DataFrame(pca_opt.transform(test_df), index = test_df.index, columns = cols)

    return train_pca, val_pca, test_pca, pca_opt


# ------------------------------------------------------------------------------------------------
# HMM Model Fitting and Regime Identification
# ------------------------------------------------------------------------------------------------

# Relabel HMM states so that label order is stable and interpretable.
# States are sorted by their mean reference return (Close_log_return) on the TRAIN set:
# state 0 = lowest mean return (most bearish) ... state n-1 = highest mean return (most bullish).
# This removes the arbitrary label-switching of EM so "Regime_0" means the same thing across runs.

def _relabel_by_reference(train_regimes, val_regimes, test_regimes, ref_train, n_components):

    tmp = pd.DataFrame({"ref": np.asarray(ref_train), "state": np.asarray(train_regimes)})

    order = tmp.groupby("state")["ref"].mean().sort_values().index.tolist()

    # Make sure every possible state is covered (in case some state is absent in train)

    order = order + [s for s in range(n_components) if s not in order]

    remap = {int(old): new for new, old in enumerate(order)}

    relabel = np.vectorize(lambda s: remap[int(s)])

    return relabel(train_regimes), relabel(val_regimes), relabel(test_regimes), remap


def fit_HMM_identify_regimes(train_data, val_data, test_data,
                             ref_train, ref_val, ref_test,
                             n_components = 3, covariance_type = "diag",
                             n_iter = 1000, random_state = 42,
                             model_name = "model",
                             plot_filename = "HMM_regimes.png"):

    model_GHMM = GaussianHMM(n_components = n_components,
                             covariance_type = covariance_type,
                             n_iter = n_iter,
                             random_state = random_state)

    model_GHMM.fit(train_data)

    # Model selection info (helps justify n_components). Wrapped in try/except for older hmmlearn.

    try:

        print(f"    [{model_name}] AIC: {model_GHMM.aic(train_data):.2f} | "
              f"BIC: {model_GHMM.bic(train_data):.2f} | converged: {model_GHMM.monitor_.converged}")

    except Exception:

        print(f"    [{model_name}] converged: {model_GHMM.monitor_.converged}")

    # Predict the hidden states (regimes) for each set

    train_regimes = model_GHMM.predict(train_data)
    val_regimes = model_GHMM.predict(val_data)
    test_regimes = model_GHMM.predict(test_data)

    # Relabel states by mean reference return on train (stable, interpretable ordering)

    train_regimes, val_regimes, test_regimes, remap = _relabel_by_reference(
        train_regimes, val_regimes, test_regimes, ref_train, n_components
    )

    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()

    train_data["HMM_Regimes"] = train_regimes
    val_data["HMM_Regimes"] = val_regimes
    test_data["HMM_Regimes"] = test_regimes

    # Visualization: reference return (Close_log_return) over time, colored by regime, per split.

    color_map = {i: c for i, c in enumerate(["lightseagreen", "mediumvioletred", "gold",
                                             "royalblue", "tomato"])}

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 5))

    for ax, (name, ref, reg) in zip(
        axes,
        [("Train", ref_train, train_regimes),
         ("Validation", ref_val, val_regimes),
         ("Test", ref_test, test_regimes)]
    ):

        ax.scatter(ref.index, ref.values, c = pd.Series(reg).map(color_map).values, s = 10)

        ax.set_title(f"{model_name} | {name}: Close_log_return by HMM Regime", fontsize = 10)
        ax.set_xlabel("Date", fontsize = 8)
        ax.set_ylabel("Close_log_return (scaled)", fontsize = 8)
        ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FT_ENGINEERING_PATH, plot_filename))
    plt.show()

    return train_data, val_data, test_data, model_GHMM, remap


# ------------------------------------------------------------------------------------------------
# Persist fitted artifacts (scaler, PCA, HMMs) for later inference
# ------------------------------------------------------------------------------------------------

def save_fitted_artifacts(artifacts, local_dir = FT_ENGINEERING_PATH,
                          bucket_name = S3_BUCKET_NAME, access_key_id = S3_ACCESS_KEY_ID,
                          secret_access_key = S3_SECRET_ACCESS_KEY, key = S3_KEY_ARTIFACTS):

    local_path = os.path.join(local_dir, "hmm_artifacts.joblib")

    joblib.dump(artifacts, local_path)

    print(f"Saved fitted artifacts locally to: {local_path}")

    try:

        s3_client = boto3.client(
            "s3",
            aws_access_key_id = access_key_id,
            aws_secret_access_key = secret_access_key
        )

        s3_client.upload_file(local_path, bucket_name, key)

        print(f"Uploaded fitted artifacts to: s3://{bucket_name}/{key}")

    except Exception as e:

        print(f"Error uploading artifacts to S3: {e}")


# ------------------------------------------------------------------------------------------------
# Load the updated dataframe (original with new HMM regime feature) to the S3 bucket (silver folder)
# ------------------------------------------------------------------------------------------------

def load_updated_data_to_S3(df, bucket_name = S3_BUCKET_NAME, access_key_id = S3_ACCESS_KEY_ID,
                            secret_access_key = S3_SECRET_ACCESS_KEY, key = S3_KEY_SILVER_ENRICHED_NEW):

    # Create the S3 client

    s3_client = boto3.client(
        "s3",
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key
    )

    # Save the dataframe to a temporary CSV file

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    try:

        df.to_csv(temp_file_path, index = False)

        # Upload the file to S3

        s3_client.upload_file(temp_file_path, bucket_name, key)

        print(f"Uploaded updated data to S3 bucket: s3://{bucket_name}/{key}")

    except Exception as e:

        print(f"Error uploading file to S3: {e}")

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)


# ------------------------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Load the consolidated dataframe from the S3 bucket (in silver folder)

    df_consolidated = get_consolidated_data()

    # 2. PIPELINE

    # 2.1. Drop invalid columns + index by date + filter range

    cols_to_drop = ["Unnamed: 0_x", "Unnamed: 0_y", "Future_Return"]

    df = preprocess_data_for_HMM(df_consolidated, invalid_colums = cols_to_drop)

    # 2.2. Feature selection (everything except the Target; "date" is already the index)

    cols_to_select = [col for col in df.columns if col != "Target"]

    df = selected_features_for_HMM(df, features_to_select = cols_to_select)

    # 2.3. Numerical imputation (forward fill)

    df = impute_num_ffill(df)

    # 2.4. Feature transformations

    df = transform_features_for_HMM(df)

    # 2.5. Drop rows with NaN created by shift/diff/rolling and the few categorical NaNs

    df = df.dropna()

    df_train, df_validation, df_test = temporal_split_for_HMM(df)

    # 2.6. One-Hot encoding (fit on train only)

    one_hot_cols = ["Day_Type", "Weekly_Breakout"]

    encoder_onehot = fit_onehot(df_train, one_hot_cols)

    # 2.7. Apply OHE to all splits

    df_train_coded = apply_onehot(df_train, one_hot_cols, encoder_onehot)
    df_validation_coded = apply_onehot(df_validation, one_hot_cols, encoder_onehot)
    df_test_coded = apply_onehot(df_test, one_hot_cols, encoder_onehot)

    # 2.8. Ordinal encoding (fit on train only)

    ordinal_cols = ["Relative_Volume_Category"]

    ordinal_categories = [["Very Low", "Low", "High", "Very High"]]

    encoder_ordinal = fit_ordinal(df_train, ordinal_cols, ordinal_categories)

    # 2.9. Apply ordinal encoding to all splits

    df_train_ordinal_coded = apply_ordinal(df_train_coded, ordinal_cols, encoder_ordinal)
    df_validation_ordinal_coded = apply_ordinal(df_validation_coded, ordinal_cols, encoder_ordinal)
    df_test_ordinal_coded = apply_ordinal(df_test_coded, ordinal_cols, encoder_ordinal)

    # 2.10. Scaling: scale ONLY the continuous columns. One-hot dummies stay as 0/1.

    onehot_dummy_cols = list(encoder_onehot.get_feature_names_out(one_hot_cols))

    num_cols = [
        col for col in df_train_ordinal_coded.select_dtypes(include = ["float64", "int64"]).columns
        if col not in onehot_dummy_cols
    ]

    df_train_scaled, df_validation_scaled, df_test_scaled, scaler_RS = scale_data(
        df_train_ordinal_coded,
        df_validation_ordinal_coded,
        df_test_ordinal_coded,
        num_cols
    )

    # Reference return series per split (used for stable regime relabeling and for plotting)

    ref_train = df_train_scaled["Close_log_return"].copy()
    ref_val = df_validation_scaled["Close_log_return"].copy()
    ref_test = df_test_scaled["Close_log_return"].copy()

    # 3. PCA (Model 1) vs all features (Model 2)

    # Model 1: PCA-reduced features

    df_train_pca, df_validation_pca, df_test_pca, pca_opt = tuning_apply_PCA(
        df_train_scaled, df_validation_scaled, df_test_scaled
    )

    # Model 2: all features (without PCA)

    df_train_all, df_validation_all, df_test_all = df_train_scaled, df_validation_scaled, df_test_scaled

    # 4. Fit HMM models and identify regimes

    # Model 1: PCA-reduced features

    df_train_pca_regimes, df_validation_pca_regimes, df_test_pca_regimes, model_GHMM_pca, remap_pca = \
        fit_HMM_identify_regimes(
            df_train_pca, df_validation_pca, df_test_pca,
            ref_train, ref_val, ref_test,
            model_name = "PCA",
            plot_filename = "HMM_regimes_PCA.png"
        )

    # Model 2: all features (without PCA)

    df_train_all_regimes, df_validation_all_regimes, df_test_all_regimes, model_GHMM_all, remap_all = \
        fit_HMM_identify_regimes(
            df_train_all, df_validation_all, df_test_all,
            ref_train, ref_val, ref_test,
            model_name = "All_Features",
            plot_filename = "HMM_regimes_ALL.png"
        )

    # Protect the index of the original consolidated dataframe before concatenation (align by date)

    df_consolidated["date"] = pd.to_datetime(df_consolidated["date"])
    df_consolidated = df_consolidated.set_index("date").sort_index()

    # Add the identified regimes as new features. Labels are now ordered by mean return:
    # Regime_0 = most bearish ... Regime_2 = most bullish.
    # Change the regime labels in All_Features because it captures better the regime differences (view EDA post HMM)

    state_mapping_pca = {0: "PCA_Regime_0", 1: "PCA_Regime_1", 2: "PCA_Regime_2"}
    state_mapping_all = {0: "Volatility:Expansion_Regime", 1: "Consolidation_Regime", 2: "Bullish_Regime"}

    regimes_pca = pd.concat([df_train_pca_regimes["HMM_Regimes"],
                             df_validation_pca_regimes["HMM_Regimes"],
                             df_test_pca_regimes["HMM_Regimes"]]).map(state_mapping_pca)

    regimes_all = pd.concat([df_train_all_regimes["HMM_Regimes"],
                             df_validation_all_regimes["HMM_Regimes"],
                             df_test_all_regimes["HMM_Regimes"]]).map(state_mapping_all)

    # reindex onto the full consolidated index; rows outside the HMM coverage stay NaN

    df_consolidated["HMM_Regime_PCA"] = regimes_pca.reindex(df_consolidated.index)
    df_consolidated["HMM_Regime_All"] = regimes_all.reindex(df_consolidated.index)

    # Bring "date" back as a column so it is not lost when saving with index=False

    df_consolidated = df_consolidated.reset_index()

    # 5. Persist fitted objects for inference (scaler, PCA, both HMMs, relabel maps)

    artifacts = {
        "scaler": scaler_RS,
        "pca": pca_opt,
        "hmm_pca": model_GHMM_pca,
        "hmm_all": model_GHMM_all,
        "remap_pca": remap_pca,
        "remap_all": remap_all,
        "onehot_encoder": encoder_onehot,
        "ordinal_encoder": encoder_ordinal,
        "scaled_num_cols": num_cols,
    }

    save_fitted_artifacts(artifacts)

    # 6. Load the updated dataframe (original + HMM regime features) to the S3 bucket

    load_updated_data_to_S3(df_consolidated)

    print("HMM regime features added to the consolidated dataframe and uploaded to S3 bucket successfully.")