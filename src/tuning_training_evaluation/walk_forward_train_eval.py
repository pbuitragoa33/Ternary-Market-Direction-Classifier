# Walk-forward training and evaluation Weights & Biases

# This file aims to:

#   - Train the models with the best hyperparameters found in the tuning phase
#   - Evaluate every model with walk-forward validation:
#       * rolling window of 3 years train -> next 6 months test, sliding forward by 6 months
#       * per fold: F1-macro, accuracy, per-class precision/recall/F1, train-vs-test gap,
#         and (when the model exposes predict_proba) macro ROC-AUC / PR-AUC / log-loss / Brier
#       * compared against the trivial "always Neutral" baseline on each test window
#  - Collect every fold scalar metrics (csv + json + W&B curves), but only upload 4 folds to wandb (first, last, best F1, worst F1)


# Another considerations:

# - The overfittinf signal is the gap between train_F1 - test_F1 per fold
# - The gold data was scaled once with RobustScaler fit on the original
#   training period, so folds whose test window lies inside thta period
#   see slightly-future scaling statistics (mild, median, IQR)


# Libraries

import os
import json
import tempfile
 
import boto3
import wandb
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from dotenv import load_dotenv
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from pandas.tseries.offsets import DateOffset
 
from sklearn.utils import class_weight
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve, auc,
                             precision_recall_curve, log_loss, confusion_matrix)
 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, StackingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# ----------------------------------------------------------------------------------------------------
# Important definitions
# ----------------------------------------------------------------------------------------------------
 
load_dotenv()
 
TUNING_PATH = os.getenv("TUNING_PATH", ".")
WALKFORWARD_PATH = os.getenv("WALKFORWARD_PATH", TUNING_PATH)
DASHBOARD_DIR = os.path.join(WALKFORWARD_PATH, "fold_dashboards")
 
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
 
GOLD_FILES = {
    "X_train": "gold/X_train.csv", "y_train": "gold/y_train.csv",
    "X_validation": "gold/X_validation.csv", "y_validation": "gold/y_validation.csv",
    "X_test": "gold/X_test.csv", "y_test": "gold/y_test.csv",
}
 
# Weights & Biases
 
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "TMDC_walkforward")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_MODE = os.getenv("WANDB_MODE", "online")
 
# Walk-forward window configuration
 
TRAIN_YEARS = 3
TEST_MONTHS = 6
STEP_MONTHS = 6
 
# N_WANDB_DASHBOARDS: Number of dashboards per model to be uploaded
# SAVE_ALL_FOLD_DASHBOARDS: False, to save only the 4 dashboardss

N_WANDB_DASHBOARDS = 4
SAVE_ALL_FOLD_DASHBOARDS = False
 
# Class labels and balancing policy 
 
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
CUSTOM_WEIGHTS = {0: 2, 1: 1, 2: 1}
 
MODELS_TO_RUN = ["lightgbm", "adaboost", "decision_tree", "knn",
                 "random_forest", "xgboost", "stacking", "voting_hard", "gradient_boosting"]


# ------------------------------------------------------------------------------------------------
# Data loading from S3 gold layer
# ------------------------------------------------------------------------------------------------
 
def download_gold_data(s3_key):
 
    s3 = boto3.client("s3", aws_access_key_id = S3_ACCESS_KEY_ID, aws_secret_access_key = S3_SECRET_ACCESS_KEY)
 
    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:
 
        path = tmp.name
 
    try:
 
        s3.download_file(S3_BUCKET_NAME, s3_key, path)
 
        return pd.read_csv(path, index_col = 0)
 
    finally:
 
        if os.path.exists(path):
 
            os.remove(path)


# ------------------------------------------------------------------------------------------------
# Merge the datasets to make the walkforward splits
# ------------------------------------------------------------------------------------------------

def get_full_dataset():

    X_fractions, y_fractions = [], []

    for split in ["train", "validation", "test"]:

        X_fractions.append(download_gold_data(GOLD_FILES[f"X_{split}"]))
        y_fractions.append(download_gold_data(GOLD_FILES[f"y_{split}"]))

    X_all = pd.concat(X_fractions)
    y_all = pd.concat(y_fractions)

    # Index to datatime and sort them

    X_all.index = pd.to_datetime(X_all.index)
    y_all.index = pd.to_datetime(y_all.index)

    X_all = X_all.sort_index()
    y_all = y_all.sort_index()

    # Constrain if there is a mismatch

    if not X_all.index.equals(y_all.index):

        raise ValueError("Index mismatch between X and y afeter concatenation")
    
    y_all = y_all.iloc[:, 0].astype(int)

    return X_all, y_all


# ------------------------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------------------------
 
# Function to resolve the class_weight parameter

def resolve_cw(value):
 
    return {"balanced": "balanced", "none": None, "custom": CUSTOM_WEIGHTS}.get(value, value)


# Function to extract the values of hyperparameters from the json file

def load_best_params(model_name):

    with open(os.path.join(TUNING_PATH, f"best_params_{model_name}.json")) as f:

        return json.load(f)["best_params"]
    

# ------------------------------------------------------------------------------------------------
# Build models with the best hyperparameters
# ------------------------------------------------------------------------------------------------
 
def build_model_from_best_params(model_name, p):

    # Return (estimator, needs_sample_weight)

    # XGBoost

    if model_name ==  "xgboost":

        return XGBClassifier(
            n_estimators = p["n_estimators"],
            max_depth = p["max_depth"],
            learning_rate = p["learning_rate"],
            subsample = p["subsample"],
            colsample_bytree = p["colsample_bytree"],
            booster = p["booster"],
            reg_lambda = p["reg_lambda"],
            eval_metric = "mlogloss",
            random_state = 42, verbosity = 0
        ), True
    
    # LightGBM

    if model_name == "lightgbm":

        return LGBMClassifier(
            n_estimators = p["n_estimators"],
            max_depth = p["max_depth"],
            learning_rate = p["learning_rate"],
            subsample = p["subsample"],
            num_leaves = p["num_leaves"],
            colsample_bytree = p["colsample_bytree"],
            class_weight = resolve_cw(p.get("class_weight")),
            random_state = 42, verbose = -1
        ), False
    
    # AdaBoost

    if model_name == "adaboost":
 
        if p["base_estimator"] == "decision_tree":
 
            base = DecisionTreeClassifier(max_depth = p["base_dt_max_depth"], random_state = 42)
 
        else:
 
            base = LogisticRegression(max_iter = 1000)
 
        return AdaBoostClassifier(estimator = base, n_estimators = p["n_estimators"],
                                  learning_rate = p["learning_rate"], random_state = 42), False
 
    # Stacking
    
    if model_name == "stacking":
 
        if p["base_combo"] == "ada_rf":
 
            estimators = [("ada", AdaBoostClassifier(random_state = 42)),
                          ("rf", RandomForestClassifier(random_state = 42))]
 
        else:
 
            estimators = [("gb", GradientBoostingClassifier(random_state = 42)),
                          ("dt", DecisionTreeClassifier(random_state = 42))]
 
        final = LogisticRegression(max_iter = 1000) if p["final_estimator"] == "logistic_regression" else DecisionTreeClassifier(random_state = 42)
 
        return StackingClassifier(estimators = estimators, final_estimator = final), False
    
    # Random Forest

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators = p["n_estimators"],
            criterion = p["criterion"],
            max_depth = p["max_depth"],
            min_samples_leaf = p["min_samples_leaf"],
            min_samples_split = p["min_samples_split"],
            class_weight = resolve_cw(p.get("class_weight")),
            random_state = 42
        ), False
    
    # Gradient Boosting

    if model_name == "gradient_boosting":

        return GradientBoostingClassifier(
            n_estimators = p["n_estimators"],
            learning_rate = p["learning_rate"],
            loss = p["log_loss"],
            max_depth = p["max_depth"],
            min_samples_leaf = p["min_samples_leaf"], 
            random_state = 42
        ), True
    
    # Decision Tree

    if model_name == "decision_tree":

        return DecisionTreeClassifier(
            criterion = p["criterion"],
            max_depth = p["max_depth"],
            min_samples_leaf = p["min_samples_leaf"],
            min_samples_split = p["min_samples_split"],
            class_weight = resolve_cw(p.get("class_weight")),
            random_state = 42
        ), False
    
    # Voting Hard

    if model_name == "voting_hard":
 
        if p["base_estimator"] == "decision_tree":
 
            base = DecisionTreeClassifier(max_depth = p["base_dt_max_depth"], random_state = 42)
 
        elif p["base_estimator"] == "neural_network":
 
            hidden = {"50": (50,), "100": (100,), "50_50": (50, 50)}[p["nn_hidden_layers"]]
            base = MLPClassifier(max_iter = 1000, hidden_layer_sizes = hidden, random_state = 42)
 
        else:
 
            base = LogisticRegression(max_iter = 1000)
 
        return VotingClassifier(estimators = [("base", base)], voting = "hard"), False
    
    # KNN

    if model_name == "knn":

        return KNeighborsClassifier(
            n_neighbors = p["n_neighbors"],
            weights = p["weights"],
            algorithm = p["algorithm"]
        ), False

    raise ValueError(f"Unknown model name: {model_name}")

 
# ------------------------------------------------------------------------------------------------
# Probability helpers (multiclass, robust to a class missing in a fold)
# --------------------------------------------------------------------------------------------

def full_probabilities(model, X, labels = LABELS):

    """
    Return an (n, 3) probabilioty array aligned to labels,
    even if the model never saw a class
    """

    # Get the predicted probabilities foe the classes

    proba = model.predict_proba(X)

    # Array full of zeros with shape (n, 3) to hold the aligned probabilities

    full = np.zeros((proba.shape[0], len(labels)))

    for j, c in enumerate(model.classes_):

        if c in labels:

            full[:, labels.index(c)] = proba[:, j]

    # Renormalize to sum to 1

    row_sums = full.sum(axis = 1, keepdims = True)
    row_sums[row_sums == 0] = 1.0
    full = full / row_sums

    return full


def proba_metrics(y_true, proba):

    """
    Macro one vs rest ROC-AUC and PR-AUC
    Multiclass logloss and Brier socore
    Nan-safe
    """

    y_oh = label_binarize(y_true, classes = LABELS)
 
    aucs, aps = [], []
 
    for k in range(len(LABELS)):
 
        yk = y_oh[:, k]
        pos = yk.sum()
 
        if pos == 0 or pos == len(yk):
 
            aucs.append(np.nan)
            aps.append(np.nan)
 
        else:
 
            try:
 
                aucs.append(roc_auc_score(yk, proba[:, k]))
 
            except Exception:
 
                aucs.append(np.nan)
 
            try:
 
                aps.append(average_precision_score(yk, proba[:, k]))
 
            except Exception:
 
                aps.append(np.nan)
 
    try:
 
        ll = log_loss(y_true, proba, labels = LABELS)
 
    except Exception:
 
        ll = np.nan
 
    brier = float(np.mean(np.sum((proba - y_oh) ** 2, axis = 1)))
 
    return {
        "roc_auc_macro": float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else np.nan,
        "pr_auc_macro": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else np.nan,
        "log_loss": float(ll) if ll == ll else np.nan,
        "brier_multiclass": brier,
    }


# ------------------------------------------------------------------------------------------------
# Per-fold Dashboard (3-class: ROC OvR, PR OvR, calibration OvR, confusion matrix)
# ------------------------------------------------------------------------------------------------

def build_fold_dashboard(art, model_name, fold_id, save_path):

    """
    art = {'y_true', 'y_pred', 'proba' (or None), 'test_range'}
    """

    y_true = art["y_true"]
    y_pred = art["y_pred"]
    proba = art["proba"]

    # Confusion Matriz

    cm = confusion_matrix(y_true, y_pred, labels = LABELS, normalize = "true")
    cm = np.nan_to_num(cm)
    names = [LABEL_NAMES[l] for l in LABELS]

    fig, axes = plt.subplots(2, 2, figsize = (14, 12))

    fig.suptitle(f"{model_name} - fold {fold_id} - ({art['test_range']})", fontsize = 15)

    if proba is not None:
 
        y_oh = label_binarize(y_true, classes = LABELS)
 
        # 1. ROC one-vs-rest (one curve per class)
 
        for k, l in enumerate(LABELS):
 
            if 0 < y_oh[:, k].sum() < len(y_oh):
 
                fpr, tpr, _ = roc_curve(y_oh[:, k], proba[:, k])
                axes[0, 0].plot(fpr, tpr, lw = 2, label = f"{LABEL_NAMES[l]} (AUC {auc(fpr, tpr):.2f})")
 
        axes[0, 0].plot([0, 1], [0, 1], "k--", alpha = 0.4)
        axes[0, 0].set_title("ROC One-vs-Rest")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].legend(loc = "lower right")
        axes[0, 0].grid(alpha = 0.3)
 
        # 2. Precision-Recall one-vs-rest
 
        for k, l in enumerate(LABELS):
 
            if 0 < y_oh[:, k].sum() < len(y_oh):
 
                prec, rec, _ = precision_recall_curve(y_oh[:, k], proba[:, k])
                axes[0, 1].plot(rec, prec, lw = 2, label = f"{LABEL_NAMES[l]} (AP {auc(rec, prec):.2f})")
 
        axes[0, 1].set_title("Precision-Recall One-vs-Rest")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].legend(loc = "lower left")
        axes[0, 1].grid(alpha = 0.3)
 
        # 3. Calibration one-vs-rest
 
        for k, l in enumerate(LABELS):
 
            if y_oh[:, k].sum() >= 10:
 
                try:
 
                    pt, pp = calibration_curve(y_oh[:, k], proba[:, k], n_bins = 10)
                    axes[1, 0].plot(pp, pt, "s-", lw = 1.5, label = LABEL_NAMES[l])
 
                except Exception:
 
                    pass
 
        axes[1, 0].plot([0, 1], [0, 1], "k:", label = "Perfect")
        axes[1, 0].set_title("Calibration (Reliability) One-vs-Rest")
        axes[1, 0].set_xlabel("Mean Predicted Probability")
        axes[1, 0].set_ylabel("Fraction of Positives")
        axes[1, 0].legend(loc = "lower right")
        axes[1, 0].grid(alpha = 0.3)
 
    else:
 
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
 
            ax.text(0.5, 0.5, "Without predict_proba\n(hard voting)", ha = "center", va = "center", fontsize = 12)
            ax.axis("off")
 
    # 4. Normalized confusion matrix (recall per row)
 
    sns.heatmap(cm, annot = True, fmt = ".2f", cmap = "viridis", vmin = 0, vmax = 1,
                xticklabels = names, yticklabels = names, ax = axes[1, 1])
    axes[1, 1].set_title("Confusion Matrix (normalized by class)")
    axes[1, 1].set_ylabel("Real Class")
    axes[1, 1].set_xlabel("Prediction")
 
    plt.tight_layout(rect = [0, 0, 1, 0.97])
    fig.savefig(save_path, dpi = 120, bbox_inches = "tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Walk-forward fold generation
# ------------------------------------------------------------------------------------------------

def generate_folds(index, train_years = TRAIN_YEARS, test_months = TEST_MONTHS, step_months = STEP_MONTHS):

    folds = []
    start = index.min()
    data_end = index.max()

    while True:

        train_start = start
        train_end = train_start + DateOffset(years = train_years)
        test_end = train_end + DateOffset(months = test_months)

        if train_end > data_end:

            break

        folds.append((train_start, train_end, min(test_end, data_end + DateOffset(days = 1))))

        if test_end > data_end:

            break

        start = start + DateOffset(months = step_months)

    return folds


# ------------------------------------------------------------------------------------------------
# Evaluate one model across all walk-forward folds
# ------------------------------------------------------------------------------------------------
 
def evaluate_walkforward(model_name, best_params, X_all, y_all, folds):
 
    rows = []
    artifacts = {}     # fold_id -> {y_true, y_pred, proba, test_range} for dashboards
 
    for i, (tr_start, tr_end, te_end) in enumerate(folds):
 
        tr_mask = (X_all.index >= tr_start) & (X_all.index < tr_end)
        te_mask = (X_all.index >= tr_end) & (X_all.index < te_end)
 
        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
 
        if len(X_te) < 20 or y_tr.nunique() < 2:
 
            print(f"    fold {i}: skipped (n_test={len(X_te)}, train_classes={y_tr.nunique()})")

            continue
 
        model, needs_sw = build_model_from_best_params(model_name, best_params)
 
        fit_kwargs = {}
 
        if needs_sw:
 
            fit_kwargs["sample_weight"] = class_weight.compute_sample_weight(CUSTOM_WEIGHTS, y_tr.values)
 
        try:
 
            model.fit(X_tr, y_tr.values, **fit_kwargs)
            y_pred = model.predict(X_te)
            y_tr_pred = model.predict(X_tr)
 
        except Exception as e:
 
            print(f"    fold {i}: model error -> {str(e)[:120]}")

            continue
 
        # Hard-label metrics
 
        f1m = f1_score(y_te, y_pred, average = "macro", labels = LABELS, zero_division = 0)
        acc = accuracy_score(y_te, y_pred)
        train_f1 = f1_score(y_tr, y_tr_pred, average = "macro", labels = LABELS, zero_division = 0)
 
        prec = precision_score(y_te, y_pred, average = None, labels = LABELS, zero_division = 0)
        rec = recall_score(y_te, y_pred, average = None, labels = LABELS, zero_division = 0)
        f1c = f1_score(y_te, y_pred, average = None, labels = LABELS, zero_division = 0)
 
        # Probabilistic metrics (only if the model exposes predict_proba)
 
        proba = None
        pm = {"roc_auc_macro": np.nan, "pr_auc_macro": np.nan, "log_loss": np.nan, "brier_multiclass": np.nan}
 
        if hasattr(model, "predict_proba"):
 
            try:
 
                proba = full_probabilities(model, X_te)
                pm = proba_metrics(y_te.values, proba)
 
            except Exception as e:
 
                print(f"    fold {i}: proba metrics failed -> {str(e)[:90]}")
                proba = None
 
        # Baseline: always predict Neutral (label 1) on this test window
 
        base_pred = np.ones(len(y_te), dtype = int)
        base_f1 = f1_score(y_te, base_pred, average = "macro", labels = LABELS, zero_division = 0)
        base_acc = accuracy_score(y_te, base_pred)
 
        rows.append({
            "fold": i,
            "train_start": str(tr_start.date()), "test_start": str(tr_end.date()),
            "test_end": str(te_end.date()),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "f1_macro": round(float(f1m), 4), "accuracy": round(float(acc), 4),
            "train_f1_macro": round(float(train_f1), 4),
            "overfit_gap": round(float(train_f1 - f1m), 4),
            "precision_bearish": round(float(prec[0]), 4),
            "precision_neutral": round(float(prec[1]), 4),
            "precision_bullish": round(float(prec[2]), 4),
            "recall_bearish": round(float(rec[0]), 4),
            "recall_neutral": round(float(rec[1]), 4),
            "recall_bullish": round(float(rec[2]), 4),
            "f1_bearish": round(float(f1c[0]), 4),
            "f1_neutral": round(float(f1c[1]), 4),
            "f1_bullish": round(float(f1c[2]), 4),
            "roc_auc_macro": round(pm["roc_auc_macro"], 4) if pm["roc_auc_macro"] == pm["roc_auc_macro"] else np.nan,
            "pr_auc_macro": round(pm["pr_auc_macro"], 4) if pm["pr_auc_macro"] == pm["pr_auc_macro"] else np.nan,
            "log_loss": round(pm["log_loss"], 4) if pm["log_loss"] == pm["log_loss"] else np.nan,
            "brier_multiclass": round(pm["brier_multiclass"], 4) if pm["brier_multiclass"] == pm["brier_multiclass"] else np.nan,
            "baseline_f1_macro": round(float(base_f1), 4),
            "baseline_accuracy": round(float(base_acc), 4),
            "f1_lift_vs_baseline": round(float(f1m - base_f1), 4),
        })
 
        artifacts[i] = {"y_true": y_te.values, "y_pred": y_pred, "proba": proba,
                        "test_range": f"{tr_end.date()} -> {te_end.date()}"}
 
    return pd.DataFrame(rows), artifacts


# ------------------------------------------------------------------------------------------------
# Dashboard folds
# ------------------------------------------------------------------------------------------------

# Select the folds to be uploaded

def select_dashboard_folds(folds_df, k = N_WANDB_DASHBOARDS):
  
    ids = [int(folds_df["fold"].iloc[0]), int(folds_df["fold"].iloc[-1]),
           int(folds_df.loc[folds_df["f1_macro"].idxmax(), "fold"]),
           int(folds_df.loc[folds_df["f1_macro"].idxmin(), "fold"])]
 
    seen, out = set(), []
 
    for i in ids:
 
        if i not in seen:
 
            seen.add(i)
            out.append(i)
 
    return out[:k]
 

# Summarize the folds with stats of the metrics
 
def summarize(folds_df):
 
    if len(folds_df) == 0:
 
        return {}
 
    def stats(col):
 
        s = folds_df[col].dropna()
 
        if len(s) == 0:
 
            return {"mean": None, "std": None, "min": None, "max": None}
 
        return {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4), "max": round(float(s.max()), 4)}
 
    return {
        "n_folds": int(len(folds_df)),
        "f1_macro": stats("f1_macro"),
        "accuracy": stats("accuracy"),
        "overfit_gap": stats("overfit_gap"),
        "recall_bearish": stats("recall_bearish"),
        "recall_neutral": stats("recall_neutral"),
        "recall_bullish": stats("recall_bullish"),
        "roc_auc_macro": stats("roc_auc_macro"),
        "pr_auc_macro": stats("pr_auc_macro"),
        "log_loss": stats("log_loss"),
        "f1_lift_vs_baseline": stats("f1_lift_vs_baseline"),
        "pct_folds_beating_baseline": round(float((folds_df["f1_lift_vs_baseline"] > 0).mean()), 4),
    }


# Veridict of stability

def stability_verdict(mean_gap):
 
    if mean_gap is None:
 
        return "n/a"
 
    if mean_gap < 0.05:
 
        return "VERY STABLE"
 
    if mean_gap < 0.10:
 
        return "MODERATELY STABLE"
 
    return "UNSTABLE (overfit)"



# ------------------------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    print("Walk-forward validation...")

    X_all, y_all = get_full_dataset()

    print(f"Full dataset: {X_all.shape[0]} rows | ({X_all.index.min().date()} → {X_all.index.max().date()})")
    print(f"Class balance: {y_all.value_counts(normalize = True).round(3).to_dict()}")

    folds = generate_folds(X_all.index)

    print(f"Generated {len(folds)} walk-forward folds")
    print(f"{TRAIN_YEARS}y train | {TEST_MONTHS}m test, sliding {STEP_MONTHS}m")

    os.makedirs(WALKFORWARD_PATH, exist_ok = True)
    os.makedirs(DASHBOARD_DIR, exist_ok = True)

    all_summaries = []
    f1_distributions = {}

    for model_name in MODELS_TO_RUN:

        print(f"\n======================== Walk-forward: {model_name} ========================")

        best_params = load_best_params(model_name)

        folds_df, artifacts = evaluate_walkforward(model_name, best_params, X_all, y_all, folds)

        if len(folds_df) == 0:

            print(f"No valid folds for {model_name}, skipping...")

            continue

        summary = summarize(folds_df)
        summary["model"] = model_name
        all_summaries.append(summary)
        f1_distributions[model_name] = folds_df["f1_macro"].tolist()

        verdict = stability_verdict(summary["overfit_gap"]["mean"])

        print(f"    F1-macro: {summary['f1_macro']['mean']} +/- {summary['f1_macro']['std']} "
              f"(min {summary['f1_macro']['min']}, max {summary['f1_macro']['max']})")
        print(f"    beats always-Neutral baseline in {summary['pct_folds_beating_baseline']:.0%} of folds")
        print(f"    Bearish recall (mean): {summary['recall_bearish']['mean']} | "
              f"ROC-AUC macro (mean): {summary['roc_auc_macro']['mean']}")
        print(f"    overfit gap (mean train-test): {summary['overfit_gap']['mean']} -> {verdict}")


        # Save ALL folds locally (full per-fold collection)
 
        folds_df.to_csv(os.path.join(WALKFORWARD_PATH, f"walkforward_{model_name}.csv"), index = False)

        # Render dashboards: the selected 4 for W&B (+ optionally all 16 locally)
 
        selected = select_dashboard_folds(folds_df)
        to_render = set(selected)
 
        if SAVE_ALL_FOLD_DASHBOARDS:
 
            to_render = set(artifacts.keys())
 
        dashboard_paths = {}
 
        for fid in sorted(to_render):
 
            if fid in artifacts:
 
                path = os.path.join(DASHBOARD_DIR, f"{model_name}_fold{fid}.png")
                build_fold_dashboard(artifacts[fid], model_name, fid, path)
                dashboard_paths[fid] = path
 
        # W&B tracking → one run per mdoel
 
        run = wandb.init(project = WANDB_PROJECT, entity = WANDB_ENTITY, mode = WANDB_MODE,
                         group = model_name, job_type = "walkforward", name = model_name, reinit = True,
                         config = {"train_years": TRAIN_YEARS, "test_months": TEST_MONTHS, "step_months": STEP_MONTHS, "best_params": best_params},
                         settings = wandb.Settings(silent = True))
 
        # All folds' scalar metrics as a curve
 
        for _, r in folds_df.iterrows():
 
            run.log({k: (None if (isinstance(v, float) and v != v) else v)
                     for k, v in r.items() if k not in ("train_start", "test_start", "test_end")})
 
        # All folds in one Table (so nothing is lost), plus only the 4 selected images
 
        run.log({"folds_table": wandb.Table(dataframe = folds_df)})
 
        for fid in selected:
 
            if fid in dashboard_paths:
 
                run.log({f"dashboard_fold_{fid}": wandb.Image(dashboard_paths[fid])})
 
        for stat in ["f1_macro", "accuracy", "overfit_gap", "recall_bearish", "roc_auc_macro", "pr_auc_macro"]:
 
            if summary[stat]["mean"] is not None:
 
                run.summary[f"{stat}_mean"] = summary[stat]["mean"]
 
        run.summary["pct_folds_beating_baseline"] = summary["pct_folds_beating_baseline"]
        run.summary["stability_verdict"] = verdict
        run.summary["dashboards_uploaded"] = selected
 
        run.finish()
 
    # Save the cross-model summary
 
    summary_path = os.path.join(WALKFORWARD_PATH, "walkforward_summary.json")
 
    with open(summary_path, "w") as f:
 
        json.dump(all_summaries, f, indent = 2)
 
    # Comparison boxplot of F1-macro distributions across folds
 
    if f1_distributions:
 
        order = sorted(f1_distributions, key = lambda m: np.mean(f1_distributions[m]), reverse = True)
 
        fig, ax = plt.subplots(figsize = (11, 6))
        ax.boxplot([f1_distributions[m] for m in order], labels = order, showmeans = True)
        ax.axhline(0.33, color = "gray", linestyle = "--", linewidth = 1, label = "≈ uniform-random (0.33)")
        ax.axhline(0.19, color = "crimson", linestyle = ":", linewidth = 1, label = "always-Neutral (0.19)")
        ax.set_ylabel("F1-macro (por fold)")
        ax.set_title(f"Walk-forward: F1-macro distributions ({TRAIN_YEARS}y/{TEST_MONTHS}m)")
        ax.tick_params(axis = "x", rotation = 45)
        ax.legend()
        ax.grid(True, alpha = 0.3)
        fig.tight_layout()
 
        plot_path = os.path.join(WALKFORWARD_PATH, "walkforward_f1_boxplot.png")
        fig.savefig(plot_path, dpi = 130, bbox_inches = "tight")
        plt.close(fig)
 
        print(f"\nSaved comparison plot: {plot_path}")
 
    print("\n" + "=" * 78)
    print("Walk-forward ranking (mean F1-macro across folds):")
 
    all_summaries.sort(key = lambda s: (s["f1_macro"]["mean"] if s["f1_macro"]["mean"] is not None else -1),
                       reverse = True)
 
    for s in all_summaries:
 
        print(f" * {s['model']:<18} F1 = {s['f1_macro']['mean']:.4f} +/- {s['f1_macro']['std']:.4f}  "
              f"| beats baseline {s['pct_folds_beating_baseline']:.0%}  "
              f"| Bearish rec {s['recall_bearish']['mean']:.3f}  "
              f"| gap {s['overfit_gap']['mean']:.3f}")
 
    print("=" * 78)
    print(f"Saved summary: {summary_path}")
    print("Walk-forward finished.")