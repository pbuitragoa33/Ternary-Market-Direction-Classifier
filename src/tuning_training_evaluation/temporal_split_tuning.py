# Hyperparameters tuning for Temporal Split
#
# This file aims to:

#   - Define the search space for each model as a pure function (returns the estimator)
#   - Optimize each model with Optuna using TimeSeriesSplit cross-validation on X_train
#   - Track everything with Weights & Biases (one W&B run per model/study):
#       * each trial's metrics logged as a step (optimization curve)
#       * all trials collected into a W&B Table (sortable/filterable for HP analysis)
#       * best result + Optuna plots logged, and a W&B Artifact versions the JSON + study
#   - Save best hyperparameters + the 2 metrics + optimization time as JSON, plus Optuna plots


# Libraries

import os
import time
import json
import tempfile
import re

import boto3
import joblib
import optuna
import wandb
import numpy as np
import pandas as pd
import matplotlib
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import class_weight
import sklearn.metrics as metrics

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, StackingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------------------------------
# Important definitions
# ----------------------------------------------------------------------------------------------------

load_dotenv()

TUNING_PATH = os.getenv("TUNING_PATH", ".")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

files_in_gold = ["gold/X_train.csv", "gold/y_train.csv"]

# W&B configuration

WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_MODE = os.getenv("WANDB_MODE")

# Tuning configuration

N_SPLITS = 5
N_TRIALS = 500
TIMEOUT = 60 * 60
PRIMARY_METRIC = "f1_macro"

# Uniform balancing policy: extra weight on Bearish (class 0)

CUSTOM_WEIGHTS = {0: 2, 1: 1, 2: 1}

MODELS_TO_RUN = ["xgboost", "lightgbm", "random_forest", "gradient_boosting",
                 "decision_tree", "adaboost", "stacking", "voting_hard", "knn"]


# ------------------------------------------------------------------------------------------------
# Download data from S3 bucket to perform hyperparameters tuning
# ------------------------------------------------------------------------------------------------

def get_training_data_s3(s3_key, bucket_name = S3_BUCKET_NAME, access_key_id = S3_ACCESS_KEY_ID, secret_access_key = S3_SECRET_ACCESS_KEY):

    s3_client = boto3.client("s3", aws_access_key_id = access_key_id, aws_secret_access_key = secret_access_key)

    with tempfile.NamedTemporaryFile(suffix = ".csv", delete = False) as tmp:

        temp_file_path = tmp.name

    try:

        s3_client.download_file(bucket_name, s3_key, temp_file_path)

        # Preserve the chronological order

        df = pd.read_csv(temp_file_path, index_col = 0)

        return df

    except Exception as e:

        print(f"Error at downloading {s3_key}: {e}")

        return None

    finally:

        if os.path.exists(temp_file_path):

            os.remove(temp_file_path)


# ------------------------------------------------------------------------------------------------
# Search spaces for each model
# ------------------------------------------------------------------------------------------------

#  each returns (estimator, fit_spec)
#  fit_spec = {"sample_weight_map": {...}}  -> objective computes per-fold sample_weight
#  fit_spec = {}                            -> balancing handled internally via class_weight


# Helper to resolve class_weight choice for models thah support it

def _resolve_class_weight(trial):

    choice = trial.suggest_categorical("class_weight", ["balanced", "none", "custom"])

    return {"balanced": "balanced", "none": None, "custom": CUSTOM_WEIGHTS}[choice]


# 1. XGBoost

def space_xgb(trial):

    model = XGBClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 500),
        max_depth = trial.suggest_int("max_depth", 3, 12),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-1, log = True),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        booster = trial.suggest_categorical("booster", ["gbtree", "dart"]),
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log = True),
        eval_metric = "mlogloss", random_state = 42, verbosity = 0
    )

    return model, {"sample_weight_map": CUSTOM_WEIGHTS}


# 2. LightGBM

def space_lgbm(trial):

    model = LGBMClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 300),
        max_depth = trial.suggest_int("max_depth", 3, 12),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-1, log = True),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        num_leaves = trial.suggest_int("num_leaves", 20, 150),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        class_weight = _resolve_class_weight(trial),
        random_state = 42, verbose = -1
    )

    return model, {}


# 3. AdaBoost

def space_adaboost(trial):

    base_name = trial.suggest_categorical("base_estimator", ["decision_tree", "logistic_regression"])

    if base_name == "decision_tree":

        base = DecisionTreeClassifier(max_depth = trial.suggest_int("base_dt_max_depth", 1, 5), random_state = 42)

    else:

        base = LogisticRegression(max_iter = 1000)

    model = AdaBoostClassifier(
        estimator = base,
        n_estimators = trial.suggest_int("n_estimators", 50, 500),
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 2.0, log = True),
        random_state = 42
    )

    return model, {}


# 4. Stacking

def space_stacking(trial):

    combo = trial.suggest_categorical("base_combo", ["ada_rf", "gb_dt"])

    if combo == "ada_rf":

        estimators = [("ada", AdaBoostClassifier(random_state = 42)),
                      ("rf", RandomForestClassifier(random_state = 42))]

    else:

        estimators = [("gb", GradientBoostingClassifier(random_state = 42)),
                      ("dt", DecisionTreeClassifier(random_state = 42))]

    final_name = trial.suggest_categorical("final_estimator", ["logistic_regression", "decision_tree"])

    final = LogisticRegression(max_iter = 1000) if final_name == "logistic_regression" else DecisionTreeClassifier(random_state = 42)

    model = StackingClassifier(estimators = estimators, final_estimator = final)

    return model, {}


# 5. Random Forest

def space_rf(trial):

    model = RandomForestClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 500),
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        max_depth = trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
        class_weight = _resolve_class_weight(trial),
        random_state = 42
    )

    return model, {}


# 6. Gradient Boosting

def space_gb(trial):

    model = GradientBoostingClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 500),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-1, log = True),
        loss = "log_loss",
        max_depth = trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        random_state = 42
    )

    return model, {"sample_weight_map": CUSTOM_WEIGHTS}


# 7. Decision Tree

def space_dt(trial):

    model = DecisionTreeClassifier(
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        max_depth = trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
        class_weight = _resolve_class_weight(trial),
        random_state = 42
    )

    return model, {}


# 8. Voting

def space_voting_hard(trial):

    base_name = trial.suggest_categorical("base_estimator", ["decision_tree", "neural_network", "logistic_regression"])

    if base_name == "decision_tree":

        base = DecisionTreeClassifier(max_depth = trial.suggest_int("base_dt_max_depth", 1, 5), random_state = 42)

    elif base_name == "neural_network":

        # Optuna categoricals must be primitives -> map a string key to the tuple

        nn_choice = trial.suggest_categorical("nn_hidden_layers", ["50", "100", "50_50"])
        hidden = {"50": (50,), "100": (100,), "50_50": (50, 50)}[nn_choice]
        base = MLPClassifier(max_iter = 1000, hidden_layer_sizes = hidden, random_state = 42)

    else:

        base = LogisticRegression(max_iter = 1000)

    model = VotingClassifier(estimators = [("base", base)], voting = "hard")

    return model, {}


# 9. KNN

def space_knn(trial):

    model = KNeighborsClassifier(
        n_neighbors = trial.suggest_int("n_neighbors", 3, 20),
        weights = trial.suggest_categorical("weights", ["uniform", "distance"]),
        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
    )

    return model, {}


# Spaces dictionary for access in the objective function

SPACES = {
    "xgboost": space_xgb,
    "lightgbm": space_lgbm,
    "adaboost": space_adaboost,
    "stacking": space_stacking,
    "random_forest": space_rf,
    "gradient_boosting": space_gb,
    "decision_tree": space_dt,
    "voting_hard": space_voting_hard,
    "knn": space_knn,
}


# ------------------------------------------------------------------------------------------------
# Generic objective: TimeSeriesSplit CV, mean F1-macro across folds, W&B per-trial logging
# ------------------------------------------------------------------------------------------------

def objective(trial, model_name, X, y, cv, metric = PRIMARY_METRIC, records = None):

    model, fit_spec = SPACES[model_name](trial)

    f1_scores, accuracies = [], []

    try:

        for train_idx, val_idx in cv.split(X):

            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

            fit_kwargs = {}

            if "sample_weight_map" in fit_spec:

                fit_kwargs["sample_weight"] = class_weight.compute_sample_weight(
                    fit_spec["sample_weight_map"], y_tr.values.ravel()
                )

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr.values.ravel(), **fit_kwargs)

            y_pred = fold_model.predict(X_vl)

            f1_scores.append(metrics.f1_score(y_vl, y_pred, average = "macro"))
            accuracies.append(metrics.accuracy_score(y_vl, y_pred))

    except Exception as e:

        # Prune the trial

        if wandb.run is not None:

            wandb.log({"trial": trial.number, "trial_error": 1})

        print(f"    trial {trial.number} pruned: {str(e)[:10]}")

        raise optuna.TrialPruned()

    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    mean_acc = float(np.mean(accuracies))

    # Per-trial metrics as a step on the active (study) W&B run -> optimization curve

    if wandb.run is not None:

        wandb.log({
            "trial": trial.number,
            "f1_macro_mean": mean_f1,
            "f1_macro_std": std_f1,
            "accuracy_mean": mean_acc,
        })

    # Collect a row for the trials Table (full per-trial hyperparameters + metrics)

    if records is not None:

        records.append({
            "trial": trial.number,
            **trial.params,
            "f1_macro_mean": round(mean_f1, 6),
            "f1_macro_std": round(std_f1, 6),
            "accuracy_mean": round(mean_acc, 6),
        })

    # Secondary metrics for later retrieval from the best trial

    trial.set_user_attr("accuracy_mean", mean_acc)
    trial.set_user_attr("f1_macro_std", std_f1)

    return mean_f1 if metric == "f1_macro" else mean_acc


# ------------------------------------------------------------------------------------------------
# Run a study for one model: optimize, track with W&B, save JSON + study + Optuna plots
# ------------------------------------------------------------------------------------------------

def save_optuna_plots(study, model_name):

    saved = []

    for plot_fn, suffix in [(plot_optimization_history, "history"),
                            (plot_param_importances, "param_importances")]:

        try:

            ax = plot_fn(study)
            fig = ax.figure
            fig.tight_layout()
            path = os.path.join(TUNING_PATH, f"optuna_{model_name}_{suffix}.png")
            fig.savefig(path, dpi = 130, bbox_inches = "tight")
            plt.close(fig)

            saved.append((suffix, path))
            print(f"    saved plot: {path}")

        except Exception as e:

            print(f"    could not generate {suffix} plot for {model_name}: {e}")

    return saved


def _build_trials_table(records):

    # Trials can have different param keys (conditional params) -> union of keys, fill missing with None

    all_keys = []

    for r in records:

        for k in r:

            if k not in all_keys:

                all_keys.append(k)

    data = [[r.get(k) for k in all_keys] for r in records]

    return wandb.Table(columns = all_keys, data = data)


def run_study(model_name, X, y, cv, n_trials = N_TRIALS, timeout = TIMEOUT, metric = PRIMARY_METRIC):

    print(f"\n=== Tuning: {model_name} ===")

    study = optuna.create_study(direction = "maximize", study_name = model_name, sampler = TPESampler(seed = 42))

    # One W&B run per model (the study). Trials log as steps; we add a Table + artifacts at the end.

    run = wandb.init(
        project = WANDB_PROJECT, entity = WANDB_ENTITY, mode = WANDB_MODE,
        group = model_name, job_type = "study", name = model_name, reinit = True,
        config = {"cv_n_splits": cv.get_n_splits(), "n_trials": n_trials,
                  "timeout_sec": timeout, "primary_metric": metric, "custom_weights": CUSTOM_WEIGHTS},
        settings = wandb.Settings(silent = True),
    )

    records = []
    start = time.time()

    study.optimize(
        lambda trial: objective(trial, model_name, X, y, cv, metric, records),
        n_trials = n_trials, timeout = timeout, show_progress_bar = False,
    )

    elapsed = time.time() - start

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])

    # Guard: if every trial failed, there is no best_trial

    if n_complete == 0:

        print(f"WARNING: no completed trials for {model_name}; skipping summary.")

        if records:

            run.log({"trials_table": _build_trials_table(records)})

        run.finish(exit_code = 1)

        return {"model": model_name, "best_params": {}, "best_f1_macro": float("nan"),
                "best_accuracy": float("nan"), "n_trials_completed": 0}

    best = study.best_trial

    # Build the result payload (JSON-serializable: Optuna params are primitives/strings)

    result = {
        "model": model_name,
        "best_params": study.best_params,
        "best_f1_macro": round(study.best_value, 6),
        "best_accuracy": round(best.user_attrs.get("accuracy_mean", float("nan")), 6),
        "f1_macro_std": round(best.user_attrs.get("f1_macro_std", float("nan")), 6),
        "n_trials_completed": n_complete,
        "n_trials_pruned": n_pruned,
        "cv_n_splits": cv.get_n_splits(),
        "primary_metric": metric,
        "optimization_time_sec": round(elapsed, 1),
        "custom_weights": CUSTOM_WEIGHTS,
    }

    os.makedirs(TUNING_PATH, exist_ok = True)

    json_path = os.path.join(TUNING_PATH, f"best_params_{model_name}.json")

    with open(json_path, "w") as f:

        json.dump(result, f, indent = 2)

    study_path = os.path.join(TUNING_PATH, f"study_{model_name}.joblib")
    joblib.dump(study, study_path)

    plots = save_optuna_plots(study, model_name)

    print(f"    best F1-macro: {result['best_f1_macro']} | accuracy: {result['best_accuracy']} "
          f"| time: {result['optimization_time_sec']}s | completed: {n_complete} | pruned: {n_pruned}")
    print(f"    saved params: {json_path}")

    # --- W&B tracking: trials table, best summary, plots, and a versioned artifact ---

    if records:

        run.log({"trials_table": _build_trials_table(records)})

    run.summary["best_f1_macro"] = result["best_f1_macro"]
    run.summary["best_accuracy"] = result["best_accuracy"]
    run.summary["f1_macro_std"] = result["f1_macro_std"]
    run.summary["optimization_time_sec"] = result["optimization_time_sec"]
    run.summary["n_trials_completed"] = n_complete
    run.summary["n_trials_pruned"] = n_pruned

    run.config.update({f"best_{k}": v for k, v in study.best_params.items()})

    for suffix, path in plots:

        run.log({f"optuna_{suffix}": wandb.Image(path)})

    artifact = wandb.Artifact(f"tuning_{model_name}", type = "tuning_results",
                              metadata = {k: result[k] for k in
                                          ["best_f1_macro", "best_accuracy", "n_trials_completed"]})
    artifact.add_file(json_path)
    artifact.add_file(study_path)

    for _, path in plots:

        artifact.add_file(path)

    run.log_artifact(artifact)

    run.finish()

    return result



# ------------------------------------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Starting hyperparameters tuning with temporal split")
    print("=" * 60)

    # 1. Download training data (date stays as index, not as a feature)

    X_train = get_training_data_s3(files_in_gold[0])
    y_train = get_training_data_s3(files_in_gold[1])

    if X_train is None or y_train is None:

        raise RuntimeError("Could not download training data from S3.")

    # 2. Parse the index to datetime and ensure chronological order (TimeSeriesSplit needs order)

    X_train.index = pd.to_datetime(X_train.index)
    y_train.index = pd.to_datetime(y_train.index)

    X_train = X_train.sort_index()
    y_train = y_train.sort_index()

    # 3. Align check

    if not X_train.index.equals(y_train.index):

        raise ValueError("Index mismatch between X_train and y_train after sorting")

    print(f"Index aligned. X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"Class balance: {y_train.iloc[:, 0].value_counts(normalize = True).round(3).to_dict()}")
    print(f"W&B mode: {WANDB_MODE} | project: {WANDB_PROJECT}")

    # 4. Temporal cross-validation splitter (expanding windows, moving forward in time)

    cv = TimeSeriesSplit(n_splits = N_SPLITS)

    # 5. Run a study per model

    all_results = []

    for model_name in MODELS_TO_RUN:

        result = run_study(model_name, X_train, y_train, cv)
        all_results.append(result)

    # 6. Summary across models (best by primary metric ; NaN-safe sort)

    all_results.sort(key = lambda r: (r["best_f1_macro"] if r["best_f1_macro"] == r["best_f1_macro"] else -1),
                     reverse = True)

    summary_path = os.path.join(TUNING_PATH, "tuning_summary.json")

    with open(summary_path, "w") as f:

        json.dump(all_results, f, indent = 2)

    print("\n" + "=" * 60)
    print("Ranking by F1-macro (CV mean):")

    for r in all_results:

        print(f" * {r['model']:<18} F1 = {r['best_f1_macro']}  acc = {r['best_accuracy']}")

    print("=" * 60)
    print(f"Saved summary: {summary_path}")
    print("Tuning finished.")