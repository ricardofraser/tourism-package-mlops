
# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

import os
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from pathlib import Path

from sklearn.base import clone
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MLFLOW CONFIGURATION
# ---------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5001"
)

EXPERIMENT_NAME = "Tourism_Package_Prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------
# LOAD PREPARED TRAINING AND TEST DATA
# ---------------------------------------------------------

Xtrain = pd.read_csv(DATA_DIR / "Xtrain.csv")
Xtest = pd.read_csv(DATA_DIR / "Xtest.csv")

ytrain = pd.read_csv(DATA_DIR / "ytrain.csv")["ProdTaken"]
ytest = pd.read_csv(DATA_DIR / "ytest.csv")["ProdTaken"]

print("Prepared datasets loaded successfully.")
print(f"Xtrain shape: {Xtrain.shape}")
print(f"Xtest shape:  {Xtest.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"ytest shape:  {ytest.shape}")

# ---------------------------------------------------------
# DEFINE MODEL FEATURE SETS
# ---------------------------------------------------------

PRE_FEATURES = [
    "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
    "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome"
]

POST_FEATURES = PRE_FEATURES + [
    "PitchSatisfactionScore", "ProductPitched",
    "NumberOfFollowups", "DurationOfPitch"
]


# ---------------------------------------------------------
# DEFINE PREPROCESSING COLUMNS
# ---------------------------------------------------------

PRE_NUMERIC_FEATURES = [
    "Age", "NumberOfPersonVisiting", "NumberOfTrips", "Passport",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"
]

PRE_ORDINAL_FEATURES = [
    "CityTier", "PreferredPropertyStar"
]

PRE_CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "Designation"
]

POST_NUMERIC_FEATURES = PRE_NUMERIC_FEATURES + [
    "NumberOfFollowups", "DurationOfPitch"
]

POST_ORDINAL_FEATURES = PRE_ORDINAL_FEATURES + [
    "PitchSatisfactionScore"
]

POST_CATEGORICAL_FEATURES = PRE_CATEGORICAL_FEATURES + [
    "ProductPitched"
]

# ---------------------------------------------------------
# VALIDATE REQUIRED MODEL COLUMNS
# ---------------------------------------------------------

assert set(PRE_FEATURES).issubset(Xtrain.columns), \
    "Missing one or more PRE_FEATURES columns."

assert set(POST_FEATURES).issubset(Xtrain.columns), \
    "Missing one or more POST_FEATURES columns."

print("Feature validation passed: all required model columns are present.")
print(f"Pre-engagement features:  {len(PRE_FEATURES)}")
print(f"Post-engagement features: {len(POST_FEATURES)}")
# ---------------------------------------------------------
# HANDLE CLASS IMBALANCE
# ---------------------------------------------------------

class_weight_max = ytrain.value_counts()[0] / ytrain.value_counts()[1]
scale_pos_weight_values = np.round(
    np.linspace(1.0, class_weight_max, 6),
    2
)

print(f"Observed class imbalance ratio: {class_weight_max:.3f}")
print(f"scale_pos_weight search values: {scale_pos_weight_values}")

# ---------------------------------------------------------
# DEFINE PREPROCESSING PIPELINES
# ---------------------------------------------------------

pre_preprocessor = make_column_transformer(
    (StandardScaler(), PRE_NUMERIC_FEATURES),
    ("passthrough", PRE_ORDINAL_FEATURES),
    (OneHotEncoder(handle_unknown="ignore"), PRE_CATEGORICAL_FEATURES)
)

post_preprocessor = make_column_transformer(
    (StandardScaler(), POST_NUMERIC_FEATURES),
    ("passthrough", POST_ORDINAL_FEATURES),
    (OneHotEncoder(handle_unknown="ignore"), POST_CATEGORICAL_FEATURES)
)

# ---------------------------------------------------------
# DEFINE BASE XGBOOST MODEL
# ---------------------------------------------------------

xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

print("Preprocessing pipelines and base XGBoost model created successfully.")
# ---------------------------------------------------------
# DEFINE RANDOMIZED SEARCH SPACES
# ---------------------------------------------------------

# Pre-engagement search space
pre_random_param_dist = {
    "xgbclassifier__n_estimators": [75, 100, 150, 200, 250],              # Number of boosting trees
    "xgbclassifier__max_depth": [1, 2, 3, 4],                             # Maximum tree depth
    "xgbclassifier__colsample_bytree": [0.5, 0.6, 0.7, 0.8],              # Feature fraction sampled per tree
    "xgbclassifier__colsample_bylevel": [0.5, 0.6, 0.7, 0.8],             # Feature fraction sampled per level
    "xgbclassifier__learning_rate": [0.025, 0.05, 0.075, 0.1],            # Contribution of each new tree
    "xgbclassifier__reg_lambda": [5.0, 10.0, 15.0, 20.0, 30.0],           # L2 regularisation strength
    "xgbclassifier__scale_pos_weight": scale_pos_weight_values             # Minority-class weighting
}

# Post-engagement search space
post_random_param_dist_strong = {
    "xgbclassifier__n_estimators": [150, 175, 200, 225, 250, 300],         # Number of boosting trees
    "xgbclassifier__max_depth": [4, 5, 6, 7],                              # Maximum tree depth
    "xgbclassifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],               # Feature fraction sampled per tree
    "xgbclassifier__colsample_bylevel": [0.8, 0.9, 1.0],                    # Feature fraction sampled per level
    "xgbclassifier__learning_rate": [0.05, 0.075, 0.1, 0.125, 0.15],       # Contribution of each new tree
    "xgbclassifier__reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],                # L2 regularisation strength
    "xgbclassifier__scale_pos_weight": scale_pos_weight_values              # Minority-class weighting
}

print("Randomized search spaces created successfully.")

def refit_pareto_recall_precision(cv_results):
    recall = np.asarray(cv_results["mean_test_recall"])
    precision = np.asarray(cv_results["mean_test_precision"])

    # ---------------------------------------------------------
    # IDENTIFY PARETO-EFFICIENT CANDIDATES
    # ---------------------------------------------------------

    pareto_indices = []

    for i in range(len(recall)):
        dominated = False

        for j in range(len(recall)):
            if i == j:
                continue

            # j dominates i if it is at least as good on both
            # metrics and strictly better on at least one.
            if (
                recall[j] >= recall[i]
                and precision[j] >= precision[i]
                and (
                    recall[j] > recall[i]
                    or precision[j] > precision[i]
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto_indices.append(i)

    # ---------------------------------------------------------
    # SORT FRONTIER BY HIGHEST RECALL FIRST
    # ---------------------------------------------------------

    pareto_indices = sorted(
        pareto_indices,
        key=lambda i: recall[i],
        reverse=True
    )

    selected = pareto_indices[0]

    # ---------------------------------------------------------
    # EVALUATE THE FULL PARETO FRONTIER
    # ---------------------------------------------------------

    for candidate in pareto_indices[1:]:
        recall_loss = recall[selected] - recall[candidate]
        precision_gain = precision[candidate] - precision[selected]

        # Same recall: prefer higher precision.
        if np.isclose(recall_loss, 0):
            if precision[candidate] > precision[selected]:
                selected = candidate
            continue

        # Move when precision gained exceeds recall sacrificed.
        if precision_gain > recall_loss:
            selected = candidate

    # ---------------------------------------------------------
    # HANDLE EXACT RECALL / PRECISION TIES
    # ---------------------------------------------------------

    tied = [
        i for i in pareto_indices
        if np.isclose(recall[i], recall[selected])
        and np.isclose(precision[i], precision[selected])
    ]

    if len(tied) > 1:
        rng = np.random.default_rng(42)
        selected = rng.choice(tied)

    return int(selected)

print("Pareto recall/precision selector created successfully.")

# ---------------------------------------------------------
# TRAIN + TRACK MODEL USING RSCV + PARETO SELECTION
# ---------------------------------------------------------

def train_and_track_model_RSCV_Pareto(
    model_name,
    feature_list,
    preprocessor,
    param_distributions,
    classification_threshold=0.50,
    n_iter=20
):

    # Select model-specific features
    Xtrain_model = Xtrain[feature_list]
    Xtest_model = Xtest[feature_list]

    # Complete preprocessing + XGBoost pipeline
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # Recall and precision drive model selection.
    # F1 and ROC-AUC are retained for diagnostics.
    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    # RandomizedSearchCV calls this after CV has completed.
    # The external Pareto function returns the selected candidate index.
    def refit_pareto(cv_results):
        return refit_pareto_recall_precision(cv_results)

    # ---------------------------------------------------------
    # MLFLOW PARENT RUN
    # ---------------------------------------------------------

    with mlflow.start_run(run_name=model_name) as run:

        parent_run_id = run.info.run_id

        # Parent-run tags
        mlflow.set_tag("run_role", "parent_search")
        mlflow.set_tag("model_name", model_name)

        # Log search configuration
        mlflow.log_param("model_stage", model_name)
        mlflow.log_param("feature_count", len(feature_list))
        mlflow.log_param("feature_list", ",".join(feature_list))
        mlflow.log_param("base_classification_threshold", classification_threshold)

        mlflow.log_param("search_method", "RandomizedSearchCV")
        mlflow.log_param("selection_method", "Pareto_Recall_Precision")
        mlflow.log_param("random_search_iterations", n_iter)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scoring_metrics", "recall,precision,f1,roc_auc")

        mlflow.log_param("pareto_objective_1", "cv_recall")
        mlflow.log_param("pareto_objective_2", "cv_precision")

        mlflow.log_param(
            "class_imbalance_ratio",
            round(class_weight_max, 3)
        )

        mlflow.log_param(
            "scale_pos_weight_search",
            str(scale_pos_weight_values.tolist())
        )

        # ---------------------------------------------------------
        # RANDOMIZED SEARCH
        # ---------------------------------------------------------

        random_search = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            refit=refit_pareto,
            cv=5,
            n_jobs=-1,
            random_state=42,
            return_train_score=True
        )

        random_search.fit(Xtrain_model, ytrain)

        # Full CV search history
        results = random_search.cv_results_

        # ---------------------------------------------------------
        # LOG EVERY SEARCH CANDIDATE
        # ---------------------------------------------------------

        for i, param_set in enumerate(results["params"]):

            train_recall = results["mean_train_recall"][i]
            cv_recall = results["mean_test_recall"][i]
            cv_precision = results["mean_test_precision"][i]
            cv_f1 = results["mean_test_f1"][i]
            cv_roc_auc = results["mean_test_roc_auc"][i]
            std_recall = results["std_test_recall"][i]

            recall_gap = abs(train_recall - cv_recall)

            with mlflow.start_run(
                run_name=f"{model_name}_Trial_{i + 1}",
                nested=True
            ):

                mlflow.set_tag("run_role", "cv_trial")
                mlflow.log_param("trial_number", i + 1)

                mlflow.log_params(param_set)

                mlflow.log_metrics({
                    "mean_train_recall": train_recall,
                    "mean_cv_recall": cv_recall,
                    "mean_cv_precision": cv_precision,
                    "mean_cv_f1": cv_f1,
                    "mean_cv_roc_auc": cv_roc_auc,
                    "std_cv_recall": std_recall,
                    "recall_train_validation_gap": recall_gap
                })

        # ---------------------------------------------------------
        # RETRIEVE PARETO-SELECTED CANDIDATE
        # ---------------------------------------------------------

        best_index = random_search.best_index_
        best_params = random_search.best_params_

        best_train_recall = results["mean_train_recall"][best_index]
        best_cv_recall = results["mean_test_recall"][best_index]
        best_cv_precision = results["mean_test_precision"][best_index]
        best_cv_f1 = results["mean_test_f1"][best_index]
        best_cv_roc_auc = results["mean_test_roc_auc"][best_index]

        best_recall_gap = abs(
            best_train_recall - best_cv_recall
        )

        # Log selected parameters and CV metrics
        mlflow.log_param("selected_trial_number", best_index + 1)
        mlflow.log_params(best_params)

        mlflow.log_metrics({
            "selected_cv_train_recall": best_train_recall,
            "selected_cv_recall": best_cv_recall,
            "selected_cv_precision": best_cv_precision,
            "selected_cv_recall_gap": best_recall_gap,
            "selected_cv_f1": best_cv_f1,
            "selected_cv_roc_auc": best_cv_roc_auc
        })

        # RandomizedSearchCV has already refitted this estimator
        # on the complete training dataset using the Pareto-selected params.
        best_model = random_search.best_estimator_

        # ---------------------------------------------------------
        # FINAL TRAIN / TEST PREDICTIONS AT BASE THRESHOLD
        # ---------------------------------------------------------

        train_proba = best_model.predict_proba(Xtrain_model)[:, 1]
        test_proba = best_model.predict_proba(Xtest_model)[:, 1]

        y_pred_train = (
            train_proba >= classification_threshold
        ).astype(int)

        y_pred_test = (
            test_proba >= classification_threshold
        ).astype(int)

        train_report = classification_report(
            ytrain,
            y_pred_train,
            output_dict=True,
            zero_division=0
        )

        test_report = classification_report(
            ytest,
            y_pred_test,
            output_dict=True,
            zero_division=0
        )

        # ---------------------------------------------------------
        # LOG BASE-THRESHOLD MODEL METRICS
        # ---------------------------------------------------------

        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1_score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1_score": test_report["1"]["f1-score"]
        })

        # ---------------------------------------------------------
        # DISPLAY DEVELOPMENT RESULTS
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        print(f"Random-search candidates: {n_iter}")
        print(f"Total CV fits: {n_iter * 5}")

        print("\nPareto-selected CV performance:")
        print(f"CV Train Recall:              {best_train_recall:.4f}")
        print(f"CV Validation Recall:         {best_cv_recall:.4f}")
        print(f"CV Validation Precision:      {best_cv_precision:.4f}")
        print(f"Train-Validation Recall Gap:  {best_recall_gap:.4f}")
        print(f"CV F1 (diagnostic):           {best_cv_f1:.4f}")
        print(f"CV ROC-AUC (diagnostic):      {best_cv_roc_auc:.4f}")

        print("\nPareto-selected parameters:")
        print(best_params)

        print("\nTest classification report:")
        print(
            classification_report(
                ytest,
                y_pred_test,
                zero_division=0
            )
        )

        # Return fitted model, CV results, search object and MLflow parent run
        return best_model, results, random_search, parent_run_id
print("Train and Track Function using pareto selection sucessfully added")

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score

def tune_threshold_pareto(
    model,
    X,
    y,
    model_name,
    parent_run_id,
    thresholds=None,
    cv=5
):

    if thresholds is None:
        thresholds = [0.50, 0.48, 0.45, 0.43, 0.40, 0.38, 0.35]

    # ---------------------------------------------------------
    # GENERATE OUT-OF-FOLD TRAINING PROBABILITIES
    # ---------------------------------------------------------

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    oof_proba = cross_val_predict(
        clone(model),
        X,
        y,
        cv=cv_strategy,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # ---------------------------------------------------------
    # EVALUATE CANDIDATE THRESHOLDS
    # ---------------------------------------------------------

    rows = []

    for threshold in thresholds:
        y_pred = (oof_proba >= threshold).astype(int)

        rows.append({
            "Threshold": threshold,
            "Recall": recall_score(y, y_pred, zero_division=0),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0)
        })

    threshold_results = pd.DataFrame(rows)

    # Use the same Pareto rule used for model selection
    pareto_input = {
        "mean_test_recall": threshold_results["Recall"].to_numpy(),
        "mean_test_precision": threshold_results["Precision"].to_numpy()
    }

    selected_index = refit_pareto_recall_precision(pareto_input)

    selected_threshold = float(
        threshold_results.loc[selected_index, "Threshold"]
    )

    selected_recall = float(
        threshold_results.loc[selected_index, "Recall"]
    )

    selected_precision = float(
        threshold_results.loc[selected_index, "Precision"]
    )

    selected_f1 = float(
        threshold_results.loc[selected_index, "F1"]
    )

    threshold_results["Selected"] = False
    threshold_results.loc[selected_index, "Selected"] = True

    # ---------------------------------------------------------
    # LOG THRESHOLD SELECTION TO ORIGINAL MLFLOW PARENT RUN
    # ---------------------------------------------------------

    with mlflow.start_run(run_id=parent_run_id):

        mlflow.set_tag(
            "threshold_selection_method",
            "OOF_Pareto_Recall_Precision"
        )

        mlflow.log_param(
            "selected_inference_threshold",
            selected_threshold
        )

        mlflow.log_metrics({
            "threshold_oof_recall": selected_recall,
            "threshold_oof_precision": selected_precision,
            "threshold_oof_f1": selected_f1
        })

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{model_name} - Threshold Tuning")
    print("=" * 60)

    print(threshold_results.round(4))

    print(f"\nPareto-selected threshold: {selected_threshold:.2f}")
    print(f"Selected OOF Recall:       {selected_recall:.4f}")
    print(f"Selected OOF Precision:    {selected_precision:.4f}")
    print(f"Selected OOF F1:           {selected_f1:.4f}")

    return selected_threshold, threshold_results, selected_index
print ("Threshold Tuning Helper Fuction Created Successfully")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# FINAL HOLDOUT EVALUATION
# ---------------------------------------------------------

# Evaluate the fitted Pareto-selected model on the previously unseen
# holdout test data using the locked classification threshold selected
# from out-of-fold training predictions.
#
# Final holdout metrics are logged to the corresponding MLflow parent
# run so model selection, threshold tuning and final evaluation remain
# linked within the same experiment record.

def evaluate_holdout_and_log(
    model,
    Xtest_model,
    ytest,
    threshold,
    model_name,
    parent_run_id
):
    test_proba = model.predict_proba(Xtest_model)[:, 1]
    y_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(ytest, y_pred),
        "precision": precision_score(ytest, y_pred, zero_division=0),
        "recall": recall_score(ytest, y_pred, zero_division=0),
        "f1": f1_score(ytest, y_pred, zero_division=0)
    }

    report = classification_report( ytest, y_pred,digits=4,zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Append final holdout results to the original MLflow parent run.
    with mlflow.start_run(run_id=parent_run_id):
        mlflow.log_metrics({
            "final_holdout_accuracy": metrics["accuracy"],
            "final_holdout_precision": metrics["precision"],
            "final_holdout_recall": metrics["recall"],
            "final_holdout_f1": metrics["f1"]
        })

        mlflow.set_tag(
            "final_evaluation_stage",
            "locked_threshold_holdout"
        )

    print("\n" + "=" * 60)
    print(f"{model_name} - Final Holdout Evaluation")
    print("=" * 60)
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    return metrics
print("Evaluate hold out test and log function created sucessfully")

# ---------------------------------------------------------
# DEFINE BASE XGBOOST MODEL
# ---------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# ---------------------------------------------------------
# RANDOMIZED SEARCH SPACES
# ---------------------------------------------------------

# Pre-engagement: refined around strongest earlier candidates.
pre_random_param_dist = {
    "xgbclassifier__n_estimators": [75, 100, 150, 200, 250],              # Number of boosting trees
    "xgbclassifier__max_depth": [1, 2, 3, 4],                             # Maximum tree depth
    "xgbclassifier__colsample_bytree": [0.5, 0.6, 0.7, 0.8],              # Feature fraction sampled per tree
    "xgbclassifier__colsample_bylevel": [0.5, 0.6, 0.7, 0.8],             # Feature fraction sampled per tree level
    "xgbclassifier__learning_rate": [0.025, 0.05, 0.075, 0.1],            # Contribution of each new tree
    "xgbclassifier__reg_lambda": [5.0, 10.0, 15.0, 20.0, 30.0],           # L2 regularisation strength
    "xgbclassifier__scale_pos_weight": scale_pos_weight_values             # Minority-class weighting
}

# Post-engagement: stronger region identified during development.
post_random_param_dist_strong = {
    "xgbclassifier__n_estimators": [150, 175, 200, 225, 250, 300],         # Number of boosting trees
    "xgbclassifier__max_depth": [4, 5, 6, 7],                              # Maximum tree depth
    "xgbclassifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],               # Feature fraction sampled per tree
    "xgbclassifier__colsample_bylevel": [0.8, 0.9, 1.0],                    # Feature fraction sampled per tree level
    "xgbclassifier__learning_rate": [0.05, 0.075, 0.1, 0.125, 0.15],       # Contribution of each new tree
    "xgbclassifier__reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],                # L2 regularisation strength
    "xgbclassifier__scale_pos_weight": scale_pos_weight_values              # Minority-class weighting
}


# ---------------------------------------------------------
# TRAIN PRE-ENGAGEMENT MODEL
# RANDOMIZED SEARCH + PARETO SELECTION
# ---------------------------------------------------------
print("\n Commencing Training of the pre-engagement model")

pre_model_pareto, pre_cv_results, pre_search, pre_parent_run_id = (
    train_and_track_model_RSCV_Pareto(
        model_name="Pre_Engagement_Model_RSCV_Pareto",
        feature_list=PRE_FEATURES,
        preprocessor=pre_preprocessor,
        param_distributions=pre_random_param_dist,
        classification_threshold=0.50,
        n_iter=300
    )
)
# ---------------------------------------------------------
# TRAIN POST-ENGAGEMENT MODEL
# RANDOMIZED SEARCH + PARETO SELECTION
# ---------------------------------------------------------
print("\n Commencing Training of the post-engagement model")
post_model_strong, post_cv_results_strong, post_search_strong, post_parent_run_id = (
    train_and_track_model_RSCV_Pareto(
        model_name="Post_Engagement_Model_RSCV_Strong_Region",
        feature_list=POST_FEATURES,
        preprocessor=post_preprocessor,
        param_distributions=post_random_param_dist_strong,
        classification_threshold=0.50,
        n_iter=180
    )
)

print("\nModel training and Pareto selection completed successfully.")

# ---------------------------------------------------------
# CONFIRM RETURNED OBJECTS
# ---------------------------------------------------------

print("\nModel training completed.")

print(
    f"Pre-engagement CV candidates stored: "
    f"{len(pre_cv_results['params'])}"
)

print(
    f"Post-engagement CV candidates stored: "
    f"{len(post_cv_results_strong['params'])}"
)

print(
    "\nPre-engagement Pareto-selected parameters:"
)
print(pre_search.best_params_)

print(
    "\nPost-engagement Pareto-selected parameters:"
)
print(post_search_strong.best_params_)

# ---------------------------------------------------------
# TUNE PRE-ENGAGEMENT THRESHOLD
# ---------------------------------------------------------
print("Commencing Threshold Tuning of the Pre-Engagement Model")
pre_threshold, pre_threshold_results, pre_threshold_index = (
    tune_threshold_pareto(
        model=pre_model_pareto,
        X=Xtrain[PRE_FEATURES],
        y=ytrain,
        model_name="Pre_Engagement_Model_RSCV_Pareto",
        parent_run_id=pre_parent_run_id,
        thresholds=[0.50, 0.48, 0.45, 0.43, 0.40, 0.38, 0.35, 0.33, 0.31]
    )
)

# ---------------------------------------------------------
# TUNE POST-ENGAGEMENT THRESHOLD
# ---------------------------------------------------------
print("Commencing Threshold Tuning of the Post-Engagement Model")
post_threshold, post_threshold_results, post_threshold_index = (
    tune_threshold_pareto(
        model=post_model_strong,
        X=Xtrain[POST_FEATURES],
        y=ytrain,
        model_name="Post_Engagement_Model_RSCV_Strong_Region",
        parent_run_id=post_parent_run_id,
        thresholds=[0.50, 0.48, 0.45, 0.43, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
    )
)

print("\nThreshold tuning completed successfully.")

# ---------------------------------------------------------
# FINAL HOLDOUT EVALUATION
# ---------------------------------------------------------

# Run inference on the previously unseen holdout test data using the
# fitted Pareto-selected models and their tuned classification thresholds.
# Log the final holdout metrics to the corresponding MLflow parent runs.

pre_holdout_metrics = evaluate_holdout_and_log(
    model=pre_model_pareto,
    Xtest_model=Xtest[PRE_FEATURES],
    ytest=ytest,
    threshold=pre_threshold,
    model_name="Pre_Engagement_Model_RSCV_Pareto",
    parent_run_id=pre_parent_run_id
)

post_holdout_metrics = evaluate_holdout_and_log(
    model=post_model_strong,
    Xtest_model=Xtest[POST_FEATURES],
    ytest=ytest,
    threshold=post_threshold,
    model_name="Post_Engagement_Model_RSCV_Strong_Region",
    parent_run_id=post_parent_run_id
)

print("\nFinal holdout evaluation completed successfully.")

# ---------------------------------------------------------
# SAVE DEPLOYMENT ARTIFACTS
# ---------------------------------------------------------

PRE_MODEL_PATH = DEPLOYMENT_DIR / "pre_engagement_model.joblib"
POST_MODEL_PATH = DEPLOYMENT_DIR / "post_engagement_model.joblib"
CONFIG_PATH = DEPLOYMENT_DIR / "model_config.json"

# Save fitted pipelines
joblib.dump(pre_model_pareto, PRE_MODEL_PATH)
joblib.dump(post_model_strong, POST_MODEL_PATH)

# Save thresholds, features and MLflow traceability information
model_config = {
    "pre_engagement": {
        "threshold": float(pre_threshold),
        "features": PRE_FEATURES,
        "mlflow_run_id": pre_parent_run_id
    },
    "post_engagement": {
        "threshold": float(post_threshold),
        "features": POST_FEATURES,
        "mlflow_run_id": post_parent_run_id
    }
}

with open(CONFIG_PATH, "w") as f:
    json.dump(model_config, f, indent=2)

print(f"Pre-engagement model saved to:  {PRE_MODEL_PATH}")
print(f"Post-engagement model saved to: {POST_MODEL_PATH}")
print(f"Model configuration saved to:   {CONFIG_PATH}")


# ---------------------------------------------------------
# LOG DEPLOYMENT ARTIFACTS TO MLFLOW
# ---------------------------------------------------------

with mlflow.start_run(run_id=pre_parent_run_id):
    mlflow.log_artifact(
        str(PRE_MODEL_PATH),
        artifact_path="deployment"
    )
    mlflow.log_artifact(
        str(CONFIG_PATH),
        artifact_path="deployment"
    )

with mlflow.start_run(run_id=post_parent_run_id):
    mlflow.log_artifact(
        str(POST_MODEL_PATH),
        artifact_path="deployment"
    )
    mlflow.log_artifact(
        str(CONFIG_PATH),
        artifact_path="deployment"
    )

print("\nDeployment artifacts saved and logged to MLflow successfully.")

def model_summary(
    model_name,
    selection_method,
    parent_run_id,
    search,
    cv_results,
    threshold_results,
    threshold_index,
    holdout_metrics
):
    best_index = search.best_index_

    return {
        "Model": model_name,
        "Selection Method": selection_method,
        "MLflow Run ID": parent_run_id,

        "CV Recall": cv_results["mean_test_recall"][best_index],
        "CV Precision": cv_results["mean_test_precision"][best_index],
        "CV F1": cv_results["mean_test_f1"][best_index],
        "CV ROC-AUC": cv_results["mean_test_roc_auc"][best_index],

        "Selected Threshold": threshold_results.loc[
            threshold_index, "Threshold"
        ],
        "OOF Recall": threshold_results.loc[
            threshold_index, "Recall"
        ],
        "OOF Precision": threshold_results.loc[
            threshold_index, "Precision"
        ],
        "OOF F1": threshold_results.loc[
            threshold_index, "F1"
        ],

        "Test Accuracy": holdout_metrics["accuracy"],
        "Test Recall": holdout_metrics["recall"],
        "Test Precision": holdout_metrics["precision"],
        "Test F1": holdout_metrics["f1"],

        "Best Parameters": search.best_params_
    }

def print_model_summary(summary):

    print("\n" + "=" * 70)
    print(summary["Model"])
    print("=" * 70)

    print(f"\nSelection Method: {summary['Selection Method']}")
    print(f"MLflow Run ID:    {summary['MLflow Run ID']}")

    print("\nSelected Hyperparameters:")
    for param, value in summary["Best Parameters"].items():
        print(f"{param}: {value}")

    print("\nOOF and CV Performance:")
    print(f"CV Recall:       {summary['CV Recall']:.4f}")
    print(f"CV Precision:    {summary['CV Precision']:.4f}")
    print(f"CV F1:           {summary['CV F1']:.4f}")
    print(f"CV ROC-AUC:      {summary['CV ROC-AUC']:.4f}")
    print(f"Threshold:       {summary['Selected Threshold']:.2f}")
    print(f"OOF Recall:      {summary['OOF Recall']:.4f}")
    print(f"OOF Precision:   {summary['OOF Precision']:.4f}")
    print(f"OOF F1:          {summary['OOF F1']:.4f}")

    print("\nFinal Holdout Performance:")
    print(f"Test Accuracy:   {summary['Test Accuracy']:.4f}")
    print(f"Test Recall:     {summary['Test Recall']:.4f}")
    print(f"Test Precision:  {summary['Test Precision']:.4f}")
    print(f"Test F1:         {summary['Test F1']:.4f}")

pre_summary = model_summary(
    model_name="Pre_Engagement_Model_RSCV_Pareto",
    selection_method="RandomizedSearchCV + Pareto Recall/Precision",
    parent_run_id=pre_parent_run_id,
    search=pre_search,
    cv_results=pre_cv_results,
    threshold_results=pre_threshold_results,
    threshold_index=pre_threshold_index,
    holdout_metrics=pre_holdout_metrics
)

post_summary = model_summary(
    model_name="Post_Engagement_Model_RSCV_Strong_Region",
    selection_method="RandomizedSearchCV + Pareto Recall/Precision",
    parent_run_id=post_parent_run_id,
    search=post_search_strong,
    cv_results=post_cv_results_strong,
    threshold_results=post_threshold_results,
    threshold_index=post_threshold_index,
    holdout_metrics=post_holdout_metrics
)

print_model_summary(pre_summary)
print_model_summary(post_summary)

print("\n" + "=" * 70)
print("END-TO-END MLOPS TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Both models were trained, Pareto-selected, threshold-tuned,")
print("evaluated on holdout data, saved for deployment, and logged to MLflow.")
