"""
UWB ANTENNA BANDWIDTH PREDICTION — COMPLETE TRAINING PIPELINE

Input:
    dataset/antenna_ML_ready_dataset.csv

Target:
    bandwidth_GHz

Features:
    Antenna geometry/material columns ending in "_mm", plus epsilon_r.
    Constant features are removed automatically.

Important:
    - Uses the existing dataset_split column (train/test).
    - Does not use S11/bandwidth/frequency outputs as input features.
    - Optionally excludes samples whose -10 dB band touches the sweep boundary.
    - Hyperparameter tuning uses training data only.
    - The test set is used only for final evaluation.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "antenna_ML_ready_dataset.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"

for folder in (RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "bandwidth_GHz"
SPLIT_COLUMN = "dataset_split"
BOUNDARY_COLUMN = "band_touches_2_or_14_GHz_sweep_boundary"

# Set False for a first run that keeps all valid bandwidth samples.
# Set True for a sensitivity/cleaner evaluation excluding sweep-truncated samples.
EXCLUDE_BOUNDARY_CENSORED = False

RANDOM_STATE = 42
RUN_HYPERPARAMETER_TUNING = True

# ============================================================
# 2. LOAD AND INSPECT DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Put antenna_ML_ready_dataset.csv inside the dataset folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nDataset split counts:\n", df[SPLIT_COLUMN].value_counts(dropna=False))

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' is missing.")
if SPLIT_COLUMN not in df.columns:
    raise ValueError(f"Split column '{SPLIT_COLUMN}' is missing.")

# Keep only rows with a valid target and known split.
data = df.dropna(subset=[TARGET, SPLIT_COLUMN]).copy()
data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
data = data.dropna(subset=[TARGET])

# Optional: remove cases where the impedance band touches the sweep limits.
if EXCLUDE_BOUNDARY_CENSORED and BOUNDARY_COLUMN in data.columns:
    before = len(data)
    data = data[data[BOUNDARY_COLUMN].astype(str).str.lower().isin(["false", "0"])].copy()
    print(f"\nExcluded {before - len(data)} boundary-touching samples.")

# ============================================================
# 3. SELECT INPUT FEATURES — AVOID TARGET LEAKAGE
# ============================================================

# Geometry/material features only.
candidate_features = [
    col for col in data.columns
    if col.endswith("_mm") or col == "epsilon_r"
]

# Ensure numeric values.
for col in candidate_features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop feature columns that are entirely missing.
candidate_features = [
    col for col in candidate_features
    if data[col].notna().any()
]

# Median-impute missing feature values using TRAINING data only below.
# First split rows by the supplied split label.
train_data = data[data[SPLIT_COLUMN].astype(str).str.lower() == "train"].copy()
test_data = data[data[SPLIT_COLUMN].astype(str).str.lower() == "test"].copy()

if train_data.empty or test_data.empty:
    raise ValueError("Both 'train' and 'test' rows are required in dataset_split.")

# Remove constant features based on training data only.
features = [
    col for col in candidate_features
    if train_data[col].nunique(dropna=True) > 1
]

if not features:
    raise ValueError("No varying numeric geometry/material features were found.")

print("\n========== FEATURES USED ==========")
print(features)
print("Number of usable features:", len(features))
print("Training rows before imputation:", len(train_data))
print("Testing rows before imputation:", len(test_data))

# Impute missing feature values using training medians.
medians = train_data[features].median()
X_train = train_data[features].fillna(medians)
X_test = test_data[features].fillna(medians)

y_train = train_data[TARGET].astype(float)
y_test = test_data[TARGET].astype(float)

# Save exact feature list and imputation values.
with open(RESULTS_DIR / "preprocessing_info.json", "w") as f:
    json.dump(
        {
            "target": TARGET,
            "features": features,
            "training_medians": medians.to_dict(),
            "excluded_boundary_censored": EXCLUDE_BOUNDARY_CENSORED,
            "random_state": RANDOM_STATE,
        },
        f,
        indent=4,
    )

# ============================================================
# 4. DEFINE THE FIVE REGRESSION MODELS
# ============================================================

models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
}

# XGBoost is included as the fifth model.
try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
except ImportError:
    print("\nWARNING: xgboost is not installed. Run: pip install xgboost")
    print("The other four models will still be trained.")

# ============================================================
# 5. OPTIONAL HYPERPARAMETER TUNING
#    Cross-validation is performed ONLY on training data.
# ============================================================

param_grids = {
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3],
    },
    "Extra Trees": {
        "n_estimators": [200, 300],
        "max_features": [1.0, "sqrt"],
        "min_samples_leaf": [1, 2],
    },
    "Random Forest": {
        "n_estimators": [200, 300],
        "max_features": [1.0, "sqrt"],
        "min_samples_leaf": [1, 2],
    },
    "Decision Tree": {
        "max_depth": [None, 3, 5, 8],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

trained_models = {}
cv_results = []

print("\n========== TRAINING MODELS ==========")

for name, model in models.items():
    print(f"\nTraining: {name}")

    if RUN_HYPERPARAMETER_TUNING and name in param_grids:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_rmse = -search.best_score_
        print("Best parameters:", best_params)
        print("Training CV RMSE:", round(best_cv_rmse, 6))
    else:
        best_model = model.fit(X_train, y_train)
        best_params = {}
        best_cv_rmse = np.nan

    trained_models[name] = best_model

    cv_results.append({
        "Algorithm": name,
        "Best_CV_RMSE_GHz": best_cv_rmse,
        "Best_Parameters": json.dumps(best_params),
    })

# ============================================================
# 6. TEST-SET EVALUATION
# ============================================================

metric_rows = []
predictions_by_model = {}

print("\n========== FINAL TEST-SET EVALUATION ==========")

for name, model in trained_models.items():
    predictions = model.predict(X_test)
    predictions_by_model[name] = predictions

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    evs = explained_variance_score(y_test, predictions)

    metric_rows.append({
        "Algorithm": name,
        "MAE_GHz": mae,
        "MSE_GHz_squared": mse,
        "RMSE_GHz": rmse,
        "R2": r2,
        "R2_percent": r2 * 100,
        "Explained_Variance": evs,
        "Explained_Variance_percent": evs * 100,
    })

metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE_GHz")
cv_df = pd.DataFrame(cv_results)

metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
cv_df.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)

print(metrics_df.to_string(index=False))

# Select the model using training-only cross-validation, NOT test-set results.
# The held-out test set is reserved for final performance reporting.
if cv_df["Best_CV_RMSE_GHz"].notna().any():
    best_name = cv_df.sort_values("Best_CV_RMSE_GHz").iloc[0]["Algorithm"]
else:
    # If tuning/CV is disabled, choose a model using a separate validation set
    # before using this script for publication-quality evaluation.
    raise ValueError(
        "No CV scores available. Enable RUN_HYPERPARAMETER_TUNING "
        "or implement a separate validation-based model-selection step."
    )

best_model = trained_models[best_name]
print("\nSelected model using training-only CV:", best_name)

# Save every trained model and the selected model.
for name, model in trained_models.items():
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, MODELS_DIR / f"{safe_name}.joblib")

joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

# Save test predictions for reproducibility.
prediction_df = pd.DataFrame({
    "actual_bandwidth_GHz": y_test.values,
})
for name, pred in predictions_by_model.items():
    prediction_df[f"{name}_predicted_GHz"] = pred
prediction_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

# ============================================================
# 7. PLOT MODEL COMPARISON
# ============================================================

sns.set_theme()

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_df, x="Algorithm", y="RMSE_GHz")
plt.title("Test RMSE Comparison of Regression Models")
plt.ylabel("RMSE (GHz)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_rmse_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_df, x="Algorithm", y="R2")
plt.title("Test R-squared Comparison of Regression Models")
plt.ylabel("R-squared")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_r2_comparison.png", dpi=300)
plt.close()

# ============================================================
# 8. ACTUAL VS PREDICTED AND RESIDUAL PLOTS
# ============================================================

best_predictions = best_model.predict(X_test)

low = min(float(y_test.min()), float(np.min(best_predictions)))
high = max(float(y_test.max()), float(np.max(best_predictions)))

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_predictions, alpha=0.75)
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual bandwidth (GHz)")
plt.ylabel("Predicted bandwidth (GHz)")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

residuals = y_test.values - best_predictions

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual (actual - predicted), GHz")
plt.title(f"Residual Distribution — {best_name}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(best_predictions, residuals, alpha=0.75)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted bandwidth (GHz)")
plt.ylabel("Residual (GHz)")
plt.title(f"Residuals vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residuals_vs_predicted.png", dpi=300)
plt.close()

# ============================================================
# 9. TREE FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=300)
    plt.close()

# ============================================================
# 10. PERMUTATION IMPORTANCE ON HELD-OUT TEST DATA
# ============================================================

perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=30,
    random_state=RANDOM_STATE,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

permutation_df = pd.DataFrame({
    "Feature": features,
    "Importance_Mean": perm.importances_mean,
    "Importance_Std": perm.importances_std,
}).sort_values("Importance_Mean", ascending=False)

permutation_df.to_csv(
    RESULTS_DIR / "permutation_importance.csv", index=False
)

plt.figure(figsize=(9, 6))
sns.barplot(
    data=permutation_df,
    x="Importance_Mean",
    y="Feature"
)
plt.title("Permutation Importance on Test Data")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "permutation_importance.png", dpi=300)
plt.close()

# ============================================================
# 11. SHAP EXPLAINABILITY
# ============================================================

try:
    import shap

    print("\n========== SHAP ANALYSIS ==========")

    # TreeExplainer works with the tree-based regressors used here.
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # Summary / beeswarm plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
        max_display=len(features),
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Mean absolute SHAP importance
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
        max_display=len(features),
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dependence plot for ground length, if available.
    if "gl_mm" in features:
        plt.figure()
        shap.dependence_plot(
            "gl_mm",
            shap_values,
            X_test,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / "shap_dependence_ground_length.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save SHAP values in tabular form.
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.to_csv(RESULTS_DIR / "shap_values.csv", index=False)

    print("SHAP plots and values saved.")

except ImportError:
    print("\nSHAP is not installed. Run: pip install shap")
except Exception as exc:
    print("\nSHAP analysis could not be completed:", repr(exc))

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

summary = {
    "dataset_rows_after_target_cleaning": int(len(data)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "features_used": features,
    "best_model_by_test_RMSE": best_name,
    "best_model_test_RMSE_GHz": float(
        metrics_df.iloc[0]["RMSE_GHz"]
    ),
    "best_model_test_MAE_GHz": float(
        metrics_df.iloc[0]["MAE_GHz"]
    ),
    "best_model_test_R2": float(
        metrics_df.iloc[0]["R2"]
    ),
    "boundary_censored_excluded": EXCLUDE_BOUNDARY_CENSORED,
}

with open(RESULTS_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n========== TRAINING COMPLETED ==========")
print(json.dumps(summary, indent=4))
print(f"\nModels saved in: {MODELS_DIR}")
print(f"Metrics and plots saved in: {RESULTS_DIR}")
