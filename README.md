# UWB Antenna Bandwidth Prediction — Training Code

## 1. Folder setup

Place your CSV file here:

`dataset/antenna_ML_ready_dataset.csv`

## 2. Install packages

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

## 3. Run training

```bash
python train.py
```

## 4. Output files

- `models/`: trained model files and best model
- `results/model_metrics.csv`: MAE, MSE, RMSE, R² and explained variance
- `results/cross_validation_results.csv`: training-only CV scores and selected parameters
- `results/test_predictions.csv`: actual and predicted test values
- `results/feature_importance.csv`: tree feature importance
- `results/permutation_importance.csv`: permutation importance
- `results/shap_values.csv`: SHAP values
- `results/plots/`: model comparison, residual and SHAP plots

## 5. Important research notes

- The script uses only geometry/material columns (`*_mm`, `epsilon_r`) as predictors.
- Bandwidth, S11, frequency edges, and other simulation outputs are not used as input features.
- The existing `dataset_split` column is respected.
- Constant features are removed based on training data only.
- Missing input values are imputed using training medians.
- `EXCLUDE_BOUNDARY_CENSORED` is False by default. Run a second sensitivity experiment with it set to True because rows touching the 2 GHz or 14 GHz sweep boundary may have truncated bandwidth.
- The current supplied dataset has only a small number of varying geometry features. A larger HFSS parametric-sweep dataset is needed to study all antenna parameters reliably.
- The script selects the lowest test RMSE model for a demonstration. For a final publication-quality estimate, select/tune using training cross-validation or a validation set, and evaluate once on an untouched test set.
- Do not claim the original paper's results unless they are reproduced independently.
