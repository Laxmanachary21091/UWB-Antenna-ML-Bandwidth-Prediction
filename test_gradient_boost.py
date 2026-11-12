
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# ==========================================
# 1. LOAD TRAINED GRADIENT BOOSTING MODEL
# ==========================================

gb_model = joblib.load(
    "models/gradient_boosting.joblib"
)

# Load training feature information
with open("results/preprocessing_info.json", "r") as f:
    info = json.load(f)

features = info["features"]
medians = pd.Series(info["training_medians"])

print("Model loaded successfully!")
print("Features:", features)


# ==========================================
# 2. ENTER NEW ANTENNA GEOMETRY
# ==========================================

# Example geometry - change these values
new_sample = pd.DataFrame([{
    "gl_mm": 12.0,
    "a4_mm": 2.2,
    "a9_mm": 1.0,
    "a11_mm": 8.0
}])

# Ensure the input matches training columns
new_sample = new_sample[features]

# Apply same missing-value handling as training
new_sample = new_sample.fillna(medians)


# ==========================================
# 3. PREDICT BANDWIDTH
# ==========================================

predicted_bw = gb_model.predict(new_sample)[0]

print("\n========== ML PREDICTION ==========")

print(
    f"Predicted Bandwidth: {predicted_bw:.4f} GHz"
)


# ==========================================
# 4. ENTER HFSS SIMULATED BANDWIDTH
# ==========================================

# Replace with the actual HFSS result
simulated_bw = 7.62

print("\n========== HFSS COMPARISON ==========")

print(
    f"HFSS Simulated Bandwidth: {simulated_bw:.4f} GHz"
)

print(
    f"ML Predicted Bandwidth: {predicted_bw:.4f} GHz"
)


# ==========================================
# 5. CALCULATE PREDICTION ERROR
# ==========================================

absolute_error = abs(
    simulated_bw - predicted_bw
)

error_percentage = (
    absolute_error / abs(simulated_bw)
) * 100 if simulated_bw != 0 else np.nan

print("\n========== PREDICTION ERROR ==========")

print(
    f"Absolute Error: {absolute_error:.4f} GHz"
)

print(
    f"Error Percentage: {error_percentage:.2f}%"
)


# ==========================================
# 6. VISUALIZE COMPARISON
# ==========================================

labels = [
    "HFSS Simulated",
    "Gradient Boosting Predicted"
]

values = [
    simulated_bw,
    predicted_bw
]

colors = [
    "steelblue",
    "yellowgreen"
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    labels,
    values,
    color=colors,
    width=0.5
)

# Add values above bars
for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{value:.3f} GHz",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.ylabel("Bandwidth (GHz)", fontsize=12)

plt.title(
    "HFSS vs Gradient Boosting Bandwidth Prediction",
    fontsize=14
)

plt.ylim(
    0,
    max(values) * 1.2
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/plots/gradient_boosting_test.png",
    dpi=300
)

plt.show()


# ==========================================
# 7. SAVE PREDICTION RESULTS
# ==========================================

result = pd.DataFrame({
    "HFSS_Simulated_Bandwidth_GHz": [simulated_bw],
    "ML_Predicted_Bandwidth_GHz": [predicted_bw],
    "Absolute_Error_GHz": [absolute_error],
    "Error_Percentage": [error_percentage]
})

result.to_csv(
    "results/gradient_boosting_test.csv",
    index=False
)

print("\nPrediction results saved successfully!")
