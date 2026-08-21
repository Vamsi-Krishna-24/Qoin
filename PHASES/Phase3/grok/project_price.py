import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import joblib

# -------------------------------------------------
# 1. Load and prepare data (same as before)
# -------------------------------------------------
path = "/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/Qoin/Model data/master_raw_aligned.csv"
df = pd.read_csv(path, parse_dates=["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

target_col = "ICICI_Close"
feature_cols = [c for c in df.columns if c not in ["Date", target_col]]

# -------------------------------------------------
# 2. Fit the scalers and the model once
# -------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df[feature_cols].values)
y = scaler_y.fit_transform(df[[target_col]].values).ravel()

model = RidgeCV(alphas=np.logspace(-3, 3, 40), cv=TimeSeriesSplit(n_splits=5))
model.fit(X, y)

print("Model ready.")
print("Chosen alpha:", round(model.alpha_, 4))
print("\nWeights:")
for name, w in zip(feature_cols, model.coef_):
    print(f"  {name:30s} {w:+.4f}")

# -------------------------------------------------
# 3. Function: give me today's factors → get projected price + residual
# -------------------------------------------------
def project_price(factor_values: dict):
    """
    factor_values : dictionary with the 21 factor names as keys
                    and today's raw values as values
    
    Returns:
        projected_price  (in ₹)
        residual         (actual - projected)  [only if actual is also given]
    """
    # Arrange values in the exact same order as training
    raw = np.array([factor_values[col] for col in feature_cols]).reshape(1, -1)
    
    # Normalize
    x_scaled = scaler_X.transform(raw)
    
    # Project
    y_scaled_pred = model.predict(x_scaled)[0]
    
    # Back to rupees
    projected_price = scaler_y.inverse_transform([[y_scaled_pred]])[0][0]
    
    return projected_price

# -------------------------------------------------
# Example usage
# -------------------------------------------------
# Take the last day in the data as a test
last_row = df.iloc[-1]
example_factors = {col: last_row[col] for col in feature_cols}
actual_price = last_row[target_col]

predicted = project_price(example_factors)

print("\n----- Test on last day -----")
print(f"Date:            {last_row['Date'].date()}")
print(f"Actual ICICI:    ₹{actual_price:.2f}")
print(f"Projected:       ₹{predicted:.2f}")
print(f"Residual:        ₹{actual_price - predicted:.2f}")