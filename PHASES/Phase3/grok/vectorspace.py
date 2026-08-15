import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
path = "/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/Qoin/Model data/master_raw_aligned.csv"
df = pd.read_csv(path, parse_dates=["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

print("Data shape:", df.shape)
print("Date range:", df["Date"].min().date(), "→", df["Date"].max().date())

# -------------------------------------------------
# 2. Define target and factors
# -------------------------------------------------
target_col = "ICICI_Close"
feature_cols = [c for c in df.columns if c not in ["Date", target_col]]

print(f"\nNumber of factor axes: {len(feature_cols)}")
print("Factors:", feature_cols)

# -------------------------------------------------
# 3. Normalize (z-score) – this builds the vector space
# -------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_raw = df[feature_cols].values
y_raw = df[[target_col]].values

X = scaler_X.fit_transform(X_raw)          # shape: (n_days, n_factors)
y = scaler_y.fit_transform(y_raw).ravel()  # shape: (n_days,)

# Put normalized data back into a DataFrame for easy inspection
df_norm = pd.DataFrame(X, columns=feature_cols)
df_norm["ICICI_Close_z"] = y
df_norm["Date"] = df["Date"].values

print("\nNormalized data (first 5 rows):")
print(df_norm.head().round(3))

print("\nNormalization check (should be ~0 mean, ~1 std):")
print(df_norm[feature_cols].mean().abs().max())
print(df_norm[feature_cols].std().mean())