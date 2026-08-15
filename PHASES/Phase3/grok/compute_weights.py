import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
path = "/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/Qoin/Model data/master_raw_aligned.csv"
df = pd.read_csv(path, parse_dates=["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

target_col = "ICICI_Close"
feature_cols = [c for c in df.columns if c not in ["Date", target_col]]

print(f"Factors ({len(feature_cols)}):")
for i, c in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {c}")

# -------------------------------------------------
# 2. Normalize
# -------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df[feature_cols].values)
y = scaler_y.fit_transform(df[[target_col]].values).ravel()

# -------------------------------------------------
# 3. Find weight vector w (Ridge with time-series CV)
# -------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
alphas = np.logspace(-3, 3, 40)

model = RidgeCV(alphas=alphas, cv=tscv)
model.fit(X, y)

print(f"\nChosen alpha: {model.alpha_:.4f}")

weights = pd.Series(model.coef_, index=feature_cols)
weights_sorted = weights.reindex(weights.abs().sort_values(ascending=False).index)

print("\nWeight vector w (sorted by absolute importance):")
print(weights_sorted.round(4).to_string())

# Save weights
weights_sorted.to_csv("icici_factor_weights.csv", header=["weight"])
print("\nSaved: icici_factor_weights.csv")
