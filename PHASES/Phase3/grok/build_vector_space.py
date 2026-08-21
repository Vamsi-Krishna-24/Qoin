"""
Vector Space engine with multi-lag analysis for Template 5.
Lags: -2, -1, 0, +1, +2, +3
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from scipy.optimize import minimize

DATA_PATH = "/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/Qoin/Model data/master_raw_aligned.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

target_col = "ICICI_Close"
feature_cols = [c for c in df.columns if c not in ["Date", target_col]]
n_factors = len(feature_cols)
n_days = len(df)

print(f"Loaded {n_days} rows | {n_factors} factors")
print("Date range:", df["Date"].min().date(), "→", df["Date"].max().date())

# ============================================================
# Contemporaneous geometry (for Templates 1-4)
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(df[feature_cols].values)
y = scaler_y.fit_transform(df[[target_col]].values).ravel()

corr = np.corrcoef(X, rowvar=False)
corr = np.clip(corr, -0.999999, 0.999999)

def spherical_embedding_loss(flat_vecs, target_cos):
    n = target_cos.shape[0]
    vecs = flat_vecs.reshape(n, 3)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    dots = vecs @ vecs.T
    mask = np.triu(np.ones_like(target_cos), k=1).astype(bool)
    return np.mean((dots[mask] - target_cos[mask]) ** 2)

def embed_directions(target_cos, seed=42):
    n = target_cos.shape[0]
    rng = np.random.RandomState(seed)
    init = rng.normal(size=(n, 3))
    init /= np.linalg.norm(init, axis=1, keepdims=True)
    res = minimize(spherical_embedding_loss, init.ravel(), args=(target_cos,),
                   method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    vecs = res.x.reshape(n, 3)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

print("Embedding axes...")
directions_3d = embed_directions(corr)
stress = float(spherical_embedding_loss(directions_3d.ravel(), corr))
print(f"3D embedding stress: {stress:.6f}")

model = RidgeCV(alphas=np.logspace(-3, 3, 40), cv=TimeSeriesSplit(n_splits=5))
model.fit(X, y)
weights = model.coef_.copy()
print(f"Contemporaneous alpha: {model.alpha_:.4f}")

price_dir = directions_3d.T @ weights
price_dir_unit = price_dir / (np.linalg.norm(price_dir) + 1e-12)

daily_pos = (directions_3d.T @ X.T).T
y_proj = X @ weights
residual = y - y_proj

step = 2
idx = np.arange(0, n_days, step)
if idx[-1] != n_days - 1:
    idx = np.append(idx, n_days - 1)

daily = []
for i in idx:
    daily.append({
        "date": str(df["Date"].iloc[i].date()),
        "x": float(daily_pos[i, 0]),
        "y": float(daily_pos[i, 1]),
        "z": float(daily_pos[i, 2]),
        "residual_rupee": float(
            scaler_y.inverse_transform([[y[i]]])[0][0] -
            scaler_y.inverse_transform([[y_proj[i]]])[0][0]
        )
    })

axes = []
for i, name in enumerate(feature_cols):
    axes.append({
        "id": i,
        "name": name,
        "direction": directions_3d[i].tolist(),
        "weight": float(weights[i]),
        "abs_weight": float(abs(weights[i]))
    })
axes_sorted = sorted(axes, key=lambda a: a["abs_weight"], reverse=True)

# ============================================================
# Multi-lag analysis for Template 5
# lag > 0  : factors(t) → price(t+lag)     (forward)
# lag = 0  : factors(t) → price(t)         (contemporaneous)
# lag < 0  : factors(t) → price(t+lag)     (backward)
# ============================================================
print("\nBuilding multi-lag models...")
LAGS = [-2, -1, 0, 1, 2, 3]
lag_results = {}

for lag in LAGS:
    df_l = df.copy()
    if lag >= 0:
        df_l["target"] = df_l[target_col].shift(-lag)   # future price
    else:
        df_l["target"] = df_l[target_col].shift(-lag)   # past price (shift positive)

    df_l = df_l.dropna().reset_index(drop=True)

    X_l = StandardScaler().fit_transform(df_l[feature_cols].values)
    y_l = StandardScaler().fit_transform(df_l[["target"]].values).ravel()

    m = RidgeCV(alphas=np.logspace(-3, 3, 30), cv=TimeSeriesSplit(n_splits=5))
    m.fit(X_l, y_l)

    y_hat = m.predict(X_l)
    r2 = float(r2_score(y_l, y_hat))
    resid = y_l - y_hat

    lag_results[str(lag)] = {
        "lag": lag,
        "description": (
            f"factors(t) → price(t{lag:+d})" if lag != 0
            else "factors(t) → price(t)  [contemporaneous]"
        ),
        "alpha": float(m.alpha_),
        "r2": r2,
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
        "n_samples": int(len(df_l)),
        "weights": {name: float(w) for name, w in zip(feature_cols, m.coef_)}
    }
    print(f"  lag {lag:+d}  R² = {r2:.4f}  α = {m.alpha_:.4f}  n = {len(df_l)}")

# ============================================================
# Export
# ============================================================
output = {
    "meta": {
        "n_factors": n_factors,
        "n_days": n_days,
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "contemporaneous_alpha": float(model.alpha_),
        "lagged_alpha": lag_results["1"]["alpha"],
        "embedding_stress_mse": stress,
        "scrubber_points": len(daily)
    },
    "axes": axes_sorted,
    "price_direction": {
        "name": "ICICI_Close (projection direction)",
        "direction": price_dir_unit.tolist()
    },
    "daily": daily,
    "feature_order": feature_cols,
    "weights_contemporaneous": {name: float(w) for name, w in zip(feature_cols, weights)},
    "weights_lagged": lag_results["1"]["weights"],
    "multi_lag": lag_results
}

out_path = Path("vector_space.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nExported → {out_path.resolve()}")
print("Template 5 multi-lag data included.")
print("Done.")
