"""
Build a geometrically faithful 3D representation of the 21-factor vector space.
Now also exports the price projection direction.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize

# ============================================================
# 1. Load & clean
# ============================================================
DATA_PATH = "/Users/surisettivamsikrishna/Downloads/Vamsi Pc/CODES/Qoin/Model data/master_raw_aligned.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

target_col = "ICICI_Close"
feature_cols = [c for c in df.columns if c not in ["Date", target_col]]
n_factors = len(feature_cols)

print(f"Loaded {len(df)} rows | {n_factors} factors")
print("Date range:", df["Date"].min().date(), "→", df["Date"].max().date())

# ============================================================
# 2. Normalize
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df[feature_cols].values)
y = scaler_y.fit_transform(df[[target_col]].values).ravel()

# ============================================================
# 3. Correlation → target angles
# ============================================================
corr = np.corrcoef(X, rowvar=False)
corr = np.clip(corr, -0.999999, 0.999999)
target_cos = corr.copy()

print("Correlation matrix computed.")

# ============================================================
# 4. Embed 21 directions into 3D
# ============================================================
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

    res = minimize(
        spherical_embedding_loss,
        init.ravel(),
        args=(target_cos,),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12}
    )
    vecs = res.x.reshape(n, 3)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    final_loss = spherical_embedding_loss(vecs.ravel(), target_cos)
    print(f"3D angle-embedding stress (MSE on cosines): {final_loss:.6f}")
    return vecs

print("Embedding 21 axes into 3D (preserving angles)...")
directions_3d = embed_directions(target_cos)

# ============================================================
# 5. Weight vector w (Ridge)
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)
model = RidgeCV(alphas=np.logspace(-3, 3, 40), cv=tscv)
model.fit(X, y)
weights = model.coef_.copy()
alpha_chosen = float(model.alpha_)

print(f"Ridge alpha: {alpha_chosen:.4f}")

# ============================================================
# 6. Price projection direction in the same 3D embedding
#    d_price = sum( w_i * d_i )
# ============================================================
price_dir_3d = directions_3d.T @ weights
price_dir_norm = np.linalg.norm(price_dir_3d)
price_dir_unit = price_dir_3d / (price_dir_norm + 1e-12)

print(f"Price direction magnitude (pre-normalise): {price_dir_norm:.4f}")

# ============================================================
# 7. Export single JSON
# ============================================================
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

output = {
    "meta": {
        "n_factors": n_factors,
        "n_days": int(len(df)),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "ridge_alpha": alpha_chosen,
        "embedding_stress_mse": float(spherical_embedding_loss(directions_3d.ravel(), target_cos)),
        "note": "Directions preserve real pairwise correlations. Price direction is the weighted sum of factor directions."
    },
    "axes": axes_sorted,
    "price_direction": {
        "name": "ICICI_Close (projection direction)",
        "direction": price_dir_unit.tolist(),
        "magnitude_before_normalise": float(price_dir_norm),
        "description": "Direction of the linear combination sum(w_i * factor_i) inside the 3D embedding"
    },
    "correlation_matrix": corr.tolist(),
    "feature_order": feature_cols,
    "weights_original_order": weights.tolist()
}

out_path = Path("vector_space.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nExported → {out_path.resolve()}")
print("JSON now also contains the price projection direction.")
print("Done.")
