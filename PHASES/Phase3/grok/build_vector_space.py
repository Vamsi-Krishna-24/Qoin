"""
Build a geometrically faithful 3D representation of the 21-factor vector space.
Exports a single JSON that any HTML front-end can consume.
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
# 2. Normalize (this creates the proper vector space)
# ============================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df[feature_cols].values)          # (T, 21)
y = scaler_y.fit_transform(df[[target_col]].values).ravel()  # (T,)

# ============================================================
# 3. Correlation matrix → target angles
# ============================================================
corr = np.corrcoef(X, rowvar=False)          # (21, 21)
# numerical safety
corr = np.clip(corr, -0.999999, 0.999999)
target_cos = corr.copy()                     # we will match dot products of unit vectors

print("\nCorrelation matrix computed.")

# ============================================================
# 4. Embed 21 directions into 3D while preserving angles
#    We optimise unit vectors so that their dot products
#    stay as close as possible to the real correlations.
# ============================================================
def spherical_embedding_loss(flat_vecs, target_cos):
    """flat_vecs: (n*3,) → reshape to (n,3), normalise, compute stress"""
    n = target_cos.shape[0]
    vecs = flat_vecs.reshape(n, 3)
    # re-normalise to be safe
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    dots = vecs @ vecs.T
    # only upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(target_cos), k=1).astype(bool)
    return np.mean((dots[mask] - target_cos[mask]) ** 2)

def embed_directions(target_cos, seed=42):
    n = target_cos.shape[0]
    rng = np.random.RandomState(seed)
    # good initialisation: random on sphere
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
# 6. Daily coordinates, projection, residual (optional but useful)
# ============================================================
# Projected price in z-space
y_proj = X @ weights
residual = y - y_proj

# ============================================================
# 7. Export single JSON
# ============================================================
# Make directions and weights easy to use in JS
axes = []
for i, name in enumerate(feature_cols):
    axes.append({
        "id": i,
        "name": name,
        "direction": directions_3d[i].tolist(),   # unit vector [x,y,z]
        "weight": float(weights[i]),
        "abs_weight": float(abs(weights[i]))
    })

# sort by absolute weight for convenience
axes_sorted = sorted(axes, key=lambda a: a["abs_weight"], reverse=True)

output = {
    "meta": {
        "n_factors": n_factors,
        "n_days": int(len(df)),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "ridge_alpha": alpha_chosen,
        "embedding_stress_mse": float(spherical_embedding_loss(directions_3d.ravel(), target_cos)),
        "note": "Directions are unit vectors in 3D optimised to preserve real pairwise correlations (angles)."
    },
    "axes": axes_sorted,
    "correlation_matrix": corr.tolist(),
    "feature_order": feature_cols,           # original order matching X columns
    "weights_original_order": weights.tolist()
}

out_path = Path("vector_space.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nExported → {out_path.resolve()}")
print("JSON contains:")
print("  • meta information")
print("  • 21 axes with 3D directions + weights")
print("  • full correlation matrix")
print("  • original feature order & weights")
print("\nDone. HTML can now be 100% data-driven from this file.")
