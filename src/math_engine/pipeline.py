"""
Math engine: three rolling windows, no lookahead.

as_of(df, date)          — single entry point for slice-to-date; used everywhere
rolling_zscore()         — vectorized, rolling 126-day z-score per factor
compute_snapshot()       — per-date PCA + geometric projection (biplot)
compute_rolling_results()— full history of OLS residuals, R², MR signals
build_fingerprint_index()— assemble all historical 5-day z-scored vectors
search_fingerprints()    — cosine-similarity top-K search, exclude recent 21d
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ── Core no-lookahead helper ──────────────────────────────────────────────────

def as_of(df: pd.DataFrame | pd.Series, date) -> pd.DataFrame | pd.Series:
    """Return rows with index <= date. Single entry point for all slice ops."""
    return df[df.index <= pd.Timestamp(date)]


# ── Rolling z-score (vectorized, no lookahead) ────────────────────────────────

def rolling_zscore(df: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """
    Z-score each column using its own trailing `window`-day mean and std.
    Constant-valued series (std=0) return z=0 — the value is exactly at its mean.
    Requires the full window to be present (min_periods=window); no partial windows.
    """
    rm = df.rolling(window, min_periods=window).mean()
    rs = df.rolling(window, min_periods=window).std(ddof=1)
    z  = (df - rm) / rs
    # Where std is 0 (constant series), z-score is 0 by definition
    z  = z.where(rs.fillna(1.0) > 1e-12, other=0.0)
    return z


# ── PCA snapshot for a single date (biplot) ───────────────────────────────────

def compute_snapshot(
    factors_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    date,
    pca_window: int = 126,
    min_variance: float = 0.90,
) -> dict:
    """
    Compute PCA snapshot for `date`.  All data sliced via as_of — no lookahead.

    Returns dict with keys:
      loadings, pc_scores, active_factors, n_components, cum_variance,
      eigenvalues, stock_positions, stock_ols_r2, error (if failure)
    """
    date = pd.Timestamp(date)

    # Step 1: slice to date, compute rolling z-scores
    f_asof  = as_of(factors_df, date)
    z_asof  = rolling_zscore(f_asof, pca_window)

    if len(z_asof) < pca_window:
        return {"error": f"Only {len(z_asof)} days available; need {pca_window}."}

    z_win = z_asof.iloc[-pca_window:]          # (window, n_factors)

    # Step 2: drop any factor column with NaN in this window
    valid   = z_win.notna().all()
    z_clean = z_win.loc[:, valid]
    active  = z_clean.columns.tolist()

    if len(active) < 2:
        return {"error": f"Fewer than 2 valid factors in {pca_window}-day window at {date.date()}."}

    # Step 3: PCA
    X   = z_clean.values                       # (window, n_active)
    pca = PCA()
    try:
        pca.fit(X)
    except Exception as e:
        return {"error": f"PCA failed: {e}"}

    cum_var     = np.cumsum(pca.explained_variance_ratio_)
    n_comp      = int(np.searchsorted(cum_var, min_variance)) + 1
    n_comp      = min(n_comp, len(active))
    loadings    = pca.components_[:n_comp]     # (n_comp, n_active)
    pc_scores   = X @ loadings.T               # (window, n_comp)

    # Step 4: geometric projection of each stock onto PC directions (in time-series space)
    z_stocks_asof = rolling_zscore(as_of(stocks_df, date), pca_window)
    if len(z_stocks_asof) < pca_window:
        z_stocks_win = None
    else:
        z_stocks_win = z_stocks_asof.iloc[-pca_window:]

    stock_positions: dict[str, tuple[float, float]] = {}
    if z_stocks_win is not None:
        for stock in stocks_df.columns:
            sv = z_stocks_win[stock].values if stock in z_stocks_win.columns else None
            if sv is None or np.any(np.isnan(sv)):
                continue
            # Inner product with each PC score vector, normalised by window length
            coords = [float(sv @ pc_scores[:, k]) / pca_window for k in range(min(2, n_comp))]
            while len(coords) < 2:
                coords.append(0.0)
            stock_positions[stock] = (coords[0], coords[1])

    # Step 5: OLS of stock returns on PC scores (for R²)
    s_asof      = as_of(stocks_df, date)
    s_win       = s_asof.iloc[-pca_window:] if len(s_asof) >= pca_window else None
    stock_ols_r2: dict[str, float] = {}
    if s_win is not None:
        Xb = np.column_stack([np.ones(pca_window), pc_scores])
        for stock in stocks_df.columns:
            y = s_win[stock].values if stock in s_win.columns else None
            if y is None or np.any(np.isnan(y)):
                continue
            try:
                beta, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
                y_hat   = Xb @ beta
                ss_res  = float(np.sum((y - y_hat) ** 2))
                ss_tot  = float(np.sum((y - y.mean()) ** 2))
                stock_ols_r2[stock] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
            except Exception:
                continue

    return {
        "pca":            pca,
        "loadings":       loadings,
        "pc_scores":      pc_scores,
        "active_factors": active,
        "n_components":   n_comp,
        "cum_variance":   cum_var[:n_comp].tolist(),
        "eigenvalues":    pca.explained_variance_[:n_comp].tolist(),
        "stock_positions": stock_positions,
        "stock_ols_r2":   stock_ols_r2,
        "date":           date,
    }


# ── Full rolling history: residuals, R², MR signals ──────────────────────────

def compute_rolling_results(
    factors_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    pca_window: int = 126,
    mr_window: int = 15,
    min_variance: float = 0.90,
) -> dict[str, pd.DataFrame]:
    """
    Loop over all dates with sufficient history; compute rolling PCA + OLS per stock.
    No lookahead: each iteration uses only data[i-pca_window+1 : i+1].

    Returns dict with DataFrames:
      residuals, r2, mr_sigma, signal, fitted_returns, n_components_series
    """
    n          = len(factors_df)
    dates      = factors_df.index
    stocks     = stocks_df.columns.tolist()

    residuals_out      = {s: np.full(n, np.nan) for s in stocks}
    r2_out             = {s: np.full(n, np.nan) for s in stocks}
    fitted_ret_out     = {s: np.full(n, np.nan) for s in stocks}
    n_comp_out         = np.full(n, np.nan)

    # Precompute rolling z-scores (vectorized) — used inside window slices
    z_all = rolling_zscore(factors_df, pca_window)

    for i in range(pca_window - 1, n):
        z_win = z_all.iloc[i - pca_window + 1 : i + 1]     # (pca_window, n_factors)

        # Valid factors: no NaN in this window
        valid_mask  = z_win.notna().all()
        z_clean     = z_win.loc[:, valid_mask].values        # (pca_window, n_valid)

        if z_clean.shape[1] < 2:
            continue

        # PCA
        try:
            pca  = PCA()
            pca.fit(z_clean)
            cv   = np.cumsum(pca.explained_variance_ratio_)
            nc   = min(int(np.searchsorted(cv, min_variance)) + 1, z_clean.shape[1])
            L    = pca.components_[:nc]                      # (nc, n_valid)
            P    = z_clean @ L.T                             # (pca_window, nc)
        except Exception:
            continue

        n_comp_out[i] = nc
        Xb = np.column_stack([np.ones(pca_window), P])      # (pca_window, nc+1)

        for s in stocks:
            y_series = stocks_df[s].iloc[i - pca_window + 1 : i + 1]
            y = y_series.values
            if np.any(np.isnan(y)):
                continue
            try:
                beta, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
                y_hat  = Xb @ beta
                resid  = float(y[-1] - y_hat[-1])
                fit_r  = float(y_hat[-1])
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
                residuals_out[s][i]  = resid
                fitted_ret_out[s][i] = fit_r
                r2_out[s][i]         = r2
            except Exception:
                continue

    residuals_df    = pd.DataFrame(residuals_out,    index=dates)
    r2_df           = pd.DataFrame(r2_out,           index=dates)
    fitted_ret_df   = pd.DataFrame(fitted_ret_out,   index=dates)
    n_comp_series   = pd.Series(n_comp_out,          index=dates, name="n_components")

    mr_sigma_df = residuals_df.rolling(mr_window, min_periods=max(3, mr_window // 2)).std(ddof=1)
    signal_df   = (residuals_df.abs() > 2.0 * mr_sigma_df)

    return {
        "residuals":       residuals_df,
        "r2":              r2_df,
        "fitted_returns":  fitted_ret_df,
        "mr_sigma":        mr_sigma_df,
        "signal":          signal_df,
        "n_components":    n_comp_series,
    }


# ── Fingerprint index ─────────────────────────────────────────────────────────

def build_fingerprint_index(
    factors_df: pd.DataFrame,
    pca_window: int = 126,
    fp_window: int = 5,
    max_nan_pct: float = 0.50,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Build matrix of all historical fp_window-day z-scored factor vectors.

    Factors with >max_nan_pct NaN across the full history are excluded from
    the fingerprint to ensure consistent vector dimensions across all dates.

    Returns:
        matrix     : (n_valid_dates, fp_window * n_fp_factors)
        valid_dates: DatetimeIndex aligned to matrix rows
        factor_cols: list of factor column names included in fingerprint
    """
    # Exclude high-NaN factors so the fingerprint covers the full history
    nan_pct     = factors_df.isna().mean()
    fp_cols     = nan_pct[nan_pct <= max_nan_pct].index.tolist()
    f_fp        = factors_df[fp_cols]

    z_df        = rolling_zscore(f_fp, pca_window)
    n           = len(z_df)

    rows, row_dates = [], []
    for i in range(fp_window - 1, n):
        block = z_df.iloc[i - fp_window + 1 : i + 1]        # (fp_window, n_fp_factors)
        if block.isna().any().any():
            continue
        rows.append(block.values.flatten())
        row_dates.append(z_df.index[i])

    if not rows:
        return np.empty((0, 0)), pd.DatetimeIndex([]), fp_cols

    return np.array(rows), pd.DatetimeIndex(row_dates), fp_cols


def search_fingerprints(
    fp_matrix: np.ndarray,
    fp_dates: pd.DatetimeIndex,
    stocks_df: pd.DataFrame,
    query_date,
    factors_df: pd.DataFrame,
    pca_window: int = 126,
    fp_window: int = 5,
    top_k: int = 5,
    exclude_recent: int = 21,
    forward_windows: tuple[int, ...] = (5, 10, 15),
) -> list[dict]:
    """
    Cosine-similarity search for the top_k historical 5-day windows most similar
    to the query_date window.  Excludes the most recent `exclude_recent` trading days.

    Returns list of dicts with keys: date, similarity, forward_returns {5d, 10d, 15d}
    """
    if fp_matrix.shape[0] == 0:
        return []

    query_date = pd.Timestamp(query_date)

    # Build query vector using only the same factor columns as the fingerprint index
    n_fp_factors = fp_matrix.shape[1] // fp_window
    fp_factor_cols_needed = factors_df.columns[
        factors_df.isna().mean() <= 0.50
    ].tolist()
    f_fp_only = factors_df[fp_factor_cols_needed] if fp_factor_cols_needed else factors_df
    z_df  = rolling_zscore(as_of(f_fp_only, query_date), pca_window)
    if len(z_df) < fp_window or z_df.iloc[-fp_window:].isna().any().any():
        return []
    q_vec = z_df.iloc[-fp_window:].values.flatten().reshape(1, -1)
    if q_vec.shape[1] != fp_matrix.shape[1]:
        return []

    # Mask out recent dates
    cutoff     = query_date - pd.tseries.offsets.BusinessDay(exclude_recent)
    valid_mask = fp_dates <= cutoff
    if valid_mask.sum() < top_k:
        return []

    search_matrix = fp_matrix[valid_mask]
    search_dates  = fp_dates[valid_mask]

    # Cosine similarity via NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(top_k, len(search_matrix)), metric="cosine")
    nn.fit(search_matrix)
    distances, indices = nn.kneighbors(q_vec)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        match_date = search_dates[idx]
        fwd: dict[int, float | None] = {}
        for fw in forward_windows:
            future_slice = stocks_df[stocks_df.index > match_date].iloc[:fw]
            if len(future_slice) >= fw:
                fwd[fw] = float(future_slice.sum().mean())   # avg cross-stock cumulative log-ret
            else:
                fwd[fw] = None

        # Also return per-stock forward returns
        per_stock: dict[str, dict[int, float | None]] = {}
        for s in stocks_df.columns:
            per_stock[s] = {}
            for fw in forward_windows:
                future_slice = stocks_df[s][stocks_df.index > match_date].iloc[:fw]
                if len(future_slice) >= fw:
                    per_stock[s][fw] = float(future_slice.sum())
                else:
                    per_stock[s][fw] = None

        results.append({
            "date":            match_date,
            "similarity":      float(1.0 - dist),
            "forward_returns": fwd,
            "per_stock_fwd":   per_stock,
        })

    return results
