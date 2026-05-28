"""
QOIN — Quantitative Optimized Indian Nifty
Phase 1 dashboard: factor biplot, parallel coordinates, correlation heatmap,
mean reversion view.

Launch: streamlit run src/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st

from data.loader import load_all_factors, FACTOR_TRANSFORMS
from math_engine.pipeline import (
    compute_snapshot,
    compute_rolling_results,
    build_fingerprint_index,
    search_fingerprints,
)
from views.biplot import render_biplot
from views.parallel_coords import render_parallel_coords
from views.heatmap import render_heatmap
from views.mean_reversion import render_mean_reversion

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QOIN — Indian Equity Factor Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data (cached permanently — data files don't change at runtime) ───────
@st.cache_data(show_spinner="Loading CSVs and computing transforms…")
def get_data():
    return load_all_factors()

try:
    factors_df, stocks_df, meta = get_data()
except FileNotFoundError as e:
    st.error(f"**Data load failed:** {e}")
    st.stop()

all_factor_cols = list(factors_df.columns)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("QOIN Controls")

selected_factors = st.sidebar.multiselect(
    "Active factors",
    options=all_factor_cols,
    default=all_factor_cols,
    help="Deselect factors to widen the common analysis window.",
)
if not selected_factors:
    st.sidebar.warning("Select at least one factor.")
    st.stop()

pca_window  = st.sidebar.slider("PCA / Regression window (days)", 60, 252, 126, step=5)
mr_window   = st.sidebar.slider("Mean reversion σ window (days)",   5,  30,  15, step=1)
fp_window   = st.sidebar.slider("Fingerprint window (days)",         3,  10,   5, step=1)
min_var     = st.sidebar.slider("Min PCA variance explained",      0.70, 0.99, 0.90, step=0.01)
top_k       = st.sidebar.slider("Fingerprint top-K matches",         3,  10,   5, step=1)

st.sidebar.markdown("---")

# Subset factors
factors_sub = factors_df[selected_factors]

# Common analysis window (latest first-valid-index across selected factors)
first_valids = {c: factors_sub[c].first_valid_index() for c in selected_factors}
first_valids = {k: v for k, v in first_valids.items() if v is not None}
common_start = max(first_valids.values()) if first_valids else factors_sub.index[0]
common_end   = factors_sub.index[-1]

# ── Common window callout ─────────────────────────────────────────────────────
st.sidebar.info(
    f"**Common analysis window**\n\n"
    f"Start: **{common_start.date()}**\n\n"
    f"End:  **{common_end.date()}**\n\n"
    f"Driven by: **{max(first_valids, key=first_valids.get)}**"
    if first_valids else "No valid window."
)

# Date scrubber — only dates with sufficient history for PCA
min_date_idx = pca_window + mr_window
avail_dates  = factors_sub.index[min_date_idx:]
if len(avail_dates) == 0:
    st.error("Not enough data for the selected window. Reduce PCA window or add more history.")
    st.stop()

selected_date_str = st.sidebar.select_slider(
    "Analysis date",
    options=avail_dates.strftime("%Y-%m-%d").tolist(),
    value=avail_dates[-1].strftime("%Y-%m-%d"),
)
selected_date = pd.Timestamp(selected_date_str)

# ── Heavy computation (cached by parameters) ──────────────────────────────────
@st.cache_data(show_spinner="Computing rolling PCA + OLS (cached after first run)…")
def get_rolling_results(factor_key: tuple, pca_win: int, mr_win: int, min_v: float):
    f  = factors_df[list(factor_key)]
    return compute_rolling_results(f, stocks_df, pca_window=pca_win,
                                   mr_window=mr_win, min_variance=min_v)

@st.cache_data(show_spinner="Building fingerprint index…")
def get_fp_index(factor_key: tuple, pca_win: int, fp_win: int):
    f = factors_df[list(factor_key)]
    return build_fingerprint_index(f, pca_window=pca_win, fp_window=fp_win)

factor_key   = tuple(selected_factors)
rolling_res  = get_rolling_results(factor_key, pca_window, mr_window, min_var)
fp_matrix, fp_dates, fp_factor_cols = get_fp_index(factor_key, pca_window, fp_window)

# ── Sidebar diagnostics ───────────────────────────────────────────────────────
snapshot_diag = compute_snapshot(factors_sub, stocks_df, selected_date, pca_window, min_var)

with st.sidebar.expander("Diagnostics", expanded=False):
    st.markdown(f"**Data dir:** `{meta['data_dir']}`")
    st.markdown(f"**Factors loaded:** {len(selected_factors)}")
    st.markdown(f"**Fingerprint index size:** {len(fp_dates):,} days")

    if "error" not in snapshot_diag:
        st.markdown(f"**PCA components retained:** {snapshot_diag['n_components']}")
        cv_pct = [f"{v:.1%}" for v in snapshot_diag['cum_variance']]
        st.markdown(f"**Cumulative variance:** {' / '.join(cv_pct)}")
        r2_lines = [f"{s}: {v:.3f}" for s, v in snapshot_diag.get("stock_ols_r2", {}).items()]
        st.markdown(f"**OLS R² (at selected date):** {', '.join(r2_lines)}")
    else:
        st.warning(snapshot_diag["error"])

    if meta["dropped_factors"]:
        st.markdown("**Dropped factors (>80% NaN):**")
        for k, reason in meta["dropped_factors"].items():
            st.markdown(f"- `{k}`: {reason}")
    else:
        st.success("No factors dropped.")

    st.markdown("**Factor transforms:**")
    for f_name in selected_factors:
        desc = FACTOR_TRANSFORMS.get(f_name, "—")
        st.markdown(f"- `{f_name}`: {desc}")

# ── Four tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Biplot",
    "〰️ Parallel Coordinates",
    "🟦 Correlation Heatmap",
    "📉 Mean Reversion",
])

# ── Tab 1: Biplot ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Factor Biplot (PCA loadings + stock positions)")
    st.caption(
        "Arrows = factor loadings on PC1/PC2. Diamonds = stocks positioned by "
        "geometric projection of their z-scored returns onto the PC score vectors. "
        "Date scrubber in sidebar."
    )
    snapshot = compute_snapshot(factors_sub, stocks_df, selected_date, pca_window, min_var)
    fig_biplot = render_biplot(snapshot)
    st.plotly_chart(fig_biplot, use_container_width=True)

    if "error" not in snapshot:
        col1, col2, col3 = st.columns(3)
        col1.metric("PCA components", snapshot["n_components"])
        col2.metric("Variance explained", f"{snapshot['cum_variance'][-1]:.1%}")
        col3.metric("Active factors", len(snapshot["active_factors"]))

# ── Tab 2: Parallel Coordinates ───────────────────────────────────────────────
with tab2:
    st.subheader("Parallel Coordinates — Factor State")
    st.caption(
        "Bold blue = today's z-scored factor state. Faded lines = top fingerprint-matched "
        "historical 5-day windows. Line color = stock performance in the 10 days after "
        "the match (green=up, red=down)."
    )

    focus_stock = st.selectbox(
        "Color by return of:",
        options=list(stocks_df.columns),
        format_func=lambda s: {"nifty50": "NIFTY 50", "bank_nifty": "BANK NIFTY",
                                "icici_bank": "ICICI Bank"}.get(s, s),
        key="pc_stock",
    )

    matches = search_fingerprints(
        fp_matrix, fp_dates, stocks_df,
        query_date=selected_date,
        factors_df=factors_sub,
        pca_window=pca_window,
        fp_window=fp_window,
        top_k=top_k,
        exclude_recent=21,
        forward_windows=(5, 10, 15),
    )

    fig_pc = render_parallel_coords(
        factors_sub, stocks_df, matches, selected_date,
        pca_window=pca_window, focus_stock=focus_stock, forward_days=10,
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    if matches:
        st.markdown("**Fingerprint matches:**")
        rows = []
        for m in matches:
            per = m.get("per_stock_fwd", {})
            rows.append({
                "Date":         m["date"].strftime("%Y-%m-%d"),
                "Similarity":   f"{m['similarity']:.4f}",
                "NIFTY 5d":     f"{per.get('nifty50', {}).get(5, None)*100:.2f}%" if per.get('nifty50', {}).get(5) is not None else "—",
                "NIFTY 10d":    f"{per.get('nifty50', {}).get(10, None)*100:.2f}%" if per.get('nifty50', {}).get(10) is not None else "—",
                "NIFTY 15d":    f"{per.get('nifty50', {}).get(15, None)*100:.2f}%" if per.get('nifty50', {}).get(15) is not None else "—",
                "BNIFTY 5d":    f"{per.get('bank_nifty', {}).get(5, None)*100:.2f}%" if per.get('bank_nifty', {}).get(5) is not None else "—",
                "ICICI 10d":    f"{per.get('icici_bank', {}).get(10, None)*100:.2f}%" if per.get('icici_bank', {}).get(10) is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No fingerprint matches found (insufficient index or data).")

# ── Tab 3: Correlation Heatmap ────────────────────────────────────────────────
with tab3:
    st.subheader("Correlation Heatmap")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        rolling_toggle = st.toggle(f"Rolling {pca_window}d window", value=False)
        cluster_toggle = st.toggle("Hierarchical clustering", value=True)

    fig_heatmap = render_heatmap(
        factors_sub, stocks_df, selected_date,
        rolling=rolling_toggle,
        pca_window=pca_window,
        cluster=cluster_toggle,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption(
        "Upper block: factor × factor correlations. "
        "Rows below divider: each stock's return correlation with each factor. "
        "Toggle to switch between full-history and rolling window."
    )

# ── Tab 4: Mean Reversion ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Mean Reversion — Factor-Implied Fair Value")
    st.caption(
        "Orange line = factor-implied mean (close × exp(−residual)). "
        "Shaded band = ±2σ of 15-day rolling residual std. "
        "Markers = days where |residual| > 2σ (red=above, green=below). "
        "Right panel = residual distribution + KDE."
    )

    mr_stock = st.selectbox(
        "Select stock:",
        options=list(stocks_df.columns),
        format_func=lambda s: {"nifty50": "NIFTY 50", "bank_nifty": "BANK NIFTY",
                                "icici_bank": "ICICI Bank"}.get(s, s),
        key="mr_stock",
    )

    ohlcv_df  = meta["stocks_ohlcv"][mr_stock]
    resids    = rolling_res["residuals"][mr_stock]
    sigma_ser = rolling_res["mr_sigma"][mr_stock]
    sig_ser   = rolling_res["signal"][mr_stock]

    fig_mr = render_mean_reversion(
        ohlcv_df, stocks_df[mr_stock], resids, sigma_ser, sig_ser,
        stock=mr_stock, date=selected_date,
    )
    st.plotly_chart(fig_mr, use_container_width=True)

    # Current signal status
    latest_resid = resids.loc[:selected_date].dropna()
    latest_sigma = sigma_ser.loc[:selected_date].dropna()
    if len(latest_resid) > 0 and len(latest_sigma) > 0:
        lr   = float(latest_resid.iloc[-1])
        ls   = float(latest_sigma.iloc[-1])
        st.metric(
            label="Current residual",
            value=f"{lr*100:.3f}%",
            delta=f"σ threshold = ±{2*ls*100:.3f}%",
            delta_color="off",
        )
        if abs(lr) > 2 * ls:
            direction = "above" if lr > 0 else "below"
            st.warning(
                f"⚠ Mean reversion candidate: residual is {direction} the ±2σ band. "
                f"({lr*100:.3f}% vs threshold {2*ls*100:.3f}%)"
            )
        else:
            st.success("Residual within ±2σ band — no mean reversion flag.")
