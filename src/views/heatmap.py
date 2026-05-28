"""
View 3 — Correlation heatmap.
Factor × factor matrix + pinned stock-return rows.
Toggle: full-history vs rolling 126-day window.
Optional hierarchical clustering of factor order.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math_engine.pipeline import as_of, rolling_zscore

STOCK_LABELS = {
    "nifty50":    "NIFTY 50",
    "bank_nifty": "BANK NIFTY",
    "icici_bank": "ICICI Bank",
}


def _corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(min_periods=20)


def _cluster_order(corr: pd.DataFrame) -> list[str]:
    """Hierarchical clustering column order (average linkage, correlation distance)."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        dist = 1.0 - corr.abs().fillna(0).values
        np.fill_diagonal(dist, 0.0)
        Z    = linkage(dist[np.triu_indices(len(dist), k=1)], method="average")
        order = leaves_list(Z)
        return [corr.columns[i] for i in order]
    except Exception:
        return corr.columns.tolist()


def render_heatmap(
    factors_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    date,
    rolling: bool = False,
    pca_window: int = 126,
    cluster: bool = True,
) -> go.Figure:
    date = pd.Timestamp(date)

    f_slice = as_of(factors_df, date)
    s_slice = as_of(stocks_df, date)

    if rolling:
        if len(f_slice) < pca_window:
            f_slice = f_slice
            s_slice = s_slice
        else:
            f_slice = f_slice.iloc[-pca_window:]
            s_slice = s_slice.iloc[-pca_window:]
        title_tag = f"Rolling {pca_window}d"
    else:
        title_tag = "Full history"

    # Drop all-NaN columns
    f_valid = f_slice.dropna(axis=1, how="all")
    s_valid = s_slice.dropna(axis=1, how="all")

    if f_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid factor data for this window.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    ff_corr = _corr_matrix(f_valid)

    if cluster:
        col_order = _cluster_order(ff_corr)
        ff_corr   = ff_corr.loc[col_order, col_order]
        f_valid   = f_valid[col_order]

    factor_cols = ff_corr.columns.tolist()

    # Stock-to-factor correlations (pinned rows below factor × factor matrix)
    stock_rows: dict[str, list[float]] = {}
    for stock in stocks_df.columns:
        if stock not in s_valid.columns:
            continue
        row = []
        for fc in factor_cols:
            if fc not in f_valid.columns:
                row.append(np.nan)
                continue
            valid = f_valid[fc].notna() & s_valid[stock].notna()
            if valid.sum() < 10:
                row.append(np.nan)
            else:
                row.append(float(f_valid[fc][valid].corr(s_valid[stock][valid])))
        stock_rows[stock] = row

    # Build combined matrix: factors on top, stocks pinned below
    n_factors = len(factor_cols)
    n_stocks  = len(stock_rows)
    n_rows_total = n_factors + n_stocks

    combined   = np.full((n_rows_total, n_factors), np.nan)
    y_labels   = factor_cols.copy()

    for i, fc in enumerate(factor_cols):
        for j, fc2 in enumerate(factor_cols):
            combined[i, j] = ff_corr.loc[fc, fc2]

    stock_label_list = []
    for k, (stock, vals) in enumerate(stock_rows.items()):
        combined[n_factors + k, :] = vals
        stock_label_list.append(STOCK_LABELS.get(stock, stock))

    y_labels += stock_label_list

    # Separator annotation position
    sep_y = n_factors - 0.5

    colorscale = [
        [0.0, "#d73027"], [0.25, "#fc8d59"], [0.5, "#f7f7f7"],
        [0.75, "#91bfdb"], [1.0, "#4575b4"],
    ]

    text_matrix = [[
        f"{combined[i, j]:.2f}" if not np.isnan(combined[i, j]) else ""
        for j in range(n_factors)
    ] for i in range(n_rows_total)]

    fig = go.Figure(go.Heatmap(
        z=combined,
        x=factor_cols,
        y=y_labels,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="ρ", len=0.8),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))

    # Divider line between factor block and stock rows
    if n_stocks > 0:
        fig.add_shape(
            type="line", x0=-0.5, x1=n_factors - 0.5, y0=sep_y, y1=sep_y,
            line=dict(color="black", width=2.5),
        )

    fig.update_layout(
        title=f"Correlation Heatmap ({title_tag}) — {date.strftime('%Y-%m-%d')}",
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        height=max(500, 28 * n_rows_total + 120),
        margin=dict(t=80, b=120, l=160, r=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
