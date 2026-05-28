"""View 1 — Biplot: PCA factor loadings (arrows) + stock geometric positions (dots)."""

import numpy as np
import plotly.graph_objects as go

STOCK_COLORS = {
    "nifty50":    "#2196F3",
    "bank_nifty": "#FF9800",
    "icici_bank": "#9C27B0",
}

STOCK_LABELS = {
    "nifty50":    "NIFTY 50",
    "bank_nifty": "BANK NIFTY",
    "icici_bank": "ICICI Bank",
}


def render_biplot(snapshot: dict) -> go.Figure:
    if "error" in snapshot:
        fig = go.Figure()
        fig.add_annotation(text=snapshot["error"], xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        fig.update_layout(title="Biplot — insufficient data")
        return fig

    loadings       = snapshot["loadings"]         # (n_comp, n_active_factors)
    active_factors = snapshot["active_factors"]
    stock_positions = snapshot["stock_positions"]
    cum_var        = snapshot["cum_variance"]
    date_str       = snapshot["date"].strftime("%Y-%m-%d")

    pc1_label = f"PC1 ({cum_var[0]:.1%} var)"
    pc2_label = f"PC2 ({cum_var[1]:.1%} var)" if len(cum_var) > 1 else "PC2"

    fig = go.Figure()

    # Arrow scale: normalise loadings to fit in [-1, 1]
    l1 = loadings[0]
    l2 = loadings[1] if loadings.shape[0] > 1 else np.zeros_like(l1)
    scale = max(np.sqrt(l1**2 + l2**2).max(), 1e-9)
    l1, l2 = l1 / scale, l2 / scale

    # Factor loading arrows
    for i, fname in enumerate(active_factors):
        x_end, y_end = float(l1[i]), float(l2[i])
        fig.add_shape(
            type="line", x0=0, y0=0, x1=x_end, y1=y_end,
            line=dict(color="rgba(100,100,100,0.6)", width=1.5),
        )
        fig.add_annotation(
            x=x_end, y=y_end, text=fname, showarrow=False,
            font=dict(size=10, color="rgba(60,60,60,0.9)"),
            xanchor="left" if x_end >= 0 else "right",
        )
        # Arrowhead dot
        fig.add_trace(go.Scatter(
            x=[x_end], y=[y_end], mode="markers",
            marker=dict(size=5, color="gray"),
            showlegend=False, hoverinfo="skip",
        ))

    # Stock positions
    if stock_positions:
        # Rescale stock positions to same scale as arrows
        max_stock = max(
            (abs(x) + abs(y)) for x, y in stock_positions.values()
        ) or 1.0
        for stock, (sx, sy) in stock_positions.items():
            sx_s = sx / max_stock * 0.8
            sy_s = sy / max_stock * 0.8
            fig.add_trace(go.Scatter(
                x=[sx_s], y=[sy_s],
                mode="markers+text",
                name=STOCK_LABELS.get(stock, stock),
                text=[STOCK_LABELS.get(stock, stock)],
                textposition="top center",
                marker=dict(size=14, color=STOCK_COLORS.get(stock, "blue"),
                            symbol="diamond", line=dict(width=1.5, color="white")),
                hovertemplate=f"<b>{STOCK_LABELS.get(stock, stock)}</b><br>"
                              f"PC1 proj: {sx:.4f}<br>PC2 proj: {sy:.4f}<extra></extra>",
            ))

    # Origin crosshairs
    fig.add_hline(y=0, line=dict(color="lightgray", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="lightgray", width=1, dash="dot"))

    fig.update_layout(
        title=f"Factor Biplot — {date_str}",
        xaxis_title=pc1_label,
        yaxis_title=pc2_label,
        xaxis=dict(range=[-1.3, 1.3], zeroline=False),
        yaxis=dict(range=[-1.3, 1.3], zeroline=False, scaleanchor="x", scaleratio=1),
        height=600,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=60, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
