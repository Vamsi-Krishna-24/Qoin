"""
QOIN — Phase 3: Vector Space Visualization

Auto-selects the 3 highest-|weight| factors from the Johansen result and
plots them as literal spatial axes. Each trading day is one point in this
3-factor slice. Point color = how far the FULL residual (all 11 factors,
not just these 3) sits from its own mean that day -- i.e. the actual
mean-reversion state of the model. The black line is the Johansen weight
vector's direction, restricted to just these 3 dimensions -- the "axis"
your original idea describes, made literal.

IMPORTANT: this is a 3-of-11-dimensional SLICE, not the whole model. A
point that looks "off the line" here may be perfectly explained by the
other 8 factors moving that day. Don't read wobble in this picture alone
as the model failing -- the console output tells you what % of the total
weight mass these 3 axes actually represent.

Requires phase3_johansen.py in the SAME FOLDER -- this script reuses its
data loading, Johansen run, and residual logic so the two scripts can
never disagree with each other.

Run locally: python phase3_vector_space.py
Opens an interactive, rotatable 3D plot in your browser and also saves
phase3_vector_space.html so you can reopen it anytime without re-running.
"""

import numpy as np
import plotly.graph_objects as go

from johansen import (
    DATA_PATH,
    TARGET_COL,
    load_and_prepare,
    run_johansen,
    build_residual,
    adf_on_residual,
)


def pick_top3_factors(weights, factor_names):
    order = np.argsort(-np.abs(weights))[:3]
    top3 = [factor_names[i] for i in order]
    coverage = np.abs(weights[order]).sum() / np.abs(weights).sum()
    return top3, coverage


def make_figure(df, residual, weights, factor_cols, target_col):
    top3, coverage = pick_top3_factors(weights, factor_cols)
    print(f"Top 3 factors by |Johansen weight|: {top3}")
    print(f"These 3 account for {coverage:.1%} of total |weight| mass "
          f"across all {len(factor_cols)} factors -- the remaining "
          f"{1 - coverage:.1%} of the story lives outside this picture.\n")

    x, y, z = [df[f].values for f in top3]

    resid_z = (residual - residual.mean()) / residual.std()

    # Cold = below mean (oversold), warm = above mean (overbought) --
    # the color coding carries meaning for a mean-reversion model, not
    # just decoration.
    COOL, NEUTRAL, HOT = "#17B8C4", "#F2E9D8", "#E8622C"
    diverging = [[0.0, COOL], [0.5, NEUTRAL], [1.0, HOT]]

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=4,
            color=resid_z,
            colorscale=diverging,
            cmid=0,
            opacity=0.85,
            colorbar=dict(
                title=dict(text="Residual<br>(SDs from mean)", font=dict(color="#E6EDF3")),
                tickfont=dict(color="#C9D1D9"),
                bgcolor="rgba(0,0,0,0)",
                outlinecolor="#3A4450",
                x=1.02,
            ),
        ),
        text=[f"{target_col}: {v:.1f}" for v in df[target_col]],
        hovertemplate=(
            f"{top3[0]}: %{{x:.2f}}<br>"
            f"{top3[1]}: %{{y:.2f}}<br>"
            f"{top3[2]}: %{{z:.2f}}<br>"
            "%{text}<extra></extra>"
        ),
        name="Trading days",
    )

    idx = [factor_cols.index(f) for f in top3]
    w3 = np.array([weights[i] for i in idx])
    w3 = w3 / np.linalg.norm(w3)

    centroid = np.array([x.mean(), y.mean(), z.mean()])
    spread = np.array([x.std(), y.std(), z.std()]).mean() * 2.5
    p0 = centroid - w3 * spread
    p1 = centroid + w3 * spread

    # Two-layer line for a subtle glow -- wide/faint underneath, thin/bright on top
    axis_glow = go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode="lines", line=dict(color="#FFC857", width=16),
        opacity=0.25, showlegend=False, hoverinfo="skip",
    )
    axis_line = go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode="lines",
        line=dict(color="#FFC857", width=5),
        name="Johansen axis (this slice)",
        hoverinfo="skip",
    )

    fig = go.Figure(data=[scatter, axis_glow, axis_line])

    axis_style = dict(backgroundcolor="#121820", gridcolor="#2A323C",
                       zerolinecolor="#3A4450", color="#C9D1D9")

    fig.update_layout(
        title=dict(
            text=(
                f"QOIN Phase 3 — Vector Space Slice<br>"
                f"<sup>Axes: {top3[0]}, {top3[1]}, {top3[2]} "
                f"({coverage:.0%} of total weight mass) | "
                f"color = full-model residual, SDs from mean</sup>"
            ),
            font=dict(color="#E6EDF3", size=20),
        ),
        scene=dict(
            xaxis=dict(title=top3[0], **axis_style),
            yaxis=dict(title=top3[1], **axis_style),
            zaxis=dict(title=top3[2], **axis_style),
            bgcolor="#0B0F14",
        ),
        paper_bgcolor="#0B0F14",
        font=dict(color="#E6EDF3", family="Arial, Helvetica, sans-serif"),
        legend=dict(
            x=0.02, y=0.04,
            bgcolor="rgba(18,24,32,0.75)",
            bordercolor="#3A4450", borderwidth=1,
            font=dict(color="#E6EDF3"),
        ),
        hoverlabel=dict(bgcolor="#1B222B", font=dict(color="#E6EDF3", size=12), bordercolor="#3A4450"),
        width=1000, height=800,
        margin=dict(t=90),
    )
    return fig


if __name__ == "__main__":
    df = load_and_prepare(DATA_PATH)
    factor_cols = [c for c in df.columns if c != TARGET_COL]

    result, rank = run_johansen(df)
    if rank == 0:
        raise SystemExit(
            "No cointegration found (rank 0) -- there is no valid Johansen "
            "weight vector to build axes from. Resolve that in "
            "phase3_johansen.py before visualizing."
        )

    residual, weights_full = build_residual(df, result, TARGET_COL)
    _, resid_pval = adf_on_residual(residual)
    if resid_pval >= 0.05:
        raise SystemExit(
            "Residual is not stationary (p >= 0.05) -- these weights are "
            "not trustworthy. Fix the basis before visualizing it."
        )

    col_list = df.columns.tolist()
    weights = np.array([weights_full[col_list.index(f)] for f in factor_cols])

    fig = make_figure(df, residual, weights, factor_cols, TARGET_COL)
    fig.write_html("phase3_vector_space.html")
    print("Saved interactive plot to phase3_vector_space.html")
    fig.show()