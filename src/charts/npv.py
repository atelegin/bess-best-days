from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_cannibalization_figure(
    balancing_frame: pd.DataFrame,
    tb2_frame: pd.DataFrame,
    bess_capacity_frame: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    indexed_series: list[tuple[str, pd.Series, str]] = []
    for market, color in (("FCR", "#264653"), ("aFRR", "#e76f51")):
        market_frame = balancing_frame[balancing_frame["market"] == market]["capacity_price_eur_mw"].dropna()
        baseline = market_frame[market_frame.index < "2022-01-01"].mean()
        if baseline and not pd.isna(baseline):
            indexed_series.append((market, market_frame / baseline * 100, color))

    tb2_baseline = tb2_frame[tb2_frame.index < "2022-01-01"]["tb2_spread_eur_mwh"].mean()
    tb2_indexed = tb2_frame["tb2_spread_eur_mwh"] / tb2_baseline * 100 if tb2_baseline else tb2_frame["tb2_spread_eur_mwh"]

    for market, series, color in indexed_series:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=f"{market} indexed to 2021 = 100",
                line={"color": color, "width": 3},
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=tb2_indexed.index,
            y=tb2_indexed,
            mode="lines",
            name="DA TB2 indexed to 2021 = 100",
            line={"color": "#2a9d8f", "width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=bess_capacity_frame.index,
            y=bess_capacity_frame["bess_capacity_gw"],
            mode="lines",
            name="Installed BESS (GW)",
            line={"color": "#94a3b8", "width": 2, "dash": "dot"},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="What Crowding Has Already Hit, And What It Hasn't",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        height=380,
    )
    fig.update_yaxes(title_text="Indexed market value (2021 = 100)", secondary_y=False)
    fig.update_yaxes(title_text="Installed BESS (GW)", secondary_y=True)
    return fig


def build_npv_crossover_figure(npv_frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=npv_frame["decline_rate"] * 100,
            y=npv_frame["conservative_npv"],
            mode="lines",
            name="Conservative",
            line={"color": "#264653", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=npv_frame["decline_rate"] * 100,
            y=npv_frame["aggressive_npv"],
            mode="lines",
            name="Aggressive",
            line={"color": "#e76f51", "width": 3},
        )
    )
    fig.update_layout(
        title="How Much Future Revenue Decay Would Flip the Result?",
        xaxis_title="Annual decline in merchant revenue (%)",
        yaxis_title="Discounted lifetime value (€/MW)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


def build_npv_heatmap_figure(heatmap_frame: pd.DataFrame) -> go.Figure:
    pivot = heatmap_frame.pivot(index="discount_rate", columns="decline_rate", values="npv_delta_eur_per_mw")
    fig = px.imshow(
        pivot.sort_index(ascending=False),
        labels={"x": "Annual revenue decline (%)", "y": "Discount rate (%)", "color": "Aggressive - Conservative (€/MW)"},
        x=[value * 100 for value in pivot.columns],
        y=[value * 100 for value in pivot.index],
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(
        title="Aggressive Minus Restrained Lifetime Value",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig
