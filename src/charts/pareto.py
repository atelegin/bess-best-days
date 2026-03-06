from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_pareto_figure(
    curves_by_year: dict[int, pd.DataFrame],
    highlighted_year: int,
    highlight_day: int | None = None,
    target_revenue_share: float = 0.5,
) -> go.Figure:
    fig = go.Figure()
    for year, curve in curves_by_year.items():
        fig.add_trace(
            go.Scatter(
                x=curve["rank"],
                y=curve["cumulative_revenue_pct"] * 100,
                mode="lines",
                name=str(year),
                line={
                    "width": 4 if year == highlighted_year else 1.6,
                    "color": "#14213d" if year == highlighted_year else "rgba(231,111,81,0.35)",
                },
                opacity=1 if year == highlighted_year else 0.45,
                hovertemplate="Day rank %{x}<br>Cumulative revenue %{y:.1f}%<extra></extra>",
            )
        )
    if highlight_day:
        fig.add_vline(x=highlight_day, line={"color": "#e76f51", "dash": "dot", "width": 2})
        fig.add_hline(y=target_revenue_share * 100, line={"color": "#e76f51", "dash": "dot", "width": 2})
        fig.add_annotation(
            x=highlight_day,
            y=target_revenue_share * 100,
            text=f"{highlight_day} days generated {target_revenue_share:.0%} of revenue",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            bgcolor="rgba(255,255,255,0.85)",
        )
    fig.update_layout(
        title="How Fast the Best Days Capture the Year",
        xaxis_title="Days ranked from best to worst",
        yaxis_title="Cumulative revenue captured (%)",
        template="plotly_white",
        legend_title="Year",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=420,
    )
    fig.update_xaxes(dtick=25)
    return fig


def build_revenue_distribution_figure(revenue_series: pd.Series, stats: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=revenue_series,
            nbinsx=40,
            marker={"color": "#f4a261", "line": {"color": "#ffffff", "width": 0.5}},
            hovertemplate="Revenue %{x:.1f} €/MW<br>Days %{y}<extra></extra>",
        )
    )
    for quantile_label, key, color in (
        ("p50", "p50_daily_revenue_eur_per_mw", "#1d3557"),
        ("p90", "p90_daily_revenue_eur_per_mw", "#2a9d8f"),
        ("p95", "p95_daily_revenue_eur_per_mw", "#e76f51"),
        ("p99", "p99_daily_revenue_eur_per_mw", "#9d4edd"),
    ):
        fig.add_vline(
            x=stats[key],
            line={"color": color, "dash": "dot", "width": 2},
            annotation_text=quantile_label,
            annotation_position="top right",
        )
    fig.update_layout(
        title="Distribution of Daily Revenue",
        xaxis_title="Daily revenue (€/MW)",
        yaxis_title="Number of days",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        bargap=0.06,
        height=320,
    )
    return fig
