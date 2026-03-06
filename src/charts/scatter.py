from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_tail_pattern_figure(pattern_distribution: dict[str, int]) -> go.Figure:
    pattern_frame = pd.DataFrame(
        {
            "pattern": list(pattern_distribution.keys()),
            "count": list(pattern_distribution.values()),
        }
    )
    fig = go.Figure(
        go.Bar(
            x=pattern_frame["count"],
            y=pattern_frame["pattern"],
            orientation="h",
            marker={"color": ["#264653", "#e76f51", "#2a9d8f", "#94a3b8"][: len(pattern_frame)]},
            text=pattern_frame["count"],
            textposition="outside",
            hovertemplate="%{y}<br>%{x} of top 20 days<extra></extra>",
        )
    )
    fig.update_layout(
        title="What Kind of Days Dominate the Tail",
        xaxis_title="Count of top-20 days",
        yaxis_title="",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=320,
    )
    return fig


def build_price_shape_figure(profile_payload: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    sampled_profiles = profile_payload["sampled_days"]
    average_profiles = profile_payload["average_profiles"]

    for date_label, day_frame in sampled_profiles.groupby("date_label"):
        fig.add_trace(
            go.Scatter(
                x=day_frame["hour"],
                y=day_frame["price_eur_mwh"],
                mode="lines",
                name="Sample day",
                line={"color": "rgba(148,163,184,0.16)", "width": 1},
                hovertemplate=f"{date_label}<br>Hour %{{x}}<br>Price %{{y:.1f}} €/MWh<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=average_profiles["hour"],
            y=average_profiles["all_days"],
            mode="lines",
            name="Average of all days",
            line={"color": "#264653", "width": 2, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=average_profiles["hour"],
            y=average_profiles["top_days"],
            mode="lines",
            name="Average of top 10 days",
            line={"color": "#e76f51", "width": 4},
        )
    )
    fig.update_layout(
        title="What a Tail Day Looks Like Against the Rest of the Year",
        xaxis_title="Hour of day",
        yaxis_title="Price (€/MWh)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=360,
    )
    return fig
