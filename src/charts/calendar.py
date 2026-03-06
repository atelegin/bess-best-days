from __future__ import annotations

import calendar

import pandas as pd
import plotly.graph_objects as go


def build_calendar_heatmap(revenue_series: pd.Series, year: int) -> go.Figure:
    dates = pd.to_datetime(revenue_series.index)
    calendar_frame = pd.DataFrame({"date": dates, "revenue": revenue_series.to_numpy(dtype=float)}).copy()
    calendar_frame = calendar_frame.loc[calendar_frame["date"].dt.year == year].copy()
    calendar_frame["month"] = calendar_frame["date"].dt.month
    calendar_frame["day_of_month"] = calendar_frame["date"].dt.day
    calendar_frame["label"] = calendar_frame["date"].dt.strftime("%Y-%m-%d")

    month_order = list(range(12, 0, -1))
    day_order = list(range(1, 32))
    value_matrix = (
        calendar_frame.pivot(index="month", columns="day_of_month", values="revenue").reindex(
            index=month_order,
            columns=day_order,
        )
    )
    label_matrix = (
        calendar_frame.pivot(index="month", columns="day_of_month", values="label").reindex(
            index=month_order,
            columns=day_order,
        )
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=value_matrix.to_numpy(),
                x=value_matrix.columns,
                y=[calendar.month_abbr[month] for month in value_matrix.index],
                text=label_matrix.to_numpy(),
                customdata=label_matrix.to_numpy(),
                colorscale="YlOrRd",
                hovertemplate="%{customdata}<br>Revenue %{z:.1f} €/MW<extra></extra>",
                colorbar={"title": "€/MW"},
                xgap=1,
                ygap=1,
            )
        ]
    )
    fig.update_layout(
        title=f"Daily revenue heatmap ({year})",
        xaxis_title="Day of month",
        yaxis_title="Month",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=520,
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=5)
    return fig
