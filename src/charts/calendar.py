from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_calendar_heatmap(revenue_series: pd.Series, year: int) -> go.Figure:
    dates = pd.to_datetime(revenue_series.index)
    calendar_frame = pd.DataFrame({"date": dates, "revenue": revenue_series.to_numpy(dtype=float)})
    origin = pd.Timestamp(f"{year}-01-01")
    calendar_frame["week"] = ((calendar_frame["date"] - origin).dt.days + origin.weekday()) // 7
    calendar_frame["weekday"] = calendar_frame["date"].dt.weekday
    calendar_frame["label"] = calendar_frame["date"].dt.strftime("%Y-%m-%d")

    value_matrix = calendar_frame.pivot(index="weekday", columns="week", values="revenue").sort_index(ascending=False)
    label_matrix = calendar_frame.pivot(index="weekday", columns="week", values="label").sort_index(ascending=False)

    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=value_matrix.to_numpy(),
                x=value_matrix.columns,
                y=["Sun", "Sat", "Fri", "Thu", "Wed", "Tue", "Mon"],
                text=label_matrix.to_numpy(),
                customdata=label_matrix.to_numpy(),
                colorscale="YlOrRd",
                hovertemplate="%{customdata}<br>Revenue %{z:.1f} €/MW<extra></extra>",
                colorbar={"title": "€/MW"},
            )
        ]
    )
    fig.update_layout(
        title="Calendar Heatmap of Daily Revenue",
        xaxis_title="Week of year",
        yaxis_title="Day of week",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=320,
    )
    return fig
