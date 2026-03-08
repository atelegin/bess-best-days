from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_interannual_stability_figure(stability_frame: pd.DataFrame, highlighted_year: int) -> go.Figure:
    years = stability_frame.index.astype(str).tolist()
    year_values = stability_frame.index.to_numpy(dtype=float)
    top_20_share = stability_frame["top_20pct_days_pct_of_revenue"].to_numpy(dtype=float)
    bar_colors = ["#e76f51" if int(year) == highlighted_year else "#d9cdb8" for year in stability_frame.index]
    if len(year_values) >= 2:
        slope, intercept = np.polyfit(year_values, top_20_share, 1)
        trend_values = slope * year_values + intercept
    else:
        trend_values = top_20_share.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=years,
            y=top_20_share,
            marker={"color": bar_colors},
            text=[f"{value:.1f}%" for value in top_20_share],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Top 20%% of days: %{y:.1f}%% of annual revenue<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=trend_values,
            mode="lines",
            line={"color": "#264653", "width": 3, "dash": "dash"},
            hovertemplate="Trend<br>%{x}: %{y:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Top 20% day share of annual revenue by year",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 56, "b": 10},
        height=330,
        bargap=0.22,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Share of annual revenue (%)", range=[0, max(60.0, float(top_20_share.max()) + 6)])
    return fig


def build_missed_days_figure(missed_days: pd.DataFrame, highlight_day: int = 20) -> go.Figure:
    x_values = missed_days.index.to_numpy(dtype=int)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=missed_days["lost_share_pct"],
            mode="lines+markers",
            name="Revenue lost",
            line={"color": "#264653", "width": 4},
            marker={"size": 7, "color": "#264653"},
            hovertemplate="Miss top %{x} days<br>Lost share %{y:.1f}%<extra></extra>",
        )
    )
    if highlight_day in missed_days.index:
        highlight_row = missed_days.loc[highlight_day]
        fig.add_trace(
            go.Scatter(
                x=[highlight_day],
                y=[highlight_row["lost_share_pct"]],
                mode="markers",
                marker={"size": 14, "color": "#e76f51", "line": {"color": "#ffffff", "width": 2}},
                hovertemplate=(
                    f"Miss top {highlight_day} days"
                    "<br>Lost share %{y:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=highlight_day,
            y=float(highlight_row["lost_share_pct"]),
            text=f"<b>{highlight_day} days missed</b><br>{highlight_row['lost_share_pct']:.1f}% revenue loss",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=18,
            yshift=16,
            bgcolor="rgba(255, 251, 245, 0.96)",
            bordercolor="rgba(231, 111, 81, 0.35)",
            borderwidth=1,
            borderpad=6,
            font={"size": 11},
        )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=360,
    )
    fig.update_xaxes(title_text="Number of top revenue days missed")
    fig.update_yaxes(title_text="Annual revenue lost (%)")
    return fig


def build_feature_comparison_figure(feature_summary: pd.DataFrame) -> go.Figure:
    comparison = pd.DataFrame(
        {
            "feature": [
                "Midday trough",
                "Evening peak",
                "DA spread",
                "Evening minus midday ramp",
            ],
            "all_days": [
                feature_summary.loc["all_days", "median_midday_min_price_eur_mwh"],
                feature_summary.loc["all_days", "median_evening_peak_price_eur_mwh"],
                feature_summary.loc["all_days", "median_spread_eur_mwh"],
                feature_summary.loc["all_days", "median_da_evening_minus_midday_ramp_eur_mwh"],
            ],
            "top_20_days": [
                feature_summary.loc["top_20_revenue_days", "median_midday_min_price_eur_mwh"],
                feature_summary.loc["top_20_revenue_days", "median_evening_peak_price_eur_mwh"],
                feature_summary.loc["top_20_revenue_days", "median_spread_eur_mwh"],
                feature_summary.loc["top_20_revenue_days", "median_da_evening_minus_midday_ramp_eur_mwh"],
            ],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["feature"],
            y=comparison["all_days"],
            name="All days",
            marker={"color": "#d9cdb8"},
            hovertemplate="%{x}<br>All days median %{y:.1f} €/MWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["feature"],
            y=comparison["top_20_days"],
            name="Top 20 revenue days",
            marker={"color": "#e76f51"},
            hovertemplate="%{x}<br>Top 20 median %{y:.1f} €/MWh<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top Revenue Days Have a Distinct Day-Ahead Signature",
        xaxis_title="Feature",
        yaxis_title="Median value (€/MWh)",
        barmode="group",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=360,
    )
    return fig


def build_signal_frequency_figure(signal_summary: pd.DataFrame) -> go.Figure:
    frame = signal_summary.reset_index().rename(columns={"index": "signal"})
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["top_days_share_pct"],
            y=frame["signal"],
            name="Top 20 revenue days",
            orientation="h",
            marker={"color": "#e76f51"},
            hovertemplate="%{y}<br>Top 20 days %{x:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["all_days_share_pct"],
            y=frame["signal"],
            name="All days",
            orientation="h",
            marker={"color": "#264653"},
            hovertemplate="%{y}<br>All days %{x:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="What Kind of Days Dominate the Best Revenue Windows",
        xaxis_title="Share of days (%)",
        yaxis_title="",
        barmode="group",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=380,
    )
    return fig


def build_watchlist_scatter_figure(watchlist_frame: pd.DataFrame, base_rate_pct: float) -> go.Figure:
    color_map = {
        "Ramp >= 200 €/MWh": "#e76f51",
        "Spread >= 200 €/MWh": "#f4a261",
        "Ramp >= 150 €/MWh": "#2a9d8f",
    }
    fig = go.Figure()
    for _, row in watchlist_frame.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["recall_pct"]],
                y=[row["precision_pct"]],
                mode="markers",
                name=row["short_label"],
                marker={
                    "size": 18,
                    "color": color_map.get(row["short_label"], "#264653"),
                    "line": {"color": "#ffffff", "width": 1.5},
                    "opacity": 0.92,
                },
                customdata=[[row["signal_days"], row["maintenance_day_value_shift_eur_per_mw"], row["lift_x"]]],
                hovertemplate=(
                    f"{row['short_label']}<br>Recall %{{x:.1f}}%"
                    "<br>Precision %{y:.1f}%"
                    "<br>Days flagged %{customdata[0]:.0f}"
                    "<br>Lift %{customdata[2]:.2f}x"
                    "<br>Maintenance shift %{customdata[1]:+.0f} €/MW"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        if "callout" in row and pd.notna(row["callout"]):
            fig.add_annotation(
                x=row["recall_pct"],
                y=row["precision_pct"],
                text=f"<b>{row['short_label']}</b><br>{row['callout']}",
                showarrow=False,
                xanchor=row.get("xanchor", "left"),
                yanchor=row.get("yanchor", "bottom"),
                xshift=int(row.get("xshift", 12)),
                yshift=int(row.get("yshift", 12)),
                bgcolor="rgba(255, 251, 245, 0.96)",
                bordercolor="rgba(20, 33, 61, 0.12)",
                borderwidth=1,
                borderpad=5,
                align="left",
                font={"size": 11},
            )
    fig.add_hline(
        y=base_rate_pct,
        line_color="#94a3b8",
        line_width=2,
        line_dash="dash",
        annotation_text=f"Base rate = {base_rate_pct:.1f}%",
        annotation_position="bottom right",
        annotation_font={"size": 11, "color": "#5c677d"},
    )
    fig.update_layout(
        xaxis_title="Recall on top-20 days (%)",
        yaxis_title="Precision on top-20 days (%)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=420,
    )
    fig.update_xaxes(range=[0, max(65.0, float(watchlist_frame["recall_pct"].max()) + 6)])
    fig.update_yaxes(range=[0, max(30.0, float(watchlist_frame["precision_pct"].max()) + 8)])
    return fig


def build_throughput_budget_capture_figure(budget_summary: pd.DataFrame) -> go.Figure:
    frame = budget_summary.copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["captured_revenue_eur_per_mw"],
            name="Captured value",
            marker={"color": ["#264653", "#3a5f73", "#4f7d88", "#7ca982", "#e76f51"][: len(frame)]},
            hovertemplate="%{x}<br>Captured value %{y:,.0f} €/MW<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["share_of_full_flex_revenue_pct"],
            mode="lines+markers",
            name="Share of full-flex value",
            line={"color": "#e76f51", "width": 3},
            marker={"size": 8},
            hovertemplate="%{x}<br>Share of full-flex value %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Most Value Fits Inside the Warranty Pace — But Not All of It",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=380,
        bargap=0.18,
    )
    fig.update_xaxes(title_text="Annual throughput budget")
    fig.update_yaxes(title_text="Captured value (€/MW)", secondary_y=False)
    fig.update_yaxes(title_text="Share of full-flex value (%)", secondary_y=True)
    return fig


def build_same_cycles_reallocation_figure(summary_frame: pd.DataFrame) -> go.Figure:
    frame = summary_frame.reset_index().copy()
    frame["pair_label"] = frame["strict_daily_cap"].map(lambda value: f"{value:.1f}/day")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["pair_label"],
            y=frame["uplift_eur_per_mw"],
            name="Revenue uplift",
            marker={"color": "#e76f51"},
            text=[
                f"+{uplift / 1000:.1f}k €/MW<br>(+{uplift_pct:.1f}%)"
                for uplift, uplift_pct in zip(frame["uplift_eur_per_mw"], frame["uplift_pct_vs_strict"])
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=frame[["strict_realized_fec"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>Revenue uplift %{y:,.0f} €/MW"
                "<br>Same realised throughput %{customdata[0]:.1f} FEC"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=360,
        bargap=0.32,
        showlegend=False,
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Revenue uplift from reallocation (€/MW)")
    return fig
