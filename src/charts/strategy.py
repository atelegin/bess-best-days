from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_marginal_cycle_figure(marginal_frame: pd.DataFrame) -> go.Figure:
    total_days = len(marginal_frame)
    top_quintile_days = max(1, int(round(total_days * 0.20)))
    summary = pd.DataFrame(
        [
            {
                "bucket": "Top 20 days",
                "value": marginal_frame.head(min(20, total_days))["marginal_value_eur_per_mw"].mean(),
            },
            {
                "bucket": "Top 20%",
                "value": marginal_frame.head(top_quintile_days)["marginal_value_eur_per_mw"].mean(),
            },
            {
                "bucket": "Median day",
                "value": marginal_frame["marginal_value_eur_per_mw"].median(),
            },
            {
                "bucket": "Bottom 50%",
                "value": marginal_frame.tail(max(1, total_days // 2))["marginal_value_eur_per_mw"].mean(),
            },
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["bucket"],
            y=summary["value"],
            marker={"color": ["#e76f51", "#f4a261", "#264653", "#94a3b8"]},
            text=[f"{value:.0f} €/MW" for value in summary["value"]],
            textposition="outside",
            hovertemplate="%{x}<br>Extra value %{y:.1f} €/MW<extra></extra>",
            name="Aggressive uplift",
        )
    )
    fig.update_layout(
        title="Where the Second Cycle Actually Pays",
        xaxis_title="Day bucket",
        yaxis_title="Extra revenue (€/MW)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=360,
    )
    return fig


def build_annual_comparison_figure(annual_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=annual_summary["year"],
            y=annual_summary["conservative_revenue_eur_per_mw"],
            name="Conservative",
            marker={"color": "#264653"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=annual_summary["year"],
            y=annual_summary["aggressive_revenue_eur_per_mw"],
            name="Aggressive",
            marker={"color": "#e76f51"},
        )
    )
    fig.update_layout(
        title="Annual Revenue by Strategy",
        xaxis_title="Year",
        yaxis_title="Revenue (€/MW/year)",
        barmode="group",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=360,
    )
    return fig


def build_lifetime_cumulative_figure(
    conservative_profile: pd.DataFrame,
    aggressive_profile: pd.DataFrame,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=conservative_profile["year"],
            y=conservative_profile["cumulative_revenue_eur_per_mw"],
            mode="lines",
            name="Conservative",
            line={"color": "#264653", "width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=aggressive_profile["year"],
            y=aggressive_profile["cumulative_revenue_eur_per_mw"],
            mode="lines",
            name="Aggressive",
            line={"color": "#e76f51", "width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=conservative_profile["year"],
            y=conservative_profile["capacity_fraction"] * 100,
            mode="lines",
            name="Conservative capacity",
            line={"color": "#264653", "width": 2, "dash": "dot"},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=aggressive_profile["year"],
            y=aggressive_profile["capacity_fraction"] * 100,
            mode="lines",
            name="Aggressive capacity",
            line={"color": "#e76f51", "width": 2, "dash": "dot"},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="More Revenue Up Front Still Wins After Faster Degradation",
        xaxis_title="Project year",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=380,
    )
    fig.update_yaxes(title_text="Cumulative revenue (€/MW)", secondary_y=False)
    fig.update_yaxes(title_text="Remaining capacity (%)", secondary_y=True)
    return fig


def build_cycle_intensity_frontier_figure(
    frontier_frame: pd.DataFrame,
    reference_warranty_fec_per_year: float,
) -> go.Figure:
    optimum = frontier_frame.loc[frontier_frame["discounted_lifetime_value_eur_per_mw"].idxmax()]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frontier_frame["full_equivalent_cycles_per_year"],
            y=frontier_frame["discounted_lifetime_value_eur_per_mw"],
            mode="lines+markers",
            name="Discounted lifetime value",
            line={"color": "#e76f51", "width": 3},
            marker={"size": 9},
            hovertemplate=(
                "%{x:.0f} FEC/year"
                "<br>Discounted lifetime value %{y:,.0f} €/MW"
                "<br>Daily cycle cap %{customdata[0]:.2f}"
                "<br>Years to 60% SOH %{customdata[1]:.1f}"
                "<extra></extra>"
            ),
            customdata=frontier_frame[["max_cycles_per_day", "years_to_eol"]].to_numpy(),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frontier_frame["full_equivalent_cycles_per_year"],
            y=frontier_frame["years_to_eol"],
            mode="lines+markers",
            name="Years to 60% SOH",
            line={"color": "#264653", "width": 2, "dash": "dot"},
            marker={"size": 7},
            hovertemplate="%{x:.0f} FEC/year<br>Years to 60% SOH %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_vline(
        x=reference_warranty_fec_per_year,
        line={"color": "#94a3b8", "width": 2, "dash": "dash"},
    )
    fig.add_annotation(
        x=float(optimum["full_equivalent_cycles_per_year"]),
        y=float(optimum["discounted_lifetime_value_eur_per_mw"]),
        text=f"Best tested point<br>{optimum['full_equivalent_cycles_per_year']:.0f} FEC/year",
        showarrow=True,
        arrowhead=2,
        ax=45,
        ay=-45,
        bgcolor="rgba(255,255,255,0.88)",
    )
    fig.add_annotation(
        x=reference_warranty_fec_per_year,
        y=float(frontier_frame["discounted_lifetime_value_eur_per_mw"].max()) * 0.92,
        text=f"Reference warranty pace<br>{reference_warranty_fec_per_year:.0f} FEC/year",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
    )
    fig.update_layout(
        title="Where Cycling Harder Stops Paying",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=380,
    )
    fig.update_xaxes(title_text="Full-equivalent cycles per year")
    fig.update_yaxes(title_text="Discounted lifetime value (€/MW)", secondary_y=False)
    fig.update_yaxes(title_text="Years to 60% SOH", secondary_y=True)
    return fig


def build_warranty_posture_figure(
    warranty_value_eur_per_mw: float,
    overdrive_value_eur_per_mw: float,
    break_even_annual_cost_eur_per_mw: float,
    project_lifetime: int,
) -> go.Figure:
    max_cost = max(break_even_annual_cost_eur_per_mw * 2.0, 1.0)
    annual_costs = np.linspace(0.0, max_cost, 100)
    annuity_factor = (overdrive_value_eur_per_mw - warranty_value_eur_per_mw) / max(
        break_even_annual_cost_eur_per_mw,
        1e-9,
    )
    net_overdrive = overdrive_value_eur_per_mw - annual_costs * annuity_factor

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=annual_costs,
            y=net_overdrive,
            mode="lines",
            name="Extend warranty or self-insure and overdrive",
            line={"color": "#e76f51", "width": 3},
            hovertemplate="Annual cost %{x:,.0f} €/MW/yr<br>Net value %{y:,.0f} €/MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=annual_costs,
            y=np.full_like(annual_costs, warranty_value_eur_per_mw),
            mode="lines",
            name="Stay inside warranty pace",
            line={"color": "#264653", "width": 3, "dash": "dash"},
            hovertemplate="Annual cost %{x:,.0f} €/MW/yr<br>Net value %{y:,.0f} €/MW<extra></extra>",
        )
    )
    fig.add_vline(
        x=break_even_annual_cost_eur_per_mw,
        line={"color": "#94a3b8", "width": 2, "dash": "dot"},
    )
    fig.add_annotation(
        x=break_even_annual_cost_eur_per_mw,
        y=overdrive_value_eur_per_mw,
        text=f"Break-even annual cost<br>{break_even_annual_cost_eur_per_mw:,.0f} €/MW/yr",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-45,
        bgcolor="rgba(255,255,255,0.88)",
    )
    fig.update_layout(
        title=f"What Can You Afford to Pay for More Cycling Over {project_lifetime} Years?",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        height=360,
    )
    fig.update_xaxes(title_text="Annual premium or reserve for higher cycling (€/MW/year)")
    fig.update_yaxes(title_text="Discounted lifetime value (€/MW)")
    return fig
