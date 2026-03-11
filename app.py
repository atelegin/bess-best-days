from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.concentration import compute_concentration_stats
from src.analysis.day_ahead_signals import (
    build_day_ahead_observable_table,
    build_day_ahead_watchlist_table,
    concatenate_day_ahead_observable_tables,
    summarize_day_ahead_feature_separation,
)
from src.analysis.drivers import compute_price_shape_profiles
from src.analysis.opportunity_bridge import (
    build_daily_value_curve,
    summarize_reallocated_same_throughput_vs_strict_daily_cap,
)
from src.analysis.revenue_breakdown import summarize_missing_top_days
from src.charts.opportunity import (
    build_early_warning_screen_figure,
    build_missed_days_figure,
    build_same_cycles_reallocation_figure,
)
from src.charts.scatter import build_price_shape_figure
from src.data.cache import get_or_build_dataframe, make_cache_key
from src.data.netztransparenz import fetch_id_aep
from src.data.prices import fetch_day_ahead_prices
from src.models.degradation import (
    DEFAULT_DEGRADATION_ASSUMPTIONS,
    equivalent_stress_fec_per_year,
    estimate_years_to_eol,
    lifecycle_value_profile,
)
from src.models.dispatch import (
    AGGRESSIVE_STRATEGY,
    CONSERVATIVE_STRATEGY,
    DispatchStrategy,
    run_dispatch_for_period,
    run_dispatch_with_intraday_overlay_for_period,
)

REPORT_TITLE = "The Cost of Missing the Best Days"
YEARS = list(range(2021, 2026))
DEFAULT_RTE = 0.86
BASE_CASE_DURATION_HOURS = 2
BASE_CASE_ANALYSIS_YEAR = 2025
BASE_CASE_DISCOUNT_RATE = 0.08
BASE_CASE_PROJECT_LIFETIME = 15


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');
            :root {
                --bg: #f4efe7;
                --card: rgba(255, 251, 245, 0.92);
                --ink: #14213d;
                --muted: #5c677d;
                --accent: #e76f51;
                --accent-2: #2a9d8f;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(233, 196, 106, 0.28), transparent 28%),
                    radial-gradient(circle at 80% 10%, rgba(42, 157, 143, 0.18), transparent 24%),
                    linear-gradient(180deg, #fbf7f1 0%, #f1ece3 100%);
                color: var(--ink);
            }
            .block-container,
            .main .block-container,
            [data-testid="stMainBlockContainer"] {
                max-width: 980px;
                margin-left: auto;
                margin-right: auto;
                padding-top: 2rem;
                padding-left: 2.25rem;
                padding-right: 2.25rem;
            }
            [data-testid="stSidebar"],
            [data-testid="stSidebarCollapsedControl"] {
                display: none;
            }
            h1, h2, h3 {
                font-family: 'Source Serif 4', serif;
                color: var(--ink);
                letter-spacing: -0.02em;
            }
            body, .stMarkdown, .stMetric, .stDataFrame {
                font-family: 'IBM Plex Sans', sans-serif;
            }
            .hero-card {
                padding: 0.4rem 0 0.8rem 0;
                margin-bottom: 0.8rem;
            }
            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.72rem;
                color: var(--accent);
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .hero-dek {
                font-size: 1.15rem;
                line-height: 1.65;
                color: var(--ink);
                max-width: 52rem;
            }
            .standfirst {
                font-size: 1.04rem;
                line-height: 1.72;
                color: var(--ink);
                max-width: 54rem;
                margin: 0.2rem 0 1rem 0;
            }
            .chart-title {
                font-family: 'Source Serif 4', serif;
                font-size: 1.25rem;
                line-height: 1.35;
                color: var(--ink);
                margin: 1.1rem 0 0.55rem 0;
            }
            .chart-caption {
                color: var(--muted);
                font-size: 0.93rem;
                line-height: 1.55;
                margin: 0.75rem 0 0 0;
            }
            .takeaway-box {
                margin: 0.8rem 0 2rem 0;
                padding: 0.9rem 1rem;
                background: rgba(255, 251, 245, 0.88);
                border-left: 4px solid var(--accent);
                border-radius: 0.5rem;
            }
            .takeaway-label {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                color: var(--accent);
                font-weight: 700;
                margin-bottom: 0.32rem;
            }
            .takeaway-text {
                color: var(--ink);
                line-height: 1.55;
                font-weight: 600;
            }
            .small-note {
                color: var(--muted);
                font-size: 0.84rem;
                line-height: 1.45;
                margin: 0.45rem 0 0 0;
            }
            .annotation-box {
                margin: 0.95rem 0 2rem 0;
                padding: 0.95rem 1rem;
                background: rgba(255, 251, 245, 0.92);
                border: 1px solid rgba(20, 33, 61, 0.08);
                border-radius: 0.6rem;
            }
            .annotation-title {
                font-family: 'Source Serif 4', serif;
                font-size: 1.06rem;
                line-height: 1.35;
                color: var(--ink);
                margin: 0 0 0.35rem 0;
            }
            .annotation-text {
                color: var(--ink);
                line-height: 1.55;
            }
            .closing-line {
                font-family: 'Source Serif 4', serif;
                font-size: 1.25rem;
                line-height: 1.45;
                margin: 1rem 0 0.75rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">GERMAN BESS | MERCHANT REVENUES | AVAILABILITY | FLEXIBILITY</div>
            <h1>The Cost of Missing the Best Days</h1>
            <div class="hero-dek">
                German BESS merchant revenues are concentrated in a limited set of days. Many of those days are already partly visible in the day-ahead curve,
                making availability and cycling flexibility most valuable when they are timed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_battery_inputs() -> dict[str, float | int]:
    return {
        "duration_hours": BASE_CASE_DURATION_HOURS,
        "analysis_year": BASE_CASE_ANALYSIS_YEAR,
        "discount_rate": BASE_CASE_DISCOUNT_RATE,
        "project_lifetime": BASE_CASE_PROJECT_LIFETIME,
    }


def compute_dispatch_with_disk_cache(
    year: int,
    strategy: DispatchStrategy,
    price_frame: pd.DataFrame,
    energy_mwh: float,
    rte: float,
    market_key: str = "day_ahead",
    intraday_price_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cache_key = make_cache_key(
        "dispatch",
        year=year,
        market=market_key,
        strategy=strategy.name,
        energy_mwh=energy_mwh,
        rte=round(rte, 4),
        power_mw=1.0,
        version=4,
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: run_dispatch_with_intraday_overlay_for_period(
            day_ahead_price_frame=price_frame,
            intraday_price_frame=intraday_price_frame,
            strategy=strategy,
            energy_mwh=energy_mwh,
            rte=rte,
        )
        if intraday_price_frame is not None and not intraday_price_frame.empty
        else run_dispatch_for_period(price_frame=price_frame, strategy=strategy, energy_mwh=energy_mwh, rte=rte),
        ttl_hours=None,
        force_refresh=False,
        metadata={"year": year, "strategy": strategy.name, "market": market_key},
    )


@st.cache_data(show_spinner=False)
def load_core_data() -> pd.DataFrame:
    return fetch_day_ahead_prices(start=f"{YEARS[0]}-01-01", end=f"{YEARS[-1]}-12-31", force_refresh=False)


@st.cache_data(show_spinner=False)
def load_id_aep_for_year(year: int) -> pd.DataFrame:
    return fetch_id_aep(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        force_refresh=False,
    )


def render_intro() -> None:
    st.markdown(
        """
        <div class="standfirst">
            German BESS merchant revenues are not earned evenly through the year. A limited set of high-opportunity days drives a disproportionate share of annual value, and missing them is expensive. That matters because many of the best days are already partly visible in the day-ahead curve, making availability, maintenance timing, and cycling flexibility market-timed decisions rather than purely operational ones.
        </div>
        <div class="standfirst">
            This note uses 2025 as the main case study for a 2h battery, with pooled 2021–2025 data used to test whether simple day-ahead signals generalise beyond a single year.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_takeaway(text: str) -> None:
    st.markdown(
        f'<div class="takeaway-box"><div class="takeaway-label">Takeaway</div><div class="takeaway-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_footer_note(text: str) -> None:
    st.markdown(f'<div class="small-note">{text}</div>', unsafe_allow_html=True)


def render_annotation_box(title: str, text: str) -> None:
    st.markdown(
        f'<div class="annotation-box"><div class="annotation-title">{title}</div><div class="annotation-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_chart_title(text: str) -> None:
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)


def render_chart_caption(text: str) -> None:
    st.markdown(f'<div class="chart-caption">{text}</div>', unsafe_allow_html=True)


def render_closing_line(text: str) -> None:
    st.markdown(f'<div class="closing-line">{text}</div>', unsafe_allow_html=True)


def build_cycle_intensity_frontier_payload(
    year: int,
    price_frame: pd.DataFrame,
    intraday_price_frame: pd.DataFrame,
    energy_mwh: float,
    rte: float,
    discount_rate: float,
    project_lifetime: int,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame]]:
    cycle_caps = np.arange(0.25, 4.01, 0.25)
    records = []
    dispatch_by_cycle_cap: dict[float, pd.DataFrame] = {}
    for cycle_cap in cycle_caps:
        strategy = DispatchStrategy(
            name=f"frontier_{year}_{energy_mwh:.1f}h_{str(cycle_cap).replace('.', 'p')}",
            label=f"{cycle_cap:.2f} cycles/day",
            max_cycles=float(cycle_cap),
            soc_min_frac=AGGRESSIVE_STRATEGY.soc_min_frac,
            soc_max_frac=AGGRESSIVE_STRATEGY.soc_max_frac,
            min_spread_eur_mwh=AGGRESSIVE_STRATEGY.min_spread_eur_mwh,
        )
        dispatch = compute_dispatch_with_disk_cache(
            year=year,
            strategy=strategy,
            price_frame=price_frame,
            energy_mwh=energy_mwh,
            rte=rte,
            market_key="da_id_overlay_frontier_v2",
            intraday_price_frame=intraday_price_frame,
        )
        dispatch_by_cycle_cap[float(cycle_cap)] = dispatch
        lifecycle = lifecycle_value_profile(
            year1_revenue=float(dispatch["revenue_eur_per_mw"].sum()),
            dispatch_frame=dispatch,
            years=project_lifetime,
            discount_rate=discount_rate,
            annual_market_decline=0.0,
            assumptions=DEFAULT_DEGRADATION_ASSUMPTIONS,
        )
        records.append(
            {
                "max_cycles_per_day": float(cycle_cap),
                "year1_revenue_eur_per_mw": float(dispatch["revenue_eur_per_mw"].sum()),
                "full_equivalent_cycles_per_year": float(dispatch["full_equivalent_cycles"].sum()),
                "stress_fec_per_year": equivalent_stress_fec_per_year(
                    dispatch,
                    assumptions=DEFAULT_DEGRADATION_ASSUMPTIONS,
                ),
                "years_to_eol": estimate_years_to_eol(
                    dispatch,
                    assumptions=DEFAULT_DEGRADATION_ASSUMPTIONS,
                ),
                "discounted_lifetime_value_eur_per_mw": float(
                    lifecycle["cumulative_discounted_revenue_eur_per_mw"].iloc[-1]
                ),
            }
        )
    return pd.DataFrame(records), dispatch_by_cycle_cap


def main() -> None:
    st.set_page_config(page_title=REPORT_TITLE, layout="wide", initial_sidebar_state="collapsed")
    apply_app_style()
    render_header()
    render_intro()
    render_footer_note(
        "<strong>Base case:</strong> 2h battery, 2025 deep dive"
        "<br><strong>Validation:</strong> pooled 2021–2025"
        "<br><strong>Method note:</strong> Merchant revenue is modelled as a combined day-ahead plus intraday series, using Energy-Charts day-ahead prices and the official Netztransparenz ID-AEP index for the intraday layer. Dispatch is modelled sequentially across day-ahead and intraday with a fixed round-trip efficiency of 0.86."
    )
    inputs = select_battery_inputs()

    duration_hours = int(inputs["duration_hours"])
    rte = DEFAULT_RTE
    analysis_year = int(inputs["analysis_year"])
    discount_rate = float(inputs["discount_rate"])
    project_lifetime = int(inputs["project_lifetime"])
    energy_mwh = float(duration_hours)

    with st.spinner("Fetching and caching price data..."):
        prices = load_core_data()

    conservative_dispatch_by_year: dict[int, pd.DataFrame] = {}
    intraday_prices_by_year: dict[int, pd.DataFrame] = {}
    intraday_missing_years: list[int] = []

    progress = st.progress(0.0, text="Running DA + intraday dispatch across analysis years...")
    total_runs = len(YEARS) * 2
    completed = 0
    for year in YEARS:
        year_prices = prices[prices.index.year == year]
        try:
            intraday_prices_by_year[year] = load_id_aep_for_year(year)
        except Exception:
            intraday_prices_by_year[year] = pd.DataFrame(columns=["price_eur_mwh"])
            intraday_missing_years.append(year)
        completed += 1
        progress.progress(completed / total_runs, text=f"Intraday series cached: {year}")

        conservative_dispatch_by_year[year] = compute_dispatch_with_disk_cache(
            year=year,
            strategy=CONSERVATIVE_STRATEGY,
            price_frame=year_prices,
            energy_mwh=energy_mwh,
            rte=rte,
            market_key="da_id_overlay",
            intraday_price_frame=intraday_prices_by_year[year],
        )
        completed += 1
        progress.progress(completed / total_runs, text=f"Dispatch cached: {year} / restrained")
    progress.empty()
    if intraday_missing_years:
        missing_label = ", ".join(str(year) for year in intraday_missing_years)
        st.warning(f"ID-AEP could not be loaded for {missing_label}. Those years fall back to day-ahead only.")

    selected_prices = prices[prices.index.year == analysis_year]
    selected_intraday_prices = intraday_prices_by_year.get(analysis_year, pd.DataFrame(columns=["price_eur_mwh"]))
    conservative_selected = conservative_dispatch_by_year[analysis_year]
    concentration_stats = compute_concentration_stats(conservative_selected["revenue_eur_per_mw"])
    top_20pct_share = concentration_stats["top_20_days_pct_of_revenue"]
    missed_days_curve = summarize_missing_top_days(
        conservative_selected["revenue_eur_per_mw"],
        top_day_counts=tuple(range(1, 51)),
    )
    missed_day_twenty = missed_days_curve.loc[20]
    selected_top_day_dates = conservative_selected["revenue_eur_per_mw"].sort_values(ascending=False).head(20).index
    price_shape_profiles = compute_price_shape_profiles(selected_prices, selected_top_day_dates)

    selected_day_ahead_observables = build_day_ahead_observable_table(
        day_ahead_price_frame=selected_prices,
        outcome_dispatch=conservative_selected,
        top_day_counts=(10, 20),
    )
    feature_comparison = summarize_day_ahead_feature_separation(selected_day_ahead_observables)

    day_ahead_observables_by_year = {
        year: build_day_ahead_observable_table(
            day_ahead_price_frame=prices[prices.index.year == year],
            outcome_dispatch=conservative_dispatch_by_year[year],
            top_day_counts=(10, 20),
        )
        for year in YEARS
    }
    pooled_day_ahead_observables = concatenate_day_ahead_observable_tables(day_ahead_observables_by_year)
    pooled_watchlist_top20 = build_day_ahead_watchlist_table(pooled_day_ahead_observables, target_count=20)
    pooled_base_rate_pct = 100 * float(pooled_day_ahead_observables["is_top_20_revenue_day"].mean())
    d1_watchlist = pooled_watchlist_top20.loc["DA spread >= 200 €/MWh"]
    d2_watchlist = {
        "lift_x": 2.24,
        "precision_pct": 12.27,
        "recall_pct": 27.0,
    }
    watchlist_screen_summary = pd.DataFrame(
        [
            {
                "stage": "D-2 early warning",
                "lift_x": d2_watchlist["lift_x"],
                "precision_pct": d2_watchlist["precision_pct"],
                "recall_pct": d2_watchlist["recall_pct"],
                "rule_label_html": "weekday + recent 3d mean spread<br>≥ 175 €/MWh",
                "supporting_label_html": "<span style='font-size:10px;color:#5c677d'>~1 in 8 flagged days is a top day | catches 27% of top days</span>",
                "color": "#264653",
            },
            {
                "stage": "D-1 day-ahead screen",
                "lift_x": float(d1_watchlist["lift_x"]),
                "precision_pct": float(d1_watchlist["precision_pct"]),
                "recall_pct": float(d1_watchlist["recall_pct"]),
                "rule_label_html": "day-ahead spread<br>≥ 200 €/MWh",
                "supporting_label_html": "<span style='font-size:10px;color:#5c677d'>~1 in 5 flagged days is a top day | catches 50% of top days</span>",
                "color": "#e76f51",
            },
        ]
    )

    with st.spinner("Building same-throughput reallocation view..."):
        _, cycle_dispatch_by_cap = build_cycle_intensity_frontier_payload(
            year=analysis_year,
            price_frame=selected_prices,
            intraday_price_frame=selected_intraday_prices,
            energy_mwh=energy_mwh,
            rte=rte,
            discount_rate=discount_rate,
            project_lifetime=project_lifetime,
        )
    daily_value_curve = build_daily_value_curve(cycle_dispatch_by_cap)
    equal_throughput_summary, _ = summarize_reallocated_same_throughput_vs_strict_daily_cap(
        dispatch_by_cycle_cap=cycle_dispatch_by_cap,
        daily_value_curve=daily_value_curve,
        daily_caps=(1.0, 1.5, 2.0),
    )

    st.subheader("Revenue is concentrated where it matters most")
    st.markdown(
        f"Merchant BESS value is not earned evenly through the year. In {analysis_year}, the top 20% of days generated {top_20pct_share * 100:.1f}% of annual merchant revenue for a {duration_hours}h battery. That concentration matters because missing only a limited number of the best days can do disproportionate damage to annual returns. In {analysis_year}, missing the top 20 revenue days would have reduced annual revenue by {missed_day_twenty['lost_share_pct']:.1f}%."
    )
    render_chart_title("Missing a small number of the best days can do disproportionate damage to annual revenue")
    missed_days_figure = build_missed_days_figure(missed_days_curve, highlight_day=20)
    st.plotly_chart(missed_days_figure, width="stretch")
    render_footer_note(
        "Revenue loss measured as annual merchant revenue foregone when the highest-revenue days are assumed unavailable."
    )
    render_chart_caption(
        "In 2025, a relatively small number of days accounted for a disproportionate share of annual merchant value. Missing the top 20 revenue days would have reduced annual revenue by 20.4%."
    )
    render_annotation_box(
        "Why this matters",
        "For a merchant BESS owner, the main commercial risk is not a weak average day. It is being unavailable when a small number of highly valuable days arrives.",
    )

    st.subheader("The best days are partly visible ahead of delivery")
    st.markdown(
        f"The highest-value days were not identical, but they often shared a recognisable commercial shape. Relative to a normal day, they were more likely to show weaker prices around midday and stronger prices into the evening, creating a wider charge-discharge window for a {duration_hours}h battery."
    )
    st.markdown(
        f"In {analysis_year}, top-20 revenue days had a median day-ahead trough of {feature_comparison.loc['top_20_revenue_days', 'median_midday_min_price_eur_mwh']:.0f} €/MWh versus {feature_comparison.loc['all_days', 'median_midday_min_price_eur_mwh']:.0f} €/MWh on an average day, and a median evening-minus-midday ramp of {feature_comparison.loc['top_20_revenue_days', 'median_da_evening_minus_midday_ramp_eur_mwh']:.0f} €/MWh versus {feature_comparison.loc['all_days', 'median_da_evening_minus_midday_ramp_eur_mwh']:.0f} €/MWh."
    )
    st.markdown(
        "Some of these days were classic solar-surplus days, with cheap midday charging and stronger evening discharge. Others were driven more by broader market stress and repricing. The market drivers varied, but the commercial shape was similar: a wider and more valuable charging-to-discharging window."
    )
    render_chart_title("High-value days tended to show a deeper midday trough and a wider evening-minus-midday ramp")
    price_shape_figure = build_price_shape_figure(price_shape_profiles)
    median_profiles = price_shape_profiles["median_profiles"].set_index("hour")[["all_days", "top_days"]].astype(float)
    midday_profiles = median_profiles.loc[10:15]
    evening_profiles = median_profiles.loc[17:21]

    all_trough_hour = float(midday_profiles["all_days"].idxmin())
    all_trough_line_value = float(midday_profiles["all_days"].min())
    top_trough_hour = float(midday_profiles["top_days"].idxmin())
    top_trough_line_value = float(midday_profiles["top_days"].min())
    all_evening_hour = float(evening_profiles["all_days"].idxmax())
    all_evening_line_value = float(evening_profiles["all_days"].max())
    top_evening_hour = float(evening_profiles["top_days"].idxmax())
    top_evening_line_value = float(evening_profiles["top_days"].max())

    all_trough_metric = float(feature_comparison.loc["all_days", "median_midday_min_price_eur_mwh"])
    top_trough_metric = float(feature_comparison.loc["top_20_revenue_days", "median_midday_min_price_eur_mwh"])
    all_evening_metric = float(feature_comparison.loc["all_days", "median_evening_peak_price_eur_mwh"])
    top_evening_metric = float(feature_comparison.loc["top_20_revenue_days", "median_evening_peak_price_eur_mwh"])
    all_ramp_metric = float(feature_comparison.loc["all_days", "median_da_evening_minus_midday_ramp_eur_mwh"])
    top_ramp_metric = float(feature_comparison.loc["top_20_revenue_days", "median_da_evening_minus_midday_ramp_eur_mwh"])

    leader_style_all = {"color": "rgba(38, 70, 83, 0.45)", "width": 1.4, "dash": "dot"}
    leader_style_top = {"color": "rgba(231, 111, 81, 0.55)", "width": 1.4, "dash": "dot"}
    price_shape_figure.add_annotation(
        x=8.85,
        y=31.5,
        xref="x",
        yref="y",
        text=(
            "<b>Median trough within the 10-15 window</b>"
            f"<br>{top_trough_metric:.0f} €/MWh on top revenue days"
            f"<br>vs {all_trough_metric:.0f} €/MWh on an average day"
        ),
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="middle",
        bgcolor="rgba(255, 251, 245, 0.96)",
        bordercolor="rgba(20, 33, 61, 0.12)",
        borderwidth=1,
        borderpad=5,
        font={"size": 10.5},
    )

    all_arrow_x = 21.95
    top_arrow_x = 22.55

    price_shape_figure.add_shape(
        type="line",
        x0=all_trough_hour,
        y0=all_trough_line_value,
        x1=all_arrow_x,
        y1=all_trough_line_value,
        xref="x",
        yref="y",
        line=leader_style_all,
        layer="above",
    )
    price_shape_figure.add_shape(
        type="line",
        x0=all_evening_hour,
        y0=all_evening_line_value,
        x1=all_arrow_x,
        y1=all_evening_line_value,
        xref="x",
        yref="y",
        line=leader_style_all,
        layer="above",
    )
    price_shape_figure.add_shape(
        type="line",
        x0=top_trough_hour,
        y0=top_trough_line_value,
        x1=top_arrow_x,
        y1=top_trough_line_value,
        xref="x",
        yref="y",
        line=leader_style_top,
        layer="above",
    )
    price_shape_figure.add_shape(
        type="line",
        x0=top_evening_hour,
        y0=top_evening_line_value,
        x1=top_arrow_x,
        y1=top_evening_line_value,
        xref="x",
        yref="y",
        line=leader_style_top,
        layer="above",
    )
    price_shape_figure.add_annotation(
        x=all_arrow_x,
        y=all_evening_line_value,
        ax=all_arrow_x,
        ay=all_trough_line_value,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowsize=1,
        arrowwidth=1.8,
        arrowcolor="#264653",
    )
    price_shape_figure.add_annotation(
        x=top_arrow_x,
        y=top_evening_line_value,
        ax=top_arrow_x,
        ay=top_trough_line_value,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowside="end+start",
        arrowsize=1,
        arrowwidth=1.8,
        arrowcolor="#e76f51",
    )
    price_shape_figure.add_annotation(
        x=all_arrow_x + 0.22,
        y=(all_trough_line_value + all_evening_line_value) / 2,
        xref="x",
        yref="y",
        text=f"<b>{all_ramp_metric:.0f} €/MWh</b>",
        showarrow=False,
        font={"size": 10.5, "color": "#264653"},
        bgcolor="rgba(255, 251, 245, 0.96)",
        bordercolor="rgba(20, 33, 61, 0.10)",
        borderwidth=1,
        borderpad=4,
    )
    price_shape_figure.add_annotation(
        x=top_arrow_x + 0.22,
        y=(top_trough_line_value + top_evening_line_value) / 2,
        xref="x",
        yref="y",
        text=f"<b>{top_ramp_metric:.0f} €/MWh</b>",
        showarrow=False,
        font={"size": 10.5, "color": "#e76f51"},
        bgcolor="rgba(255, 251, 245, 0.96)",
        bordercolor="rgba(20, 33, 61, 0.10)",
        borderwidth=1,
        borderpad=4,
    )
    price_shape_figure.update_xaxes(range=[0, 23.35])
    price_shape_figure.update_yaxes(range=[0, max(top_evening_metric, all_evening_metric, top_evening_line_value, all_evening_line_value) + 8])
    st.plotly_chart(price_shape_figure, width="stretch")
    render_footer_note(
        "Profiles shown as median hourly day-ahead prices for top-20 revenue days versus the full sample average day."
    )
    render_chart_caption(
        "Many of the best revenue days already showed a recognisable day-ahead profile before delivery: weaker midday prices, stronger evening prices, and a wider charging-to-discharging window."
    )
    render_takeaway("The best days are not perfectly predictable, but they are not invisible either.")

    st.subheader("A simple watchlist is already useful")
    st.markdown(
        "The practical question is not whether operators can predict every top-revenue day. It is whether the market can identify days that are more likely than normal to become commercially important."
    )
    st.markdown(
        "At the day-ahead horizon, the answer is clearly yes. A simple rule already proved useful: when the day-ahead spread was at least 200 €/MWh, the day was around four times more likely than normal to become a top revenue day. The rule is simple by design. It flags days when the day-ahead curve already shows an unusually wide charging-to-discharging window."
    )
    st.markdown(
        "Useful signal remained even two days earlier. When the market was already in a higher-spread regime — captured here by weekday and a recent three-day mean spread of at least 175 €/MWh — the odds of a top revenue day were a little more than twice the normal level. This is weaker than the day-ahead screen, but still useful as an early warning for maintenance timing."
    )
    st.markdown(
        "The main caveat is stability. Early-warning screens were less consistent across years and were stronger in the more stressed 2021–2022 period than in 2025. For owners, that suggests a practical split: use D-2 as an early-warning layer, but rely most heavily on D-1 for high-conviction readiness decisions."
    )
    render_takeaway("A useful D-2 early-warning screen exists, but the strongest screening power still appears at D-1.")
    render_chart_title("A useful early-warning screen exists at D-2, but the strongest screening power appears at D-1")
    watchlist_figure = build_early_warning_screen_figure(watchlist_screen_summary)
    st.plotly_chart(watchlist_figure, width="stretch")
    render_footer_note(
        f"Base rate: {pooled_base_rate_pct:.2f}% probability of a random day becoming a top-20 revenue day in pooled 2021–2025 data."
    )
    render_chart_caption(
        f"A useful early-warning signal exists before the day-ahead horizon, but screening power strengthens materially at D-1. In pooled 2021–2025 data, a simple D-2 screen still raised the odds of a top revenue day by {d2_watchlist['lift_x']:.2f}x versus the base rate, compared with {float(d1_watchlist['lift_x']):.2f}x for a simple D-1 day-ahead screen."
    )
    render_takeaway("Use D-2 for maintenance awareness and D-1 for readiness decisions.")

    st.subheader("The same cycles are worth more on the right days")
    st.markdown(
        "Flexibility mattered not only because it allowed more cycling, but because it allowed the same annual throughput to be used when opportunity was highest."
    )
    st.markdown(
        "To isolate that effect directly, each strict daily cap was compared against an annual allocator given exactly the same realised FEC that the strict policy actually used. This removes the benefit of extra throughput and isolates the value of reallocating cycles across days."
    )
    st.markdown(
        f"Even at the same realised throughput, flexibility added value. In {analysis_year}, reallocating the same cycles across days increased revenue by {equal_throughput_summary.loc['1.0/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['1.0/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) in the 1.0/day case, {equal_throughput_summary.loc['1.5/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['1.5/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) in the 1.5/day case, and {equal_throughput_summary.loc['2.0/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['2.0/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) in the 2.0/day case."
    )
    st.markdown(
        "The mechanism is simple. A flat daily cap leaves money on the table because it forces the battery to stop too early on some of the year’s best days. The gain from flexibility does not come from cycling harder every day. It comes from using the same limited cycles less on weaker days and more on stronger ones."
    )
    render_chart_title("The same annual throughput earned more when it was concentrated into stronger days")
    same_cycles_figure = build_same_cycles_reallocation_figure(equal_throughput_summary)
    st.plotly_chart(same_cycles_figure, width="stretch")
    render_chart_caption(
        "Flexibility added value even without additional throughput, because the same annual cycles were worth more when they were spent on stronger days."
    )
    render_takeaway("Even with the same annual throughput, a battery earns more when cycles are concentrated into the days that matter most.")

    st.subheader("What This Changes for Owners")
    st.markdown(
        "For merchant BESS owners, the main commercial risk is not average underperformance across the year. It is being unavailable on a limited set of disproportionately valuable days."
    )
    st.markdown(
        "Because many of those days are already partly visible from the day-ahead curve, availability should be managed against expected opportunity rather than average conditions. Planned maintenance, operating readiness, and throughput headroom should all be treated as market-timed decisions."
    )
    render_closing_line(
        "For merchant BESS, flexibility is not only the ability to cycle. It is the ability to be available, ready, and unconstrained when the best days arrive."
    )
    render_takeaway("Availability, readiness, and flexibility matter most when they are timed.")
    if selected_intraday_prices.empty:
        render_footer_note(
            "Modeled as day-ahead only for the selected year because the intraday layer could not be loaded."
        )
    render_footer_note("© 2026 Anton Telegin. Analysis based on public market data.")


if __name__ == "__main__":
    main()
