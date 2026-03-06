from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.concentration import compute_concentration_stats, compute_pareto_curve, days_to_revenue_share
from src.analysis.drivers import (
    build_daily_driver_table,
    classify_tail_patterns,
    compute_price_shape_profiles,
    tail_day_signal_summary,
)
from src.charts.calendar import build_calendar_heatmap
from src.charts.pareto import build_pareto_figure, build_revenue_distribution_figure
from src.charts.scatter import build_price_shape_figure, build_tail_pattern_figure
from src.charts.strategy import build_cycle_intensity_frontier_figure, build_warranty_posture_figure
from src.data.cache import get_or_build_dataframe, make_cache_key
from src.data.generation import compute_daily_generation_metrics, fetch_generation_data
from src.data.netztransparenz import fetch_id_aep
from src.data.prices import fetch_day_ahead_prices
from src.models.degradation import (
    DEFAULT_DEGRADATION_ASSUMPTIONS,
    equivalent_stress_fec_per_year,
    estimate_years_to_eol,
    lifecycle_value_profile,
)
from src.models.dispatch import (
    CONSERVATIVE_STRATEGY,
    DispatchStrategy,
    run_dispatch_for_period,
    run_dispatch_with_intraday_overlay_for_period,
)

REPORT_TITLE = "The Tail Wags the Dog"
YEARS = list(range(2021, 2026))
DEFAULT_RTE = 0.86


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
            h1, h2, h3 {
                font-family: 'Source Serif 4', serif;
                color: var(--ink);
                letter-spacing: -0.02em;
            }
            body, .stMarkdown, .stMetric, .stDataFrame {
                font-family: 'IBM Plex Sans', sans-serif;
            }
            div[data-testid="stMetric"] {
                background: var(--card);
                border: 1px solid rgba(20, 33, 61, 0.08);
                border-radius: 18px;
                padding: 0.8rem 1rem;
            }
            div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
                gap: 0.75rem;
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(20,33,61,0.98), rgba(28,56,92,0.96));
            }
            section[data-testid="stSidebar"] * {
                color: #f8fafc;
            }
            section[data-testid="stSidebar"] [data-baseweb="select"] > div {
                background: rgba(255, 255, 255, 0.98);
                border-radius: 18px;
                color: #14213d !important;
            }
            section[data-testid="stSidebar"] [data-baseweb="select"] * {
                color: #14213d !important;
                fill: #14213d !important;
            }
            section[data-testid="stSidebar"] [data-baseweb="select"] svg {
                color: #14213d !important;
                fill: #14213d !important;
            }
            .hero-card {
                background: var(--card);
                border: 1px solid rgba(20, 33, 61, 0.08);
                border-radius: 24px;
                padding: 1.35rem 1.45rem;
                margin-bottom: 1rem;
                box-shadow: 0 18px 50px rgba(20, 33, 61, 0.08);
            }
            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.72rem;
                color: var(--accent);
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .hero-quote {
                font-family: 'Source Serif 4', serif;
                font-size: 2rem;
                line-height: 1.18;
                margin: 0.3rem 0 0.8rem 0;
                max-width: 1000px;
            }
            .hero-dek {
                font-size: 1.04rem;
                line-height: 1.6;
                max-width: 980px;
                color: var(--ink);
            }
            .section-lede {
                font-size: 1.08rem;
                line-height: 1.7;
                max-width: 980px;
                margin: 0 0 1rem 0;
            }
            .pull-quote {
                border-left: 4px solid var(--accent);
                padding: 0.9rem 1rem;
                margin: 0.6rem 0 1rem 0;
                background: rgba(231, 111, 81, 0.08);
                border-radius: 0 16px 16px 0;
                font-size: 1.04rem;
                line-height: 1.6;
            }
            .small-note {
                color: var(--muted);
                font-size: 0.92rem;
            }
            .source-note {
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.45;
            }
            .source-note p {
                margin: 0.18rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">German BESS | Merchant Revenue | Cycling Strategy</div>
            <h1>The Tail Wags the Dog</h1>
            <div class="hero-quote">
                Should a battery be preserved for future optionality, or pushed harder while today's tail still pays?
            </div>
            <div class="hero-dek">
                German BESS revenue is not earned evenly through the year. A small number of extreme days can do a disproportionate
                amount of the work. That changes the real investment question: whether degradation risk should make operators hold
                back, or whether today's tail is rich enough that the battery should run harder now.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_battery_inputs() -> dict[str, float | int]:
    with st.sidebar:
        st.header("Battery Configuration")
        duration_hours = st.selectbox("Duration (hours)", options=[1, 2, 4], index=1)
        analysis_year = st.selectbox("Year to analyze", options=YEARS, index=YEARS.index(2025))
        discount_rate_pct = st.slider("Discount rate (%)", min_value=4, max_value=12, value=8, step=1)
        project_lifetime = st.slider("Battery project lifetime (years)", min_value=10, max_value=20, value=15, step=1)
        st.caption(f"Fixed assumption: round-trip efficiency = {DEFAULT_RTE:.2f}.")
        st.caption("Revenue combines German day-ahead prices with an official German intraday price series.")
        st.caption("Lifecycle frontier uses a warranty-anchored degradation model and retires the asset at 60% state of health.")
    return {
        "duration_hours": duration_hours,
        "analysis_year": analysis_year,
        "discount_rate": discount_rate_pct / 100,
        "project_lifetime": project_lifetime,
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
def load_core_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = fetch_day_ahead_prices(start=f"{YEARS[0]}-01-01", end=f"{YEARS[-1]}-12-31", force_refresh=False)
    generation = fetch_generation_data(start=f"{YEARS[0]}-01-01", end=f"{YEARS[-1]}-12-31", force_refresh=False)
    return prices, generation


@st.cache_data(show_spinner=False)
def load_id_aep_for_year(year: int) -> pd.DataFrame:
    return fetch_id_aep(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        force_refresh=False,
    )

def render_section_lede(text: str) -> None:
    st.markdown(f'<div class="section-lede">{text}</div>', unsafe_allow_html=True)


def render_pull_quote(text: str) -> None:
    st.markdown(f'<div class="pull-quote">{text}</div>', unsafe_allow_html=True)


def render_public_sources() -> None:
    st.markdown(
        """
        <div class="source-note">
            <p><strong>Public sources:</strong></p>
            <p>Data: <a href="https://api.energy-charts.info/price" target="_blank">Energy-Charts price API</a>,
            <a href="https://api.energy-charts.info/public_power" target="_blank">Energy-Charts public power API</a>,
            <a href="https://www.netztransparenz.de/de-de/Regelenergie/Ausgleichsenergiepreis/Index-Ausgleichsenergiepreis" target="_blank">Netztransparenz ID-AEP page</a>,
            <a href="https://www.netztransparenz.de/xspproxy/api/staticfiles/ntp-relaunch/dokumente/web-api/dokumentation-webserviceapi-netztransparenz_v1.14.pdf" target="_blank">Netztransparenz WebAPI documentation</a></p>
            <p>Warranty / degradation framing:
            <a href="https://powin.com/wp-content/uploads/2024/11/Powin-POD-Datasheet_Q4_2024.pdf" target="_blank">Powin Pod Datasheet Q4 2024</a>,
            <a href="https://ir.tesla.com/_flysystem/s3/sec/000156459017015705/tsla-10q_20170630-gen_0.pdf" target="_blank">Tesla 10-Q warranty language</a>,
            <a href="https://info.fluenceenergy.com/hubfs/Smart_Service_Plans_BR-057-02-EN.pdf" target="_blank">Fluence Smart Service Plans</a>,
            <a href="https://powin.com/wp-content/uploads/2024/02/Service-Offerings_Brochure-3.pdf" target="_blank">Powin Service Offerings</a></p>
            <p>Public battery-life modeling references:
            <a href="https://www.nrel.gov/transportation/blast.html" target="_blank">NREL BLAST</a>,
            <a href="https://www.nrel.gov/samples/other/software-module/blast-lite--battery-lifetime-analysis-and-simulation-tool---lite" target="_blank">NREL BLAST-Lite</a></p>
            <p>The break-even annual premium / reserve shown above is a model output, not a public OEM price list.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def annuity_factor(discount_rate: float, years: int) -> float:
    if years <= 0:
        return 0.0
    if abs(discount_rate) < 1e-9:
        return float(years)
    return float((1 - (1 + discount_rate) ** (-years)) / discount_rate)


def build_cycle_intensity_frontier_table(
    year: int,
    price_frame: pd.DataFrame,
    intraday_price_frame: pd.DataFrame,
    energy_mwh: float,
    rte: float,
    discount_rate: float,
    project_lifetime: int,
) -> pd.DataFrame:
    cycle_caps = np.arange(0.25, 4.01, 0.25)
    records = []
    for cycle_cap in cycle_caps:
        strategy = DispatchStrategy(
            name=f"frontier_{year}_{energy_mwh:.1f}h_{str(cycle_cap).replace('.', 'p')}",
            label=f"{cycle_cap:.2f} cycles/day",
            max_cycles=float(cycle_cap),
            soc_min_frac=0.05,
            soc_max_frac=0.95,
            min_spread_eur_mwh=5.0,
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
    return pd.DataFrame(records)


def main() -> None:
    st.set_page_config(page_title=REPORT_TITLE, layout="wide")
    apply_app_style()
    render_header()
    inputs = select_battery_inputs()

    duration_hours = int(inputs["duration_hours"])
    rte = DEFAULT_RTE
    analysis_year = int(inputs["analysis_year"])
    discount_rate = float(inputs["discount_rate"])
    project_lifetime = int(inputs["project_lifetime"])
    energy_mwh = float(duration_hours)
    warranty_reference_fec = DEFAULT_DEGRADATION_ASSUMPTIONS.reference_warranty_fec_per_year

    with st.spinner("Fetching and caching price and generation data..."):
        prices, generation = load_core_data()

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
    selected_story_prices = selected_intraday_prices if not selected_intraday_prices.empty else selected_prices
    selected_generation = generation[generation.index.year == analysis_year]
    selected_generation_daily = compute_daily_generation_metrics(selected_generation)
    conservative_selected = conservative_dispatch_by_year[analysis_year]
    best_day = float(conservative_selected["revenue_eur_per_mw"].max())
    median_day = float(conservative_selected["revenue_eur_per_mw"].median())
    days_to_50 = days_to_revenue_share(conservative_selected["revenue_eur_per_mw"], 0.50)

    pareto_curves = {
        year: compute_pareto_curve(conservative_dispatch_by_year[year]["revenue_eur_per_mw"])
        for year in YEARS
    }
    concentration_stats = compute_concentration_stats(conservative_selected["revenue_eur_per_mw"])
    top_20_share = concentration_stats["top_20_days_pct_of_revenue"]

    driver_table = build_daily_driver_table(
        dispatch_frame=conservative_selected,
        price_frame=selected_story_prices,
        generation_daily=selected_generation_daily,
    )
    top_days_table = driver_table.sort_values("revenue_eur_per_mw", ascending=False).head(20)
    price_shape_profiles = compute_price_shape_profiles(selected_story_prices, top_days_table.index[:10])
    tail_summary = tail_day_signal_summary(top_days_table)
    tail_patterns = classify_tail_patterns(driver_table, top_n=20)

    with st.spinner("Building warranty-backed cycle frontier..."):
        cycle_frontier = build_cycle_intensity_frontier_table(
            year=analysis_year,
            price_frame=selected_prices,
            intraday_price_frame=selected_intraday_prices,
            energy_mwh=energy_mwh,
            rte=rte,
            discount_rate=discount_rate,
            project_lifetime=project_lifetime,
        )
    optimal_cycle_row = cycle_frontier.loc[cycle_frontier["discounted_lifetime_value_eur_per_mw"].idxmax()]
    optimal_cycle_is_edge = bool(
        np.isclose(
            float(optimal_cycle_row["max_cycles_per_day"]),
            float(cycle_frontier["max_cycles_per_day"].max()),
        )
    )
    warranty_row = cycle_frontier.iloc[
        int(np.argmin(np.abs(cycle_frontier["full_equivalent_cycles_per_year"].to_numpy() - warranty_reference_fec)))
    ]
    value_uplift_vs_warranty = (
        optimal_cycle_row["discounted_lifetime_value_eur_per_mw"]
        / max(warranty_row["discounted_lifetime_value_eur_per_mw"], 1e-6)
        - 1
    )
    break_even_annual_access_cost = (
        optimal_cycle_row["discounted_lifetime_value_eur_per_mw"] - warranty_row["discounted_lifetime_value_eur_per_mw"]
    ) / max(annuity_factor(discount_rate, project_lifetime), 1e-9)

    st.subheader("1. Revenue Concentration")
    render_section_lede(
        f"In {analysis_year}, the <strong>top 20% of days generated {top_20_share * 100:.1f}% of annual merchant revenue</strong> "
        f"for a {duration_hours}h battery. <strong>Half of the year's revenue was earned in just {days_to_50} days.</strong>"
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Top 20% of days", f"{top_20_share * 100:.1f}% of annual revenue")
    metric_cols[1].metric("Days for 50% of revenue", f"{days_to_50}")
    metric_cols[2].metric("Best day", f"{best_day:,.0f} €/MW")
    render_pull_quote(
        f"The best day earned {best_day / max(median_day, 1e-6):.1f}x the median day. "
        f"This is not a smooth-cashflow business. A minority of days disproportionately shapes the year."
    )
    if not selected_intraday_prices.empty:
        st.caption(
            "Modeled as one combined day-ahead + intraday revenue series, using the official Netztransparenz ID-AEP index for the intraday layer."
        )
    else:
        st.caption("Intraday data were unavailable for the selected year, so the revenue series below falls back to day-ahead only.")
    st.plotly_chart(
        build_pareto_figure(
            pareto_curves,
            highlighted_year=analysis_year,
            highlight_day=days_to_50,
            target_revenue_share=0.50,
        ),
        width="stretch",
    )
    st.plotly_chart(
        build_calendar_heatmap(conservative_selected["revenue_eur_per_mw"], year=analysis_year),
        width="stretch",
    )
    with st.expander("See the daily revenue distribution"):
        st.plotly_chart(
            build_revenue_distribution_figure(conservative_selected["revenue_eur_per_mw"], concentration_stats),
            width="stretch",
        )

    st.subheader("2. What Drives the Best Days?")
    render_section_lede(
        f"Tail days are not random. In {analysis_year}, the most common top-day shape was <strong>{tail_patterns['dominant_pattern']}</strong> "
        f"({tail_patterns['dominant_share'] * 100:.0f}% of the top 20 days). The common thread is simple: a charging window first, then a violent repricing later in the day."
    )
    render_pull_quote(
        f"{tail_summary['share_with_negative_midday'] * 100:.0f}% of the top 20 days also featured negative midday pricing. "
        f"Across the whole top-20 set, the median midday floor was {tail_summary['median_midday_floor']:.1f} €/MWh, "
        f"while the median evening peak reached {tail_summary['median_evening_peak']:.0f} €/MWh."
    )
    st.plotly_chart(build_price_shape_figure(price_shape_profiles), width="stretch")
    st.plotly_chart(
        build_tail_pattern_figure(tail_patterns["distribution"]),
        width="stretch",
    )
    with st.expander("Open the top-20 day table"):
        st.dataframe(
            top_days_table[
                [
                    "revenue_eur_per_mw",
                    "max_price_eur_mwh",
                    "min_price_eur_mwh",
                    "spread_eur_mwh",
                    "solar_generation_gwh",
                    "wind_generation_gwh",
                    "residual_load_range_mw",
                ]
            ].rename(
                columns={
                    "revenue_eur_per_mw": "Revenue (€/MW)",
                    "max_price_eur_mwh": "Max price",
                    "min_price_eur_mwh": "Min price",
                    "spread_eur_mwh": "Daily spread",
                    "solar_generation_gwh": "Solar (GWh)",
                    "wind_generation_gwh": "Wind (GWh)",
                    "residual_load_range_mw": "Residual load range (MW)",
                }
            ),
            width="stretch",
            height=420,
        )

    st.subheader("3. How Hard Should the Battery Run?")
    if optimal_cycle_is_edge:
        render_section_lede(
            f"Even under a harsher, warranty-anchored degradation curve, the tested range still peaks at the right edge: "
            f"<strong>{optimal_cycle_row['full_equivalent_cycles_per_year']:,.0f} FEC/year</strong>. "
            f"That means the chart still has not found the true turning point before {optimal_cycle_row['max_cycles_per_day']:.2f} cycles/day."
        )
    else:
        render_section_lede(
            f"With a warranty-anchored degradation curve and retirement at 60% state of health, discounted lifetime value peaks around "
            f"<strong>{optimal_cycle_row['full_equivalent_cycles_per_year']:,.0f} FEC/year</strong>. "
            f"That is about <strong>{value_uplift_vs_warranty * 100:.0f}% more value</strong> than the nearest one-cycle-per-day warranty pace."
        )
    render_pull_quote(
        f"The dashed reference line marks roughly {warranty_reference_fec:,.0f} FEC/year, the operating pace implied by 7,300 full cycles over 20 years. "
        f"The frontier uses that as a harsher vendor-backed reference curve and retires the asset at 60% state of health."
    )
    cycle_cols = st.columns(3)
    cycle_cols[0].metric("Value-maximising intensity", f"{optimal_cycle_row['full_equivalent_cycles_per_year']:,.0f} FEC/year")
    cycle_cols[1].metric("Discounted lifetime value", f"{optimal_cycle_row['discounted_lifetime_value_eur_per_mw']:,.0f} €/MW")
    cycle_cols[2].metric("Years to 60% SOH", f"{optimal_cycle_row['years_to_eol']:.1f}")
    st.plotly_chart(
        build_cycle_intensity_frontier_figure(
            cycle_frontier,
            reference_warranty_fec_per_year=warranty_reference_fec,
        ),
        width="stretch",
    )
    st.caption(
        "The frontier varies only the daily cycle cap. The trading hurdle and SoC window stay fixed, so the chart isolates cycle intensity rather than changing the whole strategy at once."
    )

    st.subheader("4. Who Should Own the Cycling Risk?")
    render_section_lede(
        f"The commercial question is not just how hard the battery should run. It is who should own the downside once cycling moves above the standard warranty pace. "
        f"In this sample, the gap between the warranty pace and the economic optimum is worth about <strong>{break_even_annual_access_cost:,.0f} €/MW/year</strong>."
    )
    render_pull_quote(
        "That number can be read two ways: as the maximum annual premium worth paying for an expanded warranty envelope, or as the maximum annual reserve worth setting aside if the owner chooses to self-insure."
    )
    risk_cols = st.columns(3)
    risk_cols[0].metric("Inside warranty value", f"{warranty_row['discounted_lifetime_value_eur_per_mw']:,.0f} €/MW")
    risk_cols[1].metric("Overdrive value", f"{optimal_cycle_row['discounted_lifetime_value_eur_per_mw']:,.0f} €/MW")
    risk_cols[2].metric("Break-even annual premium / reserve", f"{break_even_annual_access_cost:,.0f} €/MW/yr")
    st.plotly_chart(
        build_warranty_posture_figure(
            warranty_value_eur_per_mw=float(warranty_row["discounted_lifetime_value_eur_per_mw"]),
            overdrive_value_eur_per_mw=float(optimal_cycle_row["discounted_lifetime_value_eur_per_mw"]),
            break_even_annual_cost_eur_per_mw=float(break_even_annual_access_cost),
            project_lifetime=project_lifetime,
        ),
        width="stretch",
    )
    st.markdown(
        f"""
        - **Stay inside warranty:** lowest execution and replacement risk, but roughly {value_uplift_vs_warranty * 100:.0f}% less discounted value than the economic optimum in this sample.
        - **Buy more cycling from the OEM/LTSA:** rational if the all-in annual premium is below about {break_even_annual_access_cost:,.0f} €/MW/year.
        - **Self-insure and overdrive:** only rational if expected augmentation, downtime, and residual-value drag stay below the same break-even number.
        - **Traders with a hard cycle budget:** spend those cycles on the highest-conviction tail days, not on smoothing ordinary days.
        """
    )
    with st.expander("Method note"):
        st.caption(
            f"Methods: Energy-Charts day-ahead data, Netztransparenz ID-AEP intraday series, fixed round-trip efficiency of {DEFAULT_RTE:.2f}, sequential day-ahead plus intraday dispatch with SciPy HiGHS, cycle-cap sweep from 0.25 to 4.00 cycles/day, and a harsher lifecycle model anchored to 7,300 reference cycles over 20 years with retirement at 60% state of health. The break-even annual premium/reserve is computed as the lifetime value gap between the warranty pace and the economic optimum, spread over the project-life annuity at the selected discount rate. This is still a simplified economic model, not a chemistry-specific warranty simulator."
        )
        render_public_sources()


if __name__ == "__main__":
    main()
