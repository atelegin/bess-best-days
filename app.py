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
    build_missed_days_figure,
    build_same_cycles_reallocation_figure,
    build_watchlist_scatter_figure,
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
            .section-lede {
                font-size: 1.08rem;
                line-height: 1.7;
                margin: 0 0 1rem 0;
            }
            .proof-box {
                margin: 0.8rem 0 2rem 0;
                padding: 0.9rem 1rem;
                background: rgba(255, 251, 245, 0.88);
                border-left: 4px solid var(--accent);
                border-radius: 0.5rem;
            }
            .proof-label {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                color: var(--accent);
                font-weight: 700;
                margin-bottom: 0.32rem;
            }
            .proof-text {
                color: var(--ink);
                line-height: 1.55;
                font-weight: 600;
            }
            .small-note {
                color: var(--muted);
                font-size: 0.92rem;
            }
            .closing-line {
                font-family: 'Source Serif 4', serif;
                font-size: 1.25rem;
                line-height: 1.45;
                margin: 1rem 0 0.75rem 0;
            }
            .logic-line {
                margin: 0.8rem 0 0 0;
                color: var(--muted);
                font-size: 0.95rem;
                line-height: 1.55;
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
            <div class="hero-kicker">German BESS | Merchant Revenue | Availability | Flexibility</div>
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

def render_section_lede(text: str) -> None:
    st.markdown(f'<div class="section-lede">{text}</div>', unsafe_allow_html=True)


def render_proof_box(text: str) -> None:
    st.markdown(
        f'<div class="proof-box"><div class="proof-label">What this slide proves</div><div class="proof-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_base_case_note() -> None:
    st.markdown(
        (
            '<div class="small-note"><strong>Base case:</strong> 2h battery, 2025 deep dive. '
            '<strong>Validation:</strong> pooled 2021–2025. '
            f'Fixed assumptions here: round-trip efficiency = {DEFAULT_RTE:.2f}, discount rate = {BASE_CASE_DISCOUNT_RATE:.0%}, '
            f'project life = {BASE_CASE_PROJECT_LIFETIME} years.</div>'
        ),
        unsafe_allow_html=True,
    )


def render_footer_note(text: str) -> None:
    st.markdown(f'<div class="small-note">{text}</div>', unsafe_allow_html=True)


def render_closing_line(text: str) -> None:
    st.markdown(f'<div class="closing-line">{text}</div>', unsafe_allow_html=True)


def render_logic_line(text: str) -> None:
    st.markdown(f'<div class="logic-line"><strong>Story logic:</strong> {text}</div>', unsafe_allow_html=True)


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
    render_base_case_note()
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
    watchlist_signals = [
        "DA evening-midday ramp >= 200 €/MWh",
        "DA spread >= 200 €/MWh",
        "DA evening-midday ramp >= 150 €/MWh",
    ]
    watchlist_short_label = {
        "DA evening-midday ramp >= 200 €/MWh": "Ramp >= 200 €/MWh",
        "DA spread >= 200 €/MWh": "Spread >= 200 €/MWh",
        "DA evening-midday ramp >= 150 €/MWh": "Ramp >= 150 €/MWh",
    }
    watchlist_callouts = {
        "DA evening-midday ramp >= 200 €/MWh": "tighter, higher-conviction signal",
        "DA spread >= 200 €/MWh": "most balanced practical screen",
        "DA evening-midday ramp >= 150 €/MWh": "broader readiness watchlist",
    }
    watchlist_annotation_positions = {
        "DA evening-midday ramp >= 200 €/MWh": {"xshift": 68, "yshift": 22, "xanchor": "left", "yanchor": "bottom"},
        "DA spread >= 200 €/MWh": {"xshift": 70, "yshift": -2, "xanchor": "left", "yanchor": "middle"},
        "DA evening-midday ramp >= 150 €/MWh": {"xshift": -18, "yshift": 18, "xanchor": "right", "yanchor": "bottom"},
    }
    pooled_watchlist_scatter = pooled_watchlist_top20.reindex(watchlist_signals).copy()
    pooled_watchlist_scatter["short_label"] = pooled_watchlist_scatter.index.map(watchlist_short_label)
    pooled_watchlist_scatter["callout"] = pooled_watchlist_scatter.index.map(watchlist_callouts)
    for column in ("xshift", "yshift", "xanchor", "yanchor"):
        pooled_watchlist_scatter[column] = pooled_watchlist_scatter.index.map(
            lambda signal, field=column: watchlist_annotation_positions[signal][field]
        )
    pooled_base_rate_pct = 100 * float(pooled_day_ahead_observables["is_top_20_revenue_day"].mean())

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

    st.subheader("1. Missing a small number of the best days can do disproportionate damage to annual revenue")
    missed_days_figure = build_missed_days_figure(missed_days_curve, highlight_day=20)
    st.plotly_chart(missed_days_figure, width="stretch")
    st.markdown(
        f"Merchant BESS value is not earned evenly through the year. In {analysis_year}, the top 20% of days generated {top_20pct_share * 100:.1f}% of annual merchant revenue for a {duration_hours}h battery. That concentration matters because missing only a limited number of the best days can do disproportionate damage to annual returns. In {analysis_year}, missing the top 20 revenue days would have reduced annual revenue by {missed_day_twenty['lost_share_pct']:.1f}%."
    )
    render_proof_box(
        "The main commercial risk is not average underperformance. It is being unavailable on a limited set of highly valuable days."
    )

    st.subheader("2. High-value days tended to show a deeper midday trough and a stronger evening ramp")
    price_shape_figure = build_price_shape_figure(price_shape_profiles)
    st.plotly_chart(price_shape_figure, width="stretch")
    st.markdown(
        f"The highest-value days were not identical, but they often shared a recognisable day-ahead shape. Relative to a normal day, they were more likely to show weaker prices around midday and stronger prices into the evening, creating a wider charge-discharge window for a {duration_hours}h battery. In {analysis_year}, top-20 revenue days had a median day-ahead trough of {feature_comparison.loc['top_20_revenue_days', 'median_midday_min_price_eur_mwh']:.0f} €/MWh versus {feature_comparison.loc['all_days', 'median_midday_min_price_eur_mwh']:.0f} €/MWh on an average day, and a median evening-minus-midday ramp of {feature_comparison.loc['top_20_revenue_days', 'median_da_evening_minus_midday_ramp_eur_mwh']:.0f} €/MWh versus {feature_comparison.loc['all_days', 'median_da_evening_minus_midday_ramp_eur_mwh']:.0f} €/MWh."
    )
    render_proof_box(
        "The best days are not perfectly predictable, but they are not invisible ahead of delivery."
    )

    st.subheader("3. Simple day-ahead rules identified days with materially higher-than-normal odds of becoming top revenue days")
    watchlist_figure = build_watchlist_scatter_figure(pooled_watchlist_scatter, pooled_base_rate_pct)
    st.plotly_chart(watchlist_figure, width="stretch")
    st.markdown(
        f"The practical question is not whether operators can predict every top-revenue day. It is whether the day-ahead curve can identify days that are more likely than normal to become commercially important. Across pooled 2021–2025 data, the answer was yes. A random day in the pooled sample had only a {pooled_base_rate_pct:.1f}% chance of becoming a top-20 revenue day, so a signal with around 20% precision already identifies days with roughly four times the normal odds."
    )
    render_proof_box(
        "A simple watchlist is already good enough to support better maintenance timing, readiness, and throughput prioritisation."
    )

    st.subheader("4. The same annual throughput earned more when it was concentrated into stronger days")
    same_cycles_figure = build_same_cycles_reallocation_figure(equal_throughput_summary)
    st.plotly_chart(same_cycles_figure, width="stretch")
    st.markdown(
        f"Flexibility mattered not only because it allowed more cycling, but because it allowed the same annual throughput to be used when opportunity was highest. Comparing strict daily caps with an annual allocator given the same realised FEC showed that reallocation alone added value. The gain came from using limited cycles less on weaker days and more on stronger ones. In {analysis_year}, that uplift was +{equal_throughput_summary.loc['1.0/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['1.0/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) at 1.0/day, +{equal_throughput_summary.loc['1.5/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['1.5/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) at 1.5/day, and +{equal_throughput_summary.loc['2.0/day vs reallocated same FEC', 'uplift_eur_per_mw'] / 1000:.1f}k €/MW (+{equal_throughput_summary.loc['2.0/day vs reallocated same FEC', 'uplift_pct_vs_strict']:.1f}%) at 2.0/day."
    )
    render_proof_box(
        "The value of flexibility is not only more throughput. It is better timing."
    )

    st.subheader("What This Changes for Owners")
    st.markdown(
        "For merchant BESS owners, the main commercial risk is not average underperformance across the year. It is being unavailable on a limited set of disproportionately valuable days. Because many of those days are already partly visible from the day-ahead curve, availability should be managed against expected opportunity rather than average conditions. Planned maintenance, operating readiness, and throughput headroom should all be treated as market-timed decisions."
    )
    render_closing_line(
        "For merchant BESS, flexibility is not only the ability to cycle. It is the ability to be available, ready, and unconstrained when the best days arrive."
    )
    render_logic_line(
        "The best days matter disproportionately, they are partly visible ahead of delivery, and flexibility is valuable because it lets operators act on that timing."
    )
    with st.expander("Method note"):
        st.caption(
            f"Methods: Energy-Charts day-ahead data, Netztransparenz ID-AEP intraday series, fixed round-trip efficiency of {DEFAULT_RTE:.2f}, sequential day-ahead plus intraday dispatch with SciPy HiGHS, cycle-cap sweep from 0.25 to 4.00 cycles/day, and a harsher lifecycle model anchored to 7,300 reference cycles over 20 years with retirement at 60% state of health. The break-even annual premium/reserve is computed as the lifetime value gap between the warranty pace and the economic optimum, spread over the project-life annuity at the selected discount rate. This is still a simplified economic model, not a chemistry-specific warranty simulator."
        )
        render_public_sources()
    if not selected_intraday_prices.empty:
        render_footer_note(
            "Modeled as one combined day-ahead + intraday revenue series, using the official Netztransparenz ID-AEP index for the intraday layer."
        )
    else:
        render_footer_note(
            "Modeled as day-ahead only for the selected year because the intraday layer could not be loaded."
        )


if __name__ == "__main__":
    main()
