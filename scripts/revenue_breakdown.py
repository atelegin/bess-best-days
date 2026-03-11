from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.concentration import compute_concentration_stats
from src.analysis.day_ahead_signals import (
    build_day_ahead_observable_table,
    build_day_ahead_watchlist_table,
    concatenate_day_ahead_observable_tables,
    day_ahead_signal_groups,
    evaluate_day_ahead_signals,
    evaluate_day_ahead_signals_by_year,
    summarize_day_ahead_feature_separation,
    summarize_day_ahead_signal_stability,
)
from src.analysis.opportunity_bridge import (
    build_daily_value_curve,
    summarize_annual_budget_vs_strict_daily_cap,
    summarize_interannual_stability,
    summarize_opportunity_day_signals,
    summarize_reallocated_same_throughput_vs_strict_daily_cap,
    summarize_throughput_budget_scenarios,
    summarize_value_outside_warranty_pace,
    summarize_within_day_concentration,
)
from src.analysis.revenue_breakdown import (
    build_interval_revenue_table,
    summarize_missing_top_days,
    summarize_top_spreads,
)
from src.data.cache import get_or_build_dataframe, make_cache_key
from src.data.netztransparenz import fetch_id_aep
from src.data.prices import fetch_day_ahead_prices
from src.models.degradation import DEFAULT_DEGRADATION_ASSUMPTIONS
from src.models.dispatch import (
    AGGRESSIVE_STRATEGY,
    CONSERVATIVE_STRATEGY,
    DispatchStrategy,
    run_dispatch_for_period,
    run_dispatch_with_intraday_overlay_for_period,
)

DEFAULT_YEAR = 2025
DEFAULT_DURATION_HOURS = 2.0
DEFAULT_RTE = 0.86
DEFAULT_BUDGETS = [365.0, 450.0, 550.0, 650.0]
DEFAULT_MISS_DAYS = [5, 10, 20, 50]
DEFAULT_INTERANNUAL_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_CYCLE_CAPS = [float(value) for value in np.arange(0.25, 4.01, 0.25)]
DEFAULT_REALLOCATION_CAPS = [1.0, 1.5, 2.0]
TOP_DAY_COUNT = 20


def _format_table(frame: pd.DataFrame, float_precision: int = 1) -> str:
    if frame.empty:
        return "(no data)"
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        220,
        "display.float_format",
        lambda value: f"{value:,.{float_precision}f}",
    ):
        return frame.to_string()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local owner-problem revenue report without Streamlit.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--duration-hours", type=float, default=DEFAULT_DURATION_HOURS)
    parser.add_argument("--rte", type=float, default=DEFAULT_RTE)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--miss-days", type=int, nargs="*", default=DEFAULT_MISS_DAYS)
    parser.add_argument("--budgets", type=float, nargs="*", default=DEFAULT_BUDGETS)
    parser.add_argument("--interannual-years", type=int, nargs="*", default=DEFAULT_INTERANNUAL_YEARS)
    parser.add_argument("--cycle-caps", type=float, nargs="*", default=DEFAULT_CYCLE_CAPS)
    parser.add_argument("--reallocation-caps", type=float, nargs="*", default=DEFAULT_REALLOCATION_CAPS)
    return parser.parse_args()


def _load_day_ahead_prices(years: list[int]) -> pd.DataFrame:
    return fetch_day_ahead_prices(
        start=f"{min(years)}-01-01",
        end=f"{max(years)}-12-31",
        force_refresh=False,
    )


def _load_intraday_prices(year: int) -> pd.DataFrame:
    try:
        return fetch_id_aep(start=f"{year}-01-01", end=f"{year}-12-31", force_refresh=False)
    except Exception:
        return pd.DataFrame(columns=["price_eur_mwh"])


def _dispatch_with_cache(
    year: int,
    strategy: DispatchStrategy,
    day_ahead_prices: pd.DataFrame,
    intraday_prices: pd.DataFrame,
    duration_hours: float,
    rte: float,
    market_key: str,
) -> pd.DataFrame:
    cache_key = make_cache_key(
        "dispatch",
        year=year,
        market=market_key,
        strategy=strategy.name,
        energy_mwh=round(duration_hours, 4),
        rte=round(rte, 4),
        power_mw=1.0,
        version=1,
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: run_dispatch_with_intraday_overlay_for_period(
            day_ahead_price_frame=day_ahead_prices,
            intraday_price_frame=intraday_prices,
            strategy=strategy,
            energy_mwh=duration_hours,
            rte=rte,
        )
        if not intraday_prices.empty
        else run_dispatch_for_period(
            price_frame=day_ahead_prices,
            strategy=strategy,
            energy_mwh=duration_hours,
            rte=rte,
        ),
        ttl_hours=None,
        force_refresh=False,
        metadata={"year": year, "strategy": strategy.name, "market": market_key},
    )


def _reconcile_interval_revenue(interval_revenue: pd.DataFrame, dispatch: pd.DataFrame) -> None:
    if interval_revenue.empty or dispatch.empty:
        return
    interval_daily = interval_revenue.groupby("date")["revenue_eur_per_mw"].sum().sort_index()
    dispatch_daily = dispatch["revenue_eur_per_mw"].sort_index()
    if not interval_daily.index.equals(dispatch_daily.index) or not np.allclose(
        interval_daily.to_numpy(),
        dispatch_daily.to_numpy(),
        atol=1e-4,
        rtol=1e-6,
    ):
        raise RuntimeError("Interval revenue decomposition does not reconcile to daily dispatch revenue.")


def _strategy_for_cycle_cap(cycle_cap: float, year: int, duration_hours: float) -> DispatchStrategy:
    cycle_token = str(cycle_cap).replace(".", "p")
    return DispatchStrategy(
        name=f"owner_bridge_{year}_{duration_hours:g}h_cap_{cycle_token}",
        label=f"{cycle_cap:.2f} cycles/day",
        max_cycles=float(cycle_cap),
        soc_min_frac=AGGRESSIVE_STRATEGY.soc_min_frac,
        soc_max_frac=AGGRESSIVE_STRATEGY.soc_max_frac,
        min_spread_eur_mwh=AGGRESSIVE_STRATEGY.min_spread_eur_mwh,
    )


def _build_report(args: argparse.Namespace) -> str:
    years = sorted({int(year) for year in args.interannual_years + [args.year]})
    day_ahead_all = _load_day_ahead_prices(years)
    intraday_by_year = {year: _load_intraday_prices(year) for year in years}

    conservative_dispatch_by_year: dict[int, pd.DataFrame] = {}
    for year in years:
        year_da = day_ahead_all[day_ahead_all.index.year == year]
        conservative_dispatch_by_year[year] = _dispatch_with_cache(
            year=year,
            strategy=CONSERVATIVE_STRATEGY,
            day_ahead_prices=year_da,
            intraday_prices=intraday_by_year[year],
            duration_hours=args.duration_hours,
            rte=args.rte,
            market_key="owner_bridge_conservative_v1",
        )

    selected_day_ahead = day_ahead_all[day_ahead_all.index.year == args.year]
    selected_intraday = intraday_by_year[args.year]
    selected_story_prices = selected_intraday if not selected_intraday.empty else selected_day_ahead
    selected_dispatch = conservative_dispatch_by_year[args.year]

    interval_revenue = build_interval_revenue_table(
        day_ahead_price_frame=selected_day_ahead,
        intraday_price_frame=selected_intraday,
        strategy=CONSERVATIVE_STRATEGY,
        energy_mwh=args.duration_hours,
        rte=args.rte,
    )
    _reconcile_interval_revenue(interval_revenue, selected_dispatch)

    concentration = compute_concentration_stats(selected_dispatch["revenue_eur_per_mw"])
    missed_days = summarize_missing_top_days(selected_dispatch["revenue_eur_per_mw"], top_day_counts=args.miss_days)
    top_spreads = summarize_top_spreads(selected_dispatch, selected_story_prices, top_n=args.top_n).rename_axis("date")
    top_day_dates = selected_dispatch["revenue_eur_per_mw"].sort_values(ascending=False).head(TOP_DAY_COUNT).index
    _, within_day_summary = summarize_within_day_concentration(interval_revenue, top_day_dates=top_day_dates)
    opportunity_signals = summarize_opportunity_day_signals(
        dispatch_frame=selected_dispatch,
        price_frame=selected_story_prices,
        top_n=TOP_DAY_COUNT,
    )
    day_ahead_observables_by_year = {
        year: build_day_ahead_observable_table(
            day_ahead_price_frame=day_ahead_all[day_ahead_all.index.year == year],
            outcome_dispatch=conservative_dispatch_by_year[year],
            top_day_counts=(10, 20),
        )
        for year in years
    }
    day_ahead_observables = build_day_ahead_observable_table(
        day_ahead_price_frame=selected_day_ahead,
        outcome_dispatch=selected_dispatch,
        top_day_counts=(10, 20),
    )
    day_ahead_feature_separation = summarize_day_ahead_feature_separation(day_ahead_observables)
    day_ahead_signal_screen = evaluate_day_ahead_signals(day_ahead_observables, top_day_counts=(10, 20))
    pooled_day_ahead_observables = concatenate_day_ahead_observable_tables(
        {year: day_ahead_observables_by_year[year] for year in sorted(args.interannual_years)}
    )
    signal_groups = day_ahead_signal_groups()
    pooled_top20_tight = build_day_ahead_watchlist_table(
        pooled_day_ahead_observables,
        target_count=20,
        signal_names=signal_groups["tight"],
    )
    pooled_top20_broad = build_day_ahead_watchlist_table(
        pooled_day_ahead_observables,
        target_count=20,
        signal_names=signal_groups["broad"],
    )
    pooled_top10_tight = build_day_ahead_watchlist_table(
        pooled_day_ahead_observables,
        target_count=10,
        signal_names=signal_groups["tight"],
    )
    pooled_top10_broad = build_day_ahead_watchlist_table(
        pooled_day_ahead_observables,
        target_count=10,
        signal_names=signal_groups["broad"],
    )
    yearly_top20_signal_eval = evaluate_day_ahead_signals_by_year(
        pooled_day_ahead_observables,
        target_count=20,
    )
    pooled_top20_stability_tight = summarize_day_ahead_signal_stability(
        yearly_top20_signal_eval,
        signal_names=signal_groups["tight"],
    )
    pooled_top20_stability_broad = summarize_day_ahead_signal_stability(
        yearly_top20_signal_eval,
        signal_names=signal_groups["broad"],
    )
    interannual_stability = summarize_interannual_stability(
        {year: conservative_dispatch_by_year[year] for year in sorted(args.interannual_years)}
    )

    aggressive_dispatch_by_cap: dict[float, pd.DataFrame] = {}
    for cycle_cap in sorted({float(value) for value in args.cycle_caps if float(value) > 0}):
        strategy = _strategy_for_cycle_cap(cycle_cap=cycle_cap, year=args.year, duration_hours=args.duration_hours)
        aggressive_dispatch_by_cap[cycle_cap] = _dispatch_with_cache(
            year=args.year,
            strategy=strategy,
            day_ahead_prices=selected_day_ahead,
            intraday_prices=selected_intraday,
            duration_hours=args.duration_hours,
            rte=args.rte,
            market_key="owner_bridge_aggressive_curve_v1",
        )

    daily_value_curve = build_daily_value_curve(aggressive_dispatch_by_cap)
    throughput_budgets, throughput_allocations = summarize_throughput_budget_scenarios(
        daily_value_curve=daily_value_curve,
        budgets=sorted({float(value) for value in args.budgets if float(value) > 0}),
        top_opportunity_day_count=TOP_DAY_COUNT,
    )
    warranty_summary, warranty_days = summarize_value_outside_warranty_pace(
        daily_value_curve=daily_value_curve,
        reference_warranty_fec_per_year=DEFAULT_DEGRADATION_ASSUMPTIONS.reference_warranty_fec_per_year,
        top_n_days=args.top_n,
    )
    reallocation_summary, reallocation_diagnostics = summarize_annual_budget_vs_strict_daily_cap(
        dispatch_by_cycle_cap=aggressive_dispatch_by_cap,
        daily_value_curve=daily_value_curve,
        daily_caps=args.reallocation_caps,
    )
    equal_throughput_summary, equal_throughput_diagnostics = summarize_reallocated_same_throughput_vs_strict_daily_cap(
        dispatch_by_cycle_cap=aggressive_dispatch_by_cap,
        daily_value_curve=daily_value_curve,
        daily_caps=args.reallocation_caps,
    )

    lines = [
        f"Owner-problem revenue report | year={args.year} | duration_hours={args.duration_hours:g} | rte={args.rte:.2f}",
        f"Conservative layer: soc {CONSERVATIVE_STRATEGY.soc_min_frac:.0%}-{CONSERVATIVE_STRATEGY.soc_max_frac:.0%}, "
        f"max {CONSERVATIVE_STRATEGY.max_cycles:.2f} cycles/day, hurdle {CONSERVATIVE_STRATEGY.min_spread_eur_mwh:.0f} €/MWh",
        f"Throughput bridge layer: soc {AGGRESSIVE_STRATEGY.soc_min_frac:.0%}-{AGGRESSIVE_STRATEGY.soc_max_frac:.0%}, "
        f"cycle-cap sweep over {min(args.cycle_caps):.2f}-{max(args.cycle_caps):.2f} cycles/day, "
        f"reference warranty pace {DEFAULT_DEGRADATION_ASSUMPTIONS.reference_warranty_fec_per_year:.0f} FEC/year",
        f"Modeled market layer: {'day-ahead + intraday overlay' if not selected_intraday.empty else 'day-ahead only'}",
        "",
        "Daily revenue distribution (conservative layer)",
        _format_table(
            pd.DataFrame(
                [
                    {
                        "annual_revenue_eur_per_mw": concentration["annual_revenue_eur_per_mw"],
                        "p50_daily_revenue_eur_per_mw": concentration["p50_daily_revenue_eur_per_mw"],
                        "p90_daily_revenue_eur_per_mw": concentration["p90_daily_revenue_eur_per_mw"],
                        "p95_daily_revenue_eur_per_mw": concentration["p95_daily_revenue_eur_per_mw"],
                        "p99_daily_revenue_eur_per_mw": concentration["p99_daily_revenue_eur_per_mw"],
                        "top_10pct_days_pct_of_revenue": 100 * concentration["top_10_days_pct_of_revenue"],
                        "top_20pct_days_pct_of_revenue": 100 * concentration["top_20_days_pct_of_revenue"],
                        "gini": concentration["gini"],
                    }
                ]
            ),
            float_precision=1,
        ),
        "",
        "Miss top N days (conservative layer)",
        _format_table(missed_days, float_precision=2),
        "",
        "Throughput budget capture (aggressive bridge layer)",
        _format_table(throughput_budgets, float_precision=2),
        "",
        "Value outside warranty pace (aggressive bridge layer)",
        _format_table(warranty_summary, float_precision=2),
        "",
        "Days where revenue sits above warranty pace",
        _format_table(warranty_days, float_precision=2),
        "",
        "Same annual cycle budget: strict daily cap vs annual allocator (aggressive bridge layer)",
        _format_table(
            reallocation_summary[
                [
                    "strict_daily_cap_revenue_eur_per_mw",
                    "annual_budget_revenue_eur_per_mw",
                    "uplift_eur_per_mw",
                    "uplift_pct_vs_strict",
                    "strict_share_of_full_flex_revenue_pct",
                    "annual_budget_share_of_full_flex_revenue_pct",
                ]
            ].rename(
                columns={
                    "strict_daily_cap_revenue_eur_per_mw": "strict_daily_cap_revenue_eur_per_mw",
                    "annual_budget_revenue_eur_per_mw": "annual_budget_revenue_eur_per_mw",
                    "uplift_eur_per_mw": "uplift_eur_per_mw",
                    "uplift_pct_vs_strict": "uplift_pct_vs_strict",
                    "strict_share_of_full_flex_revenue_pct": "strict_share_of_full_flex_revenue_pct",
                    "annual_budget_share_of_full_flex_revenue_pct": "annual_budget_share_of_full_flex_revenue_pct",
                }
            ),
            float_precision=2,
        ),
        "",
        "Same annual cycle budget diagnostics (top opportunity days defined by full-flex revenue)",
        _format_table(reallocation_diagnostics, float_precision=2),
        "",
        "Apples-to-apples reallocation: strict daily cap vs reallocated same realized FEC",
        _format_table(
            equal_throughput_summary[
                [
                    "strict_realized_fec",
                    "strict_daily_cap_revenue_eur_per_mw",
                    "reallocated_same_fec_revenue_eur_per_mw",
                    "uplift_eur_per_mw",
                    "uplift_pct_vs_strict",
                    "strict_share_of_full_flex_revenue_pct",
                    "reallocated_same_fec_share_of_full_flex_revenue_pct",
                ]
            ],
            float_precision=2,
        ),
        "",
        "Apples-to-apples reallocation diagnostics",
        _format_table(equal_throughput_diagnostics, float_precision=2),
        "",
        "Inter-annual stability (conservative layer)",
        _format_table(interannual_stability, float_precision=2),
        "",
        "Within-day concentration (conservative layer)",
        _format_table(within_day_summary, float_precision=2),
        "",
        "Opportunity-day signals (top 20 days vs all days, conservative layer)",
        _format_table(opportunity_signals, float_precision=2),
        "",
        "Day-ahead observable feature separation (DA-only inputs, outcome = actual revenue days)",
        _format_table(day_ahead_feature_separation, float_precision=2),
        "",
        "Day-ahead signal screen (DA-only inputs, outcome = actual revenue days)",
        _format_table(day_ahead_signal_screen, float_precision=2),
        "",
        "Pooled 2021-2025 DA watchlist | tight screens | top-20",
        _format_table(pooled_top20_tight, float_precision=2),
        "",
        "Pooled 2021-2025 DA watchlist | broad screens | top-20",
        _format_table(pooled_top20_broad, float_precision=2),
        "",
        "Pooled 2021-2025 DA watchlist | tight screens | top-10",
        _format_table(pooled_top10_tight, float_precision=2),
        "",
        "Pooled 2021-2025 DA watchlist | broad screens | top-10",
        _format_table(pooled_top10_broad, float_precision=2),
        "",
        "Pooled 2021-2025 DA stability by year | tight screens | top-20",
        _format_table(pooled_top20_stability_tight, float_precision=2),
        "",
        "Pooled 2021-2025 DA stability by year | broad screens | top-20",
        _format_table(pooled_top20_stability_broad, float_precision=2),
        "",
        "Top spreads (conservative layer)",
        _format_table(top_spreads, float_precision=2),
    ]

    if "365" in throughput_allocations and "full_flex" in throughput_allocations:
        warranty_capture_days = (
            throughput_allocations["full_flex"].daily.rename(
                columns={
                    "allocated_fec": "full_flex_fec",
                    "captured_revenue_eur_per_mw": "full_flex_revenue_eur_per_mw",
                }
            )
            .join(
                throughput_allocations["365"].daily.rename(
                    columns={
                        "allocated_fec": "budget_365_fec",
                        "captured_revenue_eur_per_mw": "budget_365_revenue_eur_per_mw",
                    }
                ),
                how="left",
            )
            .fillna(0.0)
        )
        warranty_capture_days["incremental_revenue_vs_365_eur_per_mw"] = (
            warranty_capture_days["full_flex_revenue_eur_per_mw"] - warranty_capture_days["budget_365_revenue_eur_per_mw"]
        )
        warranty_capture_days["incremental_fec_vs_365"] = (
            warranty_capture_days["full_flex_fec"] - warranty_capture_days["budget_365_fec"]
        )
        warranty_capture_days = warranty_capture_days.sort_values(
            "incremental_revenue_vs_365_eur_per_mw", ascending=False
        ).head(args.top_n)
        lines.extend(
            [
                "",
                "Top days that need throughput above 365 FEC/year",
                _format_table(warranty_capture_days, float_precision=2),
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    print(_build_report(args))


if __name__ == "__main__":
    main()
