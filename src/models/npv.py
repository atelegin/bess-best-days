from __future__ import annotations

import numpy as np
import pandas as pd


def compute_npv(
    year1_revenue: float,
    annual_revenue_decline: float,
    capacity_trajectory: np.ndarray,
    discount_rate: float,
    years: int,
) -> float:
    npv = 0.0
    for year in range(years):
        market_factor = (1 + annual_revenue_decline) ** year
        capacity_factor = float(capacity_trajectory[year])
        revenue = year1_revenue * market_factor * capacity_factor
        npv += revenue / (1 + discount_rate) ** year
    return float(npv)


def build_npv_curve(
    conservative_year1_revenue: float,
    aggressive_year1_revenue: float,
    conservative_capacity: np.ndarray,
    aggressive_capacity: np.ndarray,
    discount_rate: float,
    years: int,
    decline_grid: np.ndarray,
) -> pd.DataFrame:
    records = []
    for decline_rate in decline_grid:
        records.append(
            {
                "decline_rate": float(decline_rate),
                "conservative_npv": compute_npv(
                    year1_revenue=conservative_year1_revenue,
                    annual_revenue_decline=-decline_rate,
                    capacity_trajectory=conservative_capacity,
                    discount_rate=discount_rate,
                    years=years,
                ),
                "aggressive_npv": compute_npv(
                    year1_revenue=aggressive_year1_revenue,
                    annual_revenue_decline=-decline_rate,
                    capacity_trajectory=aggressive_capacity,
                    discount_rate=discount_rate,
                    years=years,
                ),
            }
        )
    return pd.DataFrame(records)


def build_npv_heatmap(
    conservative_year1_revenue: float,
    aggressive_year1_revenue: float,
    conservative_capacity: np.ndarray,
    aggressive_capacity: np.ndarray,
    years: int,
    decline_grid: np.ndarray,
    discount_grid: np.ndarray,
) -> pd.DataFrame:
    records = []
    for decline_rate in decline_grid:
        for discount_rate in discount_grid:
            conservative = compute_npv(
                year1_revenue=conservative_year1_revenue,
                annual_revenue_decline=-decline_rate,
                capacity_trajectory=conservative_capacity,
                discount_rate=discount_rate,
                years=years,
            )
            aggressive = compute_npv(
                year1_revenue=aggressive_year1_revenue,
                annual_revenue_decline=-decline_rate,
                capacity_trajectory=aggressive_capacity,
                discount_rate=discount_rate,
                years=years,
            )
            records.append(
                {
                    "decline_rate": float(decline_rate),
                    "discount_rate": float(discount_rate),
                    "npv_delta_eur_per_mw": aggressive - conservative,
                }
            )
    return pd.DataFrame(records)

