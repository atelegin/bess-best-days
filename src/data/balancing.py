from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import warnings

import pandas as pd
import requests

from src.data.cache import get_or_build_dataframe, make_cache_key

REGELLEISTUNG_URL = "https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/resultsoverview"
BESS_CAPACITY_GW = {
    "2021-01": 0.20,
    "2021-06": 0.25,
    "2022-06": 0.50,
    "2023-06": 1.00,
    "2024-01": 1.20,
    "2024-06": 1.40,
    "2025-01": 1.70,
    "2025-06": 2.00,
    "2025-12": 2.40,
    "2026-12": 3.40,
}


@dataclass(frozen=True)
class BalancingSnapshot:
    market: str
    month: pd.Timestamp
    capacity_price_eur_mw: float
    offered_capacity_mw: float
    sampled_from: pd.Timestamp


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _mean_numeric(frame: pd.DataFrame, candidates: list[str]) -> float:
    column = _first_existing_column(frame, candidates)
    if column is None:
        return float("nan")
    series = pd.to_numeric(frame[column], errors="coerce")
    return float(series.mean()) if not series.dropna().empty else float("nan")


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), pd.Timestamp(end).to_period("M").to_timestamp(), freq="MS"))


def _download_market_sheet(market: str, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    response = requests.get(
        REGELLEISTUNG_URL,
        params={
            "date": snapshot_date.date().isoformat(),
            "exportFormat": "xlsx",
            "market": "CAPACITY",
            "productTypes": market,
        },
        timeout=60,
    )
    response.raise_for_status()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style", category=UserWarning)
        return pd.read_excel(BytesIO(response.content))


def _parse_fcr_snapshot(frame: pd.DataFrame, month: pd.Timestamp, sampled_from: pd.Timestamp) -> BalancingSnapshot:
    price = _mean_numeric(
        frame,
        [
            "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",
            "DE_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",
        ],
    )
    demand_col = _first_existing_column(frame, ["GERMANY_DEMAND_[MW]", "DE_DEMAND_[MW]"])
    surplus_col = _first_existing_column(
        frame,
        [
            "GERMANY_DEFICIT(-)_SURPLUS(+)_[MW]",
            "GERMANY_IMPORT(-)_EXPORT(+)_[MW]",
            "DE_DEFICIT(-)_SURPLUS(+)_[MW]",
            "DE_IMPORT(-)_EXPORT(+)_[MW]",
        ],
    )
    demand = pd.to_numeric(frame[demand_col], errors="coerce") if demand_col else pd.Series(dtype=float)
    surplus = pd.to_numeric(frame[surplus_col], errors="coerce") if surplus_col else pd.Series(dtype=float)
    offered = float((demand + surplus.fillna(0)).mean()) if not demand.empty else float("nan")
    return BalancingSnapshot(
        market="FCR",
        month=month,
        capacity_price_eur_mw=float(price),
        offered_capacity_mw=offered,
        sampled_from=sampled_from,
    )


def _parse_afrr_snapshot(frame: pd.DataFrame, month: pd.Timestamp, sampled_from: pd.Timestamp) -> BalancingSnapshot:
    price = _mean_numeric(
        frame,
        [
            "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]",
            "GERMANY_AVERAGE_CAPACITY_PRICE_[EUR/MW]",
        ],
    )
    offered = _mean_numeric(frame, ["GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]"])
    return BalancingSnapshot(
        market="aFRR",
        month=month,
        capacity_price_eur_mw=float(price),
        offered_capacity_mw=float(offered),
        sampled_from=sampled_from,
    )


def _fetch_month_snapshot(month: pd.Timestamp) -> list[BalancingSnapshot]:
    fcr_frame = _download_market_sheet("FCR", month)
    afrr_frame = _download_market_sheet("aFRR", month)
    return [
        _parse_fcr_snapshot(fcr_frame, month=month, sampled_from=month),
        _parse_afrr_snapshot(afrr_frame, month=month, sampled_from=month),
    ]


def _build_balancing_frame(start: str, end: str) -> pd.DataFrame:
    months = _month_starts(start, end)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for snapshots in executor.map(_fetch_month_snapshot, months):
            for snapshot in snapshots:
                records.append(
                    {
                        "month": snapshot.month,
                        "market": snapshot.market,
                        "capacity_price_eur_mw": snapshot.capacity_price_eur_mw,
                        "offered_capacity_mw": snapshot.offered_capacity_mw,
                        "sampled_from": snapshot.sampled_from,
                    }
                )
    return pd.DataFrame(records).sort_values(["market", "month"]).set_index("month")


def fetch_balancing_capacity_prices(
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_key = make_cache_key("balancing_capacity", start=start, end=end, source="regelleistung_monthly_snapshot_v1")
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: _build_balancing_frame(start=start, end=end),
        ttl_hours=24 * 30,
        force_refresh=force_refresh,
        metadata={"source": REGELLEISTUNG_URL, "method": "monthly_first_day_snapshot"},
    )


def build_bess_capacity_series(start: str = "2021-01-01", end: str = "2026-12-31") -> pd.DataFrame:
    anchor = pd.Series(BESS_CAPACITY_GW, dtype=float)
    anchor.index = pd.to_datetime(anchor.index)
    monthly_index = pd.date_range(start=start, end=end, freq="MS")
    monthly = anchor.reindex(anchor.index.union(monthly_index)).sort_index().interpolate(method="time").reindex(monthly_index)
    return pd.DataFrame({"bess_capacity_gw": monthly})
