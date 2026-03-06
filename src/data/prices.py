from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import pandas as pd
import requests

from src.data.cache import get_or_build_dataframe, make_cache_key

ENERGY_CHARTS_PRICE_URL = "https://api.energy-charts.info/price"
DEFAULT_TIMEZONE = "Europe/Berlin"


@dataclass(frozen=True)
class PriceRequest:
    start: str
    end: str
    bidding_zone: str = "DE-LU"


def _chunk_dates(start: str, end: str, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    periods = max(1, ceil(((end_ts - start_ts).days + 1) / chunk_days))
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_ts
    for _ in range(periods):
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end_ts)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
        if current > end_ts:
            break
    return chunks


def _fetch_prices_chunk(request: PriceRequest) -> pd.DataFrame:
    response = requests.get(
        ENERGY_CHARTS_PRICE_URL,
        params={"bzn": request.bidding_zone, "start": request.start, "end": request.end},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True).tz_convert(DEFAULT_TIMEZONE),
            "price_eur_mwh": payload["price"],
        }
    )
    return frame.set_index("timestamp").sort_index()


def _build_price_frame(start: str, end: str, bidding_zone: str) -> pd.DataFrame:
    chunks = _chunk_dates(start, end, chunk_days=365)
    frames = [
        _fetch_prices_chunk(PriceRequest(start=chunk_start.date().isoformat(), end=chunk_end.date().isoformat(), bidding_zone=bidding_zone))
        for chunk_start, chunk_end in chunks
    ]
    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame["price_eur_mwh"] = pd.to_numeric(frame["price_eur_mwh"], errors="coerce")
    return frame.dropna(subset=["price_eur_mwh"])


def fetch_day_ahead_prices(
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    bidding_zone: str = "DE-LU",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_key = make_cache_key(
        "day_ahead_prices",
        start=start,
        end=end,
        bidding_zone=bidding_zone,
        source="energy_charts_v1",
    )
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: _build_price_frame(start=start, end=end, bidding_zone=bidding_zone),
        ttl_hours=24 * 7,
        force_refresh=force_refresh,
        metadata={"source": ENERGY_CHARTS_PRICE_URL},
    )


def compute_daily_price_metrics(price_frame: pd.DataFrame) -> pd.DataFrame:
    daily = []
    prices = price_frame.sort_index()
    for day, group in prices.groupby(prices.index.normalize()):
        series = group["price_eur_mwh"].astype(float)
        hourly = series.resample("1h").mean()
        midday = series[(series.index.hour >= 10) & (series.index.hour <= 15)]
        evening = series[(series.index.hour >= 17) & (series.index.hour <= 21)]
        daily.append(
            {
                "date": pd.Timestamp(day).tz_localize(None),
                "min_price_eur_mwh": float(series.min()),
                "max_price_eur_mwh": float(series.max()),
                "spread_eur_mwh": float(series.max() - series.min()),
                "tb2_spread_eur_mwh": float(hourly.nlargest(min(2, len(hourly))).mean() - hourly.nsmallest(min(2, len(hourly))).mean()),
                "mean_price_eur_mwh": float(series.mean()),
                "negative_intervals": int((series < 0).sum()),
                "midday_min_price_eur_mwh": float(midday.min()) if not midday.empty else float("nan"),
                "evening_peak_price_eur_mwh": float(evening.max()) if not evening.empty else float("nan"),
            }
        )
    return pd.DataFrame(daily).set_index("date").sort_index()


def compute_monthly_tb2_spread(price_frame: pd.DataFrame) -> pd.DataFrame:
    daily = compute_daily_price_metrics(price_frame)
    monthly = (
        daily.assign(month=lambda frame: frame.index.to_period("M").to_timestamp())
        .groupby("month")
        .agg(tb2_spread_eur_mwh=("tb2_spread_eur_mwh", "mean"))
    )
    return monthly
