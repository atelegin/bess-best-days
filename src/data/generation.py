from __future__ import annotations

from math import ceil

import pandas as pd
import requests

from src.data.cache import get_or_build_dataframe, make_cache_key

ENERGY_CHARTS_GENERATION_URL = "https://api.energy-charts.info/public_power"
SERIES_MAP = {
    "Solar": "solar_mw",
    "Wind onshore": "wind_onshore_mw",
    "Wind offshore": "wind_offshore_mw",
    "Load": "load_mw",
    "Residual load": "residual_load_mw",
}


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


def _fetch_generation_chunk(start: str, end: str) -> pd.DataFrame:
    response = requests.get(
        ENERGY_CHARTS_GENERATION_URL,
        params={"country": "de", "start": start, "end": end},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Berlin")
    frame = pd.DataFrame(index=index)
    for series in payload["production_types"]:
        target_column = SERIES_MAP.get(series["name"])
        if target_column:
            frame[target_column] = pd.to_numeric(series["data"], errors="coerce")
    return frame.sort_index()


def _build_generation_frame(start: str, end: str) -> pd.DataFrame:
    chunks = _chunk_dates(start, end, chunk_days=180)
    frames = [
        _fetch_generation_chunk(chunk_start.date().isoformat(), chunk_end.date().isoformat())
        for chunk_start, chunk_end in chunks
    ]
    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    return frame.dropna(how="all")


def fetch_generation_data(
    start: str = "2021-01-01",
    end: str = "2025-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_key = make_cache_key("generation", start=start, end=end, source="energy_charts_public_power_v1")
    return get_or_build_dataframe(
        cache_key=cache_key,
        builder=lambda: _build_generation_frame(start=start, end=end),
        ttl_hours=24 * 7,
        force_refresh=force_refresh,
        metadata={"source": ENERGY_CHARTS_GENERATION_URL},
    )


def _infer_step_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 3600
    return float(deltas.mode().iloc[0])


def compute_daily_generation_metrics(generation_frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    frame = generation_frame.sort_index()
    for day, group in frame.groupby(frame.index.normalize()):
        dt_hours = _infer_step_hours(group.index)
        records.append(
            {
                "date": pd.Timestamp(day).tz_localize(None),
                "solar_generation_gwh": float(group["solar_mw"].sum() * dt_hours / 1000),
                "wind_onshore_generation_gwh": float(group["wind_onshore_mw"].sum() * dt_hours / 1000),
                "wind_offshore_generation_gwh": float(group["wind_offshore_mw"].sum() * dt_hours / 1000),
                "wind_generation_gwh": float((group["wind_onshore_mw"] + group["wind_offshore_mw"]).sum() * dt_hours / 1000),
                "load_gwh": float(group["load_mw"].sum() * dt_hours / 1000),
                "residual_load_range_mw": float(group["residual_load_mw"].max() - group["residual_load_mw"].min()),
                "residual_load_min_mw": float(group["residual_load_mw"].min()),
                "residual_load_max_mw": float(group["residual_load_mw"].max()),
                "solar_peak_mw": float(group["solar_mw"].max()),
                "wind_peak_mw": float((group["wind_onshore_mw"] + group["wind_offshore_mw"]).max()),
            }
        )
    return pd.DataFrame(records).set_index("date").sort_index()
