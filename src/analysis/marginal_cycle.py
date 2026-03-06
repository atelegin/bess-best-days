from __future__ import annotations

import pandas as pd


def compute_marginal_cycle_value(
    conservative_dispatch: pd.DataFrame,
    aggressive_dispatch: pd.DataFrame,
) -> pd.DataFrame:
    combined = conservative_dispatch[["revenue_eur_per_mw"]].rename(columns={"revenue_eur_per_mw": "conservative_revenue_eur_per_mw"})
    combined = combined.join(
        aggressive_dispatch[["revenue_eur_per_mw"]].rename(columns={"revenue_eur_per_mw": "aggressive_revenue_eur_per_mw"}),
        how="inner",
    )
    combined["marginal_value_eur_per_mw"] = (
        combined["aggressive_revenue_eur_per_mw"] - combined["conservative_revenue_eur_per_mw"]
    )
    combined = combined.sort_values("conservative_revenue_eur_per_mw", ascending=False)
    combined["rank"] = range(1, len(combined) + 1)
    combined["rank_pct"] = combined["rank"] / max(len(combined), 1)
    return combined
