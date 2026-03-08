# The Cost of Missing the Best Days

## Revenue Concentration and Optimal Cycling Strategy in German BESS

---

## 1. Research Question

German BESS revenue is not evenly distributed across the year. A small number of extreme-spread days likely generates a disproportionate share of annual income. This has direct consequences for how operators should manage their batteries.

This project investigates three linked questions:

1. **How concentrated is BESS revenue?** What share of annual DA arbitrage income comes from the top 10%, 20%, 50% of days?

2. **What makes the best days?** Are they random, or driven by identifiable fundamentals (solar surplus, Dunkelflaute, weather events)?

3. **Does concentration justify aggressive cycling?** If most revenue comes from tail days — and those tail days reward multiple cycles — then the extra degradation from aggressive cycling is paid for on boring days but earned back on extreme days. And if future revenue per MW is declining as the fleet grows, front-loading cycles has higher NPV than preserving the battery.

The connection: revenue concentration determines **when** extra cycles are valuable. Cannibalization determines **whether** to spend capacity now or save it. Together they answer: what is the NPV-optimal cycling strategy for a German BESS today?

---

## 2. Who This Is For

- **Asset owners** deciding between conservative O&M and aggressive merchant strategies
- **Investors** underwriting BESS projects who need to understand revenue variance, not just averages
- **Optimizers** whose real job is not to trade well on average days, but to never miss the tail
- **Anyone modeling BESS revenue** who assumes smooth annual cashflows — this research challenges that assumption

---

## 3. Format

Streamlit interactive research report. Guided narrative with live data and configurable parameters. Not a generic dashboard — an analytical tool that walks through an argument.

---

## 4. Data Sources

### 4.1 Day-Ahead Prices (primary revenue layer)

**Source:** Energy-Charts API (Fraunhofer ISE)
- Endpoint: `https://api.energy-charts.info/price?bzn=DE-LU&start={date}&end={date}`
- Auth: None
- Format: JSON → pandas DataFrame
- Granularity: hourly (pre Oct 2025), 15-min (post Oct 2025)
- Range needed: 2020-01-01 to 2025-12-31

```python
import requests, pandas as pd

def fetch_da_prices(start: str, end: str) -> pd.DataFrame:
    r = requests.get("https://api.energy-charts.info/price",
                     params={"bzn": "DE-LU", "start": start, "end": end})
    data = r.json()
    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
        "price_eur_mwh": data["price"]
    }).set_index("timestamp").tz_convert("Europe/Berlin")
```

**Fallback:** SMARD.de API (`https://www.smard.de/app/chart_data/4169/DE-LU/...`), no auth, same data.

### 4.2 Imbalance Price / reBAP (real-time volatility layer)

**Source:** Netztransparenz.de API
- Endpoint: `https://ds.netztransparenz.de/api/v1/data/reBAP`
- Auth: OAuth 2.0 (free registration at `https://extranet.netztransparenz.de`)
- Granularity: 15-min
- Purpose: amplifier layer showing how concentration intensifies in real-time vs DA

**Risk:** OAuth registration may take time. If unavailable by build time, proceed with DA-only and note as limitation. reBAP strengthens the concentration thesis but is not required for it.

### 4.3 Generation Mix (driver analysis)

**Source:** Energy-Charts API
- Endpoint: `https://api.energy-charts.info/public_power?country=de&start={date}&end={date}`
- Auth: None
- Needed: solar, wind_onshore, wind_offshore, load → compute residual load
- Same date range as prices

### 4.4 Balancing Capacity Prices (cannibalization evidence)

Three indicators, each showing a different saturation timeline:

**FCR capacity prices:**
- Source: regelleistung.net
- Download: `https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/resultsoverview?date={date}&exportFormat=xlsx&market=CAPACITY&productTypes=FCR`
- Already saturated: 800+ MW prequalified for ~564 MW market
- Shows what happens first when fleet grows

**aFRR capacity prices:**
- Same source, `productTypes=aFRR`
- Currently saturating: ~550 MW prequalified, growing fast
- Shows what's happening now

**DA TB2 spreads:**
- Computed from DA prices (Section 4.1)
- Still growing due to solar penetration — but will compress with more BESS competing for same spreads
- Shows what comes next

### 4.5 Installed BESS Capacity

Hardcoded lookup from Modo Energy Battery Buildout Reports and MaStR:

```python
BESS_CAPACITY_GW = {
    "2020-06": 0.15, "2021-06": 0.25, "2022-06": 0.50,
    "2023-06": 1.00, "2024-01": 1.20, "2024-06": 1.40,
    "2025-01": 1.70, "2025-06": 2.00, "2025-12": 2.40,
    "2026-12": 3.40,  # projected
}
```

---

## 5. App Structure

### Sidebar — Battery Configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Duration (hours) | 2 | 1, 2, 4 | Dropdown. Determines energy capacity relative to 1 MW |
| Round-trip efficiency | 86% | 75-95% | Slider, AC-to-AC |
| Year to analyze | 2024 | 2020-2025 | Dropdown |
| Revenue decline rate | -5%/year | -20% to 0% | Slider, for NPV section |
| Discount rate | 8% | 4-12% | Slider |
| Battery project lifetime | 15 years | 10-20 | Slider |

All results normalized to €/MW/day or €/MW/year. Power fixed at 1 MW for simplicity.

### Section 1 — Revenue Concentration

Goal: establish the core fact. How unevenly distributed is BESS revenue?

**Dispatch model:** for each day in the selected year, run perfect-foresight LP optimization on DA hourly prices. Battery charges at low prices, discharges at high prices, subject to SoC and efficiency constraints. Conservative parameters (1 cycle/day, 20-80% SoC) as the baseline. This gives a daily revenue time series.

**Chart 1a — Pareto Curve**
- X: days ranked best-to-worst (cumulative %)
- Y: cumulative % of annual revenue
- 45° reference line (uniform distribution)
- Multiple years overlaid (faded) to show whether concentration is increasing

**Chart 1b — Calendar Heatmap**
- 365 cells colored by daily revenue intensity
- Reveals clustering: are best days random or seasonal?

**Chart 1c — Revenue Distribution**
- Histogram of daily revenues with p50, p90, p95, p99 annotated
- Box: "Top 20 days earned €X/MW — equivalent to Y months of median-day revenue"

**Headline metric:** "In {year}, the top N% of days generated M% of annual revenue."

### Section 2 — What Drives the Best Days?

Goal: show that tail days are not random — they're driven by identifiable market fundamentals.

**Chart 2a — Top-20 Days Table**
- Date, revenue (€/MW), max price, min price, daily spread, solar generation (GWh), wind generation (GWh), residual load range
- Sortable columns

**Chart 2b — Revenue vs Fundamentals**
- Scatter: daily revenue (y) vs residual load range or solar generation (x)
- Color by season
- Show correlation: extreme days = extreme solar swings or Dunkelflaute

**Chart 2c — Price Shape: Best Days vs Average**
- Two overlaid 24h price profiles: mean of top-10 days vs mean of all days
- Shows that best days have both deeper troughs AND higher peaks — not just one extreme

**Insight:** "X% of top-20 days involved negative midday prices below €Y/MWh combined with evening peaks above €Z/MWh"

### Section 3 — The Value of the Second Cycle

Goal: bridge between concentration and degradation strategy. Show that on tail days, a second cycle is extremely valuable — but on average days, it's barely worth the wear.

**Two strategies defined by dispatch constraints:**

| | Conservative | Aggressive |
|---|---|---|
| Max cycles/day | 1 | 2 |
| SoC operating range | 20-80% (60% usable) | 5-95% (90% usable) |
| Min spread to trade | €20/MWh | €5/MWh |

Both use perfect-foresight LP dispatch. Difference is in constraints.

**Chart 3a — Marginal Value of 2nd Cycle by Day Rank**
- X: days ranked by conservative-strategy revenue (best → worst)
- Y: extra revenue from aggressive strategy (€/MW)
- Key visual: tall bars on the left (tail days: 2nd cycle worth €100-500/MW), tiny bars on the right (average days: €5-20/MW)
- This is the bridge chart: concentration means aggressive cycling is justified by tail days

**Chart 3b — Annual Revenue: Conservative vs Aggressive**
- Paired bars for each year (2020-2025)
- Label the absolute and % difference

**Chart 3c — Cumulative Revenue Over Lifetime (no decline)**
- Two curves: conservative vs aggressive, with degradation applied
- Conservative: ~2%/year capacity fade → revenue declines slowly
- Aggressive: ~4-5%/year capacity fade → more revenue early, less later
- Shows crossover in cumulative terms even without revenue decline

**Degradation model:**

```
annual_degradation = calendar_aging + cycle_aging

calendar_aging = 0.8% per year

cycle_aging = cycles_per_year × degradation_per_cycle(DoD)

degradation_per_cycle(DoD) = base_rate × (DoD / 0.8) ^ k
    where base_rate ≈ 0.003% per full equivalent cycle at 80% DoD
    k ≈ 1.5 (Schmalstieg/Xu approximation)

Conservative: 365 cycles/yr × 60% DoD → ~1.8% cycle + 0.8% calendar ≈ 2.6%/yr
Aggressive: 600 cycles/yr × 90% DoD → ~3.8% cycle + 0.8% calendar ≈ 4.6%/yr
```

These numbers are derived from inputs, not hardcoded. Changing DoD or cycle count in the model changes the degradation rate.

### Section 4 — The Cannibalization Discount

Goal: answer the final question. Given that revenue per MW is declining as the fleet grows, does it make more sense to spend capacity now?

**Chart 4a — Cannibalization Evidence**
Three small panels side by side:
- FCR capacity price (€/MW/month) vs time, with BESS installed GW overlay → "already happened"
- aFRR capacity price vs time, with BESS prequalified MW overlay → "happening now"
- DA TB2 spread (monthly avg) vs time → "still growing, but for how long?"

Each tells a different chapter of the same story. FCR is the leading indicator for what will happen to aFRR and then to DA spreads.

**Chart 4b — NPV Crossover**
- X: annual revenue decline rate (0% to -20%)
- Y: NPV over battery lifetime (€/MW)
- Two lines: conservative vs aggressive
- Crossover point highlighted and labeled: "At >{X}% annual decline, aggressive cycling has higher NPV"

**Chart 4c — Sensitivity Heatmap**
- X: revenue decline rate (0% to -20%)
- Y: discount rate (4% to 12%)
- Color: NPV(aggressive) − NPV(conservative)
- Green = aggressive wins, red = conservative wins
- Shows robustness: under what combinations of assumptions does the conclusion hold?

**Insight:** "If annual revenues decline by >{X}% — consistent with the trajectory implied by {Y} GW of BESS entering the market by 2028 — aggressive cycling delivers {Z}% higher NPV over {N} years."

### Section 5 — Key Findings

3-4 numbered findings with specific numbers. No fluff. Each one derived from the analysis above.

Example structure:
1. Revenue concentration: "In 2024, the top 15% of days generated X% of annual DA arbitrage revenue for a 2h battery."
2. Tail drivers: "Y% of top-revenue days involved midday prices below -€Z/MWh."
3. Marginal cycle value: "The 2nd daily cycle was worth €A/MW on top-20 days vs €B/MW on median days — a C:1 ratio."
4. Optimal strategy: "Under a D% annual revenue decline scenario, aggressive cycling (2 cycles/day, 5-95% SoC) delivers E% higher NPV than conservative operation over F years."

---

## 6. Technical Architecture

### Directory Structure

```
bess-best-days/
├── README.md
├── requirements.txt
├── app.py                        # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prices.py             # Energy-Charts DA price fetcher
│   │   ├── generation.py         # Solar/wind/load fetcher
│   │   ├── balancing.py          # FCR/aFRR prices from regelleistung.net
│   │   ├── rebap.py              # reBAP imbalance prices (if OAuth available)
│   │   └── cache.py              # Parquet read/write with staleness check
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dispatch.py           # LP dispatch optimizer
│   │   ├── degradation.py        # Cycle + calendar degradation
│   │   └── npv.py                # Discounted cashflow model
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── concentration.py      # Pareto, Gini, percentile stats
│   │   ├── drivers.py            # Correlation of revenue with fundamentals
│   │   └── marginal_cycle.py     # Value of 2nd cycle by day rank
│   └── charts/
│       ├── __init__.py
│       ├── pareto.py             # Pareto curve + year overlay
│       ├── calendar.py           # Calendar heatmap
│       ├── scatter.py            # Revenue vs fundamentals
│       ├── strategy.py           # Conservative vs aggressive comparisons
│       └── npv.py                # Crossover + sensitivity heatmap
├── data/
│   └── cache/                    # .gitignore'd parquet files
├── notebooks/
│   └── exploration.ipynb         # EDA showing analytical process
└── resume.pdf
```

### Dependencies

```
# requirements.txt
streamlit>=1.30
plotly>=5.18
pandas>=2.1
numpy>=1.26
scipy>=1.12
requests>=2.31
openpyxl>=3.1              # for regelleistung.net XLSX files
```

### Dispatch LP (src/models/dispatch.py)

```python
from scipy.optimize import linprog
import numpy as np

def optimize_day(
    prices: np.ndarray,        # hourly prices, length 24
    energy_mwh: float,         # e.g. 2.0 for a 1MW/2h system
    rte: float,                # round-trip efficiency, e.g. 0.86
    soc_min_frac: float,       # e.g. 0.20 or 0.05
    soc_max_frac: float,       # e.g. 0.80 or 0.95
    max_cycles: float,         # e.g. 1.0 or 2.0
    power_mw: float = 1.0,
) -> dict:
    """
    Perfect-foresight LP for a single day (24 hourly periods).

    Decision variables (48 total):
        charge[0..23]:    MW charged in each hour (>= 0)
        discharge[0..23]: MW discharged in each hour (>= 0)

    Objective (maximize → negate for linprog minimize):
        sum over t: discharge[t] * price[t] * sqrt(rte)
                   - charge[t] * price[t] / sqrt(rte)

    Constraints:
        Power limits:
            charge[t] <= power_mw
            discharge[t] <= power_mw

        SoC bounds (cumulative energy balance):
            soc[t] = soc_init + sum(charge[0..t] * sqrt(rte) - discharge[0..t] / sqrt(rte))
            soc_min <= soc[t] <= soc_max
            soc_init = (soc_min + soc_max) / 2

        Cycle limit:
            sum(discharge[t]) <= max_cycles * energy_mwh

        End-of-day SoC (soft — return to within 10% of start):
            |soc[23] - soc_init| <= 0.1 * energy_mwh

    Returns:
        {
            "revenue": float,           # €/MW for this day
            "charge": np.ndarray,       # 24h charge profile
            "discharge": np.ndarray,    # 24h discharge profile
            "soc": np.ndarray,          # 24h SoC profile
            "cycles": float,            # full equivalent cycles used
            "avg_dod": float,           # average depth of discharge
        }
    """
    pass  # Implementation ~60 lines with scipy.optimize.linprog
```

### Degradation Model (src/models/degradation.py)

```python
def compute_annual_degradation(
    cycles_per_year: float,
    avg_dod: float,
    base_cycle_degradation: float = 0.003,  # % per FEC at 80% DoD
    dod_exponent: float = 1.5,              # Schmalstieg/Xu
    calendar_aging: float = 0.008,          # 0.8% per year
) -> float:
    """
    Annual capacity fade as fraction.

    cycle_degradation = cycles_per_year * base_rate * (avg_dod / 0.8) ^ dod_exponent
    total = calendar_aging + cycle_degradation

    Returns: e.g. 0.026 for 2.6% annual degradation
    """
    dod_factor = (avg_dod / 0.8) ** dod_exponent
    cycle_deg = cycles_per_year * (base_cycle_degradation / 100) * dod_factor
    return calendar_aging + cycle_deg


def capacity_trajectory(
    annual_degradation: float,
    years: int,
) -> np.ndarray:
    """
    Remaining capacity fraction for each year.
    Returns array: [1.0, 1-d, (1-d)^2, ..., (1-d)^years]
    """
    return np.array([(1 - annual_degradation) ** y for y in range(years + 1)])
```

### NPV Model (src/models/npv.py)

```python
def compute_npv(
    year1_revenue: float,            # €/MW/year from dispatch model
    annual_revenue_decline: float,   # e.g. -0.05 for -5%/year
    capacity_trajectory: np.ndarray, # from degradation model
    discount_rate: float,            # e.g. 0.08
    years: int,
) -> float:
    """
    NPV = sum over y: revenue_y / (1 + r)^y

    where revenue_y = year1_revenue
                    * (1 + annual_revenue_decline)^y   # market decline
                    * capacity_trajectory[y]            # degradation
    """
    npv = 0.0
    for y in range(years):
        market_factor = (1 + annual_revenue_decline) ** y
        capacity_factor = capacity_trajectory[y]
        revenue_y = year1_revenue * market_factor * capacity_factor
        npv += revenue_y / (1 + discount_rate) ** y
    return npv
```

---

## 7. Build Plan

### Phase 1 — Data (2 hours)

- Set up repo, venv, requirements
- `prices.py`: fetch 2020-2025 DA hourly prices from Energy-Charts, cache as parquet
- `generation.py`: fetch solar, wind, load for same period
- `balancing.py`: download FCR + aFRR monthly capacity prices from regelleistung.net
- Validate: check for gaps, timezone handling (CET → CEST transitions), NaN values
- Compute TB1/TB2 daily spreads as derived columns

### Phase 2 — Models + Analysis (2.5 hours)

- `dispatch.py`: LP optimizer, test on single day, validate against hand calculation
- Run dispatch for all days × 2 strategies (conservative + aggressive) × selected years
- Cache results (this is the expensive step — ~2200 LP solves per year per strategy)
- `concentration.py`: Pareto stats, Gini coefficient, percentile breakdowns
- `drivers.py`: merge revenue with generation data, compute correlations
- `marginal_cycle.py`: difference between aggressive and conservative revenue per day
- `degradation.py` + `npv.py`: implement and test

### Phase 3 — Streamlit App (2.5 hours)

- `app.py`: layout with sidebar + 5 sections
- Section 1: Pareto curve, calendar heatmap, distribution histogram
- Section 2: top days table, scatter plot, price shape overlay
- Section 3: marginal cycle value chart, annual comparison bars, cumulative curves
- Section 4: cannibalization panels, NPV crossover, sensitivity heatmap
- Section 5: key findings with computed numbers
- Wire sidebar parameters to trigger recalculation

### Phase 4 — Polish (1 hour)

- README.md: methodology, findings, how to run, assumptions, limitations, extensions
- Code cleanup: docstrings, type hints, consistent naming
- `notebooks/exploration.ipynb`: show EDA process (1-2 key explorations)
- Test: fresh clone → pip install → streamlit run → works
- Add resume.pdf, push, grant access to alexmarkdone

---

## 8. README Structure

```markdown
# The Cost of Missing the Best Days
## Revenue Concentration and Optimal Cycling Strategy in German BESS

### The Question
[Problem statement — 3 sentences]

### Key Findings
1. [Revenue concentration — specific number]
2. [Tail day drivers — specific pattern]
3. [Marginal cycle value — specific ratio]
4. [NPV optimal strategy — specific threshold]

### Methodology
- Day-ahead prices from Energy-Charts API (2020-2025)
- Perfect-foresight LP dispatch optimization (scipy)
- Degradation model based on cycle count, DoD, and calendar aging
- NPV analysis with configurable revenue decline and discount rate

### How to Run
\```bash
git clone ...
cd bess-best-days
pip install -r requirements.txt
streamlit run app.py
\```

### Assumptions & Limitations
- DA-only (no intraday continuous or AS energy revenue)
- Perfect foresight (upper bound — real optimizer captures less)
- Single German bidding zone
- Simplified degradation (no temperature, no C-rate effects)
- Cannibalization modeled as constant annual decline (reality is nonlinear)

### What I'd Build Next
- Intraday continuous trading layer (reBAP / EPEX ID)
- aFRR energy revenue stacking
- Stochastic scenarios (Monte Carlo on price distributions)
- Real optimizer backtesting (forecast-based strategies)
- Temperature-dependent degradation

### AI Usage
[How AI tools were used during development]
```

---

## 9. Risks

| Risk | Mitigation |
|------|------------|
| Energy-Charts API down or rate-limited | Fallback to SMARD.de. Cache aggressively. |
| LP too slow for 6 years × 2 strategies | ~4400 LP solves × ~10ms each ≈ 44 seconds. Acceptable. Cache results. |
| regelleistung.net download blocked | Hardcode monthly FCR/aFRR averages from their published PDFs |
| reBAP OAuth registration delayed | Proceed DA-only, note in limitations |
| Degradation model challenged | Cite Schmalstieg et al., acknowledge simplification, show sensitivity |
| Scope creep into AS/ID | Hard boundary: DA-only for dispatch. Other markets in cannibalization context only. |

---

## 10. Why This Research Matters

Revenue concentration is an underexplored risk dimension. Most BESS financial models assume smooth annual revenue curves. If 80% of income comes from 20% of days, the confidence interval around any annual forecast is much wider than smooth models suggest.

For optimizers, this reframes the job: consistent daily performance matters less than not missing the tail. For asset owners choosing between conservative and aggressive operation, the answer depends on whether you believe future tails will be as fat as today's.

This research doesn't claim to have the final answer. It provides the framework and the data to explore the question seriously.
