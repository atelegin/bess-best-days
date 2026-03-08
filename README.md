# The Cost of Missing the Best Days

[Open the live app](https://bess-best-days.streamlit.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bess-best-days.streamlit.app)

![Review hero](assets/review-hero.svg)

Publication-style Streamlit article on German BESS merchant revenue concentration, day-ahead visibility, and the value of timed flexibility.

## Review Path
- Open the live app first.
- Read sections 1-4 in order.
- Read the story in this order: missed best days, best-day shape, day-ahead watchlist, same-throughput uplift.

## Publish For Review
1. Push this repository to a public GitHub repo.
2. Deploy `app.py` from that repo on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
3. In GitHub repo `About`, set `Website` to the same live URL so the review link is visible on the repo front page.
4. Send reviewers the Streamlit URL first, and the GitHub repo second.

Suggested GitHub repo description:

`Publication-style Streamlit article on German BESS merchant revenue concentration, day-ahead visibility, and timed flexibility.`

## The Question
German BESS merchant revenue is not earned evenly through the year. A limited set of high-value days drives a disproportionate share of annual income, and many of those days are already partly visible in the day-ahead curve. That changes the commercial question from average optimisation to timed availability, readiness, and throughput allocation.

## What Is Implemented
- Day-ahead price ingestion from Energy-Charts with parquet cache and native timestep handling for the 2025 quarter-hour transition.
- Official intraday series ingestion from Netztransparenz `ID-AEP`.
- Sequential day-ahead plus intraday-overlay dispatch using SciPy HiGHS.
- Revenue concentration analytics, top-day diagnostics, day-ahead watchlist scoring, and same-throughput reallocation analysis.
- Streamlit report written as a publication-style narrative for investors, traders, and asset owners.

## Methodology
- Day-ahead prices: Energy-Charts API (`DE-LU`, `2021-01-01` through `2025-12-31`).
- Intraday layer: Netztransparenz `ID-AEP` web service (`2021-01-01` through `2025-12-31` in the app), used as an official intraday series rather than a full continuous trade tape.
- Dispatch model: linear optimization with power limits, SoC bounds, usable-energy-aware cycle limits, and an end-of-day SoC return band. The publication app uses one merchant revenue series built as a sequential stack: day-ahead schedule first, then intraday re-trading on `ID-AEP`.
- Best-day shape model: top-20 revenue days are compared with all days using median day-ahead hourly price profiles.
- Watchlist model: simple day-ahead rules are evaluated on pooled `2021-2025` data using recall, precision, and lift against the top-20 revenue-day base rate.
- Reallocation model: strict daily caps are compared with an annual allocator using the same realized FEC, isolating timing value from throughput volume.

## Important Modeling Choices
- The dispatch solver uses the native market timestep per day, not a fixed 24-step hour model. This matters for Q4 2025 because Energy-Charts shifts to 15-minute prices.
- The publication app fixes round-trip efficiency at `0.86` instead of exposing it as a user control.
- The strategy `min spread to trade` input is approximated as a symmetric throughput penalty in the LP objective. That keeps the optimization linear while preserving the economic intent of a minimum spread hurdle.
- Physical cycle count is approximated from contiguous charge/discharge segments. This is practical for this report but less precise than rainflow counting on a full SoC trace.
- The same-throughput comparison isolates timing value by holding realized annual throughput constant between the strict-cap and flexible allocator cases.

## How To Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Review Checklist
- Confirm the top link in this README points to the deployed app, not `localhost`.
- Confirm GitHub `About -> Website` points to the same deployed app.
- Confirm the app opens directly into the publication narrative without sidebar controls.
- Confirm the `Public sources` note is visible on the page for external audit.

## Project Structure
```text
bess-best-days/
├── README.md
├── requirements.txt
├── app.py
├── assets/
│   └── review-hero.svg
├── src/
│   ├── analysis/
│   ├── charts/
│   ├── data/
│   └── models/
├── data/
│   └── cache/
└── notebooks/
    └── exploration.ipynb
```

## Assumptions And Limitations
- Dispatch is a perfect-foresight upper bound.
- The app uses `ID-AEP` as an audit-friendly intraday series, not as a full intraday continuous trade tape.
- Merchant revenue is modeled as a sequential `DA + ID overlay` stack, not as a fully joint `DA+ID` co-optimization.
- No `aFRR` energy, `FCR` energy, or `DA+ID` co-optimization stack is modeled.
- Reallocation uplift is still based on realized perfect-foresight opportunity ranking, so it should be read as an economic upper bound on timing value.
- Degradation excludes temperature, C-rate effects, and chemistry-specific nonlinearities.

## Better Next Steps
- Replace the cycle-event approximation with rainflow counting on the SoC series.
- Add a more realistic intraday execution layer if you want a closer approximation to gate closures and re-trading frictions.
- Replace the `ID-AEP` overlay proxy with a full `DA+ID` co-optimization layer if you get an auditable intraday continuous trade source.
- Add a maintenance-timing or operational-readiness simulator on top of the day-ahead watchlist logic.

## AI Usage
AI was used to scaffold the research app, implement the data/model/chart layers, and wire the Streamlit narrative together. External market data still comes from the cited source systems at runtime.
