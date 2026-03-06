# The Tail Wags the Dog

[Open the live app](https://bess-tail-economics.streamlit.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bess-tail-economics.streamlit.app)

![Review hero](assets/review-hero.svg)

Publication-style Streamlit article on German BESS merchant revenue, tail-day concentration, cycling intensity, and warranty economics.

## Review Path
- Open the live app first.
- Read sections 1-4 in order.
- Start with the default controls: `2h`, `2025`.

## Publish For Review
1. Push this repository to a public GitHub repo.
2. Deploy `app.py` from that repo on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
3. In GitHub repo `About`, set `Website` to the same live URL so the review link is visible on the repo front page.
4. Send reviewers the Streamlit URL first, and the GitHub repo second.

Suggested GitHub repo description:

`Publication-style Streamlit article on German BESS merchant revenue, tail-day concentration, cycling intensity, and warranty economics.`

## The Question
German BESS merchant revenue is not smooth. A relatively small number of extreme-spread days can drive a large share of annual battery trading income. That changes the real investment question: should operators preserve the battery for future optionality, or run it harder while today's tail still pays?

The publication app intentionally starts in `2021`, because the preceding partial year does not offer a full-year `ID-AEP` history.

## What Is Implemented
- Day-ahead price ingestion from Energy-Charts with parquet cache and native timestep handling for the 2025 quarter-hour transition.
- Official intraday series ingestion from Netztransparenz `ID-AEP`.
- Generation mix ingestion from Energy-Charts and daily driver metrics for solar, wind, load, and residual load.
- Sequential day-ahead plus intraday-overlay dispatch using SciPy HiGHS.
- Revenue concentration analytics, top-day diagnostics, a warranty-backed cycle-intensity frontier, and a break-even annual premium / reserve calculation for extended warranty versus self-insurance.
- Streamlit report written as a publication-style narrative for investors, traders, and asset owners.

## Methodology
- Day-ahead prices: Energy-Charts API (`DE-LU`, `2021-01-01` through `2025-12-31`).
- Intraday layer: Netztransparenz `ID-AEP` web service (`2021-01-01` through `2025-12-31` in the app), used as an official intraday series rather than a full continuous trade tape.
- Generation mix: Energy-Charts `public_power`.
- Dispatch model: linear optimization with power limits, SoC bounds, usable-energy-aware cycle limits, and an end-of-day SoC return band. The publication app uses one merchant revenue series built as a sequential stack: day-ahead schedule first, then intraday re-trading on `ID-AEP`.
- Frontier model: the app sweeps daily cycle caps from `0.25` to `4.00` while holding the trading setup fixed, then discounts lifetime merchant value under a harsher warranty-backed degradation curve.
- Degradation model: vendor-style lifecycle anchor of `7,300` full cycles over `20` years with retirement at `60%` state of health.
- Warranty-posture model: the app converts the value gap between warranty pace and economic optimum into a break-even annual premium / reserve that can be read either as OEM extension pricing or self-insurance budget.

## Important Modeling Choices
- The dispatch solver uses the native market timestep per day, not a fixed 24-step hour model. This matters for Q4 2025 because Energy-Charts shifts to 15-minute prices.
- The publication app fixes round-trip efficiency at `0.86` instead of exposing it as a user control.
- The strategy `min spread to trade` input is approximated as a symmetric throughput penalty in the LP objective. That keeps the optimization linear while preserving the economic intent of a minimum spread hurdle.
- Physical cycle count is approximated from contiguous charge/discharge segments. This is practical for this report but less precise than rainflow counting on a full SoC trace.
- The cycle-intensity frontier varies only the daily cycle cap; the SoC window and spread hurdle stay fixed so the chart isolates cycling intensity rather than changing the whole strategy at once.

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
- Confirm the app opens with the default controls already set to `2h` and `2025`.
- Confirm the `Public sources` note is visible on the page for external audit.

## Project Structure
```text
bess-strategy/
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
- Degradation excludes temperature, C-rate effects, and chemistry-specific nonlinearities.
- The warranty-backed lifecycle curve is still a simplified economic model, not a chemistry-specific warranty simulator.

## Better Next Steps
- Replace the cycle-event approximation with rainflow counting on the SoC series.
- Add a more realistic intraday execution layer if you want a closer approximation to gate closures and re-trading frictions.
- Replace the `ID-AEP` overlay proxy with a full `DA+ID` co-optimization layer if you get an auditable intraday continuous trade source.
- Calibrate the lifecycle anchor against a specific OEM warranty sheet if you want asset-specific underwriting instead of a publication-grade reference curve.

## AI Usage
AI was used to scaffold the research app, implement the data/model/chart layers, and wire the Streamlit narrative together. External market data still comes from the cited source systems at runtime.
