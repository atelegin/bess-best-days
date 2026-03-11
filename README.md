# The Cost of Missing the Best Days

[Open the live app](https://bess-best-days.streamlit.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bess-best-days.streamlit.app)

![Review hero](assets/review-hero.svg)

Publication-style Streamlit article on German BESS merchant revenue concentration, day-ahead visibility, and the value of timed availability, readiness, and flexibility.

## Review Path
- Open the live app first.
- Read sections 1-4 in order.
- Read the story in this order: missed best days, best-day shape, watchlist usefulness, same-throughput timing value.

## Publish For Review
1. Push this repository to a public GitHub repo.
2. Deploy `app.py` from that repo on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
3. In GitHub repo `About`, set `Website` to the same live URL so the review link is visible on the repo front page.
4. Send reviewers the Streamlit URL first, and the GitHub repo second.

Suggested GitHub repo description:

`Publication-style Streamlit article on German BESS merchant revenue concentration, day-ahead visibility, and timed flexibility.`

## The Question
German BESS merchant revenue is not earned evenly through the year. A limited set of disproportionately valuable days drives annual outcomes, and many of those days are already partly visible before delivery. That changes the commercial question from average optimization to timed availability, readiness, and throughput allocation.

## Article Structure
- Section 1: revenue is concentrated where it matters most.
- Section 2: high-value days show a deeper midday trough and a wider evening-minus-midday ramp.
- Section 3: a useful early-warning screen exists at D-2, but the strongest screening power appears at D-1.
- Section 4: even with the same annual throughput, concentrating cycles into the strongest days earns more revenue.
- Closing: availability, readiness, and flexibility matter most when they are timed.

## What Is Implemented
- Day-ahead price ingestion from Energy-Charts with native timestep handling through the 2025 quarter-hour transition.
- Official intraday proxy ingestion from Netztransparenz `ID-AEP`.
- Sequential day-ahead plus intraday-overlay dispatch using SciPy HiGHS.
- A publication-style Streamlit narrative built around four fixed sections and a closing owner takeaway.
- Revenue concentration analytics, best-day shape diagnostics, pooled watchlist screening, and same-throughput reallocation analysis.

## Core Method
- Scope: base case is a 2h battery with a 2025 deep dive; validation uses pooled `2021-2025` data.
- Merchant revenue is modeled as one combined day-ahead plus intraday series using Energy-Charts day-ahead prices and the official Netztransparenz `ID-AEP` index for the intraday layer.
- Dispatch is modeled sequentially across day-ahead and intraday with fixed round-trip efficiency of `0.86`.
- Best-day shape compares top-20 revenue days against the full sample average day using median hourly day-ahead price profiles.
- Watchlist screening uses pooled `2021-2025` precision, recall, and lift versus the top-20 revenue-day base rate.
- Same-throughput reallocation compares strict daily caps with an annual allocator using the same realized FEC, isolating timing value from extra throughput.

## Important Modeling Choices
- Dispatch is a perfect-foresight upper bound.
- `ID-AEP` is used as an audit-friendly intraday proxy, not as a full continuous intraday trade tape.
- The merchant stack is modeled as sequential `DA + intraday overlay`, not as a fully joint `DA+ID` co-optimization.
- The app fixes round-trip efficiency at `0.86` and does not expose it as a user control.
- Reallocation uplift is still based on realized opportunity ranking, so it should be read as an economic upper bound on timing value.
- No `FCR`, `aFRR` energy, or ancillary co-optimization stack is modeled.

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
- Confirm the top scope line shows `Base case`, `Validation`, and `Method note`.
- Confirm the four sections match the README structure above.

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
```

## Assumptions And Limitations
- The article is intentionally fixed-scope and publication-style, not an exploratory dashboard.
- D-2 metrics in Section 3 are presented as an early-warning benchmark used in the article narrative.
- The app uses public market data and simplified merchant dispatch assumptions to make the owner problem legible, not to replicate plant-level execution frictions exactly.
- Degradation excludes temperature, C-rate effects, and chemistry-specific nonlinearities.

## Better Next Steps
- Add a more realistic D-2 feature-construction pipeline if you want the early-warning screen to be fully derived inside the app instead of presented as a narrative benchmark.
- Replace the intraday proxy with a fuller auditable intraday execution dataset if one becomes available.
- Add a maintenance-timing simulator on top of the watchlist logic.
- Replace the cycle-event approximation with rainflow counting on the SoC trace.

## AI Usage
AI was used to scaffold the research app, implement the data/model/chart layers, and wire the Streamlit narrative together. External market data still comes from the cited source systems at runtime.
