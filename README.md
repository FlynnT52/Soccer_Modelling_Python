# Soccer modelling in Python

An interactive, recency-weighted Poisson football forecasting project using
current data from [football-data.co.uk](https://www.football-data.co.uk/).

**[Open the live prediction dashboard](https://thompsonflynn-epl-soccer-model.streamlit.app/)**

## Project overview

The application downloads and combines a selectable range of league seasons,
then estimates team attack, team defence and league-wide home advantage. Users
can explore expected goals, exact-score distributions, match-result
probabilities and no-margin decimal odds.

Key features:

- hourly refresh of completed match data;
- selectable leagues, seasons and recency half-life;
- interactive expected-goal and exact-score forecasts;
- chronological out-of-sample evaluation;
- log loss, Ranked Probability Score and baseline comparison;
- automated tests and continuous deployment through Streamlit Community Cloud.

## Model evaluation

The dashboard reserves the latest 20% of the selected matches as a holdout.
The model is fitted only to the earlier 80%, then evaluated against results it
did not see during training. This avoids presenting in-sample fit as forecasting
performance. Results vary with the league, date range and half-life selected.

Current limitations include the independent-Poisson assumption and the absence
of player availability, expected line-ups, injuries and match context. The
outputs are illustrative and are not betting or financial advice.

## Technology

Python, pandas, NumPy, SciPy, statsmodels, matplotlib, pytest and Streamlit.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Run the walkthrough cell by cell in [`tutorial.py`](tutorial.py), or use the
command line:

```bash
soccer-model
soccer-model "Arsenal" "Chelsea"
```

The default is the current Premier League season (`E0`). For a fixed season,
another league, or a previously downloaded file:

```bash
soccer-model --season 2025 --division E1 "Coventry" "Sunderland"
soccer-model --csv data/E0.csv "Arsenal" "Chelsea"
```

The model estimates team attack, team defence, and home advantage with a
Poisson GLM. Recent matches receive more weight via a configurable half-life.

Team names must match the source CSV. Run `soccer-model` without teams to print
the available names. Model prices are illustrative and are not betting advice.

## Web dashboard

Run the Streamlit portfolio app locally:

```bash
python -m streamlit run app.py
```

The deployed application uses `app.py` on the `main` branch.

