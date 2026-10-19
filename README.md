# Soccer modelling in Python

An updated, executable version of Betfair's soccer modelling tutorial using
current data from [football-data.co.uk](https://www.football-data.co.uk/).

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
Poisson GLM. Recent matches receive more weight via a configurable 365-day
half-life. It reports expected goals, win/draw/loss probabilities, exact-score
probabilities, and no-margin decimal odds.

Team names must match the source CSV. Run `soccer-model` without teams to print
the available names. Model prices are illustrative and are not betting advice.

## Web dashboard

Run the Streamlit portfolio app locally:

```bash
python -m streamlit run app.py
```

To publish it, create an app at [Streamlit Community
Cloud](https://share.streamlit.io/), select this repository and the `main`
branch, and use `app.py` as the main file path.

