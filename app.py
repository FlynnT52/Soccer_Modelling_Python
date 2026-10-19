"""Public Streamlit dashboard for the soccer prediction model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from soccer_model import (
    PoissonFootballModel,
    current_season_start,
    football_data_url,
    load_matches,
    matches_from_json,
)

LEAGUES = {
    "Premier League (England)": "E0",
    "Championship (England)": "E1",
    "Bundesliga (Germany)": "D1",
    "La Liga (Spain)": "SP1",
    "Serie A (Italy)": "I1",
    "Ligue 1 (France)": "F1",
}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_matches(url: str) -> pd.DataFrame:
    """Download results at most once an hour per Streamlit server."""

    return load_matches(url)


@st.cache_resource(show_spinner=False)
def fit_model(matches_csv: str, half_life_days: float) -> PoissonFootballModel:
    """Fit once for a given cleaned dataset and modelling configuration."""

    matches = matches_from_json(matches_csv)
    return PoissonFootballModel.fit(matches, half_life_days=half_life_days)


def scoreline_table(matrix: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    rows = [
        {
            "Score": f"{home_team} {home_goals}–{away_goals} {away_team}",
            "Probability": probability,
        }
        for home_goals, probabilities in matrix.iterrows()
        for away_goals, probability in probabilities.items()
    ]
    return pd.DataFrame(rows).sort_values("Probability", ascending=False).head(8)


st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Match Predictor")
st.caption(
    "A recency-weighted Poisson model trained on results from football-data.co.uk."
)

with st.sidebar:
    st.header("Model settings")
    league_name = st.selectbox("League", list(LEAGUES))
    season_start = st.number_input(
        "Season starting year",
        min_value=2000,
        max_value=2098,
        value=current_season_start(),
        step=1,
        help="For the 2026/27 season, enter 2026.",
    )
    half_life = st.slider(
        "Form half-life (days)",
        min_value=90,
        max_value=730,
        value=365,
        step=30,
        help="Lower values give recent matches more influence.",
    )

division = LEAGUES[league_name]
data_url = football_data_url(int(season_start), division)

try:
    with st.spinner("Downloading results and fitting the model…"):
        matches = download_matches(data_url)
        serialized_matches = matches.to_json(orient="split", date_format="iso")
        model = fit_model(serialized_matches, float(half_life))
except Exception as error:
    st.error("The latest results could not be downloaded or modelled.")
    st.code(str(error))
    st.markdown(f"You can check the [source CSV]({data_url}) directly.")
    st.stop()

latest_result = matches["Date"].max()
st.success(
    f"Model fitted using {len(matches)} completed matches through "
    f"{latest_result:%d %B %Y}."
)

selection_left, selection_right = st.columns(2)
with selection_left:
    home_team = st.selectbox("Home team", model.teams, index=0)
with selection_right:
    away_options = tuple(team for team in model.teams if team != home_team)
    away_team = st.selectbox("Away team", away_options, index=min(1, len(away_options) - 1))

home_xg, away_xg = model.expected_goals(home_team, away_team)
probabilities = model.outcome_probabilities(home_team, away_team)
odds = model.fair_decimal_odds(home_team, away_team)
matrix = model.score_matrix(home_team, away_team)

st.subheader(f"{home_team} vs {away_team}")
xg_home, xg_away = st.columns(2)
xg_home.metric(f"{home_team} expected goals", f"{home_xg:.2f}")
xg_away.metric(f"{away_team} expected goals", f"{away_xg:.2f}")

st.subheader("Result probabilities")
labels = {
    "home_win": f"{home_team} win",
    "draw": "Draw",
    "away_win": f"{away_team} win",
}
result_columns = st.columns(3)
for column, outcome in zip(result_columns, probabilities.index):
    with column:
        st.metric(labels[outcome], f"{probabilities[outcome]:.1%}")
        st.caption(f"Fair decimal odds: {odds[outcome]:.2f}")

chart_column, table_column = st.columns([3, 2])
with chart_column:
    st.subheader("Exact-score probabilities")
    figure, axis = plt.subplots(figsize=(7, 5))
    display_matrix = matrix.iloc[:7, :7]
    image = axis.imshow(display_matrix, cmap="Blues", origin="upper")
    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            probability = display_matrix.iloc[row, column]
            axis.text(
                column,
                row,
                f"{probability:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if probability > display_matrix.to_numpy().max() * 0.55 else "black",
            )
    axis.set_xlabel(f"{away_team} goals")
    axis.set_ylabel(f"{home_team} goals")
    axis.set_xticks(range(7))
    axis.set_yticks(range(7))
    figure.colorbar(image, ax=axis, label="Probability")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with table_column:
    st.subheader("Most likely scores")
    likely_scores = scoreline_table(matrix, home_team, away_team)
    st.dataframe(
        likely_scores.style.format({"Probability": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("How the model works"):
    st.markdown(
        """
        The model estimates an attacking and defensive effect for every team,
        plus a league-wide home advantage. Goals are modelled with a Poisson
        distribution. Recent games receive more weight than older games using
        the selected half-life.

        The displayed odds are no-margin model estimates. They are illustrative,
        not financial or betting advice. A production forecasting system should
        also be evaluated on matches that were not used for training.
        """
    )

st.divider()
st.markdown(
    f"[View the data used for this model]({data_url}) · "
    "[View the source code](https://github.com/FlynnT52/Soccer_Modelling_Python)"
)

