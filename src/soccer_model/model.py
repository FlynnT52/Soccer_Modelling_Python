"""Data preparation and Poisson score modelling.

The model follows the Betfair tutorial's attack/defence Poisson regression,
with two practical updates: recent games receive more weight, and input data
is checked explicitly before fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

REQUIRED_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def current_season_start(today: date | None = None) -> int:
    """Return the first calendar year of the current European season.

    football-data.co.uk rolls to a new season in July: October 2026 therefore
    maps to the 2026/27 season and January 2027 still maps to 2026/27.
    """

    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(season_start: int) -> str:
    """Convert 2026 to football-data.co.uk's ``2627`` directory name."""

    if not 1900 <= season_start <= 2098:
        raise ValueError("season_start must be a four-digit year")
    return f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"


def football_data_url(season_start: int, division: str = "E0") -> str:
    """Build a football-data.co.uk CSV URL (E0 is the Premier League)."""

    if not division or not division.isalnum():
        raise ValueError("division must be an alphanumeric football-data code")
    return (
        "https://www.football-data.co.uk/mmz4281/"
        f"{season_code(season_start)}/{division}.csv"
    )


def _download(url: str) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "soccer-modelling-tutorial/0.1"})
    with urlopen(request, timeout=30) as response:
        return BytesIO(response.read())


def load_matches(source: str | Path) -> pd.DataFrame:
    """Load and clean completed matches from a URL or local CSV file."""

    source_text = str(source)
    csv_source = _download(source_text) if source_text.startswith(("http://", "https://")) else source
    frame = pd.read_csv(csv_source)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    matches = frame.loc[:, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches["FTHG"] = pd.to_numeric(matches["FTHG"], errors="coerce")
    matches["FTAG"] = pd.to_numeric(matches["FTAG"], errors="coerce")
    matches = matches.dropna(subset=list(REQUIRED_COLUMNS)).sort_values("Date")
    matches[["FTHG", "FTAG"]] = matches[["FTHG", "FTAG"]].astype(int)
    matches = matches.reset_index(drop=True)
    if matches.empty:
        raise ValueError("CSV contains no completed matches")
    return matches


def matches_from_json(serialized: str) -> pd.DataFrame:
    """Restore cached matches without treating JSON text as a file path."""

    matches = pd.read_json(StringIO(serialized), orient="split")
    matches["Date"] = pd.to_datetime(matches["Date"])
    return matches


def _goal_rows(matches: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    latest = matches["Date"].max()
    weights = np.exp(-np.log(2) * (latest - matches["Date"]).dt.days / half_life_days)
    home = pd.DataFrame(
        {
            "goals": matches["FTHG"],
            "home": 1,
            "team": matches["HomeTeam"],
            "opponent": matches["AwayTeam"],
            "weight": weights,
        }
    )
    away = pd.DataFrame(
        {
            "goals": matches["FTAG"],
            "home": 0,
            "team": matches["AwayTeam"],
            "opponent": matches["HomeTeam"],
            "weight": weights,
        }
    )
    return pd.concat([home, away], ignore_index=True)


@dataclass
class PoissonFootballModel:
    """Independent-Poisson attack/defence model with home advantage."""

    result: object
    teams: tuple[str, ...]
    half_life_days: float

    @classmethod
    def fit(
        cls, matches: pd.DataFrame, half_life_days: float = 365.0
    ) -> "PoissonFootballModel":
        goal_rows = _goal_rows(matches, half_life_days)
        teams = tuple(sorted(set(goal_rows["team"])))
        if len(teams) < 2:
            raise ValueError("at least two teams are required")
        result = smf.glm(
            formula="goals ~ home + C(team) + C(opponent)",
            data=goal_rows,
            family=sm.families.Poisson(),
            freq_weights=goal_rows["weight"],
        ).fit()
        return cls(result=result, teams=teams, half_life_days=half_life_days)

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        self._validate_teams(home_team, away_team)
        fixtures = pd.DataFrame(
            {
                "home": [1, 0],
                "team": [home_team, away_team],
                "opponent": [away_team, home_team],
            }
        )
        home_xg, away_xg = self.result.predict(fixtures)
        return float(home_xg), float(away_xg)

    def score_matrix(
        self, home_team: str, away_team: str, max_goals: int = 10
    ) -> pd.DataFrame:
        """Return P(home goals, away goals), including scores 0..max_goals."""

        if max_goals < 1:
            raise ValueError("max_goals must be at least 1")
        home_xg, away_xg = self.expected_goals(home_team, away_team)
        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, home_xg), poisson.pmf(goals, away_xg))
        # Fold the tiny omitted tail into the last cell so probabilities sum to one.
        matrix[-1, -1] += 1.0 - matrix.sum()
        return pd.DataFrame(matrix, index=goals, columns=goals)

    def outcome_probabilities(
        self, home_team: str, away_team: str, max_goals: int = 10
    ) -> pd.Series:
        matrix = self.score_matrix(home_team, away_team, max_goals).to_numpy()
        probabilities = pd.Series(
            {
                "home_win": float(np.tril(matrix, -1).sum()),
                "draw": float(np.trace(matrix)),
                "away_win": float(np.triu(matrix, 1).sum()),
            }
        )
        return probabilities / probabilities.sum()

    def fair_decimal_odds(
        self, home_team: str, away_team: str, max_goals: int = 10
    ) -> pd.Series:
        return 1.0 / self.outcome_probabilities(home_team, away_team, max_goals)

    def _validate_teams(self, home_team: str, away_team: str) -> None:
        unknown = {home_team, away_team}.difference(self.teams)
        if unknown:
            raise ValueError(
                f"unknown team(s): {sorted(unknown)}. Choose from: {list(self.teams)}"
            )
        if home_team == away_team:
            raise ValueError("home_team and away_team must differ")

