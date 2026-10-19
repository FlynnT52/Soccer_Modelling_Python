from datetime import date

import pandas as pd
import pytest

from soccer_model.model import (
    PoissonFootballModel,
    current_season_start,
    football_data_url,
    load_matches,
    season_code,
)


@pytest.fixture
def matches(tmp_path):
    path = tmp_path / "matches.csv"
    pd.DataFrame(
        {
            "Date": ["01/08/2025", "08/08/2025", "15/08/2025", "22/08/2025", "29/08/2025", "05/09/2025"],
            "HomeTeam": ["Alpha", "Beta", "Gamma", "Alpha", "Beta", "Gamma"],
            "AwayTeam": ["Beta", "Gamma", "Alpha", "Gamma", "Alpha", "Beta"],
            "FTHG": [2, 1, 0, 3, 1, 2],
            "FTAG": [0, 1, 2, 1, 2, 2],
        }
    ).to_csv(path, index=False)
    return load_matches(path)


def test_season_helpers():
    assert current_season_start(date(2026, 10, 19)) == 2026
    assert current_season_start(date(2027, 2, 1)) == 2026
    assert season_code(2026) == "2627"
    assert football_data_url(2026) == "https://www.football-data.co.uk/mmz4281/2627/E0.csv"


def test_load_matches_drops_unplayed_rows(tmp_path):
    path = tmp_path / "matches.csv"
    pd.DataFrame(
        {
            "Date": ["01/08/2025", "08/08/2025"],
            "HomeTeam": ["Alpha", "Beta"],
            "AwayTeam": ["Beta", "Alpha"],
            "FTHG": [2, None],
            "FTAG": [1, None],
        }
    ).to_csv(path, index=False)
    result = load_matches(path)
    assert len(result) == 1
    assert result.loc[0, "FTHG"] == 2


def test_model_probabilities_are_valid(matches):
    model = PoissonFootballModel.fit(matches)
    home_xg, away_xg = model.expected_goals("Alpha", "Beta")
    probabilities = model.outcome_probabilities("Alpha", "Beta")
    assert home_xg > 0
    assert away_xg > 0
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.between(0, 1).all()


def test_unknown_team_has_clear_error(matches):
    model = PoissonFootballModel.fit(matches)
    with pytest.raises(ValueError, match="unknown team"):
        model.expected_goals("Alpha", "Unknown")

