"""Tools for fitting a Poisson model to football-data.co.uk results."""

from .model import (
    PoissonFootballModel,
    current_season_start,
    football_data_url,
    load_matches,
    matches_from_json,
)

__all__ = [
    "PoissonFootballModel",
    "current_season_start",
    "football_data_url",
    "load_matches",
    "matches_from_json",
]

