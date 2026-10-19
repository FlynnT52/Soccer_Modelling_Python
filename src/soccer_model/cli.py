"""Command-line entry point for the tutorial model."""

from __future__ import annotations

import argparse

from .model import PoissonFootballModel, current_season_start, football_data_url, load_matches


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Fit a Poisson football score model")
    command.add_argument("home_team", nargs="?")
    command.add_argument("away_team", nargs="?")
    command.add_argument("--season", type=int, default=current_season_start())
    command.add_argument("--division", default="E0", help="football-data league code")
    command.add_argument("--csv", help="local CSV instead of downloading")
    command.add_argument("--half-life", type=float, default=365.0, help="recency half-life in days")
    command.add_argument("--max-goals", type=int, default=10)
    return command


def main() -> None:
    args = parser().parse_args()
    source = args.csv or football_data_url(args.season, args.division)
    matches = load_matches(source)
    model = PoissonFootballModel.fit(matches, args.half_life)
    print(f"Loaded {len(matches)} completed matches from {source}")
    print("Teams:", ", ".join(model.teams))
    if not args.home_team or not args.away_team:
        print("Pass HOME_TEAM AWAY_TEAM to generate a prediction.")
        return
    home_xg, away_xg = model.expected_goals(args.home_team, args.away_team)
    probabilities = model.outcome_probabilities(args.home_team, args.away_team, args.max_goals)
    odds = 1.0 / probabilities
    print(f"Expected goals: {args.home_team} {home_xg:.2f}, {args.away_team} {away_xg:.2f}")
    for outcome in probabilities.index:
        print(f"{outcome:>9}: {probabilities[outcome]:.1%}  fair odds {odds[outcome]:.2f}")


if __name__ == "__main__":
    main()

