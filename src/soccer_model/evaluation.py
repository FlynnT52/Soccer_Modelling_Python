"""Out-of-sample evaluation for the football prediction model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import PoissonFootballModel


@dataclass(frozen=True)
class BacktestResult:
    """Metrics and match-level predictions from a chronological holdout."""

    predictions: pd.DataFrame
    training_matches: int
    test_matches: int
    skipped_matches: int
    accuracy: float
    log_loss: float
    baseline_log_loss: float
    ranked_probability_score: float


def _actual_outcome(row: pd.Series) -> str:
    if row["FTHG"] > row["FTAG"]:
        return "home_win"
    if row["FTHG"] < row["FTAG"]:
        return "away_win"
    return "draw"


def chronological_backtest(
    matches: pd.DataFrame,
    half_life_days: float = 365.0,
    test_fraction: float = 0.2,
) -> BacktestResult:
    """Train on the earliest matches and evaluate on the latest matches.

    This is a single chronological holdout, so no result from the evaluation
    period is used to fit the model. Matches involving a team absent from the
    training period are reported as skipped rather than silently fabricated.
    """

    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    ordered = matches.sort_values("Date").reset_index(drop=True)
    if len(ordered) < 20:
        raise ValueError("at least 20 completed matches are required for backtesting")

    split = int(len(ordered) * (1 - test_fraction))
    training = ordered.iloc[:split].copy()
    testing = ordered.iloc[split:].copy()
    model = PoissonFootballModel.fit(training, half_life_days=half_life_days)

    training_outcomes = training.apply(_actual_outcome, axis=1)
    outcome_order = ["home_win", "draw", "away_win"]
    baseline = training_outcomes.value_counts(normalize=True).reindex(outcome_order, fill_value=0.0)

    rows: list[dict[str, object]] = []
    skipped = 0
    for _, match in testing.iterrows():
        if match["HomeTeam"] not in model.teams or match["AwayTeam"] not in model.teams:
            skipped += 1
            continue
        probabilities = model.outcome_probabilities(match["HomeTeam"], match["AwayTeam"])
        actual = _actual_outcome(match)
        rows.append(
            {
                "Date": match["Date"],
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "Actual": actual,
                "Predicted": probabilities.idxmax(),
                **{outcome: float(probabilities[outcome]) for outcome in outcome_order},
            }
        )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("no holdout matches involved teams seen during training")

    probability_matrix = predictions[outcome_order].to_numpy(dtype=float)
    actual_indices = predictions["Actual"].map({name: i for i, name in enumerate(outcome_order)}).to_numpy()
    actual_matrix = np.eye(3)[actual_indices]
    chosen_probabilities = probability_matrix[np.arange(len(predictions)), actual_indices]
    log_loss = float(-np.log(np.clip(chosen_probabilities, 1e-15, 1)).mean())
    baseline_probabilities = np.tile(baseline.to_numpy(dtype=float), (len(predictions), 1))
    baseline_chosen = baseline_probabilities[np.arange(len(predictions)), actual_indices]
    baseline_log_loss = float(-np.log(np.clip(baseline_chosen, 1e-15, 1)).mean())
    cumulative_difference = np.cumsum(probability_matrix, axis=1)[:, :-1] - np.cumsum(
        actual_matrix, axis=1
    )[:, :-1]
    rps = float(np.mean(np.sum(cumulative_difference**2, axis=1) / 2))

    return BacktestResult(
        predictions=predictions,
        training_matches=len(training),
        test_matches=len(predictions),
        skipped_matches=skipped,
        accuracy=float((predictions["Actual"] == predictions["Predicted"]).mean()),
        log_loss=log_loss,
        baseline_log_loss=baseline_log_loss,
        ranked_probability_score=rps,
    )

