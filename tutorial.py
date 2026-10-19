# %% [markdown]
# # Modern football score modelling with Python
#
# This is an executable, notebook-style update of Betfair's soccer modelling
# tutorial. In VS Code or Jupyter, run one `# %%` cell at a time. It uses the
# current Premier League season from football-data.co.uk by default.

# %%
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

from soccer_model import (
    PoissonFootballModel,
    current_season_start,
    football_data_url,
    load_matches,
)

# %% [markdown]
# ## 1. Download completed results
#
# `E0` means the English Premier League. Other common codes include `E1`
# (Championship), `D1` (Bundesliga), `I1` (Serie A), `SP1` (La Liga), and
# `F1` (Ligue 1). Set `SEASON_START` explicitly to make an analysis repeatable.

# %%
SEASON_START = current_season_start(date.today())
DIVISION = "E0"
DATA_URL = football_data_url(SEASON_START, DIVISION)

matches = load_matches(DATA_URL)
print(DATA_URL)
print(f"{len(matches)} completed matches, {matches.Date.min():%d %b %Y} to {matches.Date.max():%d %b %Y}")
matches.tail()

# %% [markdown]
# ## 2. Explore scoring

# %%
matches[["FTHG", "FTAG"]].describe()

# %%
ax = matches[["FTHG", "FTAG"]].mean().rename(
    {"FTHG": "Home goals", "FTAG": "Away goals"}
).plot.bar(rot=0, ylabel="Goals per match", title="Home advantage")
plt.tight_layout()

# %% [markdown]
# ## 3. Fit the model
#
# Each club gets an attacking effect and a defensive effect; `home` estimates
# the league-wide home advantage. Games decay with a one-year half-life, which
# makes this more useful than treating old and recent form as equally relevant.

# %%
model = PoissonFootballModel.fit(matches, half_life_days=365)
print("Available teams:")
print(*model.teams, sep="\n")

# %% [markdown]
# ## 4. Predict a fixture
#
# Edit these names using the exact spellings printed above.

# %%
HOME_TEAM, AWAY_TEAM = model.teams[0], model.teams[1]
home_xg, away_xg = model.expected_goals(HOME_TEAM, AWAY_TEAM)
print(f"{HOME_TEAM} {home_xg:.2f} xG — {away_xg:.2f} xG {AWAY_TEAM}")

# %%
score_probabilities = model.score_matrix(HOME_TEAM, AWAY_TEAM)
sns.heatmap(
    score_probabilities.iloc[:7, :7],
    annot=True,
    fmt=".1%",
    cmap="Blues",
    cbar=False,
)
plt.xlabel(f"{AWAY_TEAM} goals")
plt.ylabel(f"{HOME_TEAM} goals")
plt.title("Exact-score probabilities")
plt.tight_layout()

# %%
summary = model.outcome_probabilities(HOME_TEAM, AWAY_TEAM).to_frame("probability")
summary["fair_decimal_odds"] = model.fair_decimal_odds(HOME_TEAM, AWAY_TEAM)
summary

# %% [markdown]
# “Fair” odds contain no bookmaker margin and are model estimates, not betting
# advice. A useful next step is honest backtesting on matches held out by date;
# do not judge the model on the same observations used to fit it.

