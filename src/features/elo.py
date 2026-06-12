"""Compute Elo ratings from historical match results."""

import polars as pl

# K-factor varies by match importance
K_FACTORS = {
    "FIFA World Cup": 60,
    "Confederations Cup": 50,
    "Copa América": 50,
    "UEFA Euro": 50,
    "AFC Asian Cup": 50,
    "African Cup of Nations": 50,
    "CONCACAF Gold Cup": 50,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "Friendly": 20,
}

DEFAULT_K = 30
INITIAL_ELO = 1500
HOME_ADVANTAGE = 100


def get_k_factor(tournament: str) -> int:
    for key, k in K_FACTORS.items():
        if key.lower() in tournament.lower():
            return k
    return DEFAULT_K


def compute_elo_ratings(matches_df: pl.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    Walk through all matches chronologically and compute Elo for each team.

    Returns a dict: team -> list of (date, elo) tuples representing the
    team's Elo after each match they played.
    """
    matches = matches_df.sort("date").to_dicts()

    current_elo = {}
    elo_history = {}  # team -> [(date, elo), ...]

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        home_score = match["home_score"]
        away_score = match["away_score"]
        tournament = match["tournament"]
        match_date = match["date"]
        is_neutral = match["neutral"]

        if home_score is None or away_score is None:
            continue

        elo_h = current_elo.get(home, INITIAL_ELO)
        elo_a = current_elo.get(away, INITIAL_ELO)

        # Home advantage only for non-neutral venues
        ha = 0 if is_neutral else HOME_ADVANTAGE

        # Expected scores
        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h - ha) / 400))
        exp_a = 1 - exp_h

        # Actual scores (1 for win, 0.5 for draw, 0 for loss)
        if home_score > away_score:
            actual_h, actual_a = 1.0, 0.0
        elif home_score == away_score:
            actual_h, actual_a = 0.5, 0.5
        else:
            actual_h, actual_a = 0.0, 1.0

        # Goal difference multiplier (larger wins = bigger Elo change)
        gd = abs(home_score - away_score)
        if gd <= 1:
            gd_mult = 1.0
        elif gd == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11 + gd) / 8

        k = get_k_factor(tournament)

        # Update
        current_elo[home] = elo_h + k * gd_mult * (actual_h - exp_h)
        current_elo[away] = elo_a + k * gd_mult * (actual_a - exp_a)

        # Record history
        if home not in elo_history:
            elo_history[home] = []
        if away not in elo_history:
            elo_history[away] = []

        elo_history[home].append((match_date, current_elo[home]))
        elo_history[away].append((match_date, current_elo[away]))

    return elo_history


def get_current_elos(elo_history: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    """Get the most recent Elo for each team."""
    return {
        team: ratings[-1][1]
        for team, ratings in elo_history.items()
        if ratings
    }


def get_elo_at_date(elo_history: dict[str, list[tuple[str, float]]], team: str, date: str) -> float:
    """Get a team's Elo rating at a specific date (last known before that date)."""
    if team not in elo_history:
        return INITIAL_ELO
    ratings = elo_history[team]
    elo = INITIAL_ELO
    for d, r in ratings:
        if d > date:
            break
        elo = r
    return elo
