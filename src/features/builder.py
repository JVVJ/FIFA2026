"""
Fast feature matrix builder using pre-computed rolling statistics.

Instead of computing form per-match on-the-fly, we walk through matches
chronologically and maintain running accumulators per team.
"""

import numpy as np
import polars as pl

from src.features.elo import (
    INITIAL_ELO,
    HOME_ADVANTAGE,
    get_k_factor,
)


def build_features_fast(
    matches_df: pl.DataFrame, min_date: str = "2000-01-01"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build feature matrix efficiently by walking through matches once.

    Returns (X, y, feature_names)
    """
    matches = matches_df.filter(
        pl.col("home_score").is_not_null()
    ).sort("date").to_dicts()

    # Running state per team
    elo = {}         # team -> current elo
    form_results = {}  # team -> deque of last 10 results (3=W, 1=D, 0=L)
    form_gf = {}     # team -> deque of last 10 goals scored
    form_gc = {}     # team -> deque of last 10 goals conceded
    h2h_wins = {}    # (teamA, teamB) sorted -> {teamA_wins, teamB_wins, draws, gd}

    features = []
    labels = []

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

        # === Snapshot current state as features (BEFORE updating) ===
        if match_date >= min_date:
            row = _snapshot_features(
                home, away, elo, form_results, form_gf, form_gc, h2h_wins
            )
            features.append(row)

            if home_score > away_score:
                labels.append(2)
            elif home_score == away_score:
                labels.append(1)
            else:
                labels.append(0)

        # === Update running state ===
        _update_elo(home, away, home_score, away_score, tournament, is_neutral, elo)
        _update_form(home, home_score, away_score, form_results, form_gf, form_gc)
        _update_form(away, away_score, home_score, form_results, form_gf, form_gc)
        _update_h2h(home, away, home_score, away_score, h2h_wins)

    feature_names = [
        "elo_home", "elo_away", "elo_diff",
        "form_home_pts_5", "form_away_pts_5",
        "form_home_pts_10", "form_away_pts_10",
        "form_home_gf_5", "form_away_gf_5",
        "form_home_gc_5", "form_away_gc_5",
        "h2h_matches", "h2h_home_win_pct", "h2h_goal_diff",
    ]

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, feature_names


def _snapshot_features(
    home, away, elo, form_results, form_gf, form_gc, h2h_wins
) -> list:
    """Take a snapshot of current features for a match."""
    elo_h = elo.get(home, INITIAL_ELO)
    elo_a = elo.get(away, INITIAL_ELO)

    # Form: points per match over last 5 and 10
    fr_h = form_results.get(home, [])
    fr_a = form_results.get(away, [])
    pts_h_5 = sum(fr_h[-5:]) / max(len(fr_h[-5:]), 1) / 3.0
    pts_a_5 = sum(fr_a[-5:]) / max(len(fr_a[-5:]), 1) / 3.0
    pts_h_10 = sum(fr_h[-10:]) / max(len(fr_h[-10:]), 1) / 3.0
    pts_a_10 = sum(fr_a[-10:]) / max(len(fr_a[-10:]), 1) / 3.0

    # Goals
    gf_h = form_gf.get(home, [])
    gf_a = form_gf.get(away, [])
    gc_h = form_gc.get(home, [])
    gc_a = form_gc.get(away, [])
    gf_h_5 = sum(gf_h[-5:]) / max(len(gf_h[-5:]), 1)
    gf_a_5 = sum(gf_a[-5:]) / max(len(gf_a[-5:]), 1)
    gc_h_5 = sum(gc_h[-5:]) / max(len(gc_h[-5:]), 1)
    gc_a_5 = sum(gc_a[-5:]) / max(len(gc_a[-5:]), 1)

    # H2H
    key = tuple(sorted([home, away]))
    h2h = h2h_wins.get(key, {"matches": 0, home: 0, away: 0, "draws": 0, "gd": {home: 0, away: 0}})
    h2h_n = h2h["matches"]
    h2h_home_wins = h2h.get(home, 0)
    h2h_win_pct = h2h_home_wins / h2h_n if h2h_n > 0 else 0.5
    h2h_gd = h2h["gd"].get(home, 0) - h2h["gd"].get(away, 0)

    return [
        elo_h, elo_a, elo_h - elo_a,
        pts_h_5, pts_a_5, pts_h_10, pts_a_10,
        gf_h_5, gf_a_5, gc_h_5, gc_a_5,
        h2h_n, h2h_win_pct, h2h_gd,
    ]


def _update_elo(home, away, home_score, away_score, tournament, is_neutral, elo):
    """Update Elo ratings after a match."""
    elo_h = elo.get(home, INITIAL_ELO)
    elo_a = elo.get(away, INITIAL_ELO)

    ha = 0 if is_neutral else HOME_ADVANTAGE
    exp_h = 1 / (1 + 10 ** ((elo_a - elo_h - ha) / 400))

    if home_score > away_score:
        actual_h = 1.0
    elif home_score == away_score:
        actual_h = 0.5
    else:
        actual_h = 0.0

    gd = abs(home_score - away_score)
    gd_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)

    k = get_k_factor(tournament)
    elo[home] = elo_h + k * gd_mult * (actual_h - exp_h)
    elo[away] = elo_a + k * gd_mult * ((1 - actual_h) - (1 - exp_h))


def _update_form(team, goals_for, goals_against, form_results, form_gf, form_gc):
    """Update rolling form stats for a team."""
    if team not in form_results:
        form_results[team] = []
        form_gf[team] = []
        form_gc[team] = []

    # Points: W=3, D=1, L=0
    if goals_for > goals_against:
        form_results[team].append(3)
    elif goals_for == goals_against:
        form_results[team].append(1)
    else:
        form_results[team].append(0)

    form_gf[team].append(goals_for)
    form_gc[team].append(goals_against)

    # Keep last 20 (we only use 10 but keep extra for safety)
    if len(form_results[team]) > 20:
        form_results[team] = form_results[team][-20:]
        form_gf[team] = form_gf[team][-20:]
        form_gc[team] = form_gc[team][-20:]


def _update_h2h(home, away, home_score, away_score, h2h_wins):
    """Update head-to-head record."""
    key = tuple(sorted([home, away]))
    if key not in h2h_wins:
        h2h_wins[key] = {"matches": 0, home: 0, away: 0, "draws": 0, "gd": {home: 0, away: 0}}

    # Ensure both teams exist in the dict
    if home not in h2h_wins[key]:
        h2h_wins[key][home] = 0
    if away not in h2h_wins[key]:
        h2h_wins[key][away] = 0
    if home not in h2h_wins[key]["gd"]:
        h2h_wins[key]["gd"][home] = 0
    if away not in h2h_wins[key]["gd"]:
        h2h_wins[key]["gd"][away] = 0

    h2h_wins[key]["matches"] += 1
    h2h_wins[key]["gd"][home] += home_score - away_score
    h2h_wins[key]["gd"][away] += away_score - home_score

    if home_score > away_score:
        h2h_wins[key][home] += 1
    elif away_score > home_score:
        h2h_wins[key][away] += 1
    else:
        h2h_wins[key]["draws"] += 1
