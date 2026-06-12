"""Prediction engine wrapper for the Streamlit app."""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import _snapshot_features

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "model.pkl"


def load_model():
    """Load model into session_state, replay saved results to update Elo/form."""
    if "_model_state" not in st.session_state:
        if not MODEL_PATH.exists():
            st.session_state["_model_state"] = None
        else:
            with open(MODEL_PATH, "rb") as f:
                st.session_state["_model_state"] = pickle.load(f)
            # Replay any saved match results into the model
            _replay_saved_results()
    return st.session_state["_model_state"]


def _replay_saved_results():
    """Replay saved match results into model to update Elo/form/H2H."""
    import polars as pl
    from app.config import DATA_DIR

    model_state = st.session_state.get("_model_state")
    if model_state is None:
        return

    results = st.session_state.get("match_results", {})
    if not results:
        return

    # Load schedule to get team names for each match number
    schedule = pl.read_csv(str(DATA_DIR / "raw" / "wc2026_schedule.csv"))
    match_lookup = {
        row["Match_Number"]: (row["Home_Team"], row["Away_Team"])
        for row in schedule.iter_rows(named=True)
    }

    # Replay in order
    for mn in sorted(results.keys()):
        if mn not in match_lookup:
            continue
        home, away = match_lookup[mn]
        res = results[mn]
        update_model_with_result(home, away, res["home_goals"], res["away_goals"])


def update_model_with_result(
    home: str, away: str, home_goals: int, away_goals: int
):
    """
    Update the model's Elo, form, and H2H state with a real match result.
    Called every time a user saves a score. Instant (microseconds).
    """
    model_state = st.session_state.get("_model_state")
    if model_state is None:
        return

    from src.features.builder import _update_elo, _update_form, _update_h2h

    TEAM_ALIASES_REV = {"Curaçao": "Curaçao", "Curacao": "Curaçao"}
    home_key = TEAM_ALIASES_REV.get(home, home)
    away_key = TEAM_ALIASES_REV.get(away, away)

    elo = model_state["elo"]
    form_results = model_state["form_results"]
    form_gf = model_state["form_gf"]
    form_gc = model_state["form_gc"]
    h2h_wins = model_state["h2h_wins"]

    # Update Elo (World Cup K-factor = 60)
    _update_elo(
        home_key, away_key, home_goals, away_goals,
        "FIFA World Cup", True, elo
    )

    # Update form
    _update_form(home_key, home_goals, away_goals, form_results, form_gf, form_gc)
    _update_form(away_key, away_goals, home_goals, form_results, form_gf, form_gc)

    # Update H2H
    _update_h2h(home_key, away_key, home_goals, away_goals, h2h_wins)


def predict_match(
    home: str,
    away: str,
    model_state: dict,
    home_unavailable_count: int = 0,
    away_unavailable_count: int = 0,
    in_tournament_form_home: float = None,
    in_tournament_form_away: float = None,
    wc_pedigree_diff: int = 0,
    rest_days_home: int = None,
    rest_days_away: int = None,
) -> dict:
    """Generate prediction using the multi-layer model."""
    if model_state is None:
        return None

    # Normalize team names for model lookup
    TEAM_ALIASES = {
        "Curacao": "Curaçao",
        "Ivory Coast": "Côte d'Ivoire",
    }
    home_lookup = TEAM_ALIASES.get(home, home)
    away_lookup = TEAM_ALIASES.get(away, away)

    model = model_state["model"]
    elo = model_state["elo"]
    form_results = model_state["form_results"]
    form_gf = model_state["form_gf"]
    form_gc = model_state["form_gc"]
    h2h_wins = model_state["h2h_wins"]

    # Check if teams exist in model data (try both names)
    dc_model = model_state.get("dixon_coles")
    dc_teams = set(getattr(dc_model, 'attack', {}).keys()) if dc_model else set()

    if home_lookup not in elo and home_lookup not in dc_teams:
        if home not in elo and home not in dc_teams:
            return None
        home_lookup = home
    if away_lookup not in elo and away_lookup not in dc_teams:
        if away not in elo and away not in dc_teams:
            return None
        away_lookup = away

    # Layer 1: Base prediction (Ensemble classifier)
    feats = _snapshot_features(
        home_lookup, away_lookup, elo, form_results, form_gf, form_gc, h2h_wins
    )
    X = np.array([feats], dtype=np.float32)
    probs = model.predict_proba(X)[0]
    p_loss, p_draw, p_win = float(probs[0]), float(probs[1]), float(probs[2])

    # Dixon-Coles goal model (if available)
    dc_model = model_state.get("dixon_coles")
    if dc_model and home_lookup in getattr(dc_model, 'attack', {}) and away_lookup in getattr(dc_model, 'attack', {}):
        dc_pred = dc_model.predict_score_probs(home_lookup, away_lookup, neutral=True)
        xg_home = dc_pred["xg_home"]
        xg_away = dc_pred["xg_away"]
        # Blend ensemble W/D/L with Dixon-Coles W/D/L (50/50)
        p_win = 0.5 * p_win + 0.5 * dc_pred["p_home_win"]
        p_draw = 0.5 * p_draw + 0.5 * dc_pred["p_draw"]
        p_loss = 0.5 * p_loss + 0.5 * dc_pred["p_away_win"]
        total = p_win + p_draw + p_loss
        p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total
    else:
        # Fallback to simple strength model
        attack_strength = model_state.get("attack_strength", {})
        defence_strength = model_state.get("defence_strength", {})
        league_avg = model_state.get("league_avg_goals", 1.36)
        atk_h = attack_strength.get(home_lookup, 1.0)
        def_h = defence_strength.get(home_lookup, 1.0)
        atk_a = attack_strength.get(away_lookup, 1.0)
        def_a = defence_strength.get(away_lookup, 1.0)
        xg_home = atk_h * def_a * league_avg
        xg_away = atk_a * def_h * league_avg

    # Layer 2: Tournament adjustments
    adjustment = wc_pedigree_diff * 0.02

    # Rest days factor
    if rest_days_home is not None and rest_days_away is not None:
        if rest_days_home < 3 and rest_days_away >= 3:
            adjustment -= 0.04
            xg_home *= 0.92
        elif rest_days_away < 3 and rest_days_home >= 3:
            adjustment += 0.04
            xg_away *= 0.92
        elif rest_days_home < rest_days_away:
            adjustment -= 0.02
        elif rest_days_home > rest_days_away:
            adjustment += 0.02

    p_win = min(0.95, max(0.02, p_win + adjustment))
    p_loss = min(0.95, max(0.02, p_loss - adjustment))
    total = p_win + p_draw + p_loss
    p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total

    # Layer 3: Live adjustments
    squad_mod_home = max(0.7, 1.0 - home_unavailable_count * 0.02)
    squad_mod_away = max(0.7, 1.0 - away_unavailable_count * 0.02)
    xg_home *= squad_mod_home
    xg_away *= squad_mod_away

    if in_tournament_form_home is not None:
        xg_home *= 1 + (in_tournament_form_home - 0.5) * 0.15
    if in_tournament_form_away is not None:
        xg_away *= 1 + (in_tournament_form_away - 0.5) * 0.15

    # Blend XGBoost + Poisson
    p_win_p, p_draw_p, p_loss_p = _poisson_probs(xg_home, xg_away)
    p_win_f = 0.6 * p_win + 0.4 * p_win_p
    p_draw_f = 0.6 * p_draw + 0.4 * p_draw_p
    p_loss_f = 0.6 * p_loss + 0.4 * p_loss_p
    total = p_win_f + p_draw_f + p_loss_f
    p_win_f /= total
    p_draw_f /= total
    p_loss_f /= total

    # Most likely score consistent with the predicted outcome
    best_score, best_p = (1, 0), 0
    if p_win_f >= p_draw_f and p_win_f >= p_loss_f:
        # Home win — find most likely home win scoreline
        for h in range(1, 6):
            for a in range(h):
                p = poisson.pmf(h, max(xg_home, 0.1)) * poisson.pmf(a, max(xg_away, 0.1))
                if p > best_p:
                    best_p = p
                    best_score = (h, a)
    elif p_loss_f >= p_draw_f:
        # Away win
        for a in range(1, 6):
            for h in range(a):
                p = poisson.pmf(h, max(xg_home, 0.1)) * poisson.pmf(a, max(xg_away, 0.1))
                if p > best_p:
                    best_p = p
                    best_score = (h, a)
    else:
        # Draw
        for h in range(6):
            p = poisson.pmf(h, max(xg_home, 0.1)) * poisson.pmf(h, max(xg_away, 0.1))
            if p > best_p:
                best_p = p
                best_score = (h, h)

    return {
        "p_home_win": round(p_win_f, 3),
        "p_draw": round(p_draw_f, 3),
        "p_away_win": round(p_loss_f, 3),
        "xg_home": round(xg_home, 2),
        "xg_away": round(xg_away, 2),
        "likely_home": best_score[0],
        "likely_away": best_score[1],
        "elo_home": round(feats[0], 0),
        "elo_away": round(feats[1], 0),
    }


def simulate_tournament(group_df: pl.DataFrame, knockout_df: pl.DataFrame, model_state: dict) -> dict:
    """
    Simulate the entire tournament. Returns predicted results for all matches,
    standings, qualified teams, and knockout bracket.
    """
    from app.config import GROUP_NAMES
    from app.standings import compute_standings, get_qualified_teams

    # Predict all group matches
    predicted_results = {}
    for row in group_df.sort("Date", "UTC_Time").iter_rows(named=True):
        mn = row["Match_Number"]
        home = row["Home_Team"]
        away = row["Away_Team"]
        pred = predict_match(home, away, model_state)
        if pred:
            predicted_results[mn] = {
                "home_goals": pred["likely_home"],
                "away_goals": pred["likely_away"],
            }
        else:
            predicted_results[mn] = {"home_goals": 0, "away_goals": 0}

    # Compute standings
    standings = compute_standings(group_df, predicted_results)
    qualified = get_qualified_teams(standings)

    # Build 3rd-place assignment map based on FIFA rules
    third_teams = {t["Team"]: t for t in qualified["third_place"]}
    third_team_groups = {}
    for t in qualified["third_place"]:
        for group in GROUP_NAMES:
            gs = standings.filter(
                (pl.col("Group") == group) & (pl.col("Team") == t["Team"])
            )
            if len(gs) > 0:
                third_team_groups[t["Team"]] = group[-1]
    assigned_third = set()

    # Resolve knockout matches round by round
    ko_results = {}
    team_last_match_date = {}

    # First populate group match dates for rest day calc
    for row in group_df.iter_rows(named=True):
        home = row["Home_Team"]
        away = row["Away_Team"]
        match_date = row["Date"]
        team_last_match_date[home] = match_date
        team_last_match_date[away] = match_date

    round_order = [
        "Round of 32", "Round of 16", "Quarterfinal",
        "Semifinal", "Third-place Match", "Final",
    ]

    for rnd in round_order:
        rnd_df = knockout_df.filter(pl.col("Group_Stage") == rnd).sort("Date", "UTC_Time")
        if len(rnd_df) == 0:
            continue

        for row in rnd_df.iter_rows(named=True):
            mn = row["Match_Number"]
            match_date = row["Date"]
            home = _resolve_team(row["Home_Team"], qualified, ko_results, third_teams, third_team_groups, assigned_third)
            away = _resolve_team(row["Away_Team"], qualified, ko_results, third_teams, third_team_groups, assigned_third)

            # Check if resolved
            if _is_placeholder(home) or _is_placeholder(away):
                ko_results[mn] = {"winner": home, "loser": away, "pred": None}
                continue

            # Rest days
            rest_h = _days_between(team_last_match_date.get(home), match_date)
            rest_a = _days_between(team_last_match_date.get(away), match_date)

            pred = predict_match(
                home, away, model_state,
                rest_days_home=rest_h,
                rest_days_away=rest_a,
            )

            if pred is None:
                ko_results[mn] = {"winner": home, "loser": away, "pred": None}
            else:
                if pred["p_home_win"] >= pred["p_away_win"]:
                    winner, loser = home, away
                else:
                    winner, loser = away, home

                ko_results[mn] = {"winner": winner, "loser": loser, "pred": pred}

            # Update last match dates
            team_last_match_date[home] = match_date
            team_last_match_date[away] = match_date

    return {
        "predicted_results": predicted_results,
        "standings": standings,
        "qualified": qualified,
        "ko_results": ko_results,
    }


def compute_predicted_standings(group_df: pl.DataFrame, model_state: dict) -> pl.DataFrame:
    """Compute standings from model-predicted group results."""
    from app.standings import compute_standings

    predicted_results = {}
    for row in group_df.sort("Date", "UTC_Time").iter_rows(named=True):
        mn = row["Match_Number"]
        home = row["Home_Team"]
        away = row["Away_Team"]
        pred = predict_match(home, away, model_state)
        if pred:
            predicted_results[mn] = {
                "home_goals": pred["likely_home"],
                "away_goals": pred["likely_away"],
            }
        else:
            predicted_results[mn] = {"home_goals": 0, "away_goals": 0}

    return compute_standings(group_df, predicted_results)


def compute_tournament_form(match_results: dict, team: str, schedule_df) -> float:
    """Compute in-tournament form. Returns 0-1 (0.5 = neutral)."""
    team_matches = schedule_df.filter(
        (pl.col("Home_Team") == team) | (pl.col("Away_Team") == team)
    )
    points = 0
    played = 0
    for row in team_matches.iter_rows(named=True):
        mn = row["Match_Number"]
        if mn not in match_results:
            continue
        res = match_results[mn]
        is_home = row["Home_Team"] == team
        team_goals = res["home_goals"] if is_home else res["away_goals"]
        opp_goals = res["away_goals"] if is_home else res["home_goals"]
        if team_goals > opp_goals:
            points += 3
        elif team_goals == opp_goals:
            points += 1
        played += 1
    if played == 0:
        return None
    return points / (played * 3)


def _resolve_team(placeholder, qualified, ko_results, third_teams, third_team_groups, assigned_third):
    """Resolve a knockout placeholder to a real team name."""
    if not _is_placeholder(placeholder):
        return placeholder

    # "Winner X" where X is a group letter
    if placeholder.startswith("Winner "):
        ref = placeholder.replace("Winner ", "")
        if len(ref) == 1:
            return qualified["winners"].get(f"Group {ref}", placeholder)
        if ref.isdigit():
            ref_mn = int(ref)
            if ref_mn in ko_results:
                return ko_results[ref_mn]["winner"]
        return placeholder

    # "Runner-up X"
    if placeholder.startswith("Runner-up "):
        ref = placeholder.replace("Runner-up ", "")
        if len(ref) == 1:
            return qualified["runners_up"].get(f"Group {ref}", placeholder)
        return placeholder

    # "3rd X/Y/Z/..." — pick the best available 3rd place team from those groups
    if placeholder.startswith("3rd "):
        eligible_groups = placeholder.replace("3rd ", "").split("/")
        # Find best unassigned 3rd-place team from eligible groups
        candidates = [
            (team, group_letter)
            for team, group_letter in third_team_groups.items()
            if group_letter in eligible_groups and team not in assigned_third
        ]
        candidates.sort(
            key=lambda x: -(third_teams.get(x[0], {}).get("Pts", 0))
        )
        if candidates:
            chosen = candidates[0][0]
            assigned_third.add(chosen)
            return chosen
        return placeholder

    # "Loser X"
    if placeholder.startswith("Loser "):
        ref = placeholder.replace("Loser ", "")
        if ref.isdigit():
            ref_mn = int(ref)
            if ref_mn in ko_results:
                return ko_results[ref_mn]["loser"]
        return placeholder

    return placeholder


def _is_placeholder(team: str) -> bool:
    return any(x in team for x in ["Winner", "Runner", "3rd", "Loser"])


def _days_between(date_str_a, date_str_b) -> int:
    if not date_str_a or not date_str_b:
        return 4
    try:
        a = datetime.strptime(date_str_a, "%Y-%m-%d")
        b = datetime.strptime(date_str_b, "%Y-%m-%d")
        return abs((b - a).days)
    except (ValueError, TypeError):
        return 4


def _poisson_probs(xg_home: float, xg_away: float) -> tuple:
    p_win, p_draw, p_loss = 0.0, 0.0, 0.0
    xg_h = max(xg_home, 0.1)
    xg_a = max(xg_away, 0.1)
    for h in range(7):
        for a in range(7):
            p = poisson.pmf(h, xg_h) * poisson.pmf(a, xg_a)
            if h > a:
                p_win += p
            elif h == a:
                p_draw += p
            else:
                p_loss += p
    return p_win, p_draw, p_loss
