import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import streamlit as st

from app.config import DATA_DIR, GROUP_NAMES, MATCH_RESULTS_FILE, UNAVAILABLE_FILE


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def init_results():
    """Load match results and squad availability from disk into session state."""
    if "match_results" not in st.session_state:
        raw = load_json(MATCH_RESULTS_FILE)
        st.session_state.match_results = {int(k): v for k, v in raw.items()}
    if "unavailable" not in st.session_state:
        raw = load_json(UNAVAILABLE_FILE)
        st.session_state.unavailable = {
            (int(k.split("|", 1)[0]), k.split("|", 1)[1]): set(v)
            for k, v in raw.items()
            if "|" in k
        }


def persist_results():
    """Save match results to disk."""
    data = {str(k): v for k, v in st.session_state.match_results.items()}
    save_json(MATCH_RESULTS_FILE, data)


def save_match_result(match_num: int, home: str, away: str, home_goals: int, away_goals: int):
    """Save a match result AND update the live model state."""
    st.session_state.match_results[match_num] = {
        "home_goals": home_goals, "away_goals": away_goals,
    }
    persist_results()

    # Update model's Elo/form/H2H with this result
    from app.predictions import update_model_with_result
    update_model_with_result(home, away, home_goals, away_goals)


def persist_unavailable():
    """Save unavailable players to disk."""
    data = {
        f"{mn}|{team}": sorted(players)
        for (mn, team), players in st.session_state.unavailable.items()
        if players
    }
    save_json(UNAVAILABLE_FILE, data)


def parse_utc_time(date_str: str, utc_time_str: str) -> datetime:
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    if utc_time_str.endswith("+1"):
        time_part = utc_time_str.replace("+1", "").strip()
        base_date += timedelta(days=1)
    else:
        time_part = utc_time_str.strip()
    hour, minute = time_part.split(":")
    return base_date.replace(hour=int(hour), minute=int(minute))


def utc_to_est(utc_dt: datetime) -> datetime:
    return utc_dt - timedelta(hours=4)


@st.cache_data
def load_schedule() -> pl.DataFrame:
    csv_path = DATA_DIR / "raw" / "wc2026_schedule.csv"
    return pl.read_csv(str(csv_path))


@st.cache_data
def load_squads() -> pl.DataFrame:
    csv_path = DATA_DIR / "raw" / "wc2026_squads.csv"
    return pl.read_csv(str(csv_path))


def process_schedule(df: pl.DataFrame) -> pl.DataFrame:
    est_times = []
    est_dates = []
    for row in df.iter_rows(named=True):
        utc_dt = parse_utc_time(row["Date"], row["UTC_Time"])
        est_dt = utc_to_est(utc_dt)
        est_times.append(est_dt.strftime("%I:%M %p"))
        est_dates.append(est_dt.strftime("%Y-%m-%d"))
    return df.with_columns(
        pl.Series("EST_Time", est_times),
        pl.Series("EST_Date", est_dates),
    )


def get_group_teams(df: pl.DataFrame) -> dict:
    """Get teams in each group."""
    groups = {}
    for g in GROUP_NAMES:
        gdf = df.filter(pl.col("Group_Stage") == g)
        teams = sorted(
            set(gdf["Home_Team"].to_list() + gdf["Away_Team"].to_list())
        )
        groups[g] = teams
    return groups
