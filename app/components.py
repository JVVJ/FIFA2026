import polars as pl
import streamlit as st

from app.config import (
    MATCHES_PER_PAGE,
    flag_img,
    normalize_team_name,
)
from app.data import load_squads, persist_results, persist_unavailable, save_match_result


def render_match_row(row: dict, interactive: bool = False):
    """Render a compact match row."""
    mn = row["Match_Number"]
    home = row["Home_Team"]
    away = row["Away_Team"]
    hf = flag_img(home, 18)
    af = flag_img(away, 18)

    result = st.session_state.match_results.get(mn)
    score_display = ""
    if result:
        score_display = (
            f' <span style="font-weight:700;color:#1B4F72;">'
            f'{result["home_goals"]} - {result["away_goals"]}</span>'
        )

    html = (
        f'<div class="match-row">'
        f'<span class="match-num">M{mn}</span>'
        f'<span class="match-home">{home} {hf}</span>'
        f'<span class="match-vs">'
        f'{score_display if score_display else "vs"}</span>'
        f'<span class="match-away">{af} {away}</span>'
        f'<span class="match-info">'
        f'<span>⏰ {row["EST_Time"]}</span>'
        f'<span>📅 {row["EST_Date"]}</span>'
        f'<span>🏟️ {row["Venue"]}</span>'
        f'</span></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    if interactive:
        render_score_input(mn, home, away)


def render_score_input(match_num: int, home: str, away: str):
    """Render inline score input for a match."""
    _, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    current = st.session_state.match_results.get(match_num)

    with col2:
        hg = st.number_input(
            f"{home}", min_value=0, max_value=20,
            value=current["home_goals"] if current else 0,
            key=f"hg_{match_num}", label_visibility="collapsed",
        )
    with col3:
        ag = st.number_input(
            f"{away}", min_value=0, max_value=20,
            value=current["away_goals"] if current else 0,
            key=f"ag_{match_num}", label_visibility="collapsed",
        )
    with col4:
        if st.button("✓", key=f"save_{match_num}", type="primary"):
            save_match_result(match_num, home, away, hg, ag)
            st.rerun()
    with col5:
        home_unavail = st.session_state.unavailable.get(
            (match_num, home), set()
        )
        away_unavail = st.session_state.unavailable.get(
            (match_num, away), set()
        )
        unavail_count = len(home_unavail) + len(away_unavail)
        squad_label = f"📋 {unavail_count}" if unavail_count else "📋"
        if st.button(squad_label, key=f"squad_btn_{match_num}"):
            st.session_state["_squad_dialog_match"] = match_num
            st.session_state["_squad_dialog_home"] = home
            st.session_state["_squad_dialog_away"] = away
            open_squad_dialog()


@st.dialog("Squad Availability", width="large")
def open_squad_dialog():
    """Modal dialog for managing squad availability."""
    match_num = st.session_state.get("_squad_dialog_match")
    home = st.session_state.get("_squad_dialog_home")
    away = st.session_state.get("_squad_dialog_away")

    if not match_num or not home or not away:
        st.error("No match selected.")
        return

    squads = load_squads()
    home_norm = normalize_team_name(home)
    away_norm = normalize_team_name(away)

    home_squad = squads.filter(pl.col("team") == home_norm)
    away_squad = squads.filter(pl.col("team") == away_norm)

    st.markdown(
        f"**Match {match_num}:** {flag_img(home, 18)} {home} vs "
        f"{flag_img(away, 18)} {away}",
        unsafe_allow_html=True,
    )
    st.caption("Uncheck players who are unavailable (injured/suspended)")
    st.markdown("---")

    squad_container = st.container(height=500)
    with squad_container:
        col_home, col_away = st.columns(2)

        with col_home:
            _render_team_squad_dialog(
                match_num, home, home_squad, (match_num, home)
            )

        with col_away:
            _render_team_squad_dialog(
                match_num, away, away_squad, (match_num, away)
            )


def _render_team_squad_dialog(
    match_num: int, team: str, squad_df: pl.DataFrame, state_key: tuple
):
    """Render one team's squad checklist inside a dialog."""
    if len(squad_df) == 0:
        st.caption(f"{team} — no squad data")
        return

    fl = flag_img(team, 16)
    st.markdown(f"{fl} **{team}**", unsafe_allow_html=True)

    current_unavailable = st.session_state.unavailable.get(state_key, set())
    new_unavailable = set()

    pos_order = ["GK", "DF", "MF", "FW"]
    for pos in pos_order:
        pos_players = squad_df.filter(
            pl.col("position") == pos
        ).sort("shirt_number")
        if len(pos_players) == 0:
            continue

        st.caption(pos)
        for row in pos_players.iter_rows(named=True):
            name = row["player_name"]
            num = row["shirt_number"]
            club = row["club"]
            label = f"#{num} {name} ({club})"
            available = st.checkbox(
                label,
                value=(name not in current_unavailable),
                key=f"dlg_{match_num}_{team}_{num}",
            )
            if not available:
                new_unavailable.add(name)

    if new_unavailable != current_unavailable:
        st.session_state.unavailable[state_key] = new_unavailable
        persist_unavailable()


def render_standings_table(
    standings: pl.DataFrame, group: str, qualified_teams: dict
):
    """Render a compact standings table for a group."""
    group_standings = standings.filter(pl.col("Group") == group)

    winners = qualified_teams.get("winners", {})
    runners = qualified_teams.get("runners_up", {})
    third_qualified = [
        t["Team"] for t in qualified_teams.get("third_place", [])
    ]

    rows_html = ""
    for i, row in enumerate(group_standings.iter_rows(named=True)):
        team = row["Team"]
        fl = flag_img(team, 16)
        css_class = ""
        if (
            team == winners.get(group)
            or team == runners.get(group)
            or team in third_qualified
        ):
            css_class = ' class="qualified"'
        rows_html += (
            f'<tr{css_class}><td>{i+1}</td><td>{fl} {team}</td>'
            f'<td>{row["P"]}</td><td>{row["W"]}</td>'
            f'<td>{row["D"]}</td><td>{row["L"]}</td>'
            f'<td>{row["GF"]}</td><td>{row["GA"]}</td>'
            f'<td>{row["GD"]}</td><td><b>{row["Pts"]}</b></td></tr>'
        )

    st.markdown(
        f'<table class="standings-table"><thead><tr>'
        f'<th>#</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th>'
        f'<th>GF</th><th>GA</th><th>GD</th><th>Pts</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )


def paginate_matches(
    df: pl.DataFrame, key_prefix: str, interactive: bool = False
):
    """Paginated match list."""
    total = len(df)
    if total == 0:
        st.info("No matches found.")
        return

    total_pages = (total + MATCHES_PER_PAGE - 1) // MATCHES_PER_PAGE
    page_key = f"page_{key_prefix}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    current_page = st.session_state[page_key]

    col_prev, col_info, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button(
            "← Prev", key=f"p_{key_prefix}", disabled=current_page <= 1
        ):
            st.session_state[page_key] -= 1
            st.rerun()
    with col_info:
        start = (current_page - 1) * MATCHES_PER_PAGE + 1
        end = min(current_page * MATCHES_PER_PAGE, total)
        st.markdown(
            f'<div class="page-nav">'
            f'Page {current_page}/{total_pages} · '
            f'{start}–{end} of {total}</div>',
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button(
            "Next →", key=f"n_{key_prefix}",
            disabled=current_page >= total_pages,
        ):
            st.session_state[page_key] += 1
            st.rerun()

    page_df = df.slice((current_page - 1) * MATCHES_PER_PAGE, MATCHES_PER_PAGE)
    for row in page_df.iter_rows(named=True):
        render_match_row(row, interactive=interactive)
