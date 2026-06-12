import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import streamlit as st

from app.components import (
    paginate_matches,
    render_match_row,
    render_standings_table,
)
from app.config import GROUP_NAMES, flag_img
from app.data import (
    get_group_teams,
    init_results,
    load_schedule,
    process_schedule,
)
from app.standings import (
    compute_standings,
    get_qualified_teams,
    resolve_knockout_team,
)

st.set_page_config(
    page_title="FIFA World Cup 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""<style>
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px; }
h1, h2, h3 { color: #1B4F72; }

.match-row {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; margin: 2px 0;
    background: #FFFFFF; border: 1px solid #D6EAF8;
    border-radius: 8px; font-size: 0.85rem;
}
.match-row:hover { background: #EBF5FB; border-color: #AED6F1; }
.match-home { flex: 1; text-align: right; font-weight: 500; display: flex; align-items: center; justify-content: flex-end; gap: 6px; }
.match-away { flex: 1; text-align: left; font-weight: 500; display: flex; align-items: center; gap: 6px; }
.match-vs { color: #1B4F72; font-weight: 700; font-size: 0.75rem; padding: 2px 8px; background: #D6EAF8; border-radius: 10px; }
.match-info { font-size: 0.72rem; color: #5D6D7E; display: flex; gap: 10px; align-items: center; margin-left: auto; white-space: nowrap; }
.match-num { font-size: 0.7rem; color: #85929E; min-width: 20px; }

.group-chip { display: inline-flex; align-items: center; gap: 4px; background: #EAF2F8; padding: 3px 8px; border-radius: 6px; margin: 2px; font-size: 0.78rem; color: #1C2833; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #1B4F72; margin: 0.8rem 0 0.4rem 0; }
.standings-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.standings-table th { background: #1B4F72; color: white; padding: 4px 8px; text-align: left; }
.standings-table td { padding: 4px 8px; border-bottom: 1px solid #EAF2F8; }
.standings-table tr:hover { background: #EBF5FB; }
.qualified { background: #D5F5E3 !important; }

.page-nav { display: flex; justify-content: center; align-items: center; gap: 12px; margin: 8px 0; font-size: 0.8rem; color: #5D6D7E; }
.ko-round-header { font-size: 0.95rem; font-weight: 600; color: #1B4F72; margin: 1rem 0 0.3rem 0; padding-bottom: 4px; border-bottom: 2px solid #D6EAF8; }

div[data-testid="stExpander"] details summary p { font-size: 0.82rem; }
div[data-testid="stExpander"] .stCheckbox label p { font-size: 0.78rem; margin: 0; }
div[data-testid="stExpander"] .stCheckbox { margin-bottom: -8px; }

.pred-card {
    border: 1px solid #D6EAF8; border-radius: 10px; padding: 12px;
    margin: 6px 0; background: #FAFCFE;
}
.pred-teams { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.pred-team { font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 6px; }
.pred-bar { display: flex; height: 6px; border-radius: 3px; overflow: hidden; margin: 4px 0; }
.pred-bar-win { background: #27AE60; }
.pred-bar-draw { background: #F39C12; }
.pred-bar-loss { background: #E74C3C; }
.pred-score { text-align: center; font-size: 1.2rem; font-weight: 700; color: #1B4F72; }
.pred-xg { text-align: center; font-size: 0.75rem; color: #5D6D7E; }
.pred-probs { display: flex; justify-content: space-between; font-size: 0.72rem; color: #5D6D7E; margin-top: 4px; }

.bracket-match {
    border: 1px solid #D6EAF8; border-radius: 8px; padding: 8px 10px;
    margin: 4px 0; background: white; font-size: 0.8rem; min-width: 180px;
}
.bracket-team { display: flex; align-items: center; gap: 5px; padding: 2px 0; }
.bracket-team-name { font-weight: 500; }
.bracket-score { margin-left: auto; font-weight: 700; color: #1B4F72; }
.bracket-connector { border-left: 2px solid #D6EAF8; height: 40px; margin-left: 50%; }
</style>""", unsafe_allow_html=True)


def _retrain_model():
    """Run the full training pipeline and reload the model."""
    import subprocess
    with st.spinner("Retraining model (ensemble + Dixon-Coles)... ~90 seconds"):
        result = subprocess.run(
            ["uv", "run", "python", "src/models/train.py"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            # Clear cached model so it reloads from the new pkl
            if "_model_state" in st.session_state:
                del st.session_state["_model_state"]
            st.success("Model retrained successfully!")
            st.code(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            st.rerun()
        else:
            st.error("Training failed!")
            st.code(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)


def main():
    init_results()

    df = load_schedule()
    df = process_schedule(df)

    # Sidebar navigation + filters
    with st.sidebar:
        st.markdown("## ⚽ World Cup 2026")
        st.caption("USA • Mexico • Canada")
        st.markdown("---")

        if "page" not in st.session_state:
            st.session_state["page"] = "⚽ Group Stage"

        nav_items = [
            ("⚽ Group Stage", "nav_gs"),
            ("🏆 Knockout Stage", "nav_ko"),
            ("🔮 Predictions (Groups)", "nav_pg"),
            ("🔮 Predictions (Knockout)", "nav_pk"),
            ("📊 Methodology", "nav_method"),
        ]
        for label, key in nav_items:
            is_active = st.session_state["page"] == label
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=key, type=btn_type, use_container_width=True):
                st.session_state["page"] = label
                st.rerun()

        page = st.session_state["page"]

        st.markdown("---")
        st.markdown("**Filters**")
        all_dates = sorted(set(df["EST_Date"].to_list()))
        min_dt = date.fromisoformat(all_dates[0])
        max_dt = date.fromisoformat(all_dates[-1])
        date_range = st.date_input(
            "Date Range", value=(min_dt, max_dt),
            min_value=min_dt, max_value=max_dt,
        )
        all_teams = sorted(
            set(df["Home_Team"].to_list() + df["Away_Team"].to_list())
        )
        selected_teams = st.multiselect("Team", all_teams)

        st.markdown("---")
        if st.button("🔄 Retrain Model", use_container_width=True):
            _retrain_model()

    group_df = df.filter(pl.col("Group_Stage").str.starts_with("Group"))
    knockout_df = df.filter(~pl.col("Group_Stage").str.starts_with("Group"))

    if page == "⚽ Group Stage":
        render_group_stage(group_df, date_range, selected_teams)
    elif page == "🏆 Knockout Stage":
        render_knockout_stage(group_df, knockout_df, date_range)
    elif page == "🔮 Predictions (Groups)":
        render_predictions_groups(group_df)
    elif page == "🔮 Predictions (Knockout)":
        render_predictions_knockout(group_df, knockout_df)
    elif page == "📊 Methodology":
        from app.methodology import render_methodology
        render_methodology()


def render_group_stage(group_df, date_range, selected_teams):

    filtered = group_df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered.filter(
            (pl.col("EST_Date") >= date_range[0].isoformat())
            & (pl.col("EST_Date") <= date_range[1].isoformat())
        )
    if selected_teams:
        filtered = filtered.filter(
            pl.col("Home_Team").is_in(selected_teams)
            | pl.col("Away_Team").is_in(selected_teams)
        )

    view_col, standings_col = st.columns([3, 2])

    with view_col:
        st.markdown(
            f'<div class="section-title">Matches ({len(filtered)})</div>',
            unsafe_allow_html=True,
        )
        group_view = st.radio(
            "View", ["All Matches", "By Group"], horizontal=True,
            key="gv", label_visibility="collapsed",
        )
        if group_view == "All Matches":
            sorted_groups = filtered.sort("EST_Date", "EST_Time")
            paginate_matches(sorted_groups, "grp_all", interactive=True)
        else:
            groups = filtered["Group_Stage"].unique().sort().to_list()
            if groups:
                selected_group = st.selectbox("Select Group", groups, key="sel_grp")
                g_df = filtered.filter(pl.col("Group_Stage") == selected_group)
                g_df = g_df.sort("Date", "UTC_Time")
                for row in g_df.iter_rows(named=True):
                    render_match_row(row, interactive=True)

    with standings_col:
        st.markdown('<div class="section-title">Standings</div>', unsafe_allow_html=True)
        if st.session_state.match_results:
            standings = compute_standings(group_df, st.session_state.match_results)
            qualified = get_qualified_teams(standings)
            for group in GROUP_NAMES:
                with st.expander(group, expanded=False):
                    render_standings_table(standings, group, qualified)
            st.markdown(f"*{len(st.session_state.match_results)} match(es) recorded*")
        else:
            st.caption("Enter match scores to see standings update live.")
            for group in GROUP_NAMES:
                teams = get_group_teams(group_df).get(group, [])
                chips = " ".join(
                    f'<span class="group-chip">{flag_img(t, 14)} {t}</span>' for t in teams
                )
                st.markdown(f'**{group}:** {chips}', unsafe_allow_html=True)


def render_knockout_stage(group_df, knockout_df, date_range):
    from app.bracket import render_bracket_tree
    from app.predictions import load_model, simulate_tournament

    # Build ko_results with ONLY user-entered data (no predictions)
    # Use simulation only to resolve team names (who plays who)
    model_state = load_model()
    ko_results = {}

    if model_state:
        sim = simulate_tournament(group_df, knockout_df, model_state)
        # Take team assignments from simulation but clear predicted scores
        for mn, kr in sim["ko_results"].items():
            actual = st.session_state.match_results.get(mn)
            if actual:
                # User entered a real score — determine winner from that
                home = kr.get("winner", "TBD")
                away = kr.get("loser", "TBD")
                if kr.get("pred") and kr["pred"]["p_home_win"] < kr["pred"]["p_away_win"]:
                    home, away = away, home
                if actual["home_goals"] > actual["away_goals"]:
                    ko_results[mn] = {"winner": home, "loser": away, "pred": None}
                elif actual["away_goals"] > actual["home_goals"]:
                    ko_results[mn] = {"winner": away, "loser": home, "pred": None}
                else:
                    ko_results[mn] = {"winner": home, "loser": away, "pred": None}
            else:
                # No user score — show teams but no predicted winner/score
                ko_results[mn] = {
                    "winner": kr.get("winner", "TBD"),
                    "loser": kr.get("loser", "TBD"),
                    "pred": None,
                }
    elif st.session_state.match_results:
        standings = compute_standings(group_df, st.session_state.match_results)
        qualified = get_qualified_teams(standings)
        for row in knockout_df.iter_rows(named=True):
            mn = row["Match_Number"]
            home = resolve_knockout_team(row["Home_Team"], qualified)
            away = resolve_knockout_team(row["Away_Team"], qualified)
            actual = st.session_state.match_results.get(mn)
            if actual:
                if actual["home_goals"] > actual["away_goals"]:
                    ko_results[mn] = {"winner": home, "loser": away, "pred": None}
                else:
                    ko_results[mn] = {"winner": away, "loser": home, "pred": None}
            else:
                ko_results[mn] = {"winner": home, "loser": away, "pred": None}
    else:
        st.info("Enter group stage scores or train the model to see the bracket.")
        return

    # Bracket with inline score entry — no predictions shown
    render_bracket_tree(ko_results, show_predictions=False, editable=True)


def render_predictions_groups(group_df):
    from app.predictions import (
        compute_predicted_standings,
        compute_tournament_form,
        load_model,
        predict_match,
    )

    st.markdown("## 🔮 Group Stage Predictions")

    model_state = load_model()
    if model_state is None:
        st.error("Model not trained yet. Run `uv run python src/models/train.py` first.")
        return

    st.caption("Predictions recalculate live when you update scores or squad availability.")

    group_list = group_df["Group_Stage"].unique().sort().to_list()
    selected = st.selectbox("Select Group", group_list, key="pred_grp")

    g_df = group_df.filter(pl.col("Group_Stage") == selected).sort("Date", "UTC_Time")

    # Show predicted standings for this group
    pred_standings = compute_predicted_standings(group_df, model_state)
    group_standings = pred_standings.filter(pl.col("Group") == selected)

    st.markdown(f'<div class="section-title">Predicted Standings — {selected}</div>', unsafe_allow_html=True)
    rows_html = ""
    for i, row in enumerate(group_standings.iter_rows(named=True)):
        team = row["Team"]
        fl = flag_img(team, 14)
        rows_html += (
            f'<tr><td>{i+1}</td><td>{fl} {team}</td>'
            f'<td>{row["P"]}</td><td>{row["W"]}</td><td>{row["D"]}</td><td>{row["L"]}</td>'
            f'<td>{row["GF"]}</td><td>{row["GA"]}</td><td>{row["GD"]}</td>'
            f'<td><b>{row["Pts"]}</b></td></tr>'
        )
    st.markdown(
        f'<table class="standings-table"><thead><tr>'
        f'<th>#</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th>'
        f'<th>GF</th><th>GA</th><th>GD</th><th>Pts</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="section-title">Match Predictions</div>', unsafe_allow_html=True)

    for row in g_df.iter_rows(named=True):
        mn = row["Match_Number"]
        home = row["Home_Team"]
        away = row["Away_Team"]

        home_unavail = len(st.session_state.unavailable.get((mn, home), set()))
        away_unavail = len(st.session_state.unavailable.get((mn, away), set()))
        form_h = compute_tournament_form(st.session_state.match_results, home, group_df)
        form_a = compute_tournament_form(st.session_state.match_results, away, group_df)

        pred = predict_match(
            home, away, model_state,
            home_unavailable_count=home_unavail,
            away_unavailable_count=away_unavail,
            in_tournament_form_home=form_h,
            in_tournament_form_away=form_a,
        )

        if pred:
            render_prediction_card(row, pred)


def render_predictions_knockout(group_df, knockout_df):
    from app.bracket import render_bracket_tree
    from app.predictions import load_model, simulate_tournament

    st.markdown("## 🔮 Knockout Stage Predictions")

    model_state = load_model()
    if model_state is None:
        st.error("Model not trained yet. Run `uv run python src/models/train.py` first.")
        return

    st.caption("Full tournament simulation — resolves all matchups including 3rd-place qualifiers.")

    sim = simulate_tournament(group_df, knockout_df, model_state)
    ko_results = sim["ko_results"]

    # Show champion
    final_mn = knockout_df.filter(pl.col("Group_Stage") == "Final")
    if len(final_mn) > 0:
        final_id = final_mn.row(0, named=True)["Match_Number"]
        if final_id in ko_results and ko_results[final_id].get("pred"):
            champion = ko_results[final_id]["winner"]
            st.markdown(
                f'<div style="text-align:center;padding:12px;background:#D5F5E3;'
                f'border-radius:10px;margin-bottom:16px;">'
                f'<span style="font-size:1.3rem;">🏆</span> '
                f'<span style="font-size:1.1rem;font-weight:700;color:#1B4F72;">'
                f'{flag_img(champion, 20)} {champion}</span>'
                f'<span style="color:#5D6D7E;font-size:0.85rem;"> predicted champion</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    render_bracket_tree(ko_results, show_predictions=True)


def render_prediction_card(row: dict, pred: dict):
    """Render a full prediction card for a match."""
    home = row["Home_Team"]
    away = row["Away_Team"]
    hf = flag_img(home, 20)
    af = flag_img(away, 20)

    p_w = pred["p_home_win"] * 100
    p_d = pred["p_draw"] * 100
    p_l = pred["p_away_win"] * 100

    # Check if actual result exists
    result = st.session_state.match_results.get(row["Match_Number"])
    actual_html = ""
    if result:
        actual_html = (
            f'<div style="text-align:center;font-size:0.75rem;color:#27AE60;font-weight:600;">'
            f'Actual: {result["home_goals"]} - {result["away_goals"]}</div>'
        )

    st.markdown(f"""<div class="pred-card">
<div class="pred-teams">
    <span class="pred-team">{hf} {home}</span>
    <span class="pred-team">{away} {af}</span>
</div>
<div class="pred-score">{pred["likely_home"]} - {pred["likely_away"]}</div>
<div class="pred-xg">xG: {pred["xg_home"]:.2f} - {pred["xg_away"]:.2f} | Elo: {pred["elo_home"]:.0f} vs {pred["elo_away"]:.0f}</div>
<div class="pred-bar">
    <div class="pred-bar-win" style="width:{p_w}%"></div>
    <div class="pred-bar-draw" style="width:{p_d}%"></div>
    <div class="pred-bar-loss" style="width:{p_l}%"></div>
</div>
<div class="pred-probs">
    <span style="color:#27AE60">{p_w:.0f}% {home}</span>
    <span style="color:#F39C12">{p_d:.0f}% Draw</span>
    <span style="color:#E74C3C">{p_l:.0f}% {away}</span>
</div>
{actual_html}
</div>""", unsafe_allow_html=True)


def render_prediction_card_compact(row: dict, pred: dict, winner: str = None):
    """Compact prediction card for bracket view."""
    home = row["Home_Team"]
    away = row["Away_Team"]
    hf = flag_img(home, 16)
    af = flag_img(away, 16)

    p_w = pred["p_home_win"] * 100
    p_l = pred["p_away_win"] * 100
    h_goals = pred["likely_home"]
    a_goals = pred["likely_away"]

    # Style winner bold + green
    h_bold = "font-weight:700;" if winner == home else ""
    a_bold = "font-weight:700;" if winner == away else ""
    h_bg = "background:#D5F5E3;" if winner == home else ""
    a_bg = "background:#FDEDEC;" if winner == away else ""

    st.markdown(f"""<div class="bracket-match">
<div class="bracket-team" style="{h_bg}padding:4px 8px;border-radius:4px;">
    {hf}<span class="bracket-team-name" style="{h_bold}">{home}</span>
    <span class="bracket-score" style="font-size:0.85rem;font-weight:700;color:#1B4F72;">{h_goals}</span>
    <span style="font-size:0.65rem;color:#5D6D7E;margin-left:4px;">({p_w:.0f}%)</span>
</div>
<div class="bracket-team" style="{a_bg}padding:4px 8px;border-radius:4px;">
    {af}<span class="bracket-team-name" style="{a_bold}">{away}</span>
    <span class="bracket-score" style="font-size:0.85rem;font-weight:700;color:#1B4F72;">{a_goals}</span>
    <span style="font-size:0.65rem;color:#5D6D7E;margin-left:4px;">({p_l:.0f}%)</span>
</div>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
