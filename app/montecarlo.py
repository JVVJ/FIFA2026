"""Monte Carlo tournament simulation — run N iterations to get probabilities."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import GROUP_NAMES, flag_img
from app.predictions import load_model, predict_match


def run_monte_carlo(
    group_df: pl.DataFrame,
    knockout_df: pl.DataFrame,
    model_state: dict,
    n_simulations: int = 1000,
    progress_bar=None,
) -> dict:
    """
    Run N tournament simulations, sampling outcomes probabilistically.
    Returns dict: team -> {group_qual, r32, r16, qf, sf, final, champion} counts.
    """
    from app.standings import compute_standings, get_qualified_teams
    from app.predictions import _resolve_team, _is_placeholder

    # Get all teams
    all_teams = sorted(set(
        group_df["Home_Team"].to_list() + group_df["Away_Team"].to_list()
    ))
    team_counts = {
        team: {"group_qual": 0, "r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0}
        for team in all_teams
    }

    # Pre-compute group match predictions (these don't change between sims)
    group_preds = {}
    for row in group_df.iter_rows(named=True):
        mn = row["Match_Number"]
        home = row["Home_Team"]
        away = row["Away_Team"]
        pred = predict_match(home, away, model_state)
        if pred:
            group_preds[mn] = {
                "home": home, "away": away,
                "p_home_win": pred["p_home_win"],
                "p_draw": pred["p_draw"],
                "p_away_win": pred["p_away_win"],
                "xg_home": pred["xg_home"],
                "xg_away": pred["xg_away"],
            }
        else:
            group_preds[mn] = {
                "home": home, "away": away,
                "p_home_win": 0.4, "p_draw": 0.3, "p_away_win": 0.3,
                "xg_home": 1.0, "xg_away": 1.0,
            }

    for sim_i in range(n_simulations):
        if progress_bar and sim_i % 50 == 0:
            progress_bar.progress(sim_i / n_simulations, text=f"Simulation {sim_i}/{n_simulations}")

        # Sample group results
        sim_results = {}
        for mn, gp in group_preds.items():
            actual = st.session_state.get("match_results", {}).get(mn)
            if actual:
                sim_results[mn] = actual
            else:
                # Sample from Poisson using xG
                h_goals = np.random.poisson(max(gp["xg_home"], 0.3))
                a_goals = np.random.poisson(max(gp["xg_away"], 0.3))
                sim_results[mn] = {"home_goals": int(h_goals), "away_goals": int(a_goals)}

        # Compute standings
        standings = compute_standings(group_df, sim_results)
        qualified = get_qualified_teams(standings)

        # Track group qualification (top 2 + best 3rd)
        for group in GROUP_NAMES:
            gs = standings.filter(pl.col("Group") == group)
            if len(gs) >= 2:
                team_counts[gs.row(0, named=True)["Team"]]["group_qual"] += 1
                team_counts[gs.row(1, named=True)["Team"]]["group_qual"] += 1
        for t in qualified["third_place"]:
            if t["Team"] in team_counts:
                team_counts[t["Team"]]["group_qual"] += 1

        # Simulate knockout
        third_teams = {t["Team"]: t for t in qualified["third_place"]}
        third_team_groups = {}
        for t in qualified["third_place"]:
            for group in GROUP_NAMES:
                gs_check = standings.filter(
                    (pl.col("Group") == group) & (pl.col("Team") == t["Team"])
                )
                if len(gs_check) > 0:
                    third_team_groups[t["Team"]] = group[-1]
        assigned_third = set()

        ko_results = {}
        round_order = [
            "Round of 32", "Round of 16", "Quarterfinal",
            "Semifinal", "Third-place Match", "Final",
        ]

        round_stage_map = {
            "Round of 32": "r32",
            "Round of 16": "r16",
            "Quarterfinal": "qf",
            "Semifinal": "sf",
            "Final": "final",
        }

        for rnd in round_order:
            rnd_df = knockout_df.filter(pl.col("Group_Stage") == rnd).sort("Date", "UTC_Time")
            if len(rnd_df) == 0:
                continue

            for row in rnd_df.iter_rows(named=True):
                mn = row["Match_Number"]
                home = _resolve_team(
                    row["Home_Team"], qualified, ko_results,
                    third_teams, third_team_groups, assigned_third
                )
                away = _resolve_team(
                    row["Away_Team"], qualified, ko_results,
                    third_teams, third_team_groups, assigned_third
                )

                if _is_placeholder(home) or _is_placeholder(away):
                    ko_results[mn] = {"winner": home, "loser": away}
                    continue

                # Track that both teams reached this stage
                stage_key = round_stage_map.get(rnd)
                if stage_key and home in team_counts:
                    team_counts[home][stage_key] += 1
                if stage_key and away in team_counts:
                    team_counts[away][stage_key] += 1

                # Check for actual result
                actual = st.session_state.get("match_results", {}).get(mn)
                if actual:
                    if actual["home_goals"] > actual["away_goals"]:
                        ko_results[mn] = {"winner": home, "loser": away}
                    else:
                        ko_results[mn] = {"winner": away, "loser": home}
                else:
                    # Predict and sample
                    pred = predict_match(home, away, model_state)
                    if pred:
                        r = np.random.random()
                        if r < pred["p_home_win"]:
                            ko_results[mn] = {"winner": home, "loser": away}
                        elif r < pred["p_home_win"] + pred["p_draw"]:
                            # Draw in knockout — coin flip weighted by strength
                            if pred["p_home_win"] >= pred["p_away_win"]:
                                ko_results[mn] = {"winner": home, "loser": away}
                            else:
                                ko_results[mn] = {"winner": away, "loser": home}
                        else:
                            ko_results[mn] = {"winner": away, "loser": home}
                    else:
                        # 50/50 if no prediction
                        if np.random.random() < 0.5:
                            ko_results[mn] = {"winner": home, "loser": away}
                        else:
                            ko_results[mn] = {"winner": away, "loser": home}

        # Track champion + finalist
        final_mn = knockout_df.filter(pl.col("Group_Stage") == "Final")
        if len(final_mn) > 0:
            fid = final_mn.row(0, named=True)["Match_Number"]
            if fid in ko_results:
                champ = ko_results[fid]["winner"]
                runner = ko_results[fid]["loser"]
                if champ in team_counts:
                    team_counts[champ]["champion"] += 1
                if runner in team_counts:
                    team_counts[runner]["final"] += 1

    return team_counts


def render_monte_carlo(group_df: pl.DataFrame, knockout_df: pl.DataFrame):
    """Render Monte Carlo simulation results."""
    st.markdown("## 🎲 Monte Carlo Simulation")

    model_state = load_model()
    if model_state is None:
        st.error("Model not trained yet. Run `uv run python src/models/train.py` first.")
        return

    st.caption(
        "Simulates the tournament thousands of times, sampling outcomes "
        "probabilistically. Shows the probability of each team reaching each stage."
    )

    n_sims = st.select_slider(
        "Number of simulations",
        options=[100, 500, 1000, 2000, 5000, 10000],
        value=1000,
        key="mc_n_sims",
    )

    if st.button("Run Simulation", type="primary", key="mc_run"):
        progress = st.progress(0, text=f"Running {n_sims:,} simulations...")
        results = run_monte_carlo(group_df, knockout_df, model_state, n_sims, progress_bar=progress)
        progress.empty()
        st.session_state["mc_results"] = results
        st.session_state["mc_n"] = n_sims
        st.rerun()

    if "mc_results" not in st.session_state:
        st.info("Click 'Run Simulation' to start.")
        return

    results = st.session_state["mc_results"]
    n = st.session_state["mc_n"]

    # Build results table
    rows = []
    for team, counts in results.items():
        rows.append({
            "Team": team,
            "Qualify": round(counts["group_qual"] / n * 100, 1),
            "R32": round(counts["r32"] / n * 100, 1),
            "R16": round(counts["r16"] / n * 100, 1),
            "QF": round(counts["qf"] / n * 100, 1),
            "SF": round(counts["sf"] / n * 100, 1),
            "Final": round(counts["final"] / n * 100, 1),
            "Champion": round(counts["champion"] / n * 100, 1),
        })

    results_df = pl.DataFrame(rows).sort("Champion", descending=True)

    # Champion probabilities chart
    st.markdown("### 🏆 Championship Probabilities")
    top_10 = results_df.head(10)

    import plotly.express as px
    fig = px.bar(
        top_10.to_pandas(), x="Champion", y="Team", orientation="h",
        labels={"Champion": "Win Probability (%)", "Team": ""},
        color="Champion", color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=350, margin=dict(t=10, b=10, l=100),
        showlegend=False, coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full probability table
    st.markdown("### Full Tournament Probabilities (%)")
    st.markdown(f"*Based on {n:,} simulations*")

    # Render as HTML table for better formatting
    header = "<tr><th>Team</th><th>Qualify</th><th>R32</th><th>R16</th><th>QF</th><th>SF</th><th>Final</th><th>🏆</th></tr>"
    rows_html = ""
    for row in results_df.iter_rows(named=True):
        fl = flag_img(row["Team"], 14)
        champ_style = "font-weight:700;color:#1B4F72;" if row["Champion"] > 5 else ""
        rows_html += (
            f'<tr><td>{fl} {row["Team"]}</td>'
            f'<td>{row["Qualify"]}</td><td>{row["R32"]}</td>'
            f'<td>{row["R16"]}</td><td>{row["QF"]}</td>'
            f'<td>{row["SF"]}</td><td>{row["Final"]}</td>'
            f'<td style="{champ_style}">{row["Champion"]}</td></tr>'
        )

    st.markdown(
        f'<div style="max-height:500px;overflow-y:auto;">'
        f'<table class="standings-table"><thead>{header}</thead>'
        f'<tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )
