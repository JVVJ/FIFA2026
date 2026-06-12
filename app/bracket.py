"""Tournament bracket rendering with inline score entry."""

import streamlit as st

from app.config import flag_img


def render_bracket_tree(ko_results: dict, show_predictions: bool = False, editable: bool = False):
    """Render the full knockout bracket."""

    # R32 (in expander to reduce widget count)
    r32 = [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]
    with st.expander("Round of 32 (16 matches)", expanded=False):
        _render_round_grid(r32, ko_results, show_predictions, editable, cols=4)

    # R16 (in expander)
    r16 = [89, 90, 91, 92, 93, 94, 95, 96]
    with st.expander("Round of 16 (8 matches)", expanded=False):
        _render_round_grid(r16, ko_results, show_predictions, editable, cols=4)

    # QF → SF → Final (converging bracket)
    st.markdown(
        '<div style="font-size:0.95rem;font-weight:600;color:#1B4F72;'
        'margin:0.5rem 0;padding-bottom:4px;border-bottom:2px solid #D6EAF8;">'
        'Quarterfinals → Semifinals → Final</div>',
        unsafe_allow_html=True,
    )

    col_lqf, col_lsf, col_final, col_rsf, col_rqf = st.columns([2, 2, 2, 2, 2])

    with col_lqf:
        st.caption("QUARTERFINAL")
        _render_match_card(97, ko_results, show_predictions, editable)
        st.markdown("")
        _render_match_card(98, ko_results, show_predictions, editable)

    with col_lsf:
        st.caption("SEMIFINAL")
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        _render_match_card(101, ko_results, show_predictions, editable)

    with col_final:
        st.caption("FINAL")
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        _render_final_card(104, ko_results, show_predictions, editable)
        st.markdown("")
        st.caption("3RD PLACE")
        _render_match_card(103, ko_results, show_predictions, editable)

    with col_rsf:
        st.caption("SEMIFINAL")
        st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)
        _render_match_card(102, ko_results, show_predictions, editable)

    with col_rqf:
        st.caption("QUARTERFINAL")
        _render_match_card(99, ko_results, show_predictions, editable)
        st.markdown("")
        _render_match_card(100, ko_results, show_predictions, editable)


def _render_round_grid(match_nums, ko_results, show_predictions, editable, cols=4):
    """Render matches in a grid."""
    columns = st.columns(cols)
    for i, mn in enumerate(match_nums):
        with columns[i % cols]:
            _render_match_card(mn, ko_results, show_predictions, editable)


def _render_match_card(mn: int, ko_results: dict, show_predictions: bool, editable: bool):
    """Render a single match card with optional score entry."""
    if mn not in ko_results:
        st.markdown(
            '<div style="border:1px solid #EAF2F8;border-radius:8px;'
            'padding:8px;margin:4px 0;font-size:0.75rem;color:#85929E;">'
            'TBD</div>',
            unsafe_allow_html=True,
        )
        return

    kr = ko_results[mn]
    winner = kr.get("winner", "TBD")
    loser = kr.get("loser", "TBD")
    pred = kr.get("pred")

    # Determine home/away display order
    if pred and pred["p_home_win"] >= pred["p_away_win"]:
        home, away = winner, loser
    elif pred:
        home, away = loser, winner
    else:
        home, away = winner, loser

    hf = flag_img(home, 14)
    af = flag_img(away, 14)

    # Check for actual entered result
    actual = st.session_state.match_results.get(mn)

    # Styling
    h_style = "font-weight:600;color:#1B4F72;"
    a_style = "color:#5D6D7E;"
    h_bg = ""
    a_bg = ""

    if actual:
        if actual["home_goals"] > actual["away_goals"]:
            h_bg = "background:#D5F5E3;"
            h_style = "font-weight:700;color:#1B4F72;"
        elif actual["away_goals"] > actual["home_goals"]:
            a_bg = "background:#D5F5E3;"
            a_style = "font-weight:700;color:#1B4F72;"
    elif pred and winner != "TBD":
        if winner == home:
            h_bg = "background:#D5F5E3;"
        else:
            a_bg = "background:#D5F5E3;"

    # Score display
    h_score = ""
    a_score = ""
    if actual:
        h_score = str(actual["home_goals"])
        a_score = str(actual["away_goals"])
    elif pred:
        h_score = str(pred["likely_home"])
        a_score = str(pred["likely_away"])

    # Probability display
    pct_h = ""
    pct_a = ""
    if show_predictions and pred:
        pct_h = f'<span style="font-size:0.6rem;color:#85929E;">{pred["p_home_win"]*100:.0f}%</span>'
        pct_a = f'<span style="font-size:0.6rem;color:#85929E;">{pred["p_away_win"]*100:.0f}%</span>'

    if not editable:
        # Read-only display (predictions tab)
        st.markdown(f"""<div style="border:1px solid #D6EAF8;border-radius:8px;padding:6px 8px;margin:4px 0;background:white;font-size:0.78rem;">
<div style="display:flex;align-items:center;gap:4px;padding:3px 4px;border-radius:3px;{h_bg}">
    {hf}<span style="{h_style}">{home}</span>
    <span style="margin-left:auto;font-weight:700;">{h_score}</span>{pct_h}
</div>
<div style="display:flex;align-items:center;gap:4px;padding:3px 4px;border-radius:3px;{a_bg}">
    {af}<span style="{a_style}">{away}</span>
    <span style="margin-left:auto;font-weight:700;">{a_score}</span>{pct_a}
</div>
</div>""", unsafe_allow_html=True)
    else:
        # Editable: team name + score input on same row
        st.markdown(
            f'<div style="border:1px solid #D6EAF8;border-radius:8px;'
            f'padding:4px 8px;margin:4px 0;background:white;font-size:0.78rem;">'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Home team row: flag + name + score input
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(f'{hf} <span style="{h_style}">{home}</span>', unsafe_allow_html=True)
        with h2:
            hg = st.number_input(
                "H", min_value=0, max_value=20,
                value=actual["home_goals"] if actual else 0,
                key=f"bkt_hg_{mn}", label_visibility="collapsed",
            )
        # Away team row: flag + name + score input
        a1, a2 = st.columns([3, 1])
        with a1:
            st.markdown(f'{af} <span style="{a_style}">{away}</span>', unsafe_allow_html=True)
        with a2:
            ag = st.number_input(
                "A", min_value=0, max_value=20,
                value=actual["away_goals"] if actual else 0,
                key=f"bkt_ag_{mn}", label_visibility="collapsed",
            )
        # Save + Squad buttons
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✓ Save", key=f"bkt_save_{mn}", type="primary", use_container_width=True):
                from app.data import save_match_result
                save_match_result(mn, home, away, hg, ag)
                st.rerun()
        with b2:
            from app.components import open_squad_dialog
            home_unavail = st.session_state.unavailable.get((mn, home), set())
            away_unavail = st.session_state.unavailable.get((mn, away), set())
            unavail_n = len(home_unavail) + len(away_unavail)
            sq_label = f"📋 {unavail_n}" if unavail_n else "📋"
            if st.button(sq_label, key=f"bkt_sq_{mn}", use_container_width=True):
                st.session_state["_squad_dialog_match"] = mn
                st.session_state["_squad_dialog_home"] = home
                st.session_state["_squad_dialog_away"] = away
                open_squad_dialog()


def _render_final_card(mn: int, ko_results: dict, show_predictions: bool, editable: bool):
    """Render the final with trophy styling."""
    if mn not in ko_results:
        st.markdown('<div style="text-align:center;color:#85929E;">TBD</div>', unsafe_allow_html=True)
        return

    kr = ko_results[mn]
    winner = kr.get("winner", "TBD")
    loser = kr.get("loser", "TBD")
    pred = kr.get("pred")

    if pred and pred["p_home_win"] >= pred["p_away_win"]:
        home, away = winner, loser
    elif pred:
        home, away = loser, winner
    else:
        home, away = winner, loser

    hf = flag_img(home, 18)
    af = flag_img(away, 18)

    actual = st.session_state.match_results.get(mn)
    if actual:
        score = f'{actual["home_goals"]} - {actual["away_goals"]}'
        champion = home if actual["home_goals"] > actual["away_goals"] else away
    elif pred:
        score = f'{pred["likely_home"]} - {pred["likely_away"]}'
        champion = winner
    else:
        score = "vs"
        champion = winner

    st.markdown(f"""<div style="border:2px solid #F7DC6F;border-radius:10px;padding:10px;background:linear-gradient(135deg,#FDFEFE,#FEF9E7);text-align:center;">
<div style="font-size:1.2rem;">🏆</div>
<div style="display:flex;justify-content:center;align-items:center;gap:8px;margin:6px 0;">
    {hf}<span style="font-size:0.85rem;font-weight:{'700' if champion==home else '400'};">{home}</span>
    <span style="font-weight:700;color:#1B4F72;">{score}</span>
    <span style="font-size:0.85rem;font-weight:{'700' if champion==away else '400'};">{away}</span>{af}
</div>
<div style="font-size:0.75rem;color:#1B4F72;font-weight:600;">🏆 {champion}</div>
</div>""", unsafe_allow_html=True)

    if editable:
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(f'{hf} **{home}**', unsafe_allow_html=True)
        with h2:
            hg = st.number_input(
                "H", min_value=0, max_value=20,
                value=actual["home_goals"] if actual else 0,
                key=f"bkt_hg_{mn}", label_visibility="collapsed",
            )
        a1, a2 = st.columns([3, 1])
        with a1:
            st.markdown(f'{af} **{away}**', unsafe_allow_html=True)
        with a2:
            ag = st.number_input(
                "A", min_value=0, max_value=20,
                value=actual["away_goals"] if actual else 0,
                key=f"bkt_ag_{mn}", label_visibility="collapsed",
            )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✓ Save", key=f"bkt_save_{mn}", type="primary", use_container_width=True):
                from app.data import save_match_result
                save_match_result(mn, home, away, hg, ag)
                st.rerun()
        with b2:
            from app.components import open_squad_dialog
            home_unavail = st.session_state.unavailable.get((mn, home), set())
            away_unavail = st.session_state.unavailable.get((mn, away), set())
            unavail_n = len(home_unavail) + len(away_unavail)
            sq_label = f"📋 {unavail_n}" if unavail_n else "📋"
            if st.button(sq_label, key=f"bkt_sq_{mn}", use_container_width=True):
                st.session_state["_squad_dialog_match"] = mn
                st.session_state["_squad_dialog_home"] = home
                st.session_state["_squad_dialog_away"] = away
                open_squad_dialog()
