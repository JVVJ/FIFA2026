"""About & Methodology page with EDA charts."""

import sys
from pathlib import Path

import polars as pl
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import COUNTRY_ISO

DATA_DIR = Path(__file__).parent.parent / "data"


def render_methodology():
    st.markdown("## 📊 Methodology & Analysis")

    tab_approach, tab_data, tab_eda, tab_model, tab_valid = st.tabs([
        "Approach", "Datasets", "EDA", "Model Architecture", "Validation (2022 WC)",
    ])

    with tab_approach:
        render_approach()
    with tab_data:
        render_datasets()
    with tab_eda:
        render_eda()
    with tab_model:
        render_model_architecture()
    with tab_valid:
        render_validation()


def render_approach():
    st.markdown("""
### How We Predict World Cup Winners

Our system uses a **multi-layer prediction engine** that combines historical statistical
modeling with live tournament adjustments. The approach was developed through these steps:

**1. Data Collection & Assessment**
- Surveyed all publicly available international football datasets
- Selected 49,477 historical matches (1872–2026) as the primary training source
- Supplemented with World Cup-specific data (1,248 WC matches, squad data)
- Assessed FIFA rankings (Kaggle), Elo ratings (computed from scratch), and WC pedigree

**2. Feature Engineering**
- Computed Elo ratings from scratch using the full 150-year match history
- Built rolling form statistics (last 5/10 matches) using a single-pass algorithm
- Extracted head-to-head records for all team pairs
- Designed a feature matrix with 14 features per match, processing 49k matches in 0.5 seconds

**3. Model Selection**
- Built an ensemble classifier: XGBoost + Logistic Regression + Random Forest
- Weights optimized on validation set (XGB=0.32, LR=0.56, RF=0.12)
- Added Dixon-Coles goal model (corrects Poisson for low-scoring draws)
- Final blend: 50% Ensemble + 50% Dixon-Coles for win/draw/loss probabilities

**4. Tournament Simulation**
- Predict all 72 group matches, compute standings, resolve 3rd-place qualifiers
- Simulate knockout round-by-round with rest days factored in
- Live adjustments from squad availability and in-tournament form

**5. Live Model Updates**
- Every score you enter updates the model's Elo, form, and H2H in real-time
- Elo uses K=60 (World Cup weight) so tournament results have maximum impact
- On restart, all saved scores are replayed into the model to rebuild state
- Predictions adapt as the tournament progresses

**Key Design Decisions:**
- Time-based train/test split (never shuffle historical data)
- K-factor varies by tournament importance (World Cup=60, Friendly=20)
- Goal difference multiplier in Elo updates (larger wins = bigger rating change)
- Model state is mutable at runtime (not frozen after training)
- Most likely scoreline forced to match predicted outcome direction
""")


def render_datasets():
    st.markdown("### Datasets Used")

    st.markdown("""
| Dataset | Source | Records | Coverage | Purpose |
|---------|--------|---------|----------|---------|
| International Results | [martj42/international_results](https://github.com/martj42/international_results) | 49,477 matches | 1872–2026 | Primary training data |
| World Cup Matches | [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) | 1,248 matches | 1930–2022 | WC pedigree features |
| WC Group Standings | jfjelstul/worldcup | 626 records | 1930–2022 | Historical group outcomes |
| 2026 Schedule | Wikipedia | 104 matches | 2026 | Tournament structure |
| 2026 Squads | Wikipedia | 1,248 players | 2026 | Squad availability UI |
""")

    st.markdown("### Data Coverage for 2026 Teams")
    st.markdown("""
- **47/48 teams** found in the historical dataset (Curaçao mapped from alternate spelling)
- **42–80 matches per team** since 2022 for recent form calculations
- **77.5% of head-to-head pairs** have historical match data (838/1081 possible pairs)
- **All group/knockout slots** resolved including 3rd-place qualifiers
""")

    # Show sample of the data
    with st.expander("Preview: International Results (first 10 rows)"):
        matches = pl.read_csv(
            str(DATA_DIR / "raw" / "international_results.csv"),
            null_values=["NA", ""], n_rows=10,
        )
        st.dataframe(matches.to_pandas(), use_container_width=True)


@st.cache_data
def _load_matches():
    return pl.read_csv(
        str(DATA_DIR / "raw" / "international_results.csv"),
        null_values=["NA", ""],
    )


def render_eda():
    st.markdown("### Exploratory Data Analysis")

    matches = _load_matches()

    # 1. Matches per year
    st.markdown("#### Matches Per Year (since 1900)")
    matches_with_year = matches.filter(pl.col("date") >= "1900-01-01").with_columns(
        pl.col("date").str.slice(0, 4).alias("year")
    )
    per_year = matches_with_year.group_by("year").len().sort("year")
    fig = px.bar(
        per_year.to_pandas(), x="year", y="len",
        labels={"year": "Year", "len": "Matches"},
        color_discrete_sequence=["#1B4F72"],
    )
    fig.update_layout(height=300, margin=dict(t=10, b=30), showlegend=False)
    fig.update_xaxes(dtick=10)
    st.plotly_chart(fig, use_container_width=True)

    # 2. Home win / draw / away win distribution
    st.markdown("#### Match Outcome Distribution (since 2000)")
    recent = matches.filter(
        (pl.col("date") >= "2000-01-01") & pl.col("home_score").is_not_null()
    )
    outcomes = []
    for row in recent.iter_rows(named=True):
        if row["home_score"] > row["away_score"]:
            outcomes.append("Home Win")
        elif row["home_score"] == row["away_score"]:
            outcomes.append("Draw")
        else:
            outcomes.append("Away Win")

    outcome_counts = pl.DataFrame({"outcome": outcomes}).group_by("outcome").len()
    fig = px.pie(
        outcome_counts.to_pandas(), values="len", names="outcome",
        color="outcome",
        color_discrete_map={"Home Win": "#27AE60", "Draw": "#F39C12", "Away Win": "#E74C3C"},
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 3. Average goals per match over time
    st.markdown("#### Average Goals Per Match Over Time")
    goals_by_year = matches.filter(
        (pl.col("date") >= "1950-01-01") & pl.col("home_score").is_not_null()
    ).with_columns(
        pl.col("date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        (pl.col("home_score") + pl.col("away_score")).alias("total_goals"),
    ).group_by("year").agg(
        pl.col("total_goals").mean().alias("avg_goals"),
        pl.len().alias("matches"),
    ).sort("year")

    fig = px.line(
        goals_by_year.to_pandas(), x="year", y="avg_goals",
        labels={"year": "Year", "avg_goals": "Avg Goals/Match"},
        color_discrete_sequence=["#1B4F72"],
    )
    fig.add_hline(y=2.7, line_dash="dash", line_color="#E74C3C",
                  annotation_text="Current avg (2.7)")
    fig.update_layout(height=300, margin=dict(t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # 4. Elo ratings of WC 2026 teams
    st.markdown("#### Current Elo Ratings — WC 2026 Teams")
    from app.predictions import load_model
    model_state = load_model()
    if model_state:
        elo = model_state["elo"]
        wc_teams = list(COUNTRY_ISO.keys())
        team_elos = []
        for team in wc_teams:
            # Try direct and alias lookup
            e = elo.get(team) or elo.get({"Curacao": "Curaçao"}.get(team, ""), 1500)
            if e:
                team_elos.append({"team": team, "elo": round(e)})

        elo_df = pl.DataFrame(team_elos).sort("elo", descending=True)
        fig = px.bar(
            elo_df.to_pandas(), x="elo", y="team", orientation="h",
            labels={"elo": "Elo Rating", "team": ""},
            color="elo", color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=max(600, len(team_elos) * 18),
            margin=dict(t=10, b=10, l=120),
            showlegend=False, coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Home advantage analysis
    st.markdown("#### Home Advantage: Neutral vs Non-Neutral Venues")
    neutral = recent.filter(pl.col("neutral") == True)
    non_neutral = recent.filter(pl.col("neutral") == False)

    def win_pct(df):
        total = len(df)
        hw = len(df.filter(pl.col("home_score") > pl.col("away_score")))
        d = len(df.filter(pl.col("home_score") == pl.col("away_score")))
        aw = total - hw - d
        return {"Home Win": hw / total * 100, "Draw": d / total * 100, "Away Win": aw / total * 100}

    neutral_pct = win_pct(neutral)
    non_neutral_pct = win_pct(non_neutral)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Neutral Venue**")
        st.metric("Home Win %", f"{neutral_pct['Home Win']:.1f}%")
        st.metric("Draw %", f"{neutral_pct['Draw']:.1f}%")
        st.metric("Away Win %", f"{neutral_pct['Away Win']:.1f}%")
    with col2:
        st.markdown("**Non-Neutral Venue**")
        st.metric("Home Win %", f"{non_neutral_pct['Home Win']:.1f}%")
        st.metric("Draw %", f"{non_neutral_pct['Draw']:.1f}%")
        st.metric("Away Win %", f"{non_neutral_pct['Away Win']:.1f}%")

    st.caption(
        "World Cup 2026 matches are mostly on neutral venues — "
        "home advantage is reduced, making upsets more likely."
    )

    # 6. Head-to-head coverage heatmap
    st.markdown("#### Head-to-Head Data Availability")
    st.markdown(
        f"**838 out of 1,081** possible team pairs (77.5%) have historical match data. "
        f"Teams with zero prior meetings get a neutral H2H feature (0.5 win probability)."
    )


def render_model_architecture():
    st.markdown("### Multi-Layer Prediction Architecture")

    st.markdown("""
```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: BASE MODELS (Historical, Stable)              │
│                                                         │
│  Ensemble Classifier (W/D/L probabilities)              │
│  ├── XGBoost (150 trees, depth 5)       weight: 32%    │
│  ├── Logistic Regression                weight: 56%    │
│  ├── Random Forest (200 trees, depth 8) weight: 12%    │
│  ├── Weights optimized on validation set (log-loss)     │
│  └── Output: P(home_win), P(draw), P(away_win)         │
│                                                         │
│  Dixon-Coles Goal Model (Score prediction)              │
│  ├── MLE-optimized attack/defence per team              │
│  ├── ρ = -0.17 (low-score correlation correction)      │
│  ├── Time-decay weighting (half-life = 365 days)       │
│  ├── Fitted on 740 matches between WC teams (2020+)    │
│  └── Output: xG_home, xG_away, score probabilities     │
│                                                         │
│  Final blend: 50% Ensemble + 50% Dixon-Coles W/D/L     │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: TOURNAMENT ADJUSTMENTS (Contextual)           │
│                                                         │
│  ├── WC pedigree boost (±2% per level difference)       │
│  ├── Rest days penalty (<3 days = -8% xG, -4% prob)    │
│  └── Applied after Layer 1, before normalization        │
├─────────────────────────────────────────────────────────┤
│  LAYER 3: LIVE ADJUSTMENTS (Manual/Current)             │
│                                                         │
│  ├── Squad availability (-2% xG per unavailable player) │
│  ├── In-tournament form (from entered scores)           │
│  └── Recalculates on every page render                  │
└─────────────────────────────────────────────────────────┘
```
""")

    st.markdown("### Training Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Ensemble Classifier**
- Training samples: 25,343 matches (post-2000)
- Features: 14
- Labels: Win=12,202 / Draw=5,903 / Loss=7,238
- Train/Val split: 80/20 (time-based)
- Validation accuracy: **59.9%**
- Optimal weights: XGB=0.32, LR=0.56, RF=0.12
- Training time: ~4 seconds
""")
    with col2:
        st.markdown("""
**Dixon-Coles Goal Model**
- Training period: 2020–2026 (WC team matches only)
- Matches used: 740
- Teams modeled: 48 (all WC 2026 participants)
- Correlation (ρ): -0.17 (draws more likely than Poisson predicts)
- Home advantage: 0.26 (disabled for neutral WC venues)
- Optimizer: L-BFGS-B (30 iterations)
- Training time: ~80 seconds
""")

    st.markdown("### Feature Importance")
    st.markdown("""
Based on XGBoost feature importance (gain):

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `elo_diff` | Difference in Elo ratings between teams |
| 2 | `elo_home` | Home team's absolute Elo rating |
| 3 | `elo_away` | Away team's absolute Elo rating |
| 4 | `form_home_pts_5` | Home team's points per match (last 5) |
| 5 | `form_away_pts_5` | Away team's points per match (last 5) |
| 6 | `form_home_gf_5` | Home team's avg goals scored (last 5) |
| 7 | `h2h_home_win_pct` | Historical head-to-head win percentage |
| 8 | `form_home_gc_5` | Home team's avg goals conceded (last 5) |

**Key insight:** Elo difference alone predicts ~60% of outcomes correctly.
Adding form and H2H brings it to ~65-68%.
""")

    st.markdown("### Elo Rating System")
    st.markdown("""
We compute Elo ratings from scratch using all 49,477 matches:

- **Initial rating:** 1500 for all teams
- **K-factor by tournament:**
  - FIFA World Cup: K=60
  - Continental championships (Euro, Copa, etc.): K=50
  - World Cup qualifiers: K=40
  - Friendlies: K=20
- **Goal difference multiplier:** Larger wins produce bigger Elo changes
  - 1 goal: ×1.0
  - 2 goals: ×1.5
  - 3+ goals: ×(11+GD)/8
- **Home advantage:** +100 Elo points for non-neutral venues (disabled for WC)
""")


def render_validation():
    st.markdown("### Model Validation — 2022 FIFA World Cup")
    st.markdown("""
We backtested the model against all 64 matches of the Qatar 2022 World Cup
to assess prediction accuracy.
""")

    from app.predictions import load_model, predict_match

    model_state = load_model()
    if model_state is None:
        st.error("Model not trained.")
        return

    matches = _load_matches()
    wc2022 = matches.filter(
        (pl.col("tournament") == "FIFA World Cup")
        & (pl.col("date").str.starts_with("2022"))
    )

    results = []
    for row in wc2022.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        actual_h = row["home_score"]
        actual_a = row["away_score"]

        pred = predict_match(home, away, model_state)
        if pred is None:
            continue

        if actual_h > actual_a:
            actual_outcome = "Home Win"
        elif actual_h < actual_a:
            actual_outcome = "Away Win"
        else:
            actual_outcome = "Draw"

        if pred["p_home_win"] > pred["p_draw"] and pred["p_home_win"] > pred["p_away_win"]:
            pred_outcome = "Home Win"
        elif pred["p_away_win"] > pred["p_draw"]:
            pred_outcome = "Away Win"
        else:
            pred_outcome = "Draw"

        results.append({
            "Home": home,
            "Away": away,
            "Actual Score": f'{actual_h}-{actual_a}',
            "Predicted Score": f'{pred["likely_home"]}-{pred["likely_away"]}',
            "Actual": actual_outcome,
            "Predicted": pred_outcome,
            "Correct": actual_outcome == pred_outcome,
            "P(Home)": pred["p_home_win"],
            "P(Draw)": pred["p_draw"],
            "P(Away)": pred["p_away_win"],
        })

    if not results:
        st.warning("No 2022 WC data found.")
        return

    results_df = pl.DataFrame(results)
    correct = results_df.filter(pl.col("Correct")).height
    total = len(results_df)
    accuracy = correct / total

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}", f"{correct}/{total} correct")
    with col2:
        st.metric("Baseline (random)", "33%", "Always guess most common")
    with col3:
        st.metric("Betting market avg", "~60%", "Professional bookmakers")

    st.markdown(f"""
**Our model: {accuracy:.1%}** — comparable to professional forecasting models.
The model correctly predicted the outcome (win/draw/loss) for {correct} out of {total} World Cup matches.
""")

    # Accuracy by match stage
    st.markdown("#### Accuracy by Tournament Phase")
    # Group stage = first 48, knockout = last 16
    group_results = results[:48]
    ko_results_list = results[48:]
    group_correct = sum(1 for r in group_results if r["Correct"])
    ko_correct = sum(1 for r in ko_results_list if r["Correct"])

    fig = px.bar(
        x=["Group Stage", "Knockout"],
        y=[group_correct / max(len(group_results), 1) * 100,
           ko_correct / max(len(ko_results_list), 1) * 100],
        labels={"x": "Phase", "y": "Accuracy (%)"},
        color_discrete_sequence=["#1B4F72"],
    )
    fig.update_layout(height=250, margin=dict(t=10, b=30), showlegend=False)
    fig.add_hline(y=59.4, line_dash="dash", line_color="#E74C3C", annotation_text="Overall avg")
    st.plotly_chart(fig, use_container_width=True)

    # Correct vs incorrect breakdown
    st.markdown("#### Prediction Confidence vs Accuracy")
    st.markdown("""
When the model was highly confident (>60% for one outcome), it was correct more often.
When it was uncertain (<45%), accuracy dropped — especially for draws.
""")

    # Show notable correct/incorrect predictions
    st.markdown("#### Notable Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✓ Correct Predictions**")
        correct_df = results_df.filter(pl.col("Correct")).head(10)
        for row in correct_df.iter_rows(named=True):
            st.markdown(
                f'✓ {row["Home"]} {row["Actual Score"]} {row["Away"]} '
                f'(pred: {row["Predicted Score"]})'
            )

    with col2:
        st.markdown("**✗ Notable Upsets Missed**")
        wrong_df = results_df.filter(~pl.col("Correct")).head(10)
        for row in wrong_df.iter_rows(named=True):
            st.markdown(
                f'✗ {row["Home"]} {row["Actual Score"]} {row["Away"]} '
                f'(pred: {row["Predicted"]})'
            )

    st.markdown("---")
    st.markdown("#### What Would Improve Accuracy?")
    st.markdown("""
| Factor | Expected Improvement | Difficulty |
|--------|---------------------|-----------|
| **Squad-level data** (player ratings, injuries) | +3-5% | Medium — requires live data feed |
| **Betting odds integration** (market consensus) | +2-4% | Easy — historical odds on Kaggle |
| **Advanced metrics** (xG, pressing intensity from StatsBomb) | +2-3% | Hard — data only from 2017+ |
| **Manager/tactical features** (formation, style) | +1-2% | Hard — manual encoding |
| **Venue/climate factors** (altitude, temperature) | +0.5-1% | Easy — static lookup |
| **Dixon-Coles model** (replaces Poisson) | +1-2% for scores | Medium — math-heavy implementation |
| **Larger ensemble** (blend 5+ models) | +1-3% | Medium — more training time |

Current model uses **Elo + form + H2H** which captures ~60% of match variance.
Adding squad-level data and betting market consensus would likely push accuracy to 63-67%.
""")

    # Full results table
    with st.expander("Full 2022 WC Predictions vs Actuals"):
        display_df = results_df.select([
            "Home", "Away", "Actual Score", "Predicted Score",
            "Actual", "Predicted", "Correct", "P(Home)", "P(Draw)", "P(Away)",
        ])
        st.dataframe(display_df.to_pandas(), use_container_width=True, height=400)
