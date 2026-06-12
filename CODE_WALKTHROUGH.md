# Code Walkthrough — Line by Line

This document explains every file in the project, what each section does, and how they connect.

---

## Architecture Overview

```
User opens browser
    → Streamlit serves app/streamlit_app.py
        → Sidebar navigation picks a page
        → Each page calls functions from other modules:
            - config.py (constants)
            - data.py (load CSV, persist JSON)
            - standings.py (compute group tables)
            - components.py (render match rows, score inputs, squad dialog)
            - predictions.py (ML predictions, tournament simulation)
            - bracket.py (knockout bracket UI)
            - methodology.py (EDA charts, model docs)

Training pipeline (run once):
    src/models/train.py
        → src/features/builder.py (build feature matrix)
        → src/features/elo.py (Elo helpers)
        → src/models/ensemble.py (XGBoost + LR + RF)
        → src/models/dixon_coles.py (score model)
        → Saves data/model.pkl
```

---

## 1. `app/config.py` — Constants & Helpers (44 lines)

```python
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"     # Points to project/data/
MATCH_RESULTS_FILE = DATA_DIR / "match_results.json" # Where scores are saved
UNAVAILABLE_FILE = DATA_DIR / "unavailable_players.json"  # Where squad data is saved

MATCHES_PER_PAGE = 12  # Pagination: how many matches per page
```

**COUNTRY_ISO** — Maps each team name to its ISO code for flag images:
```python
COUNTRY_ISO = {
    "Mexico": "mx",          # → https://flagcdn.com/w40/mx.png
    "England": "gb-eng",     # flagcdn supports UK subdivisions
    "Scotland": "gb-sct",
    ...
}
```

**GROUP_NAMES** — The 12 groups (A through L)

**TEAM_NAME_MAP** — Maps alternate spellings:
```python
TEAM_NAME_MAP = {"Curacao": "Curaçao"}  # Schedule says "Curacao", historical data says "Curaçao"
```

**flag_img()** — Returns an HTML `<img>` tag for a team's flag:
```python
def flag_img(team: str, size: int = 20) -> str:
    iso = COUNTRY_ISO.get(team)   # Look up ISO code
    if not iso:
        return ""                  # Unknown team = no flag
    return f'<img src="https://flagcdn.com/w40/{iso}.png" style="width:{size}px;..." />'
```

---

## 2. `app/data.py` — Data Loading & Persistence (102 lines)

**JSON persistence** — Simple read/write for match results and squad data:
```python
def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())  # Read file, parse JSON
    return {}                                 # First run = empty

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)  # Create dirs if needed
    path.write_text(json.dumps(data, indent=2))      # Pretty-print JSON to file
```

**init_results()** — Called once at app start. Loads saved data into Streamlit's session state:
```python
def init_results():
    if "match_results" not in st.session_state:
        raw = load_json(MATCH_RESULTS_FILE)        # Load from disk
        st.session_state.match_results = {int(k): v for k, v in raw.items()}
        # JSON keys are strings "1", "2"... we convert back to int

    if "unavailable" not in st.session_state:
        raw = load_json(UNAVAILABLE_FILE)
        # Keys are "73|Brazil" → split into tuple (73, "Brazil") → set of player names
        st.session_state.unavailable = {
            (int(k.split("|")[0]), k.split("|")[1]): set(v)
            for k, v in raw.items() if "|" in k
        }
```

**persist_results() / persist_unavailable()** — Save current state to disk:
```python
def persist_results():
    data = {str(k): v for k, v in st.session_state.match_results.items()}
    save_json(MATCH_RESULTS_FILE, data)  # {match_num: {home_goals, away_goals}}
```

**Schedule loading:**
```python
@st.cache_data  # Only runs once, then cached in memory
def load_schedule() -> pl.DataFrame:
    return pl.read_csv(str(DATA_DIR / "raw" / "wc2026_schedule.csv"))

@st.cache_data
def load_squads() -> pl.DataFrame:
    return pl.read_csv(str(DATA_DIR / "raw" / "wc2026_squads.csv"))
```

**Time conversion:**
```python
def parse_utc_time(date_str, utc_time_str):
    # "2026-06-11", "02:00+1" → datetime(2026-06-12, 02:00)
    # The "+1" means the UTC time is on the next calendar day
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    if utc_time_str.endswith("+1"):
        time_part = utc_time_str.replace("+1", "").strip()
        base_date += timedelta(days=1)  # Move to next day
    ...

def utc_to_est(utc_dt):
    return utc_dt - timedelta(hours=4)  # EDT = UTC-4 (summer time)
```

**process_schedule()** — Adds EST columns to the schedule dataframe:
```python
def process_schedule(df):
    est_times, est_dates = [], []
    for row in df.iter_rows(named=True):      # Loop through each match
        utc_dt = parse_utc_time(row["Date"], row["UTC_Time"])
        est_dt = utc_to_est(utc_dt)
        est_times.append(est_dt.strftime("%I:%M %p"))  # "03:00 PM"
        est_dates.append(est_dt.strftime("%Y-%m-%d"))   # "2026-06-11"
    return df.with_columns(...)  # Add new columns to dataframe
```

---

## 3. `app/standings.py` — Group Standings Logic (99 lines)

**compute_standings()** — Given match results, compute W/D/L/GF/GA/GD/Pts per team:
```python
def compute_standings(df, results):
    for group, teams in group_teams.items():
        for team in teams:
            w, d, lost, gf, ga = 0, 0, 0, 0, 0
            # Find all matches this team plays in this group
            for match in team's matches:
                if match_number in results:  # Only count matches with entered scores
                    # Determine if team is home or away
                    # Count W/D/L and goals
            pts = w * 3 + d
            rows.append({team, P, W, D, L, GF, GA, GD, Pts})
    
    # Sort by: Group (A-L), then Points desc, then GD desc, then GF desc
    return sorted dataframe
```

**get_qualified_teams()** — Determine who advances:
```python
def get_qualified_teams(standings):
    for each group:
        winners[group] = 1st place team      # Automatic qualification
        runners_up[group] = 2nd place team   # Automatic qualification
        third_place.append(3rd place team)   # Collected for comparison
    
    # Sort all 3rd-place teams by Pts, GD, GF — top 8 qualify
    third_place = sorted(all_thirds)[:8]
    return {winners, runners_up, third_place}
```

**resolve_knockout_team()** — Turn "Winner A" into actual team name:
```python
def resolve_knockout_team(placeholder, qualified):
    "Winner A"   → qualified["winners"]["Group A"]     → e.g., "Spain"
    "Runner-up B" → qualified["runners_up"]["Group B"]  → e.g., "Switzerland"
    "Winner 74"  → ko_results[74]["winner"]            → whoever won match 74
    "3rd A/B/C"  → best available 3rd-place team from those groups
```

---

## 4. `app/components.py` — UI Widgets (249 lines)

**render_match_row()** — Display one match as a compact horizontal row:
```python
def render_match_row(row, interactive=False):
    # Build HTML: [M1] [Mexico 🇲🇽] [vs / 2-0] [🇿🇦 South Africa] [⏰ 📅 🏟️]
    html = f'<div class="match-row">...'
    st.markdown(html, unsafe_allow_html=True)
    
    if interactive:
        render_score_input(mn, home, away)  # Show number inputs below
```

**render_score_input()** — Score entry with squad button:
```python
def render_score_input(match_num, home, away):
    # 5 columns: [spacer] [home_score_input] [away_score_input] [✓ save] [📋 squad]
    col2: number_input for home goals
    col3: number_input for away goals
    col4: ✓ button → saves to session_state + persists to disk + reruns page
    col5: 📋 button → opens squad dialog popup
```

**open_squad_dialog()** — The `@st.dialog` popup for managing player availability:
```python
@st.dialog("Squad Availability", width="large")
def open_squad_dialog():
    # Reads match/team from session_state (set by the button that triggered it)
    # Loads squad data, shows two columns (home team / away team)
    # Each player has a checkbox — unchecked = unavailable
    # Changes are saved to session_state.unavailable and persisted to disk
```

**paginate_matches()** — Show matches with Prev/Next buttons:
```python
def paginate_matches(df, key_prefix, interactive=False):
    total_pages = ceil(total / MATCHES_PER_PAGE)
    current_page = session_state[page_key]
    
    # Prev/Next buttons
    # Slice dataframe: df[start:start+12]
    # Render each match in the page
```

**render_standings_table()** — HTML table for group standings:
```python
def render_standings_table(standings, group, qualified_teams):
    # Build HTML <table> with headers: #, Team, P, W, D, L, GF, GA, GD, Pts
    # Highlight qualified teams with green background (class="qualified")
```

---

## 5. `app/predictions.py` — ML Prediction Engine (417 lines)

**load_model()** — Load the trained model once:
```python
@st.cache_resource  # Loaded once, stays in memory forever
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)  # Returns dict with all model components
```

**predict_match()** — The core prediction function (multi-layer):
```python
def predict_match(home, away, model_state, ...):
    # 1. Normalize team names (Curacao → Curaçao)
    
    # 2. Check team exists in model data
    
    # 3. LAYER 1: Ensemble classifier
    feats = _snapshot_features(home, away, elo, form, h2h)  # 14 features
    probs = model.predict_proba(X)  # [P(loss), P(draw), P(win)]
    
    # 4. LAYER 1b: Dixon-Coles goal model
    dc_pred = dc_model.predict_score_probs(home, away)  # xG, score matrix
    # Blend: 50% ensemble + 50% Dixon-Coles for W/D/L
    
    # 5. LAYER 2: Tournament adjustments
    # Rest days: <3 days rest → penalty to xG and win probability
    
    # 6. LAYER 3: Live adjustments
    # Squad modifier: each unavailable player → -2% xG (max -30%)
    # In-tournament form: scales xG ±15%
    
    # 7. Find most likely score CONSISTENT with predicted outcome
    # If model says home wins → pick best home-win scoreline from Poisson grid
    
    return {p_home_win, p_draw, p_away_win, xg_home, xg_away, likely_home, likely_away, elo_home, elo_away}
```

**simulate_tournament()** — Predict the entire knockout bracket:
```python
def simulate_tournament(group_df, knockout_df, model_state):
    # 1. Predict all 72 group matches → get scores
    # 2. Compute standings from predicted scores
    # 3. Determine qualified teams (winners, runners-up, best 3rd)
    # 4. Build 3rd-place assignment map (which group's 3rd goes where)
    # 5. Walk through knockout rounds in order:
    #    For each match:
    #      - Resolve placeholder team names
    #      - Calculate rest days since last match
    #      - Run predict_match()
    #      - Record winner/loser
    #      - Update last-match-date for rest day tracking
    # 6. Return full results dict
```

**_resolve_team()** — Turn placeholders into real teams:
```python
def _resolve_team(placeholder, qualified, ko_results, third_teams, ...):
    "Winner A"      → look up group A winner from standings
    "Runner-up B"   → look up group B runner-up
    "3rd A/B/C/D/F" → find best unassigned 3rd-place team from those groups
    "Winner 74"     → look up who won match 74 in ko_results
    "Loser 101"     → look up who lost match 101 (for 3rd-place match)
```

---

## 6. `app/bracket.py` — Tournament Bracket UI (267 lines)

**render_bracket_tree()** — Main bracket layout:
```python
def render_bracket_tree(ko_results, show_predictions, editable):
    # Round of 32: 16 matches in a 4-column grid
    # Round of 16: 8 matches in a 4-column grid
    # QF → SF → Final: 5-column converging layout
    #   [Left QF] [Left SF] [FINAL] [Right SF] [Right QF]
```

**_render_match_card()** — One bracket card:
```python
def _render_match_card(mn, ko_results, show_predictions, editable):
    # 1. Look up match in ko_results
    # 2. Determine home/away display order
    # 3. Check for actual entered result vs predicted
    # 4. Style: winner gets green background, loser grey
    # 5. Render HTML card with team names, flags, scores
    # 6. If editable: show score inputs + ✓ + 📋 buttons below card
```

**_render_final_card()** — Special trophy-styled final card:
```python
# Gold border, centered layout, trophy emoji, champion label
```

---

## 7. `app/streamlit_app.py` — Main Entry Point (423 lines)

**Top of file:**
```python
sys.path.insert(0, ...)  # So "app.*" and "src.*" imports work

st.set_page_config(...)  # Page title, icon, layout, sidebar state
st.markdown("""<style>...""")  # All CSS for the app (match rows, tables, etc.)
```

**main():**
```python
def main():
    init_results()         # Load saved scores/squads from disk
    
    df = load_schedule()   # Load + cache the 104-match schedule
    df = process_schedule(df)  # Add EST time columns
    
    # Sidebar: navigation buttons + filters
    with st.sidebar:
        # 5 buttons: Group Stage, Knockout, Predictions (Groups), Predictions (KO), Methodology
        # Active page stored in session_state
        # Date range filter, team multi-select filter
    
    # Route to the selected page:
    if page == "⚽ Group Stage":
        render_group_stage(group_df, date_range, selected_teams)
    elif page == "🏆 Knockout Stage":
        render_knockout_stage(...)
    ...
```

**render_group_stage():**
```python
# Left column (60%): Match list with score inputs
# Right column (40%): Live standings tables per group
```

**render_knockout_stage():**
```python
# 1. Run simulate_tournament() to resolve all team names
# 2. Render bracket tree with editable=True
```

**render_predictions_groups():**
```python
# 1. Load model
# 2. Show predicted standings table for selected group
# 3. For each match: predict_match() with live adjustments, render prediction card
```

**render_predictions_knockout():**
```python
# 1. Run simulate_tournament()
# 2. Show champion banner
# 3. Render bracket tree with show_predictions=True
```

---

## 8. `src/features/elo.py` — Elo Rating System (119 lines)

**Constants:**
```python
K_FACTORS = {
    "FIFA World Cup": 60,       # WC matches cause biggest rating changes
    "Friendlies": 20,           # Friendlies have little impact
    ...
}
INITIAL_ELO = 1500             # Starting rating for any new team
HOME_ADVANTAGE = 100           # +100 Elo for home team (non-neutral only)
```

**compute_elo_ratings()** — Walk through ALL matches chronologically:
```python
def compute_elo_ratings(matches_df):
    for each match in chronological order:
        elo_h = current rating of home team (or 1500 if first match)
        elo_a = current rating of away team
        
        # Expected score (0-1 scale, like probability of winning)
        exp_h = 1 / (1 + 10^((elo_a - elo_h - home_advantage) / 400))
        
        # Actual score: win=1, draw=0.5, loss=0
        actual_h = 1.0 if home won else 0.5 if draw else 0.0
        
        # Goal difference multiplier (bigger wins = bigger change)
        gd_mult = 1.0 / 1.5 / (11+gd)/8 depending on goal difference
        
        # Update: new_elo = old_elo + K * gd_mult * (actual - expected)
        # If you win when expected to lose → big rating boost
        # If you lose when expected to win → big rating drop
```

---

## 9. `src/features/builder.py` — Feature Matrix Builder (200 lines)

**build_features_fast()** — Single pass through 49k matches:
```python
def build_features_fast(matches_df, min_date="2000-01-01"):
    # Walk through ALL matches chronologically
    # For each match:
    #   1. IF date >= min_date: snapshot current state as a feature row
    #   2. ALWAYS: update running state (Elo, form, H2H)
    
    # This is O(n) — one pass through all matches
    # The "trick": we compute features BEFORE updating state
    # So each match's features reflect what was known AT THAT TIME
```

**_snapshot_features()** — Capture 14 features for a match:
```python
def _snapshot_features(home, away, elo, form_results, form_gf, form_gc, h2h_wins):
    return [
        elo_home,           # Current Elo of home team
        elo_away,           # Current Elo of away team  
        elo_diff,           # elo_home - elo_away (most important feature)
        form_home_pts_5,    # Home team's avg points in last 5 (0-1 scale, /3)
        form_away_pts_5,    # Away team's avg points in last 5
        form_home_pts_10,   # Home team's avg points in last 10
        form_away_pts_10,   # Away team's avg points in last 10
        form_home_gf_5,     # Home team's avg goals scored (last 5)
        form_away_gf_5,     # Away team's avg goals scored (last 5)
        form_home_gc_5,     # Home team's avg goals conceded (last 5)
        form_away_gc_5,     # Away team's avg goals conceded (last 5)
        h2h_matches,        # How many times these teams have played
        h2h_home_win_pct,   # % of H2H matches home team won (0.5 if never met)
        h2h_goal_diff,      # Net goal difference in H2H history
    ]
```

**_update_elo(), _update_form(), _update_h2h()** — Update running state after each match:
```python
# These maintain Python dicts that accumulate stats per team:
# elo: {team: current_rating}
# form_results: {team: [last 20 results as 3/1/0]}
# form_gf: {team: [last 20 goals scored]}
# h2h_wins: {(teamA, teamB): {matches, teamA_wins, teamB_wins, draws, gd}}
```

---

## 10. `src/models/ensemble.py` — Ensemble Classifier (97 lines)

```python
class EnsembleClassifier:
    def __init__(self):
        self.xgb = XGBClassifier(...)      # Gradient boosted trees
        self.lr = LogisticRegression(...)   # Linear model (well-calibrated probs)
        self.rf = RandomForestClassifier(...)  # Bagged trees (robust)
        self.weights = [0.45, 0.25, 0.30]  # Initial blend weights

    def fit(self, X, y):
        self.xgb.fit(X, y)  # Train all 3 on same data
        self.lr.fit(X, y)
        self.rf.fit(X, y)

    def predict_proba(self, X):
        # Each model outputs [P(loss), P(draw), P(win)] per match
        # Blend: weighted average of all 3
        blended = w[0]*xgb + w[1]*lr + w[2]*rf
        return normalize(blended)  # Ensure rows sum to 1.0

    def optimize_weights(self, X_val, y_val):
        # Try different weight combinations
        # Find the one that minimizes log-loss on validation data
        # Uses Nelder-Mead optimizer (no gradient needed)
```

---

## 11. `src/models/dixon_coles.py` — Score Prediction (200 lines)

**The Dixon-Coles model** extends Poisson with a correction factor:

Standard Poisson: `P(home=x, away=y) = Poisson(x; λ) × Poisson(y; μ)`

Problem: This underestimates draws (especially 0-0, 1-1).

Dixon-Coles fix: `P(x,y) = τ(x, y, λ, μ, ρ) × Poisson(x; λ) × Poisson(y; μ)`

```python
def tau(x, y, lambda_val, mu_val, rho):
    # Only affects low-scoring outcomes:
    if x==0, y==0: return 1 - λ*μ*ρ     # 0-0 more likely when ρ<0
    if x==0, y==1: return 1 + λ*ρ        
    if x==1, y==0: return 1 + μ*ρ        
    if x==1, y==1: return 1 - ρ          # 1-1 more likely when ρ<0
    else: return 1.0                      # No correction for 2+ goals
```

**Model parameters:**
```python
# Per team: attack strength (how good at scoring)
#           defence strength (how bad at preventing goals)  
# Global:  home_advantage (boost for home team)
#           rho (correlation correction, typically -0.1 to -0.2)

# For a match home vs away:
λ = exp(attack_home + defence_away + home_adv)  # Expected home goals
μ = exp(attack_away + defence_home)              # Expected away goals
```

**Fitting:** Maximum likelihood estimation — find parameters that make the observed results most probable, with time-decay weighting (recent matches count more).

---

## 12. `src/models/train.py` — Training Pipeline (195 lines)

```python
def main():
    # [1] Load 49,477 international matches
    # [2] Build feature matrix (single-pass, 0.5 seconds)
    # [3] Train ensemble classifier:
    #     - Split 80/20 time-based (train on older, validate on newer)
    #     - Fit XGBoost + LR + RF
    #     - Optimize blend weights on validation set
    #     - Refit on full dataset with optimal weights
    # [4] Train Dixon-Coles:
    #     - Filter to matches between WC 2026 teams only (keeps params small)
    #     - Run L-BFGS-B optimization (~80 seconds)
    # [5] Extract final Elo/form/H2H state (walk through all matches)
    # [6] Save everything to model.pkl:
    #     {model, feature_names, dixon_coles, elo, form_results, form_gf, form_gc, h2h_wins}
```

---

## Data Flow Summary

```
User enters score "2-1" for match 7 on Group Stage tab
    → st.session_state.match_results[7] = {home_goals: 2, away_goals: 1}
    → persist_results() writes to data/match_results.json
    → Page reruns (st.rerun())
    → Standings recalculate (compute_standings reads from session_state)
    → User navigates to Predictions tab
    → predict_match() calls compute_tournament_form() 
        → Reads session_state.match_results → Brazil has 3 pts from 1 match
        → form_factor = 3/(1*3) = 1.0 (perfect form)
        → xG boosted by +7.5% ((1.0 - 0.5) * 0.15)
    → Updated prediction displayed

User marks Neymar unavailable for match 7
    → st.session_state.unavailable[(7, "Brazil")] = {"Neymar"}
    → persist_unavailable() writes to data/unavailable_players.json
    → predict_match() sees home_unavailable_count=1
        → squad_mod = 1.0 - 1*0.02 = 0.98
        → xG_home *= 0.98 (2% reduction)
    → Slightly lower win probability for Brazil in that match
```
