# FIFA World Cup 2026 — Winner Prediction System

A Python-based machine learning application that predicts the winner of the FIFA World Cup 2026. Uses an ensemble classifier (XGBoost + Logistic Regression + Random Forest) combined with a Dixon-Coles goal model, trained on 49,000+ historical international matches.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
cd Predictor_WC
uv sync
```

### Train the Model

```bash
uv run python src/models/train.py
```

Trains in ~90 seconds. Outputs `data/model.pkl` (8 MB).

### Run the App

```bash
uv run streamlit run app/streamlit_app.py --server.port 3000
```

Open http://localhost:3000 in your browser.

---

## Features

### ⚽ Group Stage
- Browse all 72 group stage matches with flag images, venue, city, and EST time
- Enter match scores interactively (score inputs + confirm button per match)
- Live-updating group standings (P/W/D/L/GF/GA/GD/Pts)
- Green highlighting for qualified teams (top 2 + best 8 third-place)
- View modes: All Matches (paginated) or By Group (dropdown)
- 📋 Squad availability button per match — popup to mark players unavailable

### 🏆 Knockout Stage
- Tournament bracket layout: R32 → R16 in grid, then QF → SF → Final converging from both sides
- Team names resolved from group stage results (Winners, Runners-up, 3rd-place qualifiers)
- Score entry directly on every bracket card (inline inputs + ✓ button)
- 📋 Squad availability button on every bracket card
- Final displayed with trophy styling and champion label

### 🔮 Predictions (Groups)
- Model-predicted W/D/L probabilities with colored probability bar
- Expected goals (xG), Elo ratings, and most likely scoreline
- **Predicted standings table** per group
- Predictions **recalculate live** when scores or squad availability change

### 🔮 Predictions (Knockout)
- Full tournament simulation resolving ALL matchups including 3rd-place qualifiers
- Champion banner at the top
- Bracket view with predicted scores and win probabilities
- Winners highlighted green, losers light red
- Rest days between matches factored into predictions

### 📊 Methodology & Analysis
- **Approach** — step-by-step walkthrough of the prediction system
- **Datasets** — sources, coverage, and purpose
- **EDA** — interactive Plotly charts (matches/year, outcome distribution, goals trend, Elo ratings, home advantage)
- **Model Architecture** — 3-layer diagram, ensemble weights, Dixon-Coles parameters
- **Validation (2022 WC)** — backtested on all 64 Qatar 2022 matches: **59.4% accuracy**

### Navigation & Persistence
- Sidebar button navigation (active page highlighted)
- All scores and squad data persist to JSON files on disk
- Data survives page refreshes, browser closes, server restarts
- **🔄 Retrain Model** button in sidebar — runs the full training pipeline (~90s) to refit ensemble + Dixon-Coles with latest data

---

## Prediction Model

### Layer 1: Ensemble Classifier + Dixon-Coles

**Win/Draw/Loss prediction:**
- XGBoost (150 trees, depth 5) — weight: 32%
- Logistic Regression — weight: 56%
- Random Forest (200 trees, depth 8) — weight: 12%
- Weights optimized via log-loss on validation set
- Validation accuracy: **59.9%** (comparable to professional bookmakers)

**Score prediction (Dixon-Coles):**
- MLE-optimized attack/defence strength per team
- ρ = -0.17 (corrects for low-scoring draws being underestimated by Poisson)
- Time-decay weighting (half-life = 365 days)
- Fitted on 740 matches between WC teams (2020–2026)

**Final blend:** 50% Ensemble W/D/L + 50% Dixon-Coles W/D/L

### Layer 2: Tournament Adjustments
- WC pedigree boost (±2% per level difference)
- Rest days penalty (<3 days rest = -8% xG, -4% win probability)

### Layer 3: Live Adjustments
- Squad availability: -2% xG per unavailable player (max -30%)
- In-tournament form: computed from entered scores, adjusts xG ±15%
- **Live Elo/form/H2H updates**: every saved score immediately updates the model's feature state (Elo with K=60, rolling form, H2H records)
- On restart, all saved scores are replayed into the model to rebuild state
- Recalculates on every page render

### Features (14 per match)
| Feature | Description |
|---------|-------------|
| `elo_home` / `elo_away` / `elo_diff` | Team Elo ratings (computed from 150 years of matches) |
| `form_home_pts_5` / `form_away_pts_5` | Points per match over last 5 games |
| `form_home_pts_10` / `form_away_pts_10` | Points per match over last 10 games |
| `form_home_gf_5` / `form_away_gf_5` | Average goals scored (last 5) |
| `form_home_gc_5` / `form_away_gc_5` | Average goals conceded (last 5) |
| `h2h_matches` | Number of historical meetings |
| `h2h_home_win_pct` | Head-to-head win percentage |
| `h2h_goal_diff` | Historical goal difference in meetings |

---

## Project Structure

```
Predictor_WC/
├── app/
│   ├── streamlit_app.py      # Entry point — config, CSS, sidebar nav, page routing
│   ├── config.py             # Constants, team ISO codes, flag_img() helper
│   ├── data.py               # Data loading, JSON persistence, schedule processing
│   ├── standings.py          # Standings computation, qualification, knockout resolution
│   ├── components.py         # UI widgets (match rows, score input, squad dialog, pagination)
│   ├── predictions.py        # Prediction engine wrapper, tournament simulation
│   ├── bracket.py            # Tournament bracket rendering (converging tree layout)
│   └── methodology.py        # About page — approach, datasets, EDA charts, model details
├── src/
│   ├── features/
│   │   ├── builder.py        # Fast single-pass feature matrix builder
│   │   └── elo.py            # Elo rating computation (K-factor by tournament type)
│   └── models/
│       ├── ensemble.py       # Ensemble classifier (XGBoost + LR + RF + weight optimization)
│       ├── dixon_coles.py    # Dixon-Coles model (MLE, L-BFGS-B optimizer)
│       └── train.py          # Training script — produces model.pkl
├── data/
│   ├── raw/
│   │   ├── wc2026_schedule.csv       # 104 matches (Wikipedia, Dec 2025 draw)
│   │   ├── wc2026_squads.csv         # 1,248 players across 48 teams
│   │   ├── international_results.csv # 49,477 matches, 1872–2026
│   │   ├── wc_matches.csv           # 1,248 World Cup matches, 1930–2022
│   │   ├── wc_group_standings.csv    # 626 WC group stage records
│   │   └── DATA_SOURCES.md          # Dataset assessment
│   ├── model.pkl                     # Trained model (ensemble + Dixon-Coles + Elo + form)
│   ├── match_results.json            # User-entered scores (persisted)
│   └── unavailable_players.json      # Player availability per match (persisted)
├── .streamlit/
│   └── config.toml                   # Theme + server config
├── pyproject.toml                    # Project metadata and dependencies
└── uv.lock                           # Locked dependency versions
```

---

## Data Sources

| Source | Records | Date Range | Use |
|--------|---------|------------|-----|
| [martj42/international_results](https://github.com/martj42/international_results) | 49,477 | 1872–2026 | Training, Elo, form, H2H |
| [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) | 1,248 WC matches | 1930–2022 | Validation |
| Wikipedia (2026 squads) | 1,248 players | 2026 | Squad availability |
| Wikipedia (2026 schedule) | 104 matches | 2026 | Tournament structure |
| [flagcdn.com](https://flagcdn.com) | — | — | Flag images |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Package Manager | uv |
| Data Processing | Polars |
| ML Models | XGBoost, scikit-learn (LR, RF), scipy (Dixon-Coles MLE) |
| Charts | Plotly Express |
| UI Framework | Streamlit 1.58 |
| Flag Images | flagcdn.com |
| Theme | Light corporate blue |

---

## Model Validation

Backtested on all 64 matches of the Qatar 2022 FIFA World Cup:

| Metric | Value |
|--------|-------|
| Accuracy (outcome) | **59.4%** (38/64 correct) |
| Random baseline | 33% |
| Professional bookmakers | ~60% |

The model correctly predicted outcomes for group stage favorites and most knockout results. Notable misses include upsets (Saudi Arabia beating Argentina, Morocco's deep run).

---

## How Predictions Update in Real-Time

When you save a score (click ✓):

1. **Persisted to disk** → `data/match_results.json` (survives restarts)
2. **Model state updated instantly** → Elo (K=60, World Cup weight), form (rolling 5/10), H2H all recalculated
3. **Page reruns** → predictions use the updated model state immediately

When you toggle squad availability (📋 button):

4. **Persisted to disk** → `data/unavailable_players.json`
5. **Layer 3 adjusts xG** → -2% per unavailable player on next prediction render

On app restart:

6. `model.pkl` loaded (pre-trained on 49k historical matches)
7. `match_results.json` replayed into the model → Elo/form/H2H updated with all entered scores
8. Predictions reflect full history + your tournament data seamlessly

**Effect:** A team that beats a strong opponent sees their Elo jump (K=60 is the highest weight), their form improves, and their H2H record updates — all feeding directly into the next prediction's Layer 1 features.

**Full retrain (optional):** Click "🔄 Retrain Model" in the sidebar to refit the Dixon-Coles parameters and ensemble weights from scratch with all available data. This takes ~90 seconds and produces a new `model.pkl`.
