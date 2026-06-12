# FIFA World Cup Prediction - Public Data Sources Assessment

**Date assessed:** 2026-06-11

---

## 1. International Football Match Results (martj42)

**Repository:** https://github.com/martj42/international_results  
**Raw CSV:** https://raw.githubusercontent.com/martj42/international_results/master/results.csv  
**License:** CC0-1.0 (Public Domain)

### Accessibility
- Fully accessible via raw GitHub URLs - no authentication needed
- Can be downloaded programmatically with `curl`, `wget`, `pandas.read_csv()` directly from URL
- Also mirrored on Kaggle: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

### Files Available

| File | Columns | Description |
|------|---------|-------------|
| `results.csv` | `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral` | Core match results |
| `goalscorers.csv` | `date, home_team, away_team, team, scorer, own_goal, penalty` | Individual goal records |
| `shootouts.csv` | `date, home_team, away_team, winner, first_shooter` | Penalty shootout outcomes |

### Key Stats
- **Records:** 49,398 matches
- **Date range:** 1872 to 2024 (continuously updated)
- **Coverage:** All men's full international matches (FIFA World Cup, continental championships, friendlies, qualifiers, etc.)
- **Exclusions:** Olympic Games, B-teams, U-23, league select teams

### Limitations / Gotchas
- Scores are full-time only (excludes penalty shootout scores; use shootouts.csv for those)
- Team names use current designations (e.g., historical "Ireland" is listed as "Northern Ireland")
- Country names reflect designations at match time
- Men's matches only
- No player-level statistics beyond goalscorers
- Update frequency not guaranteed (community-maintained)

### Usefulness for Prediction: HIGH
This is the single most important dataset - provides the complete match history needed to build predictive models (head-to-head records, home/away performance, form calculations, tournament context).

---

## 2. Fjelstul World Cup Historical Database

**Repository:** https://github.com/jfjelstul/worldcup  
**CSV directory:** https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/  
**License:** CC-BY-SA 4.0

### Accessibility
- Fully accessible via raw GitHub URLs for CSV files
- Available in 4 formats: RData, CSV, JSON, SQLite
- Also available as R package: `devtools::install_github("jfjelstul/worldcup")`
- Web interface at WorldCups.ai

### Datasets (27 total)

**Core match data:**
| Dataset | Key Columns | Records (approx.) |
|---------|-------------|-------------------|
| `tournaments.csv` | key_id, tournament_id, tournament_name, year, start_date, end_date, host_country, winner, count_teams | 30 |
| `matches.csv` | 37+ columns including match_date, teams, scores, extra_time, penalty_shootout, result flags | 960+ |
| `group_standings.csv` | team, played, wins, draws, losses, goals_for, goals_against, goal_difference, points, advanced | 626 |

**Player/squad data:**
| Dataset | Key Columns | Records (approx.) |
|---------|-------------|-------------------|
| `squads.csv` | tournament, team, player, shirt_number, position_name, position_code | 10,000+ |
| `player_appearances.csv` | match_date, team, player, position, starter, substitute | 25,000+ |
| `goals.csv` | match_date, team, player, minute, own_goal, penalty | 2,500+ |
| `bookings.csv` | match_date, player, yellow/red card | since 1970 |
| `substitutions.csv` | match_date, player_in, player_out, minute | since 1970 |

**Other useful datasets:** qualified_teams, tournament_standings, award_winners, referees, stadiums

### Key Stats
- **Date range:** 1930-2022 (men's), 1991-2019 (women's)
- **Coverage:** All FIFA World Cup tournaments (22 men's + 9 women's)
- **Data points:** 1.5 million+ across all datasets

### Limitations / Gotchas
- World Cup matches ONLY (no qualifiers, friendlies, or other tournaments)
- Player appearances only from 1970 onwards
- Bookings/substitutions only from 1970 onwards
- Women's data only through 2019
- Last updated through 2022 World Cup

### Usefulness for Prediction: MEDIUM-HIGH
Excellent for World Cup-specific features: tournament stage effects, squad composition, historical World Cup performance. Complements the broader match results dataset.

---

## 3. FIFA Rankings

### Primary Source Attempted
**URL:** https://raw.githubusercontent.com/cnc-data/fifa-rankings/main/data/rankings.csv  
**Status:** 404 Not Found - repository does not exist or has moved

### Alternative Sources

#### Kaggle: FIFA World Ranking 1992-2024
**URL:** https://www.kaggle.com/datasets/cashncarry/fifaworldranking  
- **Date range:** 1992-2024
- **Access:** Requires Kaggle account; download via web UI or `kaggle` CLI API
- **Expected columns:** rank, country_full, country_abrv, total_points, previous_points, rank_change, confederation, rank_date
- **Records:** ~60,000+ (rankings published ~10 times per year for 200+ teams)

#### Kaggle: FIFA Soccer Rankings (1993-now)
**URL:** https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now  
- Similar coverage, alternative maintainer

### How to Download FIFA Rankings Programmatically
```bash
# Using Kaggle CLI (requires API key in ~/.kaggle/kaggle.json)
pip install kaggle
kaggle datasets download -d cashncarry/fifaworldranking
```

### Limitations / Gotchas
- FIFA changed its ranking methodology in 2018 (from cumulative points to Elo-based system)
- Rankings only exist from 1993 onwards (FIFA introduced rankings in December 1992)
- No single freely accessible raw CSV on GitHub found (Kaggle requires authentication)
- Ranking updates are irregular (tied to international windows)
- Historical rankings may have gaps during methodology transitions

### Usefulness for Prediction: HIGH
FIFA rankings are one of the strongest single predictors of World Cup outcomes. Essential feature for any model.

---

## 4. Elo Ratings

### eloratings.net
**URL:** https://eloratings.net  
**Status:** No programmatic access available

- Website renders data via JavaScript (empty div containers populated dynamically)
- No API endpoints, CSV downloads, or documented data access methods found
- Data must be scraped from the website (check robots.txt and ToS first)
- Provides current and historical Elo ratings for all national teams

### Alternative: Compute Your Own Elo Ratings
Given the international results dataset (Source #1), Elo ratings can be computed from scratch:
```python
# Pseudocode for Elo calculation
K = 40  # for World Cup matches (varies by tournament importance)
for match in sorted_matches:
    expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
    actual_home = 1 if home_win else (0.5 if draw else 0)
    home_elo += K * (actual_home - expected_home)
    away_elo += K * (expected_away - actual_away)
```

### Alternative: ClubElo (club football only)
**URL:** http://clubelo.com/API  
- Provides API for club team Elo ratings
- NOT useful for national team predictions directly

### Limitations / Gotchas
- eloratings.net has no official API or download
- Scraping may violate ToS
- Best approach: compute Elo from match results (Source #1)
- The 2018 FIFA ranking reform made official FIFA rankings essentially Elo-based

### Usefulness for Prediction: HIGH
Elo ratings are historically one of the best predictors. Recommend computing from scratch using Source #1 data rather than relying on external Elo sources.

---

## 5. Additional Useful Sources

### 5a. OpenFootball World Cup
**Repository:** https://github.com/openfootball/world-cup  
**License:** CC0 (Public Domain)  
- Structured text format (Football.TXT) covering 1930-2026
- Includes match schedules, scores, goal times, scorers, venues
- Can be converted to SQL/JSON/CSV via football.db toolkit
- Useful for validation but less convenient than Source #2

### 5b. Transfermarkt (Squad Market Values)
**URL:** https://www.transfermarkt.com  
- Squad and player market valuations
- No official API; requires scraping or third-party packages
- Python package: `transfermarkt-api` or web scraping
- Useful feature: total squad value correlates with tournament performance
- **Limitation:** Scraping-only access, values start mid-2000s

### 5c. FBref / StatsBomb (Advanced Player Stats)
**URL:** https://fbref.com  
- Detailed player statistics (xG, passes, defensive actions)
- StatsBomb provides free data for some tournaments
- Accessible via `soccerdata` Python package or scraping
- **Limitation:** Advanced metrics only available from ~2017 onwards

### 5d. World Cup Betting Odds (Historical)
- Historical betting odds can serve as a baseline predictor
- Sources: football-data.co.uk, oddsportal.com
- Typically requires scraping
- **Limitation:** Limited historical depth for World Cup specifically

### 5e. FIFA Official Data
**URL:** https://www.fifa.com/fifa-world-ranking/  
- Current rankings available on FIFA website
- Historical data not easily bulk-downloadable
- Better to use Kaggle mirrors (Source #3)

---

## Summary Table

| Source | Access Method | Records | Date Range | Prediction Value | Ease of Use |
|--------|-------------|---------|------------|-----------------|-------------|
| martj42 international_results | Raw GitHub CSV | 49,398 | 1872-2024 | HIGH | Easy |
| Fjelstul World Cup DB | Raw GitHub CSV/JSON/SQLite | 1.5M+ data points | 1930-2022 | MEDIUM-HIGH | Easy |
| FIFA Rankings (Kaggle) | Kaggle API | ~60,000 | 1992-2024 | HIGH | Medium |
| Elo Ratings | Compute from Source #1 | N/A | 1872-present | HIGH | Medium |
| OpenFootball | Football.TXT + converter | All WC matches | 1930-2026 | LOW-MEDIUM | Medium |
| Transfermarkt | Scraping | Varies | 2004-present | MEDIUM | Hard |
| FBref/StatsBomb | Scraping/API | Varies | 2017-present | MEDIUM | Hard |

---

## Recommended Data Pipeline

1. **Start with:** `martj42/international_results` - download results.csv as primary match database
2. **Enrich with:** `jfjelstul/worldcup` - add World Cup-specific features (squad data, tournament structure)
3. **Add rankings:** Download FIFA rankings from Kaggle
4. **Compute Elo:** Calculate Elo ratings from match history (gives historical depth beyond 1993)
5. **Optional:** Add squad market values from Transfermarkt for recent tournaments

### Quick Start Download
```python
import pandas as pd

# Source 1: All international matches
matches = pd.read_csv('https://raw.githubusercontent.com/martj42/international_results/master/results.csv')
goalscorers = pd.read_csv('https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv')
shootouts = pd.read_csv('https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv')

# Source 2: World Cup specific data
wc_matches = pd.read_csv('https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv')
wc_squads = pd.read_csv('https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/squads.csv')
wc_goals = pd.read_csv('https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/goals.csv')
wc_standings = pd.read_csv('https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/group_standings.csv')
wc_tournaments = pd.read_csv('https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/tournaments.csv')

# Source 3: FIFA Rankings (requires Kaggle download first)
# rankings = pd.read_csv('data/raw/fifa_ranking.csv')
```
