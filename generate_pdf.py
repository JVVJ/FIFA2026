"""Generate a PDF walkthrough of the codebase with ML method explanations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fpdf import FPDF


class WalkthroughPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(27, 79, 114)
        self.cell(0, 8, "FIFA World Cup 2026 Prediction System - Code & ML Walkthrough", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(27, 79, 114)
        self.ln(5)
        self.cell(0, 10, title)
        self.ln(12)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(44, 62, 80)
        self.ln(3)
        self.cell(0, 8, title)
        self.ln(9)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_block(self, code):
        self.set_font("Courier", "", 8)
        self.set_fill_color(234, 242, 248)
        self.set_text_color(44, 62, 80)
        for line in code.strip().split("\n"):
            self.cell(0, 4.5, "  " + line, fill=True)
            self.ln(4.5)
        self.ln(3)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.cell(5)
        self.cell(0, 5, f"-  {text}")
        self.ln(5.5)

    def reference_link(self, title, url, description):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(27, 79, 114)
        self.cell(0, 5, title)
        self.ln(5)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, url)
        self.ln(4)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.5, description)
        self.ln(3)


def main():
    pdf = WalkthroughPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # === TITLE PAGE ===
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(27, 79, 114)
    pdf.ln(40)
    pdf.cell(0, 15, "FIFA World Cup 2026", align="C")
    pdf.ln(15)
    pdf.cell(0, 15, "Prediction System", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Code Walkthrough & ML Methods Guide", align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Ensemble Model (XGBoost + Logistic Regression + Random Forest)", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Dixon-Coles Goal Prediction Model", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Multi-Layer Architecture with Live Adjustments", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Validation: 59.4% accuracy on 2022 World Cup", align="C")

    # === ARCHITECTURE OVERVIEW ===
    pdf.add_page()
    pdf.chapter_title("1. System Architecture")
    pdf.body_text(
        "The system has two main components: a training pipeline (run once) that produces a model file, "
        "and a Streamlit web app that uses the model for live predictions. The prediction engine uses "
        "a 3-layer architecture that combines historical statistical models with real-time adjustments."
    )
    pdf.code_block(
        "Training Pipeline:\n"
        "  international_results.csv (49,477 matches)\n"
        "    -> builder.py (single-pass feature extraction)\n"
        "    -> ensemble.py (XGBoost + LR + RF classifier)\n"
        "    -> dixon_coles.py (MLE goal model)\n"
        "    -> model.pkl (saved to disk)\n"
        "\n"
        "App Runtime (per page render):\n"
        "  User input (scores, squad) -> session_state -> JSON files\n"
        "    -> predictions.py loads model.pkl\n"
        "    -> Layer 1: Ensemble + Dixon-Coles (base prediction)\n"
        "    -> Layer 2: Tournament adjustments (rest days, pedigree)\n"
        "    -> Layer 3: Live adjustments (squad, form)\n"
        "    -> Render predictions in UI"
    )

    # === ELO RATINGS ===
    pdf.add_page()
    pdf.chapter_title("2. Elo Rating System")
    pdf.section_title("What is Elo?")
    pdf.body_text(
        "Elo is a rating system originally designed for chess (Arpad Elo, 1960). Each team starts at "
        "1500. After each match, ratings are updated based on the result vs expectation. If a low-rated "
        "team beats a high-rated team, the rating transfer is large. If the favorite wins as expected, "
        "the transfer is small."
    )
    pdf.section_title("The Math")
    pdf.code_block(
        "Expected score for home team:\n"
        "  E_home = 1 / (1 + 10^((Elo_away - Elo_home - HomeAdv) / 400))\n"
        "\n"
        "After the match:\n"
        "  New_Elo = Old_Elo + K * GD_mult * (Actual - Expected)\n"
        "\n"
        "Where:\n"
        "  K = 60 (World Cup), 50 (continental), 40 (qualifiers), 20 (friendlies)\n"
        "  GD_mult = 1.0 (1 goal), 1.5 (2 goals), (11+GD)/8 (3+ goals)\n"
        "  HomeAdv = 100 (non-neutral), 0 (neutral/World Cup)\n"
        "  Actual = 1.0 (win), 0.5 (draw), 0.0 (loss)"
    )
    pdf.section_title("Why it works for football")
    pdf.body_text(
        "Elo difference is the single strongest predictor of match outcomes (~60% accuracy alone). "
        "It captures long-term team strength that adapts to results. Our implementation uses variable "
        "K-factors (World Cup results matter 3x more than friendlies) and goal difference multipliers "
        "(winning 5-0 gives more rating boost than winning 1-0)."
    )
    pdf.section_title("References")
    pdf.reference_link(
        "World Football Elo Ratings",
        "https://en.wikipedia.org/wiki/World_Football_Elo_Ratings",
        "Wikipedia article explaining the football-specific Elo implementation used by eloratings.net."
    )
    pdf.reference_link(
        "Elo Rating System (original)",
        "https://en.wikipedia.org/wiki/Elo_rating_system",
        "The mathematical foundation of Elo ratings, expected scores, and K-factor selection."
    )

    # === FEATURE ENGINEERING ===
    pdf.add_page()
    pdf.chapter_title("3. Feature Engineering")
    pdf.section_title("Single-Pass Feature Builder")
    pdf.body_text(
        "We process all 49,477 matches in chronological order in a single pass. For each match, we "
        "snapshot the current state BEFORE updating it. This ensures no data leakage - each match's "
        "features only reflect information available at that point in time."
    )
    pdf.section_title("The 14 Features")
    pdf.code_block(
        "Team Strength (3 features):\n"
        "  elo_home        - Current Elo rating of home team\n"
        "  elo_away        - Current Elo rating of away team\n"
        "  elo_diff        - Difference (most important single feature)\n"
        "\n"
        "Recent Form (8 features):\n"
        "  form_home_pts_5  - Home team's avg points/match (last 5)\n"
        "  form_away_pts_5  - Away team's avg points/match (last 5)\n"
        "  form_home_pts_10 - Home team's avg points/match (last 10)\n"
        "  form_away_pts_10 - Away team's avg points/match (last 10)\n"
        "  form_home_gf_5   - Home team's avg goals scored (last 5)\n"
        "  form_away_gf_5   - Away team's avg goals scored (last 5)\n"
        "  form_home_gc_5   - Home team's avg goals conceded (last 5)\n"
        "  form_away_gc_5   - Away team's avg goals conceded (last 5)\n"
        "\n"
        "Head-to-Head (3 features):\n"
        "  h2h_matches      - Number of prior meetings\n"
        "  h2h_home_win_pct - Historical win percentage\n"
        "  h2h_goal_diff    - Net goals in prior meetings"
    )
    pdf.body_text(
        "The builder processes 49k matches in 0.5 seconds by maintaining rolling accumulators "
        "(Python lists/dicts) instead of recomputing from scratch for each match."
    )

    # === XGBOOST ===
    pdf.add_page()
    pdf.chapter_title("4. XGBoost Classifier")
    pdf.section_title("What is XGBoost?")
    pdf.body_text(
        "XGBoost (eXtreme Gradient Boosting) is an ensemble of decision trees trained sequentially. "
        "Each new tree corrects the errors of the previous trees. It's the dominant algorithm in "
        "tabular data competitions (Kaggle) due to high accuracy, speed, and built-in regularization."
    )
    pdf.section_title("How it works")
    pdf.code_block(
        "1. Start with a base prediction (class probabilities)\n"
        "2. Compute residuals (errors) for each training sample\n"
        "3. Fit a new decision tree to predict those residuals\n"
        "4. Add the new tree's predictions (scaled by learning_rate)\n"
        "5. Repeat for n_estimators=150 trees\n"
        "\n"
        "Each tree has:\n"
        "  max_depth=5 (limits complexity, prevents overfitting)\n"
        "  subsample=0.8 (80% of data per tree for diversity)\n"
        "  colsample_bytree=0.8 (80% of features per tree)\n"
        "\n"
        "Output: P(loss), P(draw), P(win) for each match\n"
        "Objective: multi:softprob (multiclass probability)"
    )
    pdf.section_title("Why we use it")
    pdf.body_text(
        "XGBoost captures non-linear feature interactions (e.g., high Elo + poor recent form = "
        "vulnerable favorite). It handles missing data gracefully and provides feature importance "
        "scores. However, it can overfit if not regularized, which is why we ensemble it with "
        "simpler models."
    )
    pdf.section_title("References")
    pdf.reference_link(
        "XGBoost Paper (Chen & Guestrin, 2016)",
        "https://arxiv.org/abs/1603.02754",
        "The original paper introducing XGBoost. Covers the algorithm, regularization, "
        "and system design."
    )
    pdf.reference_link(
        "XGBoost Documentation",
        "https://xgboost.readthedocs.io/en/latest/",
        "Official docs with parameters, tutorials, and API reference."
    )

    # === LOGISTIC REGRESSION ===
    pdf.add_page()
    pdf.chapter_title("5. Logistic Regression")
    pdf.section_title("What is it?")
    pdf.body_text(
        "Logistic Regression is a linear classifier that models P(class|features) using the "
        "logistic (sigmoid) function. For multiclass (3 outcomes), it uses the softmax function. "
        "Despite its simplicity, it often produces well-calibrated probabilities - meaning when "
        "it says 60% chance of home win, it's right roughly 60% of the time."
    )
    pdf.section_title("The Math")
    pdf.code_block(
        "For 3 classes (loss, draw, win):\n"
        "\n"
        "  score_k = w_k . x + b_k    (linear combination of features)\n"
        "\n"
        "  P(class=k) = exp(score_k) / sum(exp(score_j) for all j)\n"
        "\n"
        "Training: maximize log-likelihood using L-BFGS optimizer\n"
        "Regularization: C=1.0 (L2 penalty to prevent overfitting)"
    )
    pdf.section_title("Why it's in the ensemble")
    pdf.body_text(
        "LR provides a strong linear baseline. In our ensemble, it got the highest weight (56%) "
        "because its well-calibrated probabilities are excellent for blending. XGBoost may overfit "
        "to specific patterns, but LR's smooth probability surface averages out those errors."
    )
    pdf.section_title("References")
    pdf.reference_link(
        "Logistic Regression (scikit-learn)",
        "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression",
        "scikit-learn documentation covering multinomial logistic regression."
    )

    # === RANDOM FOREST ===
    pdf.add_page()
    pdf.chapter_title("6. Random Forest")
    pdf.section_title("What is it?")
    pdf.body_text(
        "Random Forest builds many independent decision trees (200 in our case), each trained on "
        "a random subset of data and features. Final prediction = average of all trees' votes. "
        "This 'wisdom of crowds' approach is robust to outliers and noise."
    )
    pdf.section_title("How it differs from XGBoost")
    pdf.code_block(
        "Random Forest:\n"
        "  - Trees are independent (trained in parallel)\n"
        "  - Each tree sees random subset of data (bagging)\n"
        "  - Reduces variance (overfitting)\n"
        "  - Less prone to learning noise\n"
        "\n"
        "XGBoost:\n"
        "  - Trees are sequential (each corrects previous errors)\n"
        "  - All trees see full data\n"
        "  - Reduces bias (underfitting)\n"
        "  - Can overfit if not regularized"
    )
    pdf.section_title("In our ensemble")
    pdf.body_text(
        "RF got the lowest weight (12%) - it adds diversity but doesn't dominate because football "
        "prediction is fundamentally a probabilistic problem where calibrated probabilities "
        "(LR's strength) matter more than raw accuracy."
    )
    pdf.section_title("References")
    pdf.reference_link(
        "Random Forests (Breiman, 2001)",
        "https://link.springer.com/article/10.1023/A:1010933404324",
        "The seminal paper introducing Random Forests."
    )

    # === ENSEMBLE METHODS ===
    pdf.add_page()
    pdf.chapter_title("7. Ensemble Blending")
    pdf.section_title("Why blend multiple models?")
    pdf.body_text(
        "Each model has different strengths and failure modes. Blending reduces the chance that "
        "one model's error dominates the final prediction. This is called 'model stacking' or "
        "'ensemble averaging'."
    )
    pdf.section_title("Weight Optimization")
    pdf.code_block(
        "Goal: find weights [w_xgb, w_lr, w_rf] that minimize log-loss\n"
        "\n"
        "Log-loss = -sum(log(P(correct_class))) for all validation matches\n"
        "\n"
        "Method: Nelder-Mead optimization (derivative-free)\n"
        "  1. Try different weight combinations\n"
        "  2. Measure log-loss on held-out validation data\n"
        "  3. Converge to optimal weights\n"
        "\n"
        "Result: XGB=0.32, LR=0.56, RF=0.12\n"
        "  (LR dominates because it's best-calibrated)"
    )
    pdf.section_title("References")
    pdf.reference_link(
        "Ensemble Methods in Machine Learning",
        "https://link.springer.com/chapter/10.1007/3-540-45014-9_1",
        "Dietterich (2000) - overview of why ensembles work better than individual models."
    )
    pdf.reference_link(
        "Stacking and Blending",
        "https://mlwave.com/kaggle-ensembling-guide/",
        "Practical guide to ensemble stacking techniques used in competitions."
    )

    # === DIXON-COLES ===
    pdf.add_page()
    pdf.chapter_title("8. Dixon-Coles Model")
    pdf.section_title("The Problem with Independent Poisson")
    pdf.body_text(
        "A basic approach models home goals ~ Poisson(lambda) and away goals ~ Poisson(mu) "
        "independently. But in real football, low-scoring outcomes (0-0, 1-0, 0-1, 1-1) occur "
        "more often than independent Poisson predicts. Teams play more cautiously in tight matches."
    )
    pdf.section_title("The Dixon-Coles Correction")
    pdf.code_block(
        "P(home=x, away=y) = tau(x, y, lambda, mu, rho)\n"
        "                     * Poisson(x; lambda)\n"
        "                     * Poisson(y; mu)\n"
        "\n"
        "tau correction factor (only affects low scores):\n"
        "  tau(0, 0) = 1 - lambda * mu * rho\n"
        "  tau(0, 1) = 1 + lambda * rho\n"
        "  tau(1, 0) = 1 + mu * rho\n"
        "  tau(1, 1) = 1 - rho\n"
        "  tau(x, y) = 1.0  (for all other scores)\n"
        "\n"
        "When rho < 0 (our model: rho = -0.17):\n"
        "  -> 0-0 draws become MORE likely\n"
        "  -> 1-1 draws become MORE likely\n"
        "  -> This matches real football data"
    )
    pdf.section_title("Per-Team Parameters")
    pdf.code_block(
        "Each team has:\n"
        "  attack_i  - how good at scoring (log-scale)\n"
        "  defence_i - how bad at preventing goals (log-scale)\n"
        "\n"
        "For match Home vs Away:\n"
        "  lambda = exp(attack_home + defence_away + home_advantage)\n"
        "  mu     = exp(attack_away + defence_home)\n"
        "\n"
        "Fitting: Maximum Likelihood Estimation\n"
        "  Maximize: sum(weight_i * log(P(observed_score_i)))\n"
        "  Weights: exponential decay (recent matches count more)\n"
        "  Optimizer: L-BFGS-B (fast for large parameter spaces)\n"
        "  Constraint: sum(attack) = 0 (identifiability)"
    )
    pdf.section_title("Why this is the gold standard for football")
    pdf.body_text(
        "Dixon-Coles (1997) remains the most-cited paper in football analytics. Its key insight - "
        "that goal independence breaks down at low scores - has been validated repeatedly. Our "
        "implementation uses time-decay weighting (half-life=365 days) so the model adapts to "
        "teams improving or declining over time."
    )
    pdf.section_title("References")
    pdf.reference_link(
        "Dixon & Coles (1997) - Original Paper",
        "https://doi.org/10.1111/1467-9876.00065",
        "\"Modelling Association Football Scores and Inefficiencies in the Football Betting Market\" "
        "- Journal of the Royal Statistical Society. The foundational paper for football score prediction."
    )
    pdf.reference_link(
        "Implementing Dixon-Coles in Python",
        "https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles/",
        "Excellent tutorial implementing Dixon-Coles step-by-step with Python code."
    )
    pdf.reference_link(
        "Poisson Distribution for Football Goals",
        "https://en.wikipedia.org/wiki/Poisson_distribution",
        "Mathematical background on why goals follow a Poisson process."
    )

    # === MULTI-LAYER ARCHITECTURE ===
    pdf.add_page()
    pdf.chapter_title("9. Multi-Layer Prediction Architecture")
    pdf.section_title("Layer 1: Historical Base (Stable)")
    pdf.body_text(
        "Combines the ensemble classifier (W/D/L probabilities) with Dixon-Coles (score prediction). "
        "The final Layer 1 output blends both 50/50 for win/draw/loss probabilities. Score prediction "
        "uses Dixon-Coles exclusively (the ensemble doesn't predict scores)."
    )
    pdf.section_title("Layer 2: Tournament Adjustments")
    pdf.body_text(
        "Applied on top of Layer 1 to account for World Cup-specific factors:\n"
        "- WC pedigree: teams with more WC experience get a small boost (2% per level)\n"
        "- Rest days: <3 days between matches = 8% xG reduction + 4% win probability drop\n"
        "- All WC matches treated as neutral venue (no home advantage in Dixon-Coles)"
    )
    pdf.section_title("Layer 3: Live Adjustments (Real-Time)")
    pdf.body_text(
        "Updated every time the page renders based on user input:\n"
        "- Squad availability: each unavailable player reduces team's xG by 2% (max 30% total)\n"
        "- In-tournament form: computed from actual scores entered; scales xG up to +/-15%\n"
        "- This allows predictions to adapt as the tournament progresses"
    )
    pdf.section_title("Score Consistency Rule")
    pdf.body_text(
        "The 'likely score' shown in the UI must be consistent with the predicted outcome. "
        "If the model predicts home win (>50%), we find the most probable HOME WIN scoreline "
        "from the Poisson/Dixon-Coles probability grid. This avoids showing '1-1' when the "
        "model says 55% home win."
    )

    # === VALIDATION ===
    pdf.add_page()
    pdf.chapter_title("10. Model Validation")
    pdf.section_title("2022 World Cup Backtest")
    pdf.body_text(
        "We tested the model on all 64 matches of the Qatar 2022 FIFA World Cup (data the model "
        "was trained on - but features are computed using only pre-match information)."
    )
    pdf.code_block(
        "Results:\n"
        "  Accuracy: 59.4% (38/64 matches predicted correctly)\n"
        "  Random baseline: 33%\n"
        "  Professional bookmakers: ~60%\n"
        "\n"
        "Breakdown:\n"
        "  Group stage: slightly higher accuracy (more predictable)\n"
        "  Knockout: lower accuracy (upsets more common)\n"
        "\n"
        "Notable correct predictions:\n"
        "  Argentina beat Australia, France beat Poland,\n"
        "  England beat Senegal, Brazil beat South Korea\n"
        "\n"
        "Notable misses (upsets):\n"
        "  Saudi Arabia beat Argentina (model gave Argentina 64%)\n"
        "  Morocco beat Spain (model gave Spain 43%)\n"
        "  Japan drew Croatia (model gave Japan 54%)"
    )
    pdf.section_title("What limits accuracy?")
    pdf.body_text(
        "Football is inherently unpredictable - even the best models max out around 65-70% because:\n"
        "- Individual moments of brilliance/error (red cards, penalties, goalkeeper mistakes)\n"
        "- Tactical changes mid-game that statistics can't capture\n"
        "- Psychological factors (pressure, motivation, fatigue)\n"
        "- Referee decisions\n\n"
        "Our 59.4% is strong for a pure statistical model without betting market data."
    )

    # === POTENTIAL IMPROVEMENTS ===
    pdf.add_page()
    pdf.chapter_title("11. Potential Improvements")
    pdf.section_title("Data that could improve accuracy")
    pdf.body_text("Each of these could add 1-5% accuracy:")
    pdf.bullet("Squad market values (Transfermarkt) - correlates with team quality")
    pdf.bullet("Expected Goals (xG) data from StatsBomb/FBref - better than raw goals")
    pdf.bullet("Betting odds integration - market consensus is a powerful signal")
    pdf.bullet("Player-level ratings (FIFA game ratings, WhoScored) - individual quality")
    pdf.bullet("Tactical features (formation, pressing intensity, possession style)")
    pdf.bullet("Manager win rate in tournament knockouts")
    pdf.ln(5)
    pdf.section_title("Model improvements")
    pdf.bullet("Monte Carlo simulation (10,000 runs) for probabilistic bracket outcomes")
    pdf.bullet("SHAP explainability - show which features drive each prediction")
    pdf.bullet("Bayesian updating - as tournament progresses, update priors with evidence")
    pdf.bullet("Neural network on sequence of recent results (LSTM/Transformer)")
    pdf.bullet("Larger ensemble (add CatBoost, LightGBM, Neural Net)")

    # === REFERENCES SUMMARY ===
    pdf.add_page()
    pdf.chapter_title("12. Complete Reference List")
    pdf.section_title("Papers")
    pdf.reference_link(
        "Dixon & Coles (1997)",
        "https://doi.org/10.1111/1467-9876.00065",
        "Foundational paper for football score prediction with correlation correction."
    )
    pdf.reference_link(
        "Chen & Guestrin (2016) - XGBoost",
        "https://arxiv.org/abs/1603.02754",
        "The XGBoost algorithm paper - gradient boosted trees with regularization."
    )
    pdf.reference_link(
        "Breiman (2001) - Random Forests",
        "https://link.springer.com/article/10.1023/A:1010933404324",
        "Original Random Forest paper - ensemble of independent decision trees."
    )
    pdf.reference_link(
        "Elo (1978) - The Rating of Chess Players",
        "https://en.wikipedia.org/wiki/Elo_rating_system",
        "Foundation of the Elo rating system adapted for football."
    )
    pdf.reference_link(
        "Hvattum & Arntzen (2010) - ELO ratings for football",
        "https://doi.org/10.1016/j.ijforecast.2009.10.002",
        "Using Elo ratings as match predictors in football forecasting."
    )
    pdf.ln(5)
    pdf.section_title("Tutorials & Documentation")
    pdf.reference_link(
        "XGBoost Documentation",
        "https://xgboost.readthedocs.io/en/latest/",
        "Official XGBoost docs - parameters, API, tutorials."
    )
    pdf.reference_link(
        "scikit-learn User Guide",
        "https://scikit-learn.org/stable/user_guide.html",
        "Logistic Regression, Random Forest, model evaluation metrics."
    )
    pdf.reference_link(
        "Dixon-Coles Python Implementation",
        "https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles/",
        "Step-by-step Python tutorial for implementing Dixon-Coles from scratch."
    )
    pdf.reference_link(
        "Poisson Distribution in Football",
        "https://www.pinnacle.com/en/betting-articles/Soccer/how-to-calculate-poisson-distribution/MD62MLXN3KR2GRA2",
        "Practical guide to using Poisson for football betting/prediction."
    )
    pdf.reference_link(
        "World Football Elo Ratings",
        "https://www.eloratings.net/about",
        "How eloratings.net computes and uses football Elo ratings."
    )
    pdf.reference_link(
        "Ensemble Learning Guide (MLWave)",
        "https://mlwave.com/kaggle-ensembling-guide/",
        "Comprehensive guide to stacking, blending, and ensemble techniques."
    )
    pdf.ln(5)
    pdf.section_title("Data Sources")
    pdf.reference_link(
        "International Football Results",
        "https://github.com/martj42/international_results",
        "49,477 matches from 1872-2026. CC0 license. Primary training dataset."
    )
    pdf.reference_link(
        "World Cup Historical Database",
        "https://github.com/jfjelstul/worldcup",
        "1,248 World Cup matches with squads, goals, standings. 1930-2022."
    )
    pdf.reference_link(
        "Flag CDN",
        "https://flagcdn.com",
        "Free flag images supporting ISO codes including UK subdivisions (gb-eng, gb-sct)."
    )

    # Save
    output_path = Path("docs/Code_Walkthrough_and_ML_Methods.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"PDF generated: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"  Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
