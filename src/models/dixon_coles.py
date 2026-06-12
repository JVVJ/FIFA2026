"""
Dixon-Coles model for football score prediction.

Extends independent Poisson by adding a correlation factor (rho)
that adjusts probabilities for low-scoring outcomes (0-0, 1-0, 0-1, 1-1).

Reference: Dixon & Coles (1997) "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market"
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def tau(x, y, lambda_val, mu_val, rho):
    """Dixon-Coles correction factor for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1 - lambda_val * mu_val * rho
    elif x == 0 and y == 1:
        return 1 + lambda_val * rho
    elif x == 1 and y == 0:
        return 1 + mu_val * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def dixon_coles_probability(x, y, lambda_val, mu_val, rho):
    """P(home=x, away=y) under Dixon-Coles model."""
    correction = tau(x, y, lambda_val, mu_val, rho)
    p = correction * poisson.pmf(x, lambda_val) * poisson.pmf(y, mu_val)
    return max(p, 1e-10)


def time_decay_weight(days_ago, half_life=180):
    """Exponential decay: matches half_life days ago get weight 0.5."""
    return np.exp(-0.693 * days_ago / half_life)


class DixonColesModel:
    """
    Dixon-Coles model estimating per-team attack/defence strengths
    and a global correlation parameter.
    """

    def __init__(self, half_life: int = 365):
        self.half_life = half_life
        self.attack = {}
        self.defence = {}
        self.home_adv = 0.0
        self.rho = 0.0
        self.teams = []

    def fit(self, matches: list[dict], reference_date: str = "2026-06-11"):
        """
        Fit the model to historical match data.

        matches: list of dicts with keys: home_team, away_team,
                 home_score, away_score, date, neutral
        """
        from datetime import datetime

        ref = datetime.strptime(reference_date, "%Y-%m-%d")

        # Filter to valid matches
        valid = [
            m for m in matches
            if m["home_score"] is not None and m["away_score"] is not None
        ]

        # Get unique teams
        self.teams = sorted(set(
            m["home_team"] for m in valid
        ) | set(
            m["away_team"] for m in valid
        ))
        team_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        # Compute weights
        weights = []
        for m in valid:
            try:
                match_date = datetime.strptime(m["date"], "%Y-%m-%d")
                days_ago = (ref - match_date).days
                weights.append(time_decay_weight(max(days_ago, 0), self.half_life))
            except (ValueError, TypeError):
                weights.append(0.01)
        weights = np.array(weights)

        # Initial parameters: [attack_1..n, defence_1..n, home_adv, rho]
        n_params = 2 * n_teams + 2
        x0 = np.zeros(n_params)
        x0[:n_teams] = 0.0  # attack (log scale)
        x0[n_teams:2*n_teams] = 0.0  # defence (log scale)
        x0[-2] = 0.25  # home advantage
        x0[-1] = -0.05  # rho

        def neg_log_likelihood(params):
            attack = params[:n_teams]
            defence = params[n_teams:2*n_teams]
            home_a = params[-2]
            rho_val = params[-1]

            # Constrain rho
            rho_val = max(-0.5, min(0.5, rho_val))

            ll = 0.0
            for i, m in enumerate(valid):
                hi = team_idx.get(m["home_team"])
                ai = team_idx.get(m["away_team"])
                if hi is None or ai is None:
                    continue

                ha = home_a if not m.get("neutral", False) else 0.0
                lambda_val = max(np.exp(attack[hi] + defence[ai] + ha), 0.01)
                mu_val = max(np.exp(attack[ai] + defence[hi]), 0.01)

                hg = int(m["home_score"])
                ag = int(m["away_score"])

                p = dixon_coles_probability(hg, ag, lambda_val, mu_val, rho_val)
                ll += weights[i] * np.log(p)

            # Regularization to prevent overfitting
            reg = 0.001 * (np.sum(attack**2) + np.sum(defence**2))
            return -(ll - reg)

        # Use L-BFGS-B (much faster for large parameter spaces)
        # Add constraint as penalty instead of explicit constraint
        def objective(params):
            attack = params[:n_teams]
            constraint_penalty = 100.0 * (np.sum(attack) ** 2)
            return neg_log_likelihood(params) + constraint_penalty

        result = minimize(
            objective, x0,
            method="L-BFGS-B",
            options={"maxiter": 30, "ftol": 1e-3},
        )

        # Extract parameters
        params = result.x
        for i, team in enumerate(self.teams):
            self.attack[team] = params[i]
            self.defence[team] = params[n_teams + i]
        self.home_adv = params[-2]
        self.rho = max(-0.5, min(0.5, params[-1]))

    def predict_score_probs(
        self, home: str, away: str, neutral: bool = True, max_goals: int = 6
    ) -> dict:
        """
        Predict score probabilities for a match.

        Returns dict with: lambda (home xG), mu (away xG), rho,
        score_matrix (max_goals+1 × max_goals+1), p_home_win, p_draw, p_away_win
        """
        atk_h = self.attack.get(home, 0.0)
        def_h = self.defence.get(home, 0.0)
        atk_a = self.attack.get(away, 0.0)
        def_a = self.defence.get(away, 0.0)

        ha = 0.0 if neutral else self.home_adv
        lambda_val = max(np.exp(atk_h + def_a + ha), 0.01)
        mu_val = max(np.exp(atk_a + def_h), 0.01)

        # Build score probability matrix
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                score_matrix[h, a] = dixon_coles_probability(
                    h, a, lambda_val, mu_val, self.rho
                )

        # Normalize (shouldn't be far from 1 but just in case)
        score_matrix /= score_matrix.sum()

        p_home_win = np.sum(np.tril(score_matrix, -1).T)
        p_draw = np.sum(np.diag(score_matrix))
        p_away_win = np.sum(np.tril(score_matrix, -1))

        # Most likely score
        idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

        return {
            "lambda": float(lambda_val),
            "mu": float(mu_val),
            "rho": float(self.rho),
            "score_matrix": score_matrix,
            "p_home_win": float(p_home_win),
            "p_draw": float(p_draw),
            "p_away_win": float(p_away_win),
            "likely_home": int(idx[0]),
            "likely_away": int(idx[1]),
            "xg_home": float(lambda_val),
            "xg_away": float(mu_val),
        }
