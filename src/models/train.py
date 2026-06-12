"""Train the match prediction model (ensemble + Dixon-Coles)."""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polars as pl

from src.features.builder import (
    build_features_fast,
    _update_elo,
    _update_form,
    _update_h2h,
    _snapshot_features,
)
from src.models.ensemble import EnsembleClassifier
from src.models.dixon_coles import DixonColesModel

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_PATH = DATA_DIR / "model.pkl"


def main():
    print("=" * 60)
    print("FIFA World Cup 2026 — Model Training")
    print("  Ensemble (XGBoost + LR + RF) + Dixon-Coles")
    print("=" * 60)
    t0 = time.time()

    # Load data
    print("\n[1/6] Loading data...")
    matches_df = pl.read_csv(
        str(DATA_DIR / "raw" / "international_results.csv"),
        null_values=["NA", ""],
    )
    print(f"  Loaded {len(matches_df):,} international matches")

    # Build features
    print("\n[2/6] Building features (single-pass)...")
    t1 = time.time()
    X, y, feature_names = build_features_fast(matches_df, min_date="2000-01-01")
    print(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]} features")
    print(f"  Labels: W={np.sum(y==2):,} D={np.sum(y==1):,} L={np.sum(y==0):,}")
    print(f"  Time: {time.time()-t1:.1f}s")

    # Train/validation split (time-based: last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    # Train ensemble
    print("\n[3/6] Training ensemble classifier...")
    t1 = time.time()
    ensemble = EnsembleClassifier()
    ensemble.fit(X_train, y_train)
    print(f"  Base models trained in {time.time()-t1:.1f}s")

    # Optimize blend weights
    print("  Optimizing blend weights on validation set...")
    ensemble.optimize_weights(X_val, y_val)
    print(f"  Optimal weights: XGB={ensemble.weights[0]:.2f}, LR={ensemble.weights[1]:.2f}, RF={ensemble.weights[2]:.2f}")

    # Validation accuracy
    val_preds = ensemble.predict_proba(X_val)
    val_pred_classes = np.argmax(val_preds, axis=1)
    val_accuracy = np.mean(val_pred_classes == y_val)
    print(f"  Validation accuracy: {val_accuracy:.1%}")

    # Refit on all data with optimized weights
    print("  Refitting on full dataset...")
    ensemble.fit(X, y)

    # Train Dixon-Coles (only WC 2026 teams to keep parameter space small)
    print("\n[4/6] Training Dixon-Coles goal model...")
    t1 = time.time()
    wc_teams = set(pl.read_csv(
        str(DATA_DIR / "raw" / "wc2026_schedule.csv")
    )["Home_Team"].unique().to_list())
    # Add alternate names
    wc_teams.add("Curaçao")
    wc_teams.discard("Curacao")
    # Filter to matches where BOTH teams are WC participants (keeps params small)
    recent_matches = matches_df.filter(
        (pl.col("date") >= "2020-01-01")
        & pl.col("home_score").is_not_null()
        & pl.col("home_team").is_in(list(wc_teams))
        & pl.col("away_team").is_in(list(wc_teams))
    ).to_dicts()
    print(f"  Using {len(recent_matches):,} matches involving WC teams (2022+)")

    dc_model = DixonColesModel(half_life=365)
    dc_model.fit(recent_matches)
    print(f"  Dixon-Coles fitted in {time.time()-t1:.1f}s")
    print(f"  Rho (correlation): {dc_model.rho:.4f}")
    print(f"  Home advantage: {dc_model.home_adv:.3f}")
    print(f"  Teams modeled: {len(dc_model.teams)}")

    # Extract Elo and form state
    print("\n[5/6] Extracting final Elo/form state...")
    t1 = time.time()
    all_matches = matches_df.filter(
        pl.col("home_score").is_not_null()
    ).sort("date").to_dicts()

    elo = {}
    form_results, form_gf_state, form_gc_state, h2h_wins = {}, {}, {}, {}
    for m in all_matches:
        _update_elo(
            m["home_team"], m["away_team"], m["home_score"],
            m["away_score"], m["tournament"], m["neutral"], elo
        )
        _update_form(m["home_team"], m["home_score"], m["away_score"],
                     form_results, form_gf_state, form_gc_state)
        _update_form(m["away_team"], m["away_score"], m["home_score"],
                     form_results, form_gf_state, form_gc_state)
        _update_h2h(m["home_team"], m["away_team"], m["home_score"],
                    m["away_score"], h2h_wins)
    print(f"  Done in {time.time()-t1:.1f}s")

    # Top Elo
    top = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 Elo:")
    for i, (team, e) in enumerate(top, 1):
        print(f"    {i:2d}. {team:20s} {e:.0f}")

    # Save
    print("\n[6/6] Saving model...")
    state = {
        "model": ensemble,
        "feature_names": feature_names,
        "dixon_coles": dc_model,
        "elo": elo,
        "form_results": form_results,
        "form_gf": form_gf_state,
        "form_gc": form_gc_state,
        "h2h_wins": h2h_wins,
        "val_accuracy": val_accuracy,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(state, f)
    print(f"  Saved to {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # Sample predictions
    print("\n" + "=" * 60)
    print("Sample predictions (Ensemble + Dixon-Coles):")
    print("=" * 60)

    samples = [
        ("Mexico", "South Africa"),
        ("Brazil", "Morocco"),
        ("France", "Senegal"),
        ("Argentina", "Algeria"),
        ("United States", "Paraguay"),
        ("England", "Croatia"),
        ("Germany", "Curaçao"),
        ("Spain", "Cape Verde"),
    ]

    for home, away in samples:
        # Ensemble W/D/L
        feats = _snapshot_features(
            home, away, elo, form_results, form_gf_state, form_gc_state, h2h_wins
        )
        X_pred = np.array([feats], dtype=np.float32)
        probs = ensemble.predict_proba(X_pred)[0]

        # Dixon-Coles scores
        dc_pred = dc_model.predict_score_probs(home, away, neutral=True)

        # Blend: 50% ensemble + 50% Dixon-Coles for W/D/L
        p_loss = 0.5 * probs[0] + 0.5 * dc_pred["p_away_win"]
        p_draw = 0.5 * probs[1] + 0.5 * dc_pred["p_draw"]
        p_win = 0.5 * probs[2] + 0.5 * dc_pred["p_home_win"]
        total = p_win + p_draw + p_loss
        p_win, p_draw, p_loss = p_win/total, p_draw/total, p_loss/total

        print(
            f"\n  {home} vs {away}:"
            f"\n    Win: {p_win:.1%}  Draw: {p_draw:.1%}  Loss: {p_loss:.1%}"
            f"\n    DC xG: {dc_pred['xg_home']:.2f} - {dc_pred['xg_away']:.2f}"
            f"  Likely: {dc_pred['likely_home']}-{dc_pred['likely_away']}"
        )

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
