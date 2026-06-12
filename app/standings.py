import polars as pl

from app.config import GROUP_NAMES
from app.data import get_group_teams


def compute_standings(df: pl.DataFrame, results: dict) -> pl.DataFrame:
    """Compute group standings from user-entered results."""
    rows = []
    group_teams = get_group_teams(df)

    for group, teams in group_teams.items():
        for team in teams:
            w, d, lost, gf, ga = 0, 0, 0, 0, 0
            team_matches = df.filter(
                (pl.col("Group_Stage") == group)
                & (
                    (pl.col("Home_Team") == team)
                    | (pl.col("Away_Team") == team)
                )
            )
            for row in team_matches.iter_rows(named=True):
                mn = row["Match_Number"]
                if mn not in results:
                    continue
                res = results[mn]
                is_home = row["Home_Team"] == team
                team_goals = (
                    res["home_goals"] if is_home else res["away_goals"]
                )
                opp_goals = (
                    res["away_goals"] if is_home else res["home_goals"]
                )
                gf += team_goals
                ga += opp_goals
                if team_goals > opp_goals:
                    w += 1
                elif team_goals == opp_goals:
                    d += 1
                else:
                    lost += 1
            pts = w * 3 + d
            rows.append({
                "Group": group, "Team": team, "P": w + d + lost,
                "W": w, "D": d, "L": lost, "GF": gf, "GA": ga,
                "GD": gf - ga, "Pts": pts,
            })

    return pl.DataFrame(rows).sort(
        ["Group", "Pts", "GD", "GF"], descending=[False, True, True, True]
    )


def get_qualified_teams(standings: pl.DataFrame) -> dict:
    """Get top 2 from each group + best 3rd place teams."""
    qualified = {"winners": {}, "runners_up": {}, "third_place": []}

    for group in GROUP_NAMES:
        group_standings = standings.filter(pl.col("Group") == group)
        if len(group_standings) >= 1:
            qualified["winners"][group] = group_standings.row(
                0, named=True
            )["Team"]
        if len(group_standings) >= 2:
            qualified["runners_up"][group] = group_standings.row(
                1, named=True
            )["Team"]
        if len(group_standings) >= 3:
            third = group_standings.row(2, named=True)
            qualified["third_place"].append(third)

    if qualified["third_place"]:
        third_df = pl.DataFrame(qualified["third_place"]).sort(
            ["Pts", "GD", "GF"], descending=[True, True, True]
        )
        qualified["third_place"] = third_df.head(8).to_dicts()

    return qualified


def resolve_knockout_team(placeholder: str, qualified: dict) -> str:
    """Resolve knockout placeholder to actual team name."""
    if placeholder.startswith("Winner "):
        group_letter = placeholder.replace("Winner ", "")
        if len(group_letter) == 1:
            group_key = f"Group {group_letter}"
            return qualified["winners"].get(group_key, placeholder)
        return placeholder
    elif placeholder.startswith("Runner-up "):
        group_letter = placeholder.replace("Runner-up ", "")
        if len(group_letter) == 1:
            group_key = f"Group {group_letter}"
            return qualified["runners_up"].get(group_key, placeholder)
        return placeholder
    elif placeholder.startswith("3rd "):
        return placeholder
    elif placeholder.startswith("Loser "):
        return placeholder
    return placeholder
