"""Cell 11 (fixture score from predictor) + cell 12 (team strength onto player board)."""
import numpy as np
import pandas as pd


def build_team_fixture_scores(test):
    # ---- FIXTURE SCORE from predictor (single GW) ----
    home_fx = test[["home_team","home_attack_fixture","home_defence_fixture","pred_away_goals"]].rename(
        columns={"home_team":"team_code","home_attack_fixture":"fixture_attack",
                 "home_defence_fixture":"fixture_defence","pred_away_goals":"opp_pred_goals"})
    away_fx = test[["away_team","away_attack_fixture","away_defence_fixture","pred_home_goals"]].rename(
        columns={"away_team":"team_code","away_attack_fixture":"fixture_attack",
                 "away_defence_fixture":"fixture_defence","pred_home_goals":"opp_pred_goals"})
    team_fx = pd.concat([home_fx, away_fx], ignore_index=True)

    team_fx = team_fx.groupby("team_code", as_index=False)[["fixture_attack", "fixture_defence", "opp_pred_goals"]].sum()

    # Poisson clean-sheet probability — convex boost as opponent's predicted goals drop
    team_fx["cs_prob"] = np.exp(-team_fx["opp_pred_goals"].clip(lower=0))

    print(team_fx["cs_prob"].describe())
    print(team_fx["cs_prob"].head(10))

    # normalize
    team_fx["fixture_attack_score"]  = team_fx["fixture_attack"] / (team_fx["fixture_attack"].abs().max() + 1e-9)
    team_fx["fixture_defence_score"] = team_fx["cs_prob"] / (team_fx["cs_prob"].max() + 1e-9)
    return team_fx


# rescale to friendly 0-100 (linear — preserves order + distances)
def to_friendly(s, lo=0, hi=100):
    rng = s.max() - s.min()
    return pd.Series(50, index=s.index) if rng == 0 else lo + (s - s.min()) / rng * (hi - lo)


def attach_team_and_fixture_scores(merged, team_fixtures, team_fx):
    # 1. latest team strength (most recent rolling value per team)
    teams_latest = (
        team_fixtures.sort_values("gameweek")
        .groupby("team").tail(1)[["team", "weighted_attack", "weighted_defence"]]
    )

    # 2. rescale to friendly 0-100
    teams_latest["team_attack_score"]  = to_friendly(teams_latest["weighted_attack"])
    teams_latest["team_defence_score"] = to_friendly(teams_latest["weighted_defence"])

    # 3. merge onto player board (team_code ↔ team)
    merged = merged.merge(
        teams_latest[["team", "team_attack_score", "team_defence_score"]],
        left_on="team_code", right_on="team", how="left"
    ).drop(columns=["team"])

    merged["team_score"] = np.select(
        [
            merged["position"] == "Forward",
            merged["position"] == "Midfielder",
            merged["position"] == "Defender",
            merged["position"] == "Goalkeeper",
        ],
        [
            merged["team_attack_score"],                                   # FWD: pure attack
            0.65 * merged["team_attack_score"] + 0.35 * merged["team_defence_score"],  # MID: mostly attack
            merged["team_defence_score"],                                  # DEF: pure defence
            merged["team_defence_score"],                                  # GK: pure defence
        ],
        default=0.0
    )

    merged = merged.merge(
        team_fx[["team_code","fixture_attack_score","fixture_defence_score"]],
        on="team_code", how="left"
    )

    merged["fixture_score"] = np.select(
        [merged["position"]=="Forward", merged["position"]=="Midfielder",
         merged["position"]=="Defender", merged["position"]=="Goalkeeper"],
        [merged["fixture_attack_score"],
         0.65*merged["fixture_attack_score"] + 0.35*merged["fixture_defence_score"],
         merged["fixture_defence_score"],
         merged["fixture_defence_score"]],
        default=0.0
    )

    merged["fixture_score_100"] = to_friendly(merged["fixture_score"].fillna(0))

    print(merged.columns.tolist())
    print(merged[["web_name", "team_name", "team_defence_score","team_attack_score", "team_score"]].head(20))
    return merged
