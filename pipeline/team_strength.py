"""Cell 7: FIXTURE GAMBLER — ELO blending, team table, diffs."""
import pandas as pd


def load_elo_all(backfill_path="data/elo_backfill.csv", history_path="data/elo_history.csv"):
    # ELO going INTO each GW:  (GW1-6 backfilled, estimated) + scraped history (GW7+)
    elo_all = pd.concat([pd.read_csv(backfill_path),
                         pd.read_csv(history_path)])
    elo_all["code"] = pd.to_numeric(elo_all["code"], errors="coerce")
    elo_all = elo_all.dropna(how="all")
    print(elo_all[elo_all["code"].isna()])
    elo_all = elo_all.drop_duplicates(["gw", "code"], keep="last")
    return elo_all


def build_elo_long(elo_all):
    return (elo_all.pivot(index="gw", columns="code", values="elo")
            .reindex(range(1, 39)).ffill()
            .stack())


def attach_match_elo(this_season_prem_matches, elo_long):
    for s in ["home", "away"]:
        idx = pd.MultiIndex.from_arrays([this_season_prem_matches["gameweek"], this_season_prem_matches[f"{s}_team"]])
        this_season_prem_matches[f"{s}_team_elo"] = elo_long.reindex(idx).values
    return this_season_prem_matches


def team_view(df, me, opp, is_home):
    return pd.DataFrame({
        "gameweek": df["gameweek"], "team": df[f"{me}_team"], "opp": df[f"{opp}_team"], "is_home": is_home,
        "elo": df[f"{me}_team_elo"],
        "goals_scored": df[f"{me}_score"],          "goals_conceded": df[f"{opp}_score"],
        "non_penalty_xg": df[f"{me}_non_penalty_xg"], "non_penalty_xg_allowed": df[f"{opp}_non_penalty_xg"],
        "xg_set_play": df[f"{me}_xg_set_play"],       "xg_set_play_allowed": df[f"{opp}_xg_set_play"],
        "xg_on_target_xgot": df[f"{me}_xg_on_target_xgot"], "xg_on_target_xgot_allowed": df[f"{opp}_xg_on_target_xgot"],
        "keeper_saves": df[f"{me}_keeper_saves"]})


def build_team_table(fx, gw):
    played = fx[fx["gameweek"] <= gw].copy()
    played["elo_context"] = (played["home_team_elo"] - played["away_team_elo"]) / 250

    team_fixtures = (pd.concat([team_view(played, "home", "away", 1), team_view(played, "away", "home", 0)],
                               ignore_index=True).sort_values(["team", "gameweek"]))

    team_fixtures["finishing_xgot_over"] = team_fixtures["xg_on_target_xgot"] - team_fixtures["non_penalty_xg"]
    team_fixtures["shot_stopping_diff"]  = team_fixtures["xg_on_target_xgot_allowed"] - team_fixtures["goals_conceded"]
    team_fixtures["keeper_intensity"]    = team_fixtures["keeper_saves"] / team_fixtures["xg_on_target_xgot_allowed"].clip(lower=0.1)
    team_fixtures["open_play_xg"]         = team_fixtures["non_penalty_xg"] - team_fixtures["xg_set_play"]
    team_fixtures["open_play_xg_allowed"] = team_fixtures["non_penalty_xg_allowed"] - team_fixtures["xg_set_play_allowed"]

    def ewm(col, by=("team",), hl=6):
        return (team_fixtures.groupby(list(by))[col]
                .transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=1).mean()))

    for c in ["xg_set_play", "xg_set_play_allowed", "open_play_xg", "open_play_xg_allowed"]:
        team_fixtures[f"ewm_{c}"] = ewm(c)
    for src, name in [("finishing_xgot_over", "finishing_over"),
                      ("shot_stopping_diff", "shot_stopping"),
                      ("keeper_intensity", "keeper_intensity")]:
        team_fixtures[f"ewm_{name}"] = ewm(src, ("team", "is_home")).fillna(0)

    def blend(c): return team_fixtures[f"ewm_{c}"]

    team_fixtures["weighted_set_play_strength"]  = blend("xg_set_play")
    team_fixtures["weighted_open_play_strength"] = blend("open_play_xg")
    team_fixtures["weighted_set_play_defence"]   = -blend("xg_set_play_allowed")
    team_fixtures["weighted_open_play_defence"]  = -blend("open_play_xg_allowed")
    team_fixtures["weighted_attack"] = (
        0.60 * blend("open_play_xg") + 0.30 * blend("xg_set_play") + 0.10 * blend("finishing_over"))
    team_fixtures["weighted_defence"] = (
        -0.60 * blend("open_play_xg_allowed") - 0.30 * blend("xg_set_play_allowed")
        + 0.10 * blend("shot_stopping"))

    return played, team_fixtures


def add_diffs(played, team_fixtures):
    def side_feats(is_home, prefix):
        f = (team_fixtures[team_fixtures["is_home"] == is_home]
             .rename(columns=lambda c: prefix + c if c not in ["gameweek", "team", "opp", "is_home"] else c))
        return f[["gameweek", "team", "opp"] + [c for c in f.columns if c.startswith(prefix + "weighted")]]

    played = played.merge(side_feats(1, "home_"),
                          left_on=["gameweek", "home_team", "away_team"],
                          right_on=["gameweek", "team", "opp"],
                          how="left").drop(columns=["team", "opp"])
    played = played.merge(side_feats(0, "away_"),
                          left_on=["gameweek", "away_team", "home_team"],
                          right_on=["gameweek", "team", "opp"],
                          how="left").drop(columns=["team", "opp"])

    played["set_play_diff"] = (
        (played["home_weighted_set_play_strength"] - played["away_weighted_set_play_defence"]) -
        (played["away_weighted_set_play_strength"] - played["home_weighted_set_play_defence"]))

    played["open_play_diff"] = (
        (played["home_weighted_open_play_strength"] - played["away_weighted_open_play_defence"]) -
        (played["away_weighted_open_play_strength"] - played["home_weighted_open_play_defence"]))

    played["attack_diff"] = (
        (played["home_weighted_attack"] - played["away_weighted_defence"]) -
        (played["away_weighted_attack"] - played["home_weighted_defence"]))

    return played.sort_values("gameweek")


def print_team_strength_diagnostics(team_fixtures, teams, this_season_prem_matches):
    latest = team_fixtures.sort_values("gameweek").groupby("team").tail(1)
    latest = latest.merge(teams[["code","short_name"]], left_on="team", right_on="code", how="left")

    print(latest[["short_name","weighted_attack"]].sort_values("weighted_attack", ascending=False).round(2).to_string(index=False))
    print(latest[["short_name","weighted_defence"]].sort_values("weighted_defence", ascending=False).round(2).to_string(index=False))
    print(latest[["short_name","elo"]].sort_values("elo", ascending=False).to_string(index=False))
    print("Blank ELO values?")
    print(this_season_prem_matches[["home_team_elo","away_team_elo"]].isna().sum())
    return latest
