"""Cell 8 (shared train/test build) + cell 9 (POISSON REGRESSION)."""
import pandas as pd
import statsmodels.api as sm

COLS_ORDER = ["home_team_name", "pred_home_goals", "pred_away_goals", "away_team_name",
                  "pred_total_goals", "pred_goal_diff", "attack_diff",
    "home_attack_fixture", "away_defence_fixture", "away_attack_fixture", "home_defence_fixture",
    "home_weighted_attack", "away_weighted_defence", "away_weighted_attack", "home_weighted_defence",
   "set_play_diff", "open_play_diff",
  "home_weighted_set_play_strength",     "home_weighted_open_play_strength",
    "home_weighted_set_play_defence",     "home_weighted_open_play_defence",
     "away_weighted_set_play_strength",     "away_weighted_open_play_strength",
    "away_weighted_set_play_defence",   "away_weighted_open_play_defence"]


def build_train_test(played, prev_played, GW, features=("elo_context",)):
    # NOTE: tags season on the passed frames in place, as the notebook did
    prev_played["season"] = 0
    played["season"] = 1

    for season_name, season_matches in [("last season", prev_played), ("this season", played)]:
        for column in ["home_days_rest", "away_days_rest"]:
            print(season_name, column, (season_matches[column] < 3.5).sum(), "of", season_matches[column].notna().sum())

    print("rest_diff" in played.columns, "rest_diff" in prev_played.columns)

    train = pd.concat(
        [prev_played, played[played["gameweek"] < GW]],
        ignore_index=True,
    )
    test  = played[played["gameweek"] == GW].copy()

    # "set_play_diff", "open_play_diff", "rest_diff"

    FEATURES = list(features)

    train = train.dropna(subset=FEATURES + ["home_non_penalty_xg", "away_non_penalty_xg"])
    print(train.shape, train.groupby("season").size())
    return train, test, FEATURES


def fit_goal_models(train, FEATURES):
    Xtr = sm.add_constant(train[FEATURES])
    pois_home = sm.GLM(train["home_non_penalty_xg"], Xtr, family=sm.families.Poisson()).fit()
    pois_away = sm.GLM(train["away_non_penalty_xg"], Xtr, family=sm.families.Poisson()).fit()
    return pois_home, pois_away


def predict_fixtures(test, train, pois_home, pois_away, FEATURES, teams,
                     output_path="predicted_fixtures_GW.csv"):
    Xte = sm.add_constant(test[FEATURES], has_constant="add")

    test["pred_home_goals"] = pois_home.predict(Xte).values
    test["pred_away_goals"] = pois_away.predict(Xte).values
    test["pred_goal_diff"]  = test["pred_home_goals"] - test["pred_away_goals"]

    mu_home = train["home_non_penalty_xg"].mean()      # league averages, for the normalization below
    mu_away = train["away_non_penalty_xg"].mean()

    ## SCORE NORMALIZATION
    test["home_attack_fixture"] = test["pred_home_goals"] - mu_home
    test["away_attack_fixture"] = test["pred_away_goals"] - mu_away

    test["home_defence_fixture"] = mu_away - test["pred_away_goals"]
    test["away_defence_fixture"] = mu_home - test["pred_home_goals"]

    test = test.merge(
        teams[['code', 'short_name']],
        left_on='home_team',
        right_on='code',
        how='left'
    ).rename(columns={
        'short_name': 'home_team_name'
    }).drop(columns=['code'])

    test = test.merge(
        teams[['code', 'short_name']],
        left_on='away_team',
        right_on='code',
        how='left'
    ).rename(columns={
        'short_name': 'away_team_name'
    }).drop(columns=['code'])

    test["pred_total_goals"] = test["pred_home_goals"] + test["pred_away_goals"]
    test["pred_total_goals"] = test["pred_total_goals"].round(1)

    test["kickoff_time"] = pd.to_datetime(test["kickoff_time"])
    test = test.sort_values("kickoff_time")

    predictions = test[COLS_ORDER]

    name_cols = ["home_team_name", "away_team_name"]
    num_cols = [c for c in predictions.columns if c not in name_cols and pd.api.types.is_numeric_dtype(predictions[c])]

    predictions = predictions.copy()
    predictions[num_cols] = predictions[num_cols].round(1)

    predictions.to_csv(output_path, index=False)

    print(train[FEATURES].corr().round(2))
    print(test.loc[test.home_team_name.isin(["MCI","ARS"]), ["home_team_name","away_team_name","open_play_diff","set_play_diff","short_rest_advantage"]])
    return test, predictions
