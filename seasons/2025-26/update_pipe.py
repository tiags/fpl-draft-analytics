import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from xlsxwriter.utility import xl_col_to_name
from flask import Flask, send_file

#TODO
# 2. Win / Draw / Loss probabilities from regression output
# 	•	Convert predicted goal_diff → win probability curve
# 	•	Use a logistic transformation or Poisson-based simulation
# 	•	Produce: P(Home Win), P(Draw), P(Away Win)

# 3. Ensemble classifier + regressor
# 	•	Regression predicts magnitude (goal_diff)
# 	•	Classifier predicts categorical outcome
# 	•	Combine for more stable predictions in low-scoring matches


#### HOUSEKEEPING

last_years_stats = pd.read_csv("lastseasonstats.csv", encoding="utf-8")
FPL_DATA_FOLDER = os.path.expanduser('~/MyProjects/FPL/FPL-Elo-Insights/data/2025-2026')
GW = 38
LAST_GW = GW - 1
TOTAL_MINUTES = GW * 90

#STATIC
fixtures_dir = os.path.join(FPL_DATA_FOLDER, "By Gameweek")
players = pd.read_csv(f"{FPL_DATA_FOLDER}/players.csv",encoding="utf-8")
teams = pd.read_csv(
    f"{FPL_DATA_FOLDER}/teams.csv",
    encoding="utf-8")

#VARIABLES

all_fixtures = []

for folder in sorted(os.listdir(fixtures_dir)):
    if folder.startswith("GW"):
        gw_path = os.path.join(fixtures_dir, folder, "fixtures.csv")
        if os.path.exists(gw_path):
            df = pd.read_csv(gw_path, encoding="utf-8")
            all_fixtures.append(df)

fixtures_master = pd.concat(all_fixtures, ignore_index=True)
fixtures_master["tournament"] = fixtures_master["tournament"].fillna("")
fixtures_master = fixtures_master[
    fixtures_master["tournament"].str.lower().str.contains("prem")
]

matches = pd.read_csv(
    f"{FPL_DATA_FOLDER}/By Gameweek/GW{GW}/matches.csv",
    encoding="utf-8")
playerstats = pd.read_csv(
    f"{FPL_DATA_FOLDER}/By Gameweek/GW{LAST_GW}/playerstats.csv",
    encoding="utf-8")

### FUNCTIONS

def scale_global(df, col):
    s = df[col]
    median = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) + 1e-6
    return (s - median) / iqr

def compute_form_score(row):
    if row["position"] == "Forward":
        return (
            0.45 * row["form_points_p90"] +
            0.45 * row["form_xgi_p90"] +
            0.10 * row["form_bps_p90"] 
        )
    elif row["position"] == "Midfielder":
        return (
            0.40 * row["form_points_p90"] +
            0.25 * row["form_xgi_p90"] +
            0.10 * row["form_bps_p90"] +
            0.25 * row["form_defcon_p90"]
        )
    elif row["position"] == "Defender":
        return (
            0.40 * row["form_points_p90"] +
            0.10 * row["form_bps_p90"] +
            0.25 * row["form_defcon_p90"] +
            0.10 * row["form_xgi_p90"] -
            0.15 * row["form_gc_p90"] 
        )
    else:  # Goalkeeper
        return (
            0.10 * row["form_bps_p90"] +
            0.40 * row["form_points_p90"] -
            0.50 * row["form_gc_p90"]
        )
        
def expected_fpl_p90(row):
    pos = row["position"]
    xg = row.get("expected_goals_per_90", 0)
    xa = row.get("expected_assists_per_90", 0)
    xgc = row.get("expected_goals_conceded_per_90", 0)
    saves = row.get("saves_per_90", 0)
    defcon = row.get("defensive_contribution_per_90", 0)

    base = 0

    # --- GOALS + ASSISTS (position-scaled) ---
    # if pos == "Goalkeeper":
    #     base = xg * 10 + xa * 3
    if pos == "Defender":
        base = xg * 6 + xa * 3
    elif pos == "Midfielder":
        base = xg * 5 + xa * 3
    elif pos == "Forward":
        base = xg * 4 + xa * 3

    # --- CLEAN SHEETS ---
    if pos == "Goalkeeper":
        clean_sheet_pts = row["clean_sheets_per_90"] * 5
    elif pos == "Defender":
        clean_sheet_pts = row["clean_sheets_per_90"] * 4
    elif pos == "Midfielder":
        clean_sheet_pts = row["clean_sheets_per_90"] * 1
    else:
        clean_sheet_pts = 0

    # --- GOALS CONCEDED  ---
    if pos in ["Goalkeeper", "Defender"]:
        conceded_penalty = -0.5 * xgc
    else:
        conceded_penalty = 0

    # --- SAVE POINTS (GK only: 1 per 3 saves) ---
    save_pts = 0
    if pos == "Goalkeeper":
        save_pts = saves / 3

    # --- DEFENSIVE BONUS ---
    if pos == "Defender":
        threshold = 10
    elif pos in ["Midfielder", "Forward"]:
        threshold = 12
    else:
        threshold = None
    if threshold is None:
        defcon_bonus = 0
    else:
        # convert per-90 defcon into expected per-match value
        defcon_match = defcon * row["minutes_factor"]
        
        # probability of hitting match-level threshold
        prob_hit = min(defcon_match / threshold, 1)
    
        # 2-point cap
        defcon_bonus = 2 * prob_hit    


    # ---- APPEARANCE POINTS ----
    mins_est = row["minutes_factor"] * 90

    if mins_est >= 60:
        appearance_pts = 2
    elif mins_est > 0:
        appearance_pts = 1
    else:
        appearance_pts = 0

    return (
        base +
        clean_sheet_pts +
        save_pts +
        defcon_bonus +
        conceded_penalty +
        appearance_pts
    )

POSITION_LIMITS = {
    "Goalkeeper": 16,
    "Defender": 40,
    "Midfielder": 40,
    "Forward": 24
}
def compute_vorp_dynamic(df, metric="expected_return"):
    output = []

    for pos, limit in POSITION_LIMITS.items():
        subset = df[df["position"] == pos].copy()
        subset = subset.sort_values(metric, ascending=False)

        if len(subset) < limit:
            baseline = subset[metric].min()
        else:
            baseline = subset.iloc[limit - 1][metric]

        print(pos, len(subset), baseline)

        subset["baseline"] = baseline
        subset["VORP"] = subset[metric] - baseline
        subset["above_replacement"] = subset["VORP"] > 0

        output.append(subset)

    return pd.concat(output, ignore_index=True)


#### FIXTURE GAMBLER MACHINE LEARNING MODEL
fixtures_master["elo_diff"] = fixtures_master["home_team_elo"] - fixtures_master["away_team_elo"]
fixtures_master["goal_diff"] = fixtures_master["home_score"] - fixtures_master["away_score"]
fixtures_master["home_win"] = (fixtures_master["goal_diff"] > 0).astype(int)

# ✅ Defensive stats
fixtures_master["home_non_penalty_xg_allowed"] = fixtures_master["away_non_penalty_xg"]
fixtures_master["away_non_penalty_xg_allowed"] = fixtures_master["home_non_penalty_xg"]

fixtures_master["home_xg_set_play_allowed"] = fixtures_master["away_xg_set_play"]
fixtures_master["away_xg_set_play_allowed"] = fixtures_master["home_xg_set_play"]

fixtures_master["home_goals_conceded"] = fixtures_master["away_score"]
fixtures_master["away_goals_conceded"] = fixtures_master["home_score"]

fixtures_master["home_points"] = fixtures_master["home_score"].gt(fixtures_master["away_score"])*3 + \
                                 fixtures_master["home_score"].eq(fixtures_master["away_score"])*1
fixtures_master["away_points"] = fixtures_master["away_score"].gt(fixtures_master["home_score"])*3 + \
                                 fixtures_master["away_score"].eq(fixtures_master["home_score"])*1

fixtures_master["home_xg_on_target_xgot_allowed"] = fixtures_master["away_xg_on_target_xgot"]
fixtures_master["away_xg_on_target_xgot_allowed"] = fixtures_master["home_xg_on_target_xgot"]

fixtures_master["home_keeper_saves_allowed"] = fixtures_master["away_keeper_saves"]
fixtures_master["away_keeper_saves_allowed"] = fixtures_master["home_keeper_saves"]

fixtures = fixtures_master.copy()

### CREATING ROLLING METRICS

# ---- HOME ----
home = fixtures[[
    "gameweek",
    "home_team",
    "home_team_elo",
    "home_score",
    "home_goals_conceded",
    "home_points",
    "home_non_penalty_xg",
    "home_xg_set_play",
    "home_non_penalty_xg_allowed",
    "home_xg_set_play_allowed",
    "home_xg_on_target_xgot",
    "home_xg_on_target_xgot_allowed",
    "home_keeper_saves"
]].copy()

home.columns = [
    "gameweek",
    "team",
    "elo",
    "goals_scored",
    "goals_conceded",
    "points",
    "non_penalty_xg",
    "xg_set_play",
    "non_penalty_xg_allowed",
    "xg_set_play_allowed",
    "xg_on_target_xgot",
    "xg_on_target_xgot_allowed",
    "keeper_saves",
]
home["is_home"] = 1

# ---- AWAY ----
away = fixtures[[
    "gameweek",
    "away_team",
    "away_team_elo",
    "away_score",
    "away_goals_conceded",
    "away_points",
    "away_non_penalty_xg",
    "away_xg_set_play",
    "away_non_penalty_xg_allowed",
    "away_xg_set_play_allowed",
    "away_xg_on_target_xgot",
    "away_xg_on_target_xgot_allowed",
    "away_keeper_saves"
]].copy()

away.columns = [
    "gameweek",
    "team",
    "elo",
    "goals_scored",
    "goals_conceded",
    "points",
    "non_penalty_xg",
    "xg_set_play",
    "non_penalty_xg_allowed",
    "xg_set_play_allowed",
    "xg_on_target_xgot",
    "xg_on_target_xgot_allowed",
    "keeper_saves",
]
away["is_home"] = 0

team_fixtures = pd.concat([home, away], ignore_index=True)
team_fixtures = team_fixtures.sort_values(["team","gameweek"])

team_fixtures = team_fixtures[team_fixtures["gameweek"] <= GW]

# Rolling set-play xG
team_fixtures["roll3_xg_set_play"] = (
    team_fixtures
    .groupby("team")["xg_set_play"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

team_fixtures["roll5_xg_set_play"] = (
    team_fixtures
    .groupby("team")["xg_set_play"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Rolling xG
team_fixtures["roll3_open_play_xg"] = (
    team_fixtures
    .groupby("team")["non_penalty_xg"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

team_fixtures["roll5_open_play_xg"] = (
    team_fixtures
    .groupby("team")["non_penalty_xg"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
## Rolling Defensive

# Rolling DEFENSIVE set-play xG allowed
team_fixtures["roll3_xg_set_play_allowed"] = (
    team_fixtures
    .groupby("team")["xg_set_play_allowed"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

team_fixtures["roll5_xg_set_play_allowed"] = (
    team_fixtures
    .groupby("team")["xg_set_play_allowed"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Rolling DEFENSIVE open-play xG allowed
team_fixtures["roll3_open_play_xg_allowed"] = (
    team_fixtures
    .groupby("team")["non_penalty_xg_allowed"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

team_fixtures["roll5_open_play_xg_allowed"] = (
    team_fixtures
    .groupby("team")["non_penalty_xg_allowed"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Rolling Actuals

# Finishing: are they turning xG into good shots?
team_fixtures["finishing_xgot_over"] = (
    team_fixtures["xg_on_target_xgot"] - team_fixtures["non_penalty_xg"]
)

# Defense / GK: are they conceding less/more than the xGOT they face?
team_fixtures["shot_stopping_diff"] = (
    team_fixtures["xg_on_target_xgot_allowed"] - team_fixtures["goals_conceded"]
)

# a simple keeper quality proxy
team_fixtures["keeper_intensity"] = (
    team_fixtures["keeper_saves"] /
    (team_fixtures["xg_on_target_xgot_allowed"].clip(lower=0.1))
)

# Keep rolling 
for col, new3, new5 in [
    ("finishing_xgot_over", "roll3_finishing_over", "roll5_finishing_over"),
    ("shot_stopping_diff",  "roll3_shot_stopping",  "roll5_shot_stopping"),
    ("keeper_intensity",    "roll3_keeper_intensity", "roll5_keeper_intensity"),
]:
    team_fixtures[new3] = (
        team_fixtures
        .groupby(["team", "is_home"])[col]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    team_fixtures[new5] = (
        team_fixtures
        .groupby(["team", "is_home"])[col]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

### WEIGHTED FORM

team_fixtures["weighted_set_play_strength"] = (
    0.7 * team_fixtures["roll3_xg_set_play"] +
    0.3 * team_fixtures["roll5_xg_set_play"]
)

team_fixtures["weighted_open_play_strength"] = (
    0.7 * team_fixtures["roll3_open_play_xg"] +
    0.3 * team_fixtures["roll5_open_play_xg"]
)

team_fixtures["weighted_set_play_defence"] = (
    0.7 * team_fixtures["roll3_xg_set_play_allowed"] +
    0.3 * team_fixtures["roll5_xg_set_play_allowed"]
)

team_fixtures["weighted_open_play_defence"] = (
    0.7 * team_fixtures["roll3_open_play_xg_allowed"] +
    0.3 * team_fixtures["roll5_open_play_xg_allowed"]
)

# Weighted ATTACK power
team_fixtures["weighted_attack"] = (
    0.50 * team_fixtures["weighted_open_play_strength"] +
    0.25 * team_fixtures["weighted_set_play_strength"] +
    0.25 * (
        0.7 * team_fixtures["roll3_finishing_over"] +
        0.3 * team_fixtures["roll5_finishing_over"]
    )
)


# Weighted DEFENSIVE power
team_fixtures["weighted_defence"] = (
    -0.50 * (0.7 * team_fixtures["roll3_open_play_xg_allowed"] +
            0.3 * team_fixtures["roll5_open_play_xg_allowed"]) +

    -0.25 * (0.7 * team_fixtures["roll3_xg_set_play_allowed"] +
            0.3 * team_fixtures["roll5_xg_set_play_allowed"]) +

    0.20 * (0.7 * team_fixtures["roll3_shot_stopping"] +
            0.3 * team_fixtures["roll5_shot_stopping"]) +

    0.05 * (0.7 * team_fixtures["roll3_keeper_intensity"] +
            0.3 * team_fixtures["roll5_keeper_intensity"])
)

home_features = (
    team_fixtures[team_fixtures["is_home"] == 1]
    .rename(columns=lambda c: "home_" + c if c not in ["gameweek", "team", "is_home"] else c)
)

away_features = (
    team_fixtures[team_fixtures["is_home"] == 0]
    .rename(columns=lambda c: "away_" + c if c not in ["gameweek", "team", "is_home"] else c)
)

fixtures_master = fixtures_master.merge(
    home_features.drop(columns=["is_home"])[
        ["gameweek", "team"] +
        [c for c in home_features.columns if c.startswith("home_")]
    ],
    left_on=["gameweek", "home_team"],
    right_on=["gameweek", "team"],
    how="left"
).drop(columns=["team"])

fixtures_master = fixtures_master.merge(
    away_features.drop(columns=["is_home"])[
        ["gameweek", "team"] +
        [c for c in away_features.columns if c.startswith("away_")]
    ],
    left_on=["gameweek", "away_team"],
    right_on=["gameweek", "team"],
    how="left"
).drop(columns=["team"])

fixtures_master["set_play_diff"] = (
    (fixtures_master["home_weighted_set_play_strength"] -
     fixtures_master["away_weighted_set_play_defence"]) -
    (fixtures_master["away_weighted_set_play_strength"] -
     fixtures_master["home_weighted_set_play_defence"])
)

fixtures_master["open_play_diff"] = (
    (fixtures_master["home_weighted_open_play_strength"] -
     fixtures_master["away_weighted_open_play_defence"]) -
    (fixtures_master["away_weighted_open_play_strength"] -
     fixtures_master["home_weighted_open_play_defence"])
)

fixtures_master["attack_diff"] = (
    (fixtures_master["home_weighted_attack"] - fixtures_master["away_weighted_defence"]) -
    (fixtures_master["away_weighted_attack"] - fixtures_master["home_weighted_defence"])
)

fixtures_master["elo_diff"] = (
    fixtures_master["home_elo"] -
    fixtures_master["away_elo"]
)

fixtures_master["elo_context"] = fixtures_master["elo_diff"] / 250

fixtures_master = fixtures_master.sort_values("gameweek")

train = fixtures_master[fixtures_master["gameweek"] < GW]
test  = fixtures_master[fixtures_master["gameweek"] == GW]

FEATURES = ["attack_diff", "set_play_diff", "open_play_diff", "elo_context"]

train = train.dropna(subset=FEATURES + ["home_score", "away_score"])
test  = test.dropna(subset=FEATURES)

X = train[FEATURES]
X_test = test[FEATURES]

y_home = train["home_score"]
y_away = train["away_score"]

### FIXTURE PREDICTOR
model_home = HistGradientBoostingRegressor()
model_away = HistGradientBoostingRegressor()

model_home.fit(X, y_home)
model_away.fit(X, y_away)

train_home_pred = model_home.predict(X)
train_away_pred = model_away.predict(X)

mu_home  = train_home_pred.mean()
std_home = train_home_pred.std()

mu_away  = train_away_pred.mean()
std_away = train_away_pred.std()

test["pred_home_goals"] = model_home.predict(X_test)
test["pred_away_goals"] = model_away.predict(X_test)

test["pred_goal_diff"] = test["pred_home_goals"] - test["pred_away_goals"]


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

cols_order = ["home_team_name", "pred_home_goals", "pred_away_goals", "away_team_name",
                  "pred_total_goals", "pred_goal_diff", "attack_diff",
    "home_attack_fixture", "away_defence_fixture", "away_attack_fixture", "home_defence_fixture", 
    "home_weighted_attack", "away_weighted_defence", "away_weighted_attack", "home_weighted_defence",
   "set_play_diff", "open_play_diff", "elo_context", "elo_diff",
    "home_elo",     "home_weighted_set_play_strength",     "home_weighted_open_play_strength", 
    "home_weighted_set_play_defence",     "home_weighted_open_play_defence",
    "away_elo",     "away_weighted_set_play_strength",     "away_weighted_open_play_strength",
    "away_weighted_set_play_defence",   "away_weighted_open_play_defence"]


predictions = test[cols_order]

name_cols = ["home_team_name", "away_team_name"]
num_cols = [c for c in predictions.columns if c not in name_cols and pd.api.types.is_numeric_dtype(predictions[c])]

predictions = predictions.copy()
predictions[num_cols] = predictions[num_cols].round(1)

predictions.to_csv("predicted_fixtures_GW.csv", index=False)

#### Calculuating Form

# Collect recent GW stats
recent_dfs = []

past_week_weights = {
    1: 0.65,
    2: 0.25,
    3: 0.06,
    4: 0.03,
    5: 0.01
}

for gw in range(GW-5, GW):
    gw_file = f"{FPL_DATA_FOLDER}/By Gameweek/GW{gw}/player_gameweek_stats.csv"

    if os.path.exists(gw_file):
        df_gw = pd.read_csv(gw_file, encoding="utf-8")
        weeks_ago = GW - gw
        weight = past_week_weights.get(weeks_ago, 0)
        df_gw["gameweek"] = gw
        df_gw["form_xgi"] = df_gw["expected_goal_involvements"] * weight
        df_gw["form_bps"] = df_gw["bonus"] * weight
        df_gw["form_defcon"] = df_gw["defensive_contribution"] * weight
        df_gw["form_points"] = df_gw["event_points"] * weight
        df_gw["form_gc"] = df_gw["goals_conceded"] * weight
        df_gw["form_minutes"] = df_gw["minutes"] * weight
        recent_dfs.append(df_gw)

micro_trends = pd.concat(recent_dfs, ignore_index=True)

micro_trends[['id', 'web_name', 'form_points']] \
    .sort_values(by='form_points', ascending=False)

form_summary = micro_trends.groupby("id").agg(
    form_xgi=("form_xgi", "sum"),
    form_bps=("form_bps", "sum"),
    form_gc=("form_gc", "sum"),
    form_defcon=("form_defcon", "sum"),
    form_points=("form_points", "sum"),
    form_minutes=("form_minutes", "sum")
).reset_index()

MIN_FORM_MINUTES = 120

# Clip once
form_summary["form_minutes_clipped"] = (
    form_summary["form_minutes"].clip(lower=MIN_FORM_MINUTES)
)

# Metrics to convert to per-90
form_rate_metrics = [
    "form_points",
    "form_gc",
    "form_xgi",
    "form_bps",
    "form_defcon"
]

# Create p90 columns
for col in form_rate_metrics:
    p90_col = f"{col}_p90"
    form_summary[p90_col] = (
        form_summary[col] /
        form_summary["form_minutes_clipped"] * 90
    )

#### FIXTURE STRENGTH

LOOKAHEAD_WEEKS = [GW + 1, GW + 2]

weights = {
    GW: 0.75,
    GW + 1: 0.17,
    GW + 2: 0.08
}

available_weeks = [
    gw for gw in [GW, GW + 1, GW + 2]
    if gw in fixtures_master["gameweek"].values
]

total = sum(weights[w] for w in available_weeks)
weights = {w: weights[w] / total for w in available_weeks}

LOOKAHEAD_WEEKS = [w for w in available_weeks if w != GW]

current = test[["home_team", "away_team", "pred_home_goals", "pred_away_goals"]].copy()
current["week_weight"] = weights[GW]

future = fixtures_master.loc[
    fixtures_master["gameweek"].isin(LOOKAHEAD_WEEKS),
    ["gameweek", "home_team", "away_team"]
].copy()

teams_latest = (
    team_fixtures
    .sort_values("gameweek")
    .groupby("team")
    .tail(1)        # <-- the most recent rolling metrics for each club
    .reset_index(drop=True)
)

future = future.merge(
    teams_latest[[
        "team",
        "elo",
        "weighted_set_play_strength",
        "weighted_open_play_strength",
        "weighted_set_play_defence",
        "weighted_open_play_defence",
        "weighted_attack",
        "weighted_defence"
    ]],
    left_on="home_team",
    right_on="team",
    how="left"
).rename(columns={
    "elo": "home_elo",
    "weighted_set_play_strength": "home_set_attack",
    "weighted_open_play_strength": "home_open_attack",
    "weighted_set_play_defence": "home_set_defence",
    "weighted_open_play_defence": "home_open_defence",
    "weighted_attack": "home_weighted_attack",
    "weighted_defence": "home_weighted_defence"
}).drop(columns=["team"])

future = future.merge(
    teams_latest[[
        "team",
        "elo",
        "weighted_set_play_strength",
        "weighted_open_play_strength",
        "weighted_set_play_defence",
        "weighted_open_play_defence",
        "weighted_attack",
        "weighted_defence",
    ]],
    left_on="away_team",
    right_on="team",
    how="left"
).rename(columns={
    "elo": "away_elo",
    "weighted_set_play_strength": "away_set_attack",
    "weighted_open_play_strength": "away_open_attack",
    "weighted_set_play_defence": "away_set_defence",
    "weighted_open_play_defence": "away_open_defence",
    "weighted_attack": "away_weighted_attack",
    "weighted_defence": "away_weighted_defence"
}).drop(columns=["team"])

future["set_play_diff"] = (
    (future["home_set_attack"] - future["away_set_defence"]) -
    (future["away_set_attack"] - future["home_set_defence"])
)

future["open_play_diff"] = (
    (future["home_open_attack"] - future["away_open_defence"]) -
    (future["away_open_attack"] - future["home_open_defence"])
)

# Aggregate ATTACK vs DEFENCE (using new weighted_* features)
future["attack_diff"] = (
    (future["home_weighted_attack"] - future["away_weighted_defence"]) -
    (future["away_weighted_attack"] - future["home_weighted_defence"])
)

future["elo_diff"] = future["home_elo"].astype(float) - future["away_elo"].astype(float)
future["elo_context"] = future["elo_diff"] / 250

X_future = future[[
    "attack_diff",
    "set_play_diff",
    "open_play_diff",
    "elo_context"
]]

if not X_future.empty:
    future["pred_home_goals"] = model_home.predict(X_future)
    future["pred_away_goals"] = model_away.predict(X_future)
else:
    future["pred_home_goals"] = []
    future["pred_away_goals"] = []

future["week_weight"] = future["gameweek"].map(weights)

# --- BASELINE FOR DEFENCE (mean expected goals conceded across horizon) ---
all_ga = pd.concat([
    current["pred_away_goals"],
    current["pred_home_goals"],
    future["pred_away_goals"],
    future["pred_home_goals"]
], ignore_index=True)

ga_baseline = all_ga.mean()

team_outlook = pd.concat([
    # --- CURRENT GW contributions ---
    # Home team: goals for = pred_home_goals; goals against = pred_away_goals
    current.assign(
        team=current["home_team"],
        attack_contrib=current["pred_home_goals"] * current["week_weight"],
        defence_contrib=(ga_baseline - current["pred_away_goals"]) * current["week_weight"]
    )[["team", "attack_contrib", "defence_contrib"]],

    # Away team: goals for = pred_away_goals; goals against = pred_home_goals
    current.assign(
        team=current["away_team"],
        attack_contrib=current["pred_away_goals"] * current["week_weight"],
        defence_contrib=(ga_baseline - current["pred_home_goals"]) * current["week_weight"]
    )[["team", "attack_contrib", "defence_contrib"]],

    # --- FUTURE GW contributions ---
    future.assign(
        team=future["home_team"],
        attack_contrib=future["pred_home_goals"] * future["week_weight"],
        defence_contrib=(ga_baseline -future["pred_away_goals"]) * future["week_weight"],
    )[["team", "attack_contrib", "defence_contrib"]],

    future.assign(
        team=future["away_team"],
        attack_contrib=future["pred_away_goals"] * future["week_weight"],
        defence_contrib=(ga_baseline -future["pred_home_goals"]) * future["week_weight"],
    )[["team", "attack_contrib", "defence_contrib"]],
], ignore_index=True)

team_outlook = (
    team_outlook
    .groupby("team", as_index=False)
    .agg(
        attack_outlook=("attack_contrib", "sum"),
        defence_outlook=("defence_contrib", "sum"),
    )
)

# team_outlook["defence_outlook"] = -team_outlook["defence_outlook"]

team_outlook["fixture_attack_score"] = (
    team_outlook["attack_outlook"] / (team_outlook["attack_outlook"].abs().max() + 1e-9)
)

team_outlook["fixture_defence_score"] = (
    team_outlook["defence_outlook"] / (team_outlook["defence_outlook"].abs().max() + 1e-9)
)

# team_outlook = pd.concat([
#     # Home team gets +pred
#     current[["home_team", "weighted_pred"]].rename(columns={"home_team": "team"}),

#     # Away team gets -pred
#     current.assign(weighted_pred = -current["weighted_pred"])[["away_team", "weighted_pred"]]
#          .rename(columns={"away_team": "team"}),

#     # Future home team
#     future[["home_team", "weighted_pred"]].rename(columns={"home_team": "team"}),

#     # Future away team (negative)
#     future.assign(weighted_pred = -future["weighted_pred"])[["away_team","weighted_pred"]]
#          .rename(columns={"away_team":"team"})
# ])

# team_outlook = team_outlook.groupby("team", as_index=False)["weighted_pred"].sum()


######### STAT LOGIC

merged = playerstats.merge(players, left_on="id", right_on="player_id", how="left")

merged.rename(columns={
    "web_name_y": "web_name",
    "first_name_y": "first_name",
    "second_name_y": "second_name"
}, inplace=True)


merged.drop(columns=[
    "web_name_x", "first_name_x", "second_name_x"
], errors="ignore", inplace=True)

current_stats = merged.merge(
    team_outlook[["team", "fixture_attack_score", "fixture_defence_score"]],
    left_on="team_code",
    right_on="team",
    how="left"
).drop(columns=["team"])

current_stats = current_stats.merge(
    teams_latest[
        ["team",
         "elo",
         "weighted_attack",
         "weighted_defence"]
    ],
    left_on="team_code",
    right_on="team",
    how="left"
).drop(columns=["team"])

current_stats = current_stats.rename(columns={
    "elo": "team_elo",
    "weighted_attack": "team_weighted_attack",
    "weighted_defence": "team_weighted_defence"
})

current_stats = current_stats.merge(
    teams[["code", "short_name"]],
    left_on="team_code",
    right_on="code",
    how="left"
).rename(columns={"short_name": "team_name"}).drop(columns=["code"])

###LAST SEASON'S STATS

# Columns to bring from last year
cols_needed = [
    "web_name",
    "fpl_points",
    "defcon",
    "minutes_played",
    "goals",
    "assists",
    "xg",
    "xa"
]

# Prepare last year slice with prefix
last_year_slim = last_years_stats[cols_needed].copy()
last_year_slim = last_year_slim.rename(
    columns={col: f"last_{col}" for col in cols_needed if col != "web_name"}
)

last_year_slim = last_year_slim[last_year_slim["last_fpl_points"] > 0]

# Merge (current is LEFT, last year secondary)
merged = current_stats.merge(
    last_year_slim,
    left_on="web_name",
    right_on="web_name",
    how="left",
    indicator=True
)

merged = merged.merge(
    form_summary,
    on="id",
    how="left"
)

to_remove = merged.set_index(["player_id", "last_fpl_points"]).index.isin([
    (525, 26), (432, 71), (437, 81), (580, 58),
    (329, 8), (671, 69), (538, 126), (32, 75),
    (59, 71), (66, 104), (235, 26), (236, 6)
])

merged = merged[~to_remove]

# for col in merged.columns:
#     print(col)
    
merged["bonus_p90"] = (
    merged["bps"] / merged["minutes"] * 90
).replace([np.inf, -np.inf], 0).fillna(0)

merged["fixture_score"] = np.select(
    [
        merged["position"] == "Forward",
        merged["position"] == "Midfielder",
        merged["position"] == "Defender",
        merged["position"] == "Goalkeeper",
    ],
    [
        merged["fixture_attack_score"],
        0.65 * merged["fixture_attack_score"] + 0.35 * merged["fixture_defence_score"],
        merged["fixture_defence_score"],
        merged["fixture_defence_score"],
    ],
    default=0.0
)

merged["form_score_raw"] = merged.apply(compute_form_score, axis=1)
merged["exposure_factor"] = (merged["form_minutes"] / 90).clip(upper=1)
merged["form_score"] = (merged["form_score_raw"] * merged["exposure_factor"])


## Last season analysis

merged["has_last_year"] = merged["last_fpl_points"].notna() & (merged["last_fpl_points"] > 0)

merged["last_year_pace"] = np.where(
    merged["has_last_year"],
    merged["last_fpl_points"] * (GW / 38),
    np.nan
)

merged["pace_diff"] = merged["total_points"] - merged["last_year_pace"]

merged["expected_true_pace"] = np.where(
    merged["has_last_year"],
    0.70 * merged["total_points"] + 0.30 * merged["last_year_pace"],
    np.nan
)

merged["pace_regression_boost"] = merged["expected_true_pace"] - merged["total_points"]

# Normalize ONLY over players with real history
max_abs_boost = merged.loc[merged["has_last_year"], "pace_regression_boost"].abs().max()
if pd.isna(max_abs_boost) or max_abs_boost == 0:
    max_abs_boost = 1

merged["pace_regression_boost_norm"] = merged["pace_regression_boost"] / max_abs_boost
merged["pace_regression_boost_norm"] = merged["pace_regression_boost_norm"].clip(-0.5, 0.5)

# New players get 0 (neutral)
merged["pace_regression_boost_adj"] = np.where(
    merged["has_last_year"],
    merged["pace_regression_boost_norm"],
    0
)

# Stability — only for players WITH history
merged["pace_stability"] = np.where(
    merged["has_last_year"],
    np.minimum(merged["total_points"] + 5, merged["last_year_pace"] + 5) /
    np.maximum(merged["total_points"] + 5, merged["last_year_pace"] + 5),
    0
)

# 4. Scale NORM (not adj)
merged["pace_regression_boost_norm_scaled"] = scale_global(
    merged[merged["has_last_year"]], "pace_regression_boost_norm"
).reindex(merged.index).fillna(0)

# 5. New players get 0 again post-scaling
merged["pace_regression_boost_adj_scaled"] = np.where(
    merged["has_last_year"],
    merged["pace_regression_boost_norm_scaled"],0)

# AVailability

merged["minutes_factor"] = (
    merged["minutes"] / TOTAL_MINUTES
).clip(upper=1)

merged["minutes_factor"] = np.sqrt(merged["minutes_factor"])

#### P90 Expected 

merged["x_fpl_p90"] = merged.apply(expected_fpl_p90, axis=1)
merged["x_fpl_p90"] = merged["x_fpl_p90"].fillna(0)

## Team Metrics

attack = merged["team_weighted_attack"].astype(float)
defence = merged["team_weighted_defence"].astype(float)

atk_min, atk_max = attack.min(), attack.max()
def_min, def_max = defence.min(), defence.max()

# Normalize attack (high = good)
attack_norm = (attack - atk_min) / (atk_max - atk_min + 1e-6)

# Normalize defence (high = good, after inversion)
defence_norm = (defence - def_min) / (def_max - def_min + 1e-6)

merged["team_attack_norm"] = attack_norm
merged["team_defence_norm"] = defence_norm

attack_weight = merged["position"].map({
    "Forward": 0.75,
    "Midfielder": 0.60,
    "Defender": 0.30,
    "Goalkeeper": 0.10
})

defence_weight = 1 - attack_weight

merged["team_strength_factor"] = np.where(
    merged["position"].isin(["Forward", "Midfielder"]),
    merged["team_attack_norm"],
    merged["team_defence_norm"]
)

for col in [
    "x_fpl_p90",
    "fixture_score",
    "points_per_game",
    "form_score",
    "team_strength_factor",
    "bonus_p90"
]:
    merged[col + "_scaled"] = scale_global(merged, col)

status_multiplier = {
    "a": 1.0,
    "d": 0.7,
    "i": 0.0,
    "s": 0.0
}

merged["availability_factor"] = merged["status"].map(status_multiplier).fillna(1.0)

merged["weekly_expected_return_raw"] = (
    0.25 * merged["x_fpl_p90_scaled"] +
    0.21 * merged["points_per_game_scaled"] +
    0.22 * merged["fixture_score_scaled"] +
    0.22 * merged["form_score"] +
    0.05 * merged["bonus_p90_scaled"] +
    0.05 * merged["team_strength_factor_scaled"]
)

merged["weekly_expected_return_raw"] *= merged["availability_factor"]

merged["overall_expected_return_raw"] = (
    0.37 * merged["x_fpl_p90_scaled"] +
    0.33 * merged["points_per_game_scaled"] +
    0.20 * merged["bonus_p90_scaled"] +
    0.09 * merged["team_strength_factor_scaled"] +
    0.01 * merged["pace_regression_boost_adj_scaled"]
)
print(
    merged.loc[
        merged["availability_factor"] == 0,
        ["web_name", "status", "weekly_expected_return_raw"]
    ].head(10)
)

# --- apply stability + availability modifiers ---
merged["weekly_expected_return"] = merged["weekly_expected_return_raw"] * merged["minutes_factor"]
merged["overall_expected_return"] = merged["overall_expected_return_raw"] * merged["minutes_factor"]

# --- VORP CALCULATION ---

weekly_vorp = compute_vorp_dynamic(
    merged,
    metric="weekly_expected_return"
).copy()

weekly_vorp = weekly_vorp.rename(columns={
    "baseline": "weekly_baseline",
    "VORP": "weekly_VORP",
    "above_replacement": "weekly_above_replacement"
})

weekly_vorp["weekly_VORP_scaled"] = scale_global(weekly_vorp, "weekly_VORP")

overall_vorp = compute_vorp_dynamic(
    merged,
    metric="overall_expected_return"
).copy()

overall_vorp = overall_vorp.rename(columns={
    "baseline": "overall_baseline",
    "VORP": "overall_VORP",
    "above_replacement": "overall_above_replacement"
})

overall_vorp["overall_VORP_scaled"] = scale_global(overall_vorp, "overall_VORP")

weekly_vorp = weekly_vorp.drop_duplicates(subset=["player_id"])
overall_vorp = overall_vorp.drop_duplicates(subset=["player_id"])

# print(overall_vorp["player_id"].value_counts().head(10))

merged_vorp = (
    overall_vorp
    .merge(
        weekly_vorp[[
            "player_id",
            "weekly_VORP",
            "weekly_VORP_scaled",
            "weekly_baseline",
            "weekly_above_replacement"
        ]],
        on="player_id",
        how="left",
        validate="one_to_one"
    )
)

# Valuation Flags

merged_vorp["valuation_signal"] = (
    0.55 * merged_vorp["overall_VORP_scaled"] -
    0.35 * merged_vorp["weekly_VORP_scaled"] -
    0.05 * merged_vorp["points_per_game_scaled"] +
    0.05 * merged_vorp["pace_regression_boost_adj_scaled"])

merged_vorp["valuation_flag"] = np.where(
    merged_vorp["valuation_signal"] > merged_vorp["valuation_signal"].quantile(0.67),
    "BUY_LOW",
    np.where(
        merged_vorp["valuation_signal"] < merged_vorp["valuation_signal"].quantile(0.33),
        "SELL_HIGH",
        "HOLD" ))

merged_vorp["weekly_vorp_rank"] = (
    merged_vorp["weekly_VORP"]
    .rank(method="dense", ascending=False)
    .astype("Int64")   # ← pandas nullable int
)

merged_vorp["rank"] = (
    merged_vorp["overall_VORP"]
    .rank(method="dense", ascending=False)
    .astype("Int64")
)

merged_vorp["rank_gap"] = (
    merged_vorp["rank"] -
    merged_vorp["weekly_vorp_rank"]
)

merged_vorp = merged_vorp.rename(columns={
    "defensive_contribution_per_90": "defcon_p90",
    "expected_goal_involvements_per_90": "xgi_p90",
})

print(merged_vorp[["x_fpl_p90", "form_score", "points_per_game", "pace_regression_boost_adj", "team_strength_factor", "fixture_score"]].mean())
print(merged_vorp[[
    "x_fpl_p90_scaled", 
    "form_score", 
    "fixture_score_scaled", 
    "points_per_game_scaled", 
    "team_strength_factor_scaled", 
    "bonus_p90_scaled",
    "pace_regression_boost_adj_scaled"
]].agg(["min", "max", "mean", "std"]))

desired_order = [
    "rank", "web_name", "position", "team_name",  
    "total_points", "minutes",  "weekly_VORP", "overall_VORP",  
    "points_per_game", "x_fpl_p90",
    "fixture_score", "form_score", 
    "form_points_p90", "form_bps_p90", "form_xgi_p90", "form_defcon_p90", 
    "bonus_p90", "defcon_p90", "xgi_p90",
    "team_strength_factor",
    "valuation_signal", "pace_diff", "valuation_flag", "player_id"
]

merged_vorp = merged_vorp[[c for c in desired_order if c in merged_vorp.columns]].copy()

# rename_map = {
#     "web_name: "name"}

# merged_vorp.rename(columns=rename_map, inplace=True)
merged_vorp = merged_vorp[ merged_vorp["minutes"] >= 180 ].copy()
merged_vorp = merged_vorp.sort_values("weekly_VORP", ascending=False)

merged_vorp = merged_vorp.replace([np.inf, -np.inf], np.nan)
num_cols = merged_vorp.select_dtypes(include=["number"]).columns
merged_vorp[num_cols] = merged_vorp[num_cols].fillna(0)

# fill text columns only
obj_cols = merged_vorp.select_dtypes(include=["object"]).columns
merged_vorp[obj_cols] = merged_vorp[obj_cols].fillna("")

merged_vorp.to_csv("tableau_fpl.csv",  index=False, encoding="utf-8-sig")

# sns.boxplot(x="position", y="overall_VORP", data=merged_vorp)
# sns.scatterplot(x="position", y="overall_VORP", data=merged_vorp, hue="x_fpl_p90")

##### League Integration

LEAGUE_ID = 1056
MY_ENTRY_ID = 168052
BASE_URL = "https://draft.premierleague.com/api"
url = f"https://draft.premierleague.com/api/league/{LEAGUE_ID}/element-status"
resp = requests.get(url)
raw = resp.json()

KNOWN_OWNERSHIP = {
    "Šeško": 168052,
    "Woltemade": 3062,
    "Hincapie": 3062,
    "Donnarumma": 53977,
    "Xhaka": 192715,
    "Stach": 168052,
    "Thiaw": 6367
}

league_players = pd.json_normalize(raw["element_status"])

league_players["owner"] = pd.to_numeric(league_players["owner"], errors="coerce").astype("Int64")
league_players["status"] = league_players["status"].astype(str)

league_players["is_available"] = league_players["status"] == "a"
league_players["is_mine"] = league_players["owner"] == MY_ENTRY_ID
league_players["is_taken"] = league_players["status"] != "a"

available_players = league_players[league_players["is_available"]]
my_players = league_players[league_players["is_mine"]]

taken_players = league_players[
    (~league_players["is_available"]) &
    (~league_players["is_mine"])
]

league_players = league_players.rename(columns={"element": "player_id"})


## NEW
entries_url = f"{BASE_URL}/league/{LEAGUE_ID}/entries"
entries_raw = requests.get(entries_url).json()

# entries_raw might be a dict like {"league_entries":[...]} or just a list
if isinstance(entries_raw, dict):
    rows = entries_raw.get("league_entries") or entries_raw.get("entries") or []
else:
    rows = entries_raw

entries = pd.json_normalize(rows)

# Normalize ID column safely
if "entry_id" not in entries.columns and "id" in entries.columns:
    entries = entries.rename(columns={"id": "entry_id"})

# Drop duplicate columns defensively
entries = entries.loc[:, ~entries.columns.duplicated()]

# Build manager name
if {"player_first_name", "player_last_name"}.issubset(entries.columns):
    entries["manager"] = (
        entries["player_first_name"].fillna("") + " " +
        entries["player_last_name"].fillna("")
    ).str.strip()
else:
    entries["manager"] = entries.get("entry_name", "")

# Keep what exists (draft endpoints vary)
keep = [c for c in ["entry_id", "entry_name", "player_first_name", "player_last_name"] if c in entries.columns]
entries = entries[keep].copy()

if {"player_first_name", "player_last_name"}.issubset(entries.columns):
    entries["manager"] = entries["player_first_name"].fillna("") + " " + entries["player_last_name"].fillna("")
else:
    entries["manager"] = ""

ENTRY_RENAMES = {
    "The Highs And Dalots": "HighsNDalots"
}

entries["entry_name"] = entries["entry_name"].replace(ENTRY_RENAMES)

merged_vorp = merged_vorp.merge(
    league_players[["player_id", "owner", "is_available", "is_mine", "is_taken"]],
    left_on="player_id",
    right_on="player_id",
    how="left"
)

for name, owner_id in KNOWN_OWNERSHIP.items():
    mask = merged_vorp["web_name"] == name

    merged_vorp.loc[mask, "owner"] = owner_id
    merged_vorp.loc[mask, "is_mine"] = owner_id == MY_ENTRY_ID
    merged_vorp.loc[mask, "is_available"] = False
    merged_vorp.loc[mask, "is_taken"] = True

merged_vorp = merged_vorp.merge(
    entries[["entry_id", "entry_name", "manager"]],
    left_on="owner",
    right_on="entry_id",
    how="left"
)


# Choose the metrics you want to compare across teams
metric_cols = [c for c in ["x_fpl_p90", "points_per_game", "VORP", "total_points", "minutes", "defensive_contribution_per_90", "bonus_p90"] if c in merged_vorp.columns]

team_summary = (
    merged_vorp[~merged_vorp["is_available"].fillna(False)]  # owned players only
    .groupby(["owner", "entry_name", "manager"], dropna=False)
    .agg(
        players=("player_id", "count"),
        minutes_sum=("minutes", "sum"),

        xfpl_p90_mean=("x_fpl_p90", "mean"),
        xfpl_p90_sum=("x_fpl_p90", "sum"),

        ppg_mean=("points_per_game", "mean"),
        weekly_vorp_sum=("weekly_VORP", "sum"),
        overall_vorp_sum=("overall_VORP", "sum"),
        total_points_sum=("total_points", "sum"),

        defcon_p90_mean=("defcon_p90", "mean"),
        bonus_p90_mean=("bonus_p90", "mean"),
    )
    .reset_index()
    .sort_values("xfpl_p90_sum", ascending=False)
)

team_summary["team_label"] = (
    team_summary["entry_name"]
    .fillna("Entry " + team_summary["owner"].astype(str))
)

team_summary.head(20)

(
    team_summary
    .sort_values("xfpl_p90_sum", ascending=False)
    .to_csv("league_team_summary.csv", index=False)
)

team_summary[["owner", "players"]].sort_values("owner")

# --- Styled Excel Export ---
merged_vorp = merged_vorp.replace([np.inf, -np.inf], np.nan)

mine_col = merged_vorp.columns.get_loc("is_mine")
avail_col = merged_vorp.columns.get_loc("is_available")
taken_col = merged_vorp.columns.get_loc("is_taken")
                                
last_col = len(merged_vorp.columns) - 1
taken_col_letter = xl_col_to_name(taken_col)
mine_col_letter = xl_col_to_name(mine_col)
avail_col_letter = xl_col_to_name(avail_col)

with pd.ExcelWriter("updated_fpl.xlsx", engine="xlsxwriter") as writer:

    merged_vorp.to_excel(writer, index=False, sheet_name="VORP")
    wb = writer.book
    ws = writer.sheets["VORP"]

    # -----------------------
    #  FORMATS
    # -----------------------

    header_fmt = wb.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#6AA84F",  # FPL green
        "border": 1,
        "align": "center"
    })

    num_fmt = wb.add_format({"num_format": "0.00"})
    text_fmt = wb.add_format({"text_wrap": True})

    my_team_fmt = wb.add_format({
        "bg_color": "#C9DAF8",
        "bold": True
    })

    taken_fmt = wb.add_format({
        "bg_color": "#EEEEEE",
    })
    
    available_fmt = wb.add_format({
        "bg_color": "#f0e2f6"  # very faint purple
    })


    # -----------------------
    #  HEADER + COLUMN WIDTH
    # -----------------------

    for col_idx, col_name in enumerate(merged_vorp.columns):

        # header text
        ws.write(0, col_idx, col_name, header_fmt)

        # --- width based on header length (minimum 10, max 40) ---
        max_len = max(
            merged_vorp[col_name].astype(str).str.len().max(),
            len(col_name)
        ) + 2
        
        col_width = min(max(max_len, 10), 50)
    
        # --- format rules ---
        if col_name == "rank":
            fmt = wb.add_format({"num_format": "0"})   # whole number only
            ws.set_column(col_idx, col_idx, col_width, fmt)
    
        else:
            is_numeric = merged_vorp[col_name].dtype != "object"
            ws.set_column(
                col_idx,
                col_idx,
                col_width,
                num_fmt if is_numeric else text_fmt
            )

    # freeze top row
    ws.freeze_panes(1, 0)

    # -----------------------
    #  CONDITIONAL FORMATTING
    # -----------------------

    color_cols = [
        "total_points",  "weekly_VORP",	"overall_VORP", "overall_vorp", "points_per_game", "x_fpl_p90",
    "form_points_p90", "", " form_defcon_p90", "form_bps_p90", "form_score", "defcon_p90",
    "fixture_score",
        "total_points", "bonus_p90", "form_xgi_p90",	"form_defcon_p90",
    "pace_ratio", "team_strength_factor", "minutes", "valuation_signal"

    ]
    
    n_rows = len(merged_vorp)

    for col_name in color_cols:
        if col_name in merged_vorp.columns:
            col_idx = merged_vorp.columns.get_loc(col_name)

            ws.conditional_format(
                1, col_idx,
                n_rows, col_idx,
                {
                    "type": "3_color_scale",
                    "min_color": "#F4CCCC",   # red
                    "mid_color": "#FFE599",   # yellow
                    "max_color": "#93C47D"    # green
                }
            )

    # -----------------------
    #  HIGHLIGHT YOUR TEAM
    # -----------------------
    
    # Highlight YOUR players
    ws.conditional_format(
        1, 0,
        n_rows, last_col,
        {
            "type": "formula",
            "criteria": f"=${mine_col_letter}2=TRUE",
            "format": my_team_fmt
        }
    )
    
    # Highlight AVAILABLE players (faint)
    ws.conditional_format(
        1, 0,
        n_rows, last_col,
        {
            "type": "formula",
            "criteria": f"=${avail_col_letter}2=TRUE",
            "format": available_fmt
        }
    )

    # TAKEN
    ws.conditional_format(
        1, 0,
        n_rows, last_col,
        {
            "type": "formula",
            "criteria": f"=${taken_col_letter}2=TRUE",
            "format": taken_fmt
        }
    )

print("✅ Exported styled file")
