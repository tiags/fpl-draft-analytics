import pandas as pd
import numpy as np
import os
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from xlsxwriter.utility import xl_col_to_name
import subprocess

GIT_REPO = os.path.expanduser("~/MyProjects/FPL/FPL-Elo-Insights")
subprocess.run(["git", "-C", GIT_REPO, "pull"], check=True)

FPL_DATA_FOLDER = os.path.expanduser('~/MyProjects/FPL/FPL-Elo-Insights/data/2026-2027')

players = pd.read_csv(f"{FPL_DATA_FOLDER}/players.csv",encoding="utf-8")
teams = pd.read_csv(
    f"{FPL_DATA_FOLDER}/teams.csv",
    encoding="utf-8")

PRESEASON_FOLDER = f"{FPL_DATA_FOLDER}/By Tournament/Friendlies/GW0"
UEFASUPERCUP_FOLDER = f"{FPL_DATA_FOLDER}/By Tournament/Uefa Super Cup/GW0"
COM_SHIELD_FOLDER = f"{FPL_DATA_FOLDER}/By Tournament/Community Shield/GW0"
STATUS_FOLDER = f"{FPL_DATA_FOLDER}/By Tournament/Premier League/GW1/playerstats.csv"

last = pd.read_excel("seasons/2025-26/End2526_Full.xlsx")  # your protected board


LEAGUE_SIZE = int(input("8 or 14? "))
print(f"League size: {LEAGUE_SIZE} teams")

NAME_FIXES = {
    "M.Fernandes": "Fernandes",
    "Dúbravka": "Dubravka",
    "J.Gomes": "Gomes",
    # add as you find them: "last_name": "preseason_name"
}
last["web_name"] = last["web_name"].replace(NAME_FIXES)

###################

TEAM_STRENGTH = {
    # Tier 1
    "ARS":1.00,
    # Tier 2
    "MCI":0.85, "LIV":0.82, "CHE":0.88, "MUN":0.78,
    # Tier 3
    "TOT":0.70, "AVL":0.68, "NFO":0.65, "BRE":0.68,
    # Tier 4
    "CRY":0.52, "EVE":0.62, "BHA":0.58, "LEE":0.58, "SUN":0.56, "BOU":0.52, "NEW":0.52, "FUL":0.50,
    # Tier 5
    "COV":0.38, "IPS":0.34, "HUL":0.20,
}

IMPOSTORS = [
    301,   # Palmer (IPS GK) 
    108,  # C. Wilson
    172, # GK Wilson
    343, # D. James
]

####

def scale_global(df, col):
    s = df[col]
    med = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) + 1e-6
    return (s - med) / iqr

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

STARTABLE_SUPPLY = {
    "Forward": 20,      # T1+T2 — then a cliff
    "Midfielder": 70,   # deep
    "Defender": 80,     # very deep
    "Goalkeeper": 20,   # adequate
}

AVG_STARTED = {"Goalkeeper":1.0, "Defender":4.0, "Midfielder":4.0, "Forward":2.0}

POSITION_LIMITS = {pos: round(LEAGUE_SIZE * n) for pos, n in AVG_STARTED.items()}
# 8-team  → {"Goalkeeper":8,  "Defender":32, "Midfielder":32, "Forward":16}   (all demand-bound)
# 14-team → {"Goalkeeper":14, "Defender":56, "Midfielder":56, "Forward":20}   (Forward supply-bound: 20 not 28)

def compute_vorp_dynamic(df, metric="expected_return"):
    output = []
    for pos, limit in POSITION_LIMITS.items():
        subset = df[df["position"] == pos].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(metric, ascending=False)
        if len(subset) < limit:
            baseline = subset[metric].min()
        else:
            baseline = subset.iloc[limit - 1][metric]
        print(pos, len(subset), "baseline:", round(baseline, 3))
        subset["baseline"] = baseline
        subset["VORP"] = subset[metric] - baseline
        subset["above_replacement"] = subset["VORP"] > 0
        output.append(subset)
    return pd.concat(output, ignore_index=True)

dfs = []
for folder in [PRESEASON_FOLDER, UEFASUPERCUP_FOLDER, COM_SHIELD_FOLDER]:
    path = f"{folder}/playermatchstats.csv"
    dfs.append(pd.read_csv(path, encoding="utf-8"))

pm = pd.concat(dfs, ignore_index=True)

num_cols = pm.columns.drop(["player_id", "match_id"])  # keep keys out
pm[num_cols] = pm[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

agg = pm.groupby("player_id", as_index=False).sum(numeric_only=True)
agg = agg[agg["minutes_played"] >= 45]

p90_cols = [
    "goals", "assists", "xg", "xa", "shots_on_target",
    "tackles_won", "interceptions", "recoveries", "clearances", "blocks",
    "saves", "goals_conceded",
]

for col in p90_cols:
    if col in agg.columns:
        agg[col + "_p90"] = agg[col] / agg["minutes_played"] * 90
        
# attach position
agg = agg.merge(
    players[["player_id", "position"]],
    on="player_id",
    how="left"
)

# rebuild defcon per-90 (per-match average of defensive actions, thresholded)
def defcon_p90(row):
    pos = str(row["position"]).strip()
    if pos == "Defender":
        actions = row["tackles_won"] + row["interceptions"] + row["clearances"] + row["blocks"]
        threshold = 10
    elif pos in ["Midfielder", "Forward"]:
        actions = row["tackles_won"] + row["interceptions"] + row["recoveries"]
        threshold = 12
    else:  # Goalkeeper
        return 0.0
    # per-90 rate of actions, then FPL points: 2 pts if hitting threshold in a match
    actions_p90 = actions / row["minutes_played"] * 90
    return 2 * min(actions_p90 / threshold, 1)

agg["defcon_fpl_p90"] = agg.apply(defcon_p90, axis=1)

# names + team_code + position from players
agg = agg.merge(
    players[["player_id", "web_name", "team_code"]],
    on="player_id",
    how="left"
)

# team short_name from teams (code <-> team_code)
agg = agg.merge(
    teams[["code", "short_name"]],
    left_on="team_code",
    right_on="code",
    how="left"
).drop(columns=["code"])

agg = agg[~agg["player_id"].isin(IMPOSTORS)]

agg["expected_goals_per_90"] = agg["goals"] / agg["minutes_played"] * 90   # actual goals as xg proxy (or xg_p90 if you trust preseason xg)
agg["expected_assists_per_90"] = agg["assists"] / agg["minutes_played"] * 90
agg["expected_goals_conceded_per_90"] = agg["goals_conceded"] / agg["minutes_played"] * 90
agg["saves_per_90"] = agg["saves"] / agg["minutes_played"] * 90
agg["defensive_contribution_per_90"] = (
    agg["tackles_won"] + agg["interceptions"] + agg["recoveries"] + agg["clearances"] + agg["blocks"]
) / agg["minutes_played"] * 90
agg["clean_sheets_per_90"] = 0   # not available preseason

matches_played = pm.groupby("player_id").size().rename("matches")
agg = agg.merge(matches_played, on="player_id", how="left")

agg["minutes_factor"] = np.sqrt((agg["minutes_played"]/agg["matches"]/90).clip(upper=1))

agg["x_fpl_p90"] = agg.apply(expected_fpl_p90, axis=1)
agg["x_fpl_p90"] = agg["x_fpl_p90"].fillna(0)

agg["ps_trust"] = (agg["minutes_played"] / 180).clip(upper=1)
agg["preseason_signal"] = agg["x_fpl_p90"] * agg["ps_trust"]

board = agg[[
    "web_name", "position", "short_name", "minutes_played",
    "preseason_signal", "x_fpl_p90", "defcon_fpl_p90", "ps_trust",
    "goals", "assists", "xg", "xa", "player_id"
]].copy()

board = board.sort_values("preseason_signal", ascending=False)

# round numerics for readability
num = board.select_dtypes(include="number").columns
board[num] = board[num].round(2)

# preseason carries CURRENT-season facts + signal
ps = agg[["player_id", "web_name", "position", "short_name", "preseason_signal", "minutes_played"]].rename(
    columns={"short_name": "team_now", "minutes_played": "ps_minutes"}
)

merged = last.merge(ps, on="web_name", how="outer", suffixes=("_x","_y"), indicator=True)
has_last = merged["_merge"] != "right_only"

# left_only = merged[merged["_merge"]=="left_only"]["web_name"].tolist()
# right_only = merged[merged["_merge"]=="right_only"]["web_name"].tolist()
# # look for near-duplicates across the two lists
# print("LEFT ONLY:", sorted(left_only))
# print("RIGHT ONLY:", sorted(right_only))

current_players = set(players["web_name"])
merged = merged[~((merged["_merge"]=="left_only") & (~merged["web_name"].isin(current_players)))]

# coalesce: prefer CURRENT (preseason) facts, fall back to last season
merged["position"] = merged["position_y"].fillna(merged["position_x"])
merged["team"]     = merged["team_now"].fillna(merged["team_name"])
merged["preseason_signal"] = merged["preseason_signal"].fillna(0)

merged = merged[~((merged["web_name"]=="Wilson") & (merged["points_per_game"]==2.4))]
merged = merged[~((merged["web_name"]=="James") & (merged["position"]=="Midfielder"))]

TEAM_FIXES = {
    "Senesi": "TOT",
    "Struijk": "BHA",
    "Awoniyi": "COV",
    "Konsa": "ARS",
    # add others you spot: "web_name": "correct_team_code"
}

for name, correct_team in TEAM_FIXES.items():
    merged.loc[merged["web_name"] == name, "team"] = correct_team



merged["team_strength_factor"] = merged["team"].map(TEAM_STRENGTH)

bps_benchmark = merged["bonus_p90"].quantile(0.95)   # ~elite bps level
merged["bonus_est_p90"] = (merged["bonus_p90"] / bps_benchmark * 3).clip(upper=3)

for col in ["x_fpl_p90", "points_per_game", "bonus_est_p90", "team_strength_factor", "preseason_signal"]:
    merged[col + "_scaled"] = scale_global(merged, col)

merged["ls_minutes_factor"] = np.sqrt((merged["minutes"] / 3420).clip(upper=1))

merged.loc[has_last, "overall_expected_return"] = (
    0.44 * merged.loc[has_last, "x_fpl_p90_scaled"] +
    0.39 * merged.loc[has_last, "points_per_game_scaled"] +
    0.14 * merged.loc[has_last, "bonus_est_p90_scaled"] +
    0.04 * merged.loc[has_last, "team_strength_factor_scaled"]
) * merged.loc[has_last, "ls_minutes_factor"]

##newcomers
merged.loc[~has_last, "overall_expected_return"] = (
    merged.loc[~has_last, "preseason_signal_scaled"] * 0.2
)

merged["overall_expected_return"] = merged["overall_expected_return"].fillna(0)

merged = merged.drop(columns=[c for c in ["overall_VORP","weekly_VORP","overall_baseline","rank"] if c in merged.columns])

status_df = pd.read_csv(STATUS_FOLDER, encoding="utf-8")

merged = merged.merge(status_df[["web_name", "status"]], on="web_name", how="left")

status_multiplier = {"a": 1.0, "d": 0.9, "i": 0.3, "s": 0.8}
merged["availability_factor"] = merged["status"].map(status_multiplier).fillna(1.0)

merged["overall_expected_return"] *= merged["availability_factor"]

TEAM_CONTEXT = {
    ("NFO", "Defender"): 0.2,   # Glasner + outlier reversion
    ("LIV", "Defender"): 0.2,   # Iraola press
    ("LIV", "Midfielder"): 0.15,
    ("CHE", "ALL"): 0.05,        # stability, no europe
    ("TOT", "ALL"): 0.05,
    ("SUN", "ALL"): -0.03,       # euro
    ("BHA", "ALL"): -0.02,       # conference
    ("MCI", "ALL"): -0.02,       # post-pep
}

def context_add(row):
    return TEAM_CONTEXT.get(
        (row["team"], row["position"]),
        TEAM_CONTEXT.get((row["team"], "ALL"), 0)   # default 0, not 1.0
    )

merged["context_factor"] = merged.apply(context_add, axis=1)
merged["overall_expected_return"] += merged["context_factor"]   # += not *=

PLAYER_OVERRIDE = {
    "Isak":   0.7,   # bounce-back + Ekitike injury clears minutes
    "A.Becker":   0.2,   # LIV
    "Palmer": 1.2,   # bounce-back, no Europe, proven ceiling
    "Murillo":     0.39,  # bounce-back, no Europe, proven ceiling, Glasner
    "Milenković":  0.4,  # bounce-back, no Europe, proven ceiling, Glasner
    "N.Williams":  0.39,  # bounce-back, no Europe, proven ceiling, Glasner
    "Sels" :  0.15,   # bounce-back, no Europe, proven ceiling, Glasner
    "Wissa":  0.3,   # returning from injury, was elite 2 seasons ago
    "Watkins":  0.15,   #  off to Arabia?
    "Tielemans":  0.1,   # defcon
    "Canvot":  0.15,   # 2nd season
    "Tzolis":  0.6,   # new player, good start
    "Vuskovic": 0.35,   # new player, good start
    "Muharemović": 0.38,   # new player, good start
    "Gonzalo": 0.18,   # new player, good start
    "Beto": 0.1,   # backup
    "Palestra": 0.2,   # new player, good start
    "Schlager": 0.1,   # new player, good start
    "Palacios": 0.1,   # new player, looks bright
    "Barry": 0.22,   # 2nd season
    "Rayan": 0.4,   # 2nd season
    "Ayari": 0.3,   # 2nd season
    "Kudus": 0.37,   # injury return
    "Barnes": 0.3,   # space with Gordon out
    "Kerkez": 0.4,   # Iraola boy, Robertson out
    "Maatsen": 0.01,   # Digne gone
    "Mosquera": 0.3,   # Saliba injury
    "Jacquet": 0.5,   # Liverpool new signing
    "Pedro Porro": 0.2,   # comeback
    "Aina": 0.1,   # not starter
    "Sánchez": 0.035,   # CHE GK
    "Guéhi": 0.69,   # no longer top dog, risk
    "Havertz": 0.4,   # starter?
    "Gakpo": 0.1,   # no longer top dog, risk
    "Keane": -0.01,   # lost starter
    "Pope": -0.05,   # lost starter
    "James": 0.35,   # injury issues
    "Welbeck": 0.1,   # transfer, less mins
    "Senesi": 0.1,   # transfer, less mins
    "Wood": 0.23,   # healthy    
    "Solanke": 0.15,   # who knows
    "Richarlison": 0.2,   # dnt believe the hype
    "Mateta": 0.2,   # not starter
    "Strand Larsen": 0.21,   #  starter
    "Kalimuendo": 0.1,   #  likely loaned out
    "Brobbey": 0.2,   #  likely starter
    "Konsa": 0.35,   #  cover Saliba
    "M.França": 0.05,   #  not starter
    "Moore": 0.05,   #  not starter
    "Koumas": 0.05,   #  not starter
    "Elliott": 0.05,   #  not starter
    "Amass": 0.05,   #  not starter
    "Ramsay": 0.05,   #  not starter
    "Tsimikas": 0.05,   #  not starter
    "Kinsky": 0.15,   #   starter
    
}

for name, val in PLAYER_OVERRIDE.items():
    merged.loc[merged["web_name"] == name, "overall_expected_return"] = val
    
new_row = {
    "web_name": "M.Sangaré (BRE)",   # unique, won't collide
    "position": "Midfielder",
    "team": "BRE",
    "overall_expected_return": 0.4,   # your manual estimate, anchored to comparable mids
}

merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

new_row = {
    "web_name": "Diomande",   
    "position": "Defender",
    "team": "NFO",
    "overall_expected_return": 0.25,   # your manual estimate, anchored to comparable mids
}

merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

new_row = {
    "web_name": "David",   
    "position": "Forward",
    "team": "BHA",
    "overall_expected_return": 0.22,   # your manual estimate, anchored to comparable mids
}

merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

new_row = {
    "web_name": "Maddison",   # unique, won't collide
    "position": "Midfielder",
    "team": "TOT",
    "overall_expected_return": 0.25,   # your manual estimate, anchored to comparable mids
}

merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

new_row = {
    "web_name": "Manzambi",
    "position": "Midfielder",
    "team": "AVL",
    "overall_expected_return": 0.5,   # talented (EL best young player) but injured — upside stash, not immediate
}
merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

new_row = {
    "web_name": "Colwill",
    "position": "Defender",
    "team": "CHE",
    "overall_expected_return": 0.55,   # return of the mack
}
merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

new_row = {
    "web_name": "Khalaili",
    "position": "Defender",
    "team": "CRY",
    "overall_expected_return": 0.1,   # lets see
}
merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)

EXCLUDE_NAMES = [
    # academy/cameo players floating too high — draft-irrelevant
    "Scarlett", "Austin", "Akpom", "Heaton", "Benitez", "Travers", "Chalobah", "Romero", "Enes Ünal", "Reijnders", "Digne", "Bayindir", "Martinez", "Spence", "Rodrigo"
    # add as you spot them in the visible tier
]

merged = merged[~merged["web_name"].isin(EXCLUDE_NAMES)]

merged["team_pos_rank"] = (
    merged.groupby(["team","position"])["overall_expected_return"]
    .rank(ascending=False, method="first")
)

def depth_penalty(row):
    r = row["team_pos_rank"]
    pos = row["position"]

    if pos == "Goalkeeper":
        return 1.0 if r <= 1 else 0.15          # only 1 GK starts

    if pos == "Forward":
        if r <= 1: return 1.0
        if r == 2: return 0.6
        return 0.2                                # 4th+ — buried

    if pos == "Defender":
        if r <= 4: return 1.0                     # top 4 start comfortably (3-5 range)
        if r == 5: return 0.85                    # 5th — plays often but rotated
        if r == 6: return 0.45                    # 6th — bench/injury cover
        return 0.2                               # 7th+ — buried

    if pos == "Midfielder":
        if r <= 4: return 1.0                     # top 4 start
        if r == 5: return 0.85                    # 5th — frequent
        if r == 6: return 0.55                    # 6th — squad depth
        return 0.2                                # 7th+ — buried

    return 1.0

merged["depth_factor"] = merged.apply(depth_penalty, axis=1)
merged["overall_expected_return"] *= merged["depth_factor"]

# VORP with CURRENT positions (compute_vorp_dynamic splits by position internally)
VORP = compute_vorp_dynamic(merged, metric="overall_expected_return").copy()

scarcity_mult = {pos: LEAGUE_SIZE * AVG_STARTED[pos] / STARTABLE_SUPPLY[pos] for pos in AVG_STARTED}
VORP["VORP"] *= VORP["position"].map(scarcity_mult)

for pos in POSITION_LIMITS:
    sub = merged[merged["position"]==pos].sort_values("overall_expected_return", ascending=False)
    limit = POSITION_LIMITS[pos]
    baseline = sub.iloc[limit-1]["overall_expected_return"]
    print(f"{pos}: limit={limit}, baseline={baseline:.3f}, top={sub.iloc[0]['overall_expected_return']:.3f}")


VORP = VORP.sort_values("VORP", ascending=False)

VORP = VORP.drop_duplicates(subset=["web_name", "team", "position"])

# rank + export
VORP["rank"] = VORP["VORP"].rank(method="dense", ascending=False).astype(int)

board = VORP[[
    "rank","web_name","position","team","VORP",
    "overall_expected_return","preseason_signal","ps_minutes","x_fpl_p90","points_per_game"
]]
board.to_csv(f"draft_board_2627_{LEAGUE_SIZE}.csv", index=False, encoding="utf-8-sig")
print(board.head(40).to_string(index=False))

