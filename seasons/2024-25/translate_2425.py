import pandas as pd
import os

def calculate_fpl_score(row):   
    position = str(row['position']).strip()
    score = 0

    # Minutes played (approximation: 1 pt per appearance <60, 2 pts ≥60)
    approx_appearances = row['minutes_played'] // 90
    score += approx_appearances * 2  # Approximation only

    # Goals
    goal_pts = {'Goalkeeper': 10, 'Defender': 6, 'Midfielder': 5, 'Forward': 4}
    score += row['goals'] * goal_pts.get(position, 0)

    # Assists
    score += row['assists'] * 3

    # Clean sheets (only estimate, accurate if you know how many 60+ min clean-sheet games)
    if 'team_goals_conceded' in row and row['minutes_played'] > 0:
        matches = row['minutes_played'] // 90
        clean_sheets = max(matches - row['team_goals_conceded'], 0)
        if position in ['Goalkeeper', 'Defender']:
            score += clean_sheets * 4
        elif position == 'Midfielder':
            score += clean_sheets * 1

    # Saves (1 pt per 3)
    if position == 'Goalkeeper':
        score += row['saves'] // 3

    # Defensive contributions (scaled by minutes)
    defcon = 0
    if position == 'Defender':
        defcon += (row['tackles_won'] + row['interceptions'] + row['clearances'] + row['blocks']) // 10 * 2
    elif position in ['Midfielder', 'Forward']:
        defcon += (row['tackles_won'] + row['interceptions'] + row['recoveries']) // 12 * 2
    score += defcon

    # Penalties missed
    score -= 2 * row.get('penalties_missed', 0)

    # Goals conceded
    if position in ['Goalkeeper', 'def']:
        score -= row['goals_conceded'] // 2

    # Cards
    score -= row.get('yellow_cards', 0)
    score -= 3 * row.get('red_cards', 0)

    return round(score), defcon

teams = pd.read_csv("/Users/tiagoabreu/Python/MyProjects/FPL/FPL-Elo-Insights/data/2025-2026/teams.csv",encoding="utf-8")
players_2425 = pd.read_csv("/Users/tiagoabreu/Python/MyProjects/FPL/FPL-Elo-Insights/data/2024-2025/players/players.csv",encoding="utf-8")
players_2526 = pd.read_csv("/Users/tiagoabreu/Python/MyProjects/FPL/FPL-Elo-Insights/data/2025-2026/players.csv",encoding="utf-8")
player_stats = pd.read_csv("/Users/tiagoabreu/Python/MyProjects/FPL/FPL-Elo-Insights/data/2024-2025/playerstats/playerstats.csv",encoding="utf-8")

players_2526["full_name"] = players_2526["first_name"].str.lower().str.strip() + " " + players_2526["second_name"].str.lower().str.strip()
players_2425["full_name"] = players_2425["first_name"].str.lower().str.strip() + " " + players_2425["second_name"].str.lower().str.strip()

player_stats.rename(columns={"id": "player_id"}, inplace=True)

base_path = "/Users/tiagoabreu/Python/MyProjects/FPL/FPL-Elo-Insights/data/2024-2025/playermatchstats"
all_playermatchstats = []


# Loop through each GW folder
for folder in sorted(os.listdir(base_path)):
    gw_path = os.path.join(base_path, folder, "playermatchstats.csv")
    
    if os.path.isfile(gw_path):
        df = pd.read_csv(gw_path, encoding="utf-8")
        all_playermatchstats.append(df)

# Combine all weeks
playermatchstats = pd.concat(all_playermatchstats, ignore_index=True)

# Collapse all matchweek rows into one row per player
agg_stats = (
    playermatchstats
    .groupby("player_id", as_index=False)
    .sum(numeric_only=True)
)

# Drop players with too few minutes
agg_stats = agg_stats[agg_stats["minutes_played"] >= 100]

# Per 90 stats
cols_to_p90 = [
    "goals", "assists", "xg", "xa", "xgot", "shots_on_target",
    "tackles_won", "interceptions", "recoveries", "blocks",
    "clearances", "dribbled_past", "duels_won", "duels_lost",
    "was_fouled", "fouls_committed", "saves", "goals_conceded",
    "xgot_faced", "goals_prevented", "sweeper_actions"
]

# ---------- P90 CALC WITH ZERO SAFETY ----------
for col in cols_to_p90:
    if col in agg_stats.columns:
        agg_stats[col + "_p90"] = (
            agg_stats[col] / agg_stats["minutes_played"].replace(0, pd.NA) * 90
        )

for col in cols_to_p90:
    if col in agg_stats.columns:
        agg_stats[col + "_p90"] = agg_stats[col] / agg_stats["minutes_played"] * 90
        
# ---------- MERGE PLAYER META ----------
agg_stats = agg_stats.merge(
    players_2425[["player_id", "web_name", "first_name", "second_name", "position", "team_code", "full_name"]],
    on="player_id",
    how="left"
)

# ---------- IDENTIFY NEW PLAYERS ----------
new_players_2526 = players_2526.loc[
    ~players_2526["full_name"].isin(players_2425["full_name"])
].copy()

agg_stats["is_new_player"] = False
new_players_2526["is_new_player"] = True


# ---------- ALIGN COLUMNS SAFELY ----------
missing_cols = [col for col in agg_stats.columns if col not in new_players_2526.columns]
for col in missing_cols:
    new_players_2526.loc[:, col] = pd.NA

# ---------- CONCAT ----------
combined = pd.concat([agg_stats, new_players_2526], ignore_index=True)

df = combined.merge(
    teams,
    left_on="team_code",
    right_on="code",
    how="left"
)

# Compute FPL score
df.fillna(0, inplace=True)
df[['fpl_points', 'defcon']] = pd.DataFrame(df.apply(calculate_fpl_score, axis=1).tolist(), index=df.index)

# Move important columns forward
preferred_order = [
    'web_name', 'position', 'short_name', 'fpl_points', 'defcon', 'minutes_played', 'goals', 'assists', 'xg', 'xa', 'shots_on_target',
    'tackles_won', 'interceptions', 'recoveries', 'clearances', 'blocks', 'saves',
    'position', 'team_code', 'elo', 'player_id'
]

# Add remaining columns that aren’t in preferred_order
final_cols = preferred_order + [col for col in df.columns if col not in preferred_order]

# Reorder the DataFrame
df = df[final_cols]

df.to_csv("lastseasonstats.csv", index=False, encoding="utf-8-sig")