"""Cell 3: PLAYER STATS BASE — playerstats + players + teams merge, transferred-out filter."""

TRANSFERRED_OUT = {"Woltemade", "Beto", "Sánchez", "Richarlison"}

BASE_COLS = [
    # --- identity ---
    "web_name",
    "position",
    "team_name",

    # --- core output / value ---
    "total_points",
    "points_per_game",
    "minutes",
    "starts",
    "bonus",
    "bps",
    "form",

    # --- expected (xG) ---
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",

    # --- defcon ---
    "defensive_contribution",
    "defensive_contribution_per_90",
    "tackles",
    "clearances_blocks_interceptions",
    "recoveries",

    # --- set pieces ---
    "penalties_order",
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",

    # --- gk-specific ---
    "saves_per_90",
    "clean_sheets_per_90",
    "goals_conceded_per_90",

    # meta
    "player_id",
    "team_code",

    # --- availability ---
    "status",
    "chance_of_playing_next_round"
]


def build_player_base(playerstats, players, teams, transferred_out=TRANSFERRED_OUT):
    # 1. playerstats + players (get names, position, team_code)
    merged = playerstats.merge(
        players,
        left_on="id",
        right_on="player_id",
        how="left",
        suffixes=("", "_p")      # keep playerstats names clean, suffix player dups
    )

    # 2. + teams (get team short_name)
    merged = merged.merge(
        teams[["code", "short_name"]],
        left_on="team_code",
        right_on="code",
        how="left"
    ).rename(columns={"short_name": "team_name"}).drop(columns=["code"])

    merged = merged[~merged["web_name"].isin(transferred_out)].copy()

    base = merged[BASE_COLS].copy()

    base = base.sort_values(by='total_points', ascending=False)
    return merged, base
