"""Cell 13: LAST SEASON'S STATS."""

# Columns to bring from last year
LAST_SEASON_COLS = [
    "web_name",
    "points_per_game",
    "x_fpl_p90",
    "total_points",
    "bonus_p90",
    "minutes",
    "defcon_p90"
]


def attach_last_season_stats(merged, last_year_stats, cols_needed=LAST_SEASON_COLS):
    # Prepare last year slice with prefix
    last_year_slim = last_year_stats[cols_needed].copy()
    last_year_slim = last_year_slim.rename(columns={col: f"last_{col}" for col in cols_needed if col != "web_name"})
    last_year_slim = last_year_slim[last_year_slim["last_total_points"] > 0]

    last_year_slim = (
        last_year_slim
        .sort_values("last_minutes", ascending=False)
        .drop_duplicates("web_name", keep="first")
    )

    # Merge (current is LEFT, last year secondary)
    merged = merged.merge(
        last_year_slim,
        left_on="web_name",
        right_on="web_name",
        how="left"
    )
    return merged
