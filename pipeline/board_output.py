"""Cell 19: final enrich + CSV export (+ trailing debug prints)."""


def enrich_and_export(board, all_boards, LAST_GW, output_path="weekly_board_enriched.csv"):
    enriched = board.copy()
    for flags in all_boards:
        enriched = enriched.merge(flags, on="player_id", how="left")

    enriched["mins_per_gw"] = enriched["minutes"] / LAST_GW

    enriched = enriched.rename(columns={
        "current_x_fpl_p90": "x_fpl_p90",   # what the HTML expects
        "defensive_contribution_per_90": "defcon_p90",
        "expected_goal_involvements_per_90": "xgi_p90"
    })

    print(enriched.columns.tolist())

    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")
    return enriched


def print_player_debug(merged):
    print(merged.loc[merged["web_name"].isin(["Bruno G.", "Gomez", "Tzolis", "Osório", "Andrés", "Penders"]),
                     ["web_name", "team_name", "player_id", "id", "form_minutes", "minutes_factor", "recency_mins_factor"]])

    print(merged.loc[merged["id"].isin([ 650, 408, 374, 634, 174, 650, 567, 93, 452, 560]),
                     ["web_name", "team_name", "player_id", "id"]])
