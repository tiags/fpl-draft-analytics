"""Cell 18: draft-league ownership flags per league."""
import pandas as pd
import requests

import config

BASE_URL = "https://draft.premierleague.com/api"

names = ["Tzolakis","M.Sangaré","Yalcouyé","Araujo","Tzolis"]


def build_all_boards(leagues=None):
    if leagues is None:
        leagues = config.LEAGUES

    all_boards = []

    for key, league in leagues.items():
        LEAGUE_ID = league["league_id"]
        MY_ENTRY_ID = league["entry_id"]
        league_name = league["name"]          # "Gegen" / "Dalot"

        url = f"{BASE_URL}/league/{LEAGUE_ID}/element-status"
        raw = requests.get(url).json()
        league_players = pd.json_normalize(raw["element_status"])

        league_players["owner"] = pd.to_numeric(league_players["owner"], errors="coerce").astype("Int64")
        league_players["is_available"] = league_players["status"].astype(str) == "a"
        league_players["is_mine"] = league_players["owner"] == MY_ENTRY_ID
        league_players["is_taken"] = (~league_players["is_available"]) & (~league_players["is_mine"])
        league_players = league_players.rename(columns={"element": "player_id"})

        mine_set  = league.get("known_mine", set())
        taken_set = league.get("known_taken", set())
        avail_set = league.get("known_available", set())

        # mine / taken first
        league_players.loc[league_players["player_id"].isin(mine_set),  "is_mine"] = True
        league_players.loc[league_players["player_id"].isin(taken_set), "is_taken"] = True
        league_players.loc[league_players["player_id"].isin(mine_set | taken_set), "is_available"] = False

        # known_available LAST — fully reset to available, override everything
        league_players.loc[league_players["player_id"].isin(avail_set), ["is_mine","is_taken"]] = False
        league_players.loc[league_players["player_id"].isin(avail_set), "is_available"] = True

        league_players["is_mine"] = league_players["is_mine"].fillna(False).astype(bool)
        league_players["is_taken"] = league_players["is_taken"].fillna(False).astype(bool)

        check = league_players[league_players["player_id"].isin({557,565,572,579,605,608})]
        print(config.LEAGUES["gegenpressure"].keys())
        print(league_name)
        print(check[["player_id","is_mine","is_available","is_taken"]].to_string(index=False))

        flags = league_players[["player_id", "is_mine", "is_available", "is_taken"]].rename(columns={
            "is_mine":      f"mine_{league_name}",
            "is_available": f"available_{league_name}",
            "is_taken":     f"taken_{league_name}",
        })
        all_boards.append(flags)

    return all_boards
