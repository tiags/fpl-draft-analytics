"""Cell 1: git pull, GW state, raw data loads, elo history."""
import os
import subprocess
from types import SimpleNamespace

import pandas as pd
import requests

from get_elo import update_elo_history
from config import FPL_DATA_FOLDER, PREV_DATA_FOLDER, FPL_ELO_INSIGHTS_ROOT as GIT_REPO

LAST_YEAR_STATS_PATH = "seasons/2025-26/End2526_Full.xlsx"


def get_gw_state():
    return requests.get("https://draft.premierleague.com/api/game").json()


def load_pipeline_context():
    subprocess.run(["git", "-C", GIT_REPO, "pull"], check=True)

    state = get_gw_state()
    
    GW = state["next_event"]
    LAST_GW = GW - 1
    print(f"Last GW played: {LAST_GW}, Next GW: {GW}")
    
    if not state["current_event_finished"]:
            print(f"Warning: GW{LAST_GW} isn't finished yet, so its stats may be partial or empty. Board will be skewed for GW{GW}.")
            if input("Continue anyway? [y/N] ").strip().lower() != "y":
                raise SystemExit("Stopped: run again once the gameweek has finished.")
    
    TOTAL_MINUTES = LAST_GW * 90

    last_year_stats = pd.read_excel(LAST_YEAR_STATS_PATH)

    fixtures_dir = os.path.join(FPL_DATA_FOLDER, "By Gameweek")
    prev_dir     = os.path.join(PREV_DATA_FOLDER, "By Gameweek")

    players = pd.read_csv(f"{FPL_DATA_FOLDER}/players.csv",encoding="utf-8")
    teams = pd.read_csv(f"{FPL_DATA_FOLDER}/teams.csv",encoding="utf-8")
    prev_teams = pd.read_csv(f"{PREV_DATA_FOLDER}/teams.csv", encoding="utf-8")

    playerstats = pd.read_csv(f"{FPL_DATA_FOLDER}/By Gameweek/GW{LAST_GW}/playerstats.csv",encoding="utf-8")

    elo_hist = update_elo_history(GW)
    elo = elo_hist[elo_hist["date"] == elo_hist["date"].max()].set_index("code")["elo"]

    return SimpleNamespace(
        state=state, GW=GW, LAST_GW=LAST_GW,
        last_year_stats=last_year_stats,
        FPL_DATA_FOLDER=FPL_DATA_FOLDER, PREV_DATA_FOLDER=PREV_DATA_FOLDER,
        fixtures_dir=fixtures_dir, prev_dir=prev_dir,
        players=players, teams=teams, prev_teams=prev_teams,
        playerstats=playerstats, elo_hist=elo_hist, elo=elo,
    )
