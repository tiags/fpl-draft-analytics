# config.py

import os

FPL_ELO_INSIGHTS_ROOT = os.path.expanduser(
    os.environ.get("FPL_ELO_INSIGHTS_ROOT", "~/MyProjects/FPL/FPL-Elo-Insights"))

CURRENT_SEASON = "2026-2027"
PREVIOUS_SEASON = "2025-2026"

FPL_DATA_FOLDER = os.path.join(FPL_ELO_INSIGHTS_ROOT, "data", CURRENT_SEASON)
PREV_DATA_FOLDER = os.path.join(FPL_ELO_INSIGHTS_ROOT, "data", PREVIOUS_SEASON)

LEAGUES = {
    "gegenpressure": {
        "name": "Gegen",
        "entry_id": 67459,
        "league_id": 13175,
        "league_size": 15,
        "format": "classic",
        "known_taken": {557, 565, 572, 579, 82, 181, 659, 561, 408, 374, 634, 174, 650, 567, 93, 639},  
        "known_mine":  {452, 560},
        "known_available": {608}    
    },
    "highs_and_dalots": {
        "name": "Dalot",
        "entry_id": 29886,
        "league_id": 6544,
        "league_size": 8,
        "format": "h2h",
        "known_taken": {557, 565, 572, 605, 567}, 
        "known_mine": {650, },
        "known_available": {608, 640, 580}    
    },
}
