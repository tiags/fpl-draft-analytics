# config.py
LEAGUES = {
    "gegenpressure": {
        "name": "Gegen",
        "entry_id": 67459,
        "league_id": 13175,
        "league_size": 15,
        "format": "classic",
        "known_taken": {557, 565, 572, 579},   # Gegen: owned by others (Tzolis, M.Sangaré, Tzolakis, Araujo)
        "known_mine":  {605},
        "known_available": {608}    
    },
    "highs_and_dalots": {
        "name": "Dalot",
        "entry_id": 29886,
        "league_id": 6544,
        "league_size": 8,
        "format": "h2h",
        "known_taken": {557, 565, 572, 605},   # Dalot: owned by others (Tzolis, M.Sangaré, Tzolakis, Yalcoye)
        "known_mine": set(),
        "known_available": {608}    
    },
}