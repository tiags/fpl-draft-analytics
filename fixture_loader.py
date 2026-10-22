import os
import pandas as pd

def load_fixtures(season_dir, played_only=False):
    frames = []
    for folder in sorted(os.listdir(season_dir)):
        p = os.path.join(season_dir, folder, "fixtures.csv")
        if folder.startswith("GW") and os.path.exists(p):
            frames.append(pd.read_csv(p, encoding="utf-8"))
    fx = pd.concat(frames, ignore_index=True)
    fx["tournament"] = fx["tournament"].fillna("")
    return fx[fx["home_score"].notna()].copy() if played_only else fx


def both_prem(fx, teams):
    codes = set(teams["code"])
    return fx[fx["home_team"].isin(codes) & fx["away_team"].isin(codes)].copy()