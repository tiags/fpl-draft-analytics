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


def filter_prem(fx):
    return fx[fx["tournament"].str.lower().str.contains("prem")].copy()


def add_rest_days(fixtures, all_tournament_fixtures, max_rest_days=10):
    rest_table = pd.concat([
        all_tournament_fixtures[["home_team", "kickoff_time"]].set_axis(["team", "ko"], axis=1),
        all_tournament_fixtures[["away_team", "kickoff_time"]].set_axis(["team", "ko"], axis=1),
    ])
    rest_table["ko"] = pd.to_datetime(rest_table["ko"], utc=True)
    rest_table = rest_table.dropna().drop_duplicates(["team", "ko"]).sort_values(["team", "ko"])
    rest_table["days_rest"] = rest_table.groupby("team")["ko"].diff().dt.total_seconds() / 86400

    fixtures = fixtures.drop(columns=["home_days_rest", "away_days_rest", "rest_diff", "ko", "short_rest_advantage"], errors="ignore")
    fixtures["ko"] = pd.to_datetime(fixtures["kickoff_time"], utc=True)
    for side in ("home", "away"):
        rest_for_side = rest_table.rename(columns={"team": f"{side}_team", "days_rest": f"{side}_days_rest"})
        fixtures = fixtures.merge(rest_for_side, on=[f"{side}_team", "ko"], how="left")
        fixtures[f"{side}_days_rest"] = fixtures[f"{side}_days_rest"].fillna(max_rest_days).clip(upper=max_rest_days)
    fixtures["rest_diff"] = (fixtures["home_days_rest"].fillna(max_rest_days).clip(upper=max_rest_days)
                             - fixtures["away_days_rest"].fillna(max_rest_days).clip(upper=max_rest_days))
    fixtures["short_rest_advantage"] = ((fixtures["away_days_rest"] < 3.5).astype(int)
                                        - (fixtures["home_days_rest"] < 3.5).astype(int))
    return fixtures