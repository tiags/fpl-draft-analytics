"""Cell 4: FORM SCORE."""
import os

import pandas as pd

FORM_WEIGHTS = [0.5, 0.3, 0.2]          # last game first
STAT_MAP = {"expected_goal_involvements": "form_xgi", "bonus": "form_bps",
            "defensive_contribution": "form_defcon", "event_points": "form_points",
            "goals_conceded": "form_gc", "minutes": "form_minutes"}


def compute_form_score(row):
    if row["position"] == "Forward":
        return (
            0.45 * row["form_points_p90"] +
            0.45 * row["form_xgi_p90"] +
            0.10 * row["form_bps_p90"]
        )
    elif row["position"] == "Midfielder":
        return (
            0.40 * row["form_points_p90"] +
            0.25 * row["form_xgi_p90"] +
            0.10 * row["form_bps_p90"] +
            0.25 * row["form_defcon_p90"]
        )
    elif row["position"] == "Defender":
        return (
            0.40 * row["form_points_p90"] +
            0.10 * row["form_bps_p90"] +
            0.25 * row["form_defcon_p90"] +
            0.10 * row["form_xgi_p90"] -
            0.15 * row["form_gc_p90"]
        )
    else:  # Goalkeeper
        return (
            0.10 * row["form_bps_p90"] +
            0.40 * row["form_points_p90"] -
            0.50 * row["form_gc_p90"]
        )


def build_form(GW, data_folder, weights=FORM_WEIGHTS, min_weighted_minutes=30):
    files = [(GW - k, w) for k, w in enumerate(weights, start=1)
             if GW - k >= 1 and os.path.exists(f"{data_folder}/By Gameweek/GW{GW - k}/player_gameweek_stats.csv")]
    total = sum(w for _, w in files)
    frames = []
    for gw, w in files:
        d = pd.read_csv(f"{data_folder}/By Gameweek/GW{gw}/player_gameweek_stats.csv", encoding="utf-8")
        for src, dst in STAT_MAP.items():
            d[dst] = d[src] * (w / total)
        frames.append(d[["id"] + list(STAT_MAP.values())])
    fs = pd.concat(frames).groupby("id", as_index=False).sum()
    mins = fs["form_minutes"].clip(lower=min_weighted_minutes)
    for c in ["form_points", "form_gc", "form_xgi", "form_bps", "form_defcon"]:
        fs[f"{c}_p90"] = fs[c] / mins * 90
    return fs


def attach_form(merged, GW, data_folder):
    form_summary = build_form(GW, data_folder)

    merged = merged.merge(form_summary.rename(columns={"id": "player_id"}),
                          on="player_id", how="left")

    merged["form_score"] = merged.apply(compute_form_score, axis=1)
    return merged
