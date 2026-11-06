"""Cells 14 + 15 + 17: scaling/xFPL/VORP helpers, minutes factors, blended signals, board assembly."""
import numpy as np
import pandas as pd

POSITION_LIMITS = {"Goalkeeper": 16, "Defender": 40, "Midfielder": 40, "Forward": 24}
MIN_MINUTES = 30   # expand as season progresses


def scale_global(df, col):
    s = df[col]
    median = s.median()
    iqr = (s.quantile(0.75) - s.quantile(0.25)) + 1e-6
    return ((s - median) / iqr)


def expected_fpl_p90(row):
    pos = row["position"]
    xg = row.get("expected_goals_per_90", 0)
    xa = row.get("expected_assists_per_90", 0)
    xgc = row.get("expected_goals_conceded_per_90", 0)
    saves = row.get("saves_per_90", 0)
    defcon = row.get("defensive_contribution_per_90", 0)

    base = 0
    typical_minutes = min(row["minutes"] / max(row["starts"], 1), 90)

    # --- GOALS + ASSISTS (position-scaled) ---
    # if pos == "Goalkeeper":
    #     base = xg * 10 + xa * 3
    if pos == "Defender":
        base = xg * 6 + xa * 3
    elif pos == "Midfielder":
        base = xg * 5 + xa * 3
    elif pos == "Forward":
        base = xg * 4 + xa * 3

    # --- CLEAN SHEETS ---
    if pos == "Goalkeeper":
        clean_sheet_pts = row["clean_sheets_per_90"] * 5
    elif pos == "Defender":
        clean_sheet_pts = row["clean_sheets_per_90"] * 4
    elif pos == "Midfielder":
        clean_sheet_pts = row["clean_sheets_per_90"] * 1
    else:
        clean_sheet_pts = 0

    # --- GOALS CONCEDED  ---
    if pos in ["Goalkeeper", "Defender"]:
        conceded_penalty = -0.5 * xgc
    else:
        conceded_penalty = 0

    # --- SAVE POINTS (GK only: 1 per 3 saves) ---
    save_pts = 0
    if pos == "Goalkeeper":
        save_pts = saves / 3

    # --- DEFENSIVE BONUS ---
    if pos == "Defender":
        threshold = 10
    elif pos in ["Midfielder", "Forward"]:
        threshold = 12
    else:
        threshold = None
    if threshold is None:
        defcon_bonus = 0
    else:
        # convert per-90 defcon into expected per-match value
        defcon_match = defcon * typical_minutes / 90

        # probability of hitting match-level threshold
        prob_hit = min(defcon_match / threshold, 1)

        # 2-point cap
        defcon_bonus = 2 * prob_hit

    # ---- APPEARANCE POINTS ----
    mins_est = typical_minutes

    if mins_est >= 60:
        appearance_pts = 2
    elif mins_est > 0:
        appearance_pts = 1
    else:
        appearance_pts = 0

    return (
        base +
        clean_sheet_pts +
        save_pts +
        defcon_bonus +
        conceded_penalty +
        appearance_pts
    )


def compute_vorp_dynamic(df, metric=""):
    output = []

    for pos, limit in POSITION_LIMITS.items():
        subset = df[df["position"] == pos].copy()
        subset = subset.sort_values(metric, ascending=False)

        if len(subset) < limit:
            baseline = subset[metric].min()
        else:
            baseline = subset.iloc[limit - 1][metric]

        print(pos, len(subset), baseline)

        subset["baseline"] = baseline
        subset["VORP"] = subset[metric] - baseline
        subset["above_replacement"] = subset["VORP"] > 0

        output.append(subset)

    return pd.concat(output, ignore_index=True)


def attach_minutes_factors(merged, LAST_GW):
    # cell 14
    print(merged.columns.tolist())
    full_minutes = max(360, 0.6 * LAST_GW * 90)
    merged["minutes_factor"] = (merged["minutes"] / full_minutes).clip(upper=1)

    merged["recency_mins_factor"] = np.sqrt((merged["form_minutes"] / 90).clip(upper=1))

    print(merged["minutes_factor"].describe())
    return merged


def attach_expected_return_signals(merged, LAST_GW, min_minutes=MIN_MINUTES):
    # cell 15
    merged = merged[merged["minutes"] >= min_minutes].copy()

    merged["current_x_fpl_p90"] = merged.apply(expected_fpl_p90, axis=1).fillna(0)
    x_fpl_median = merged["current_x_fpl_p90"].median()
    merged["current_x_fpl_p90"] = x_fpl_median + (merged["current_x_fpl_p90"] - x_fpl_median) * np.sqrt(merged["minutes_factor"])

    merged["bps_GW"] = merged["bps"] / LAST_GW

    for col in ["current_x_fpl_p90", "points_per_game", "bps_GW", "last_points_per_game", "team_score", "fixture_score", "form_score"]:
        merged[col + "_scaled"] = scale_global(merged, col)

    print(merged["fixture_score"].describe())

    status_multiplier = {"a": 1.0, "d": 0.9, "i": 0.3, "s": 0.3, "u": 0.0}
    merged["availability_factor"] = merged["status"].map(status_multiplier).fillna(1.0)

    merged["set_piece_score"] = 0.0
    merged.loc[merged["penalties_order"] == 1, "set_piece_score"] += 0.15
    merged.loc[merged["direct_freekicks_order"] == 1, "set_piece_score"] += 0.08
    merged.loc[merged["corners_and_indirect_freekicks_order"] == 1, "set_piece_score"] += 0.05

    #TODO bake the set piece score into secondary position_score function

    w_current = 0.9
    w_last = 0.1

    merged["overall_expected_return"] = (
        w_current * (
            0.43 * merged["current_x_fpl_p90_scaled"] +
            0.37 * merged["points_per_game_scaled"] +
            0.11 * merged["bps_GW_scaled"] +
            0.09 * merged["team_score_scaled"]
        ).fillna(0) +
        w_last * merged["last_points_per_game_scaled"].fillna(0)
    )

    merged["weekly_expected_return"] = (
        w_current * (
            0.30 * merged["current_x_fpl_p90_scaled"] +
            0.25 * merged["points_per_game_scaled"] +
            0.05 * merged["bps_GW_scaled"] +
            0.20 * merged["fixture_score_scaled"].fillna(0)+
            0.15 * merged["form_score_scaled"].fillna(0)+
            0.05 * merged["team_score_scaled"]
        ).fillna(0) +
        w_last * merged["last_points_per_game_scaled"].fillna(0)
    )

    # --- apply stability + availability modifiers ---
    merged["weekly_expected_return"] = np.where(
        merged["weekly_expected_return"] > 0,
        merged["weekly_expected_return"] * merged["recency_mins_factor"],
        merged["weekly_expected_return"] / merged["recency_mins_factor"].clip(lower=0.25))

    merged["overall_expected_return"] = np.where(
        merged["overall_expected_return"] > 0,
        merged["overall_expected_return"] * merged["minutes_factor"],
        merged["overall_expected_return"] / merged["minutes_factor"].clip(lower=0.25))

    merged["weekly_expected_return"] = np.where(
        merged["weekly_expected_return"] > 0,
        merged["weekly_expected_return"] * merged["availability_factor"],   # positive: multiply (penalty works)
        merged["weekly_expected_return"]                                     # negative: leave as-is (don't flip)
    )

    ### BUY LOW / SELL HIGH SIGNALS

    A = merged["current_x_fpl_p90_scaled"] - merged["points_per_game_scaled"]       # underlying vs actual this season
    B = (merged["last_points_per_game_scaled"] - merged["points_per_game_scaled"])  # last year's level vs actual
    B = B.where(merged["last_minutes"] >= 900)                                      # ignore thin last-year samples

    last_weight = 0.2
    merged["buy_low"]   = np.where(B.notna(), (1 - last_weight) * A + last_weight * B, A)
    merged["sell_high"] = -merged["buy_low"]

    print(merged.sort_values("current_x_fpl_p90_scaled", ascending=False)
          [["web_name", "position", "minutes", "current_x_fpl_p90", "current_x_fpl_p90_scaled"]]
          .head(15).to_string(index=False))
    return merged


def assemble_board(merged):
    # cell 17
    weekly = compute_vorp_dynamic(merged, metric="weekly_expected_return")[["player_id","VORP"]].rename(columns={"VORP":"weekly_VORP"})
    overall = compute_vorp_dynamic(merged, metric="overall_expected_return")[["player_id","VORP"]].rename(columns={"VORP":"overall_VORP"})

    board = merged.merge(weekly, on="player_id").merge(overall, on="player_id")
    board = board.sort_values("overall_VORP", ascending=False)
    return board
