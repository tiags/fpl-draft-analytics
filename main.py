"""Headless weekly pipeline (lift-and-shift of reformatted_pipe.ipynb). Run from repo root."""
import config
from fixture_loader import load_fixtures, both_prem, filter_prem, add_rest_days
from pipeline.context import load_pipeline_context
from pipeline.player_base import build_player_base, TRANSFERRED_OUT
from pipeline.form import attach_form
from pipeline.team_strength import (
    load_elo_all, build_elo_long, attach_match_elo,
    build_team_table, add_diffs, print_team_strength_diagnostics,
)
from pipeline.predictions import build_train_test, predict_fixtures, fit_pooled_goal_model
from pipeline.team_scores import build_team_fixture_scores, attach_team_and_fixture_scores
from pipeline.last_season import attach_last_season_stats
from pipeline.scoring import attach_minutes_factors, attach_expected_return_signals, assemble_board
from pipeline.draft_ownership import build_all_boards
from pipeline.board_output import enrich_and_export, print_player_debug


def main():
    ctx = load_pipeline_context()                                    
    GW, LAST_GW = ctx.GW, ctx.LAST_GW

    merged, base = build_player_base(ctx.playerstats, ctx.players, ctx.teams, TRANSFERRED_OUT)  
    merged = attach_form(merged, GW, ctx.FPL_DATA_FOLDER)            

    this_seasons_fixtures = load_fixtures(ctx.fixtures_dir)            
    prev_seasons_fixtures = load_fixtures(ctx.prev_dir, played_only=True)
    this_season_prem_matches = filter_prem(this_seasons_fixtures)
    prev_fx = filter_prem(both_prem(prev_seasons_fixtures, ctx.prev_teams))
    print(f"GW{GW} fixtures: {(this_season_prem_matches['gameweek'] == GW).sum()}")

    this_season_prem_matches = add_rest_days(this_season_prem_matches, this_seasons_fixtures)  
    prev_fx = add_rest_days(prev_fx, prev_seasons_fixtures)
    print(f"GW{GW} fixtures: {(this_season_prem_matches['gameweek'] == GW).sum()}")

    elo_all = load_elo_all()                                         
    elo_long = build_elo_long(elo_all)
    this_season_prem_matches = attach_match_elo(this_season_prem_matches, elo_long)
    played, team_fixtures = build_team_table(this_season_prem_matches, GW)
    prev_played, prev_team_fixtures = build_team_table(prev_fx, 38)
    played = add_diffs(played, team_fixtures)
    prev_played = add_diffs(prev_played, prev_team_fixtures)
    print_team_strength_diagnostics(team_fixtures, ctx.teams, this_season_prem_matches)

    train, test, FEATURES = build_train_test(played, prev_played, GW)  
    
    pooled_model = fit_pooled_goal_model(train, FEATURES)
    test, predictions = predict_fixtures(test, train, pooled_model, FEATURES, ctx.teams, output_path="docs/data/predicted_fixtures_GW.csv")
    
    team_fx = build_team_fixture_scores(test)                          
    print(merged["player_id"].duplicated().sum())
    merged = attach_team_and_fixture_scores(merged, team_fixtures, team_fx)  
    merged = attach_last_season_stats(merged, ctx.last_year_stats)     
    merged = attach_minutes_factors(merged, LAST_GW)                   
    merged = attach_expected_return_signals(merged, LAST_GW)          
    board = assemble_board(merged)                                     
    all_boards = build_all_boards(config.LEAGUES)                      
    enriched = enrich_and_export(board, all_boards, LAST_GW,           
                                 output_path="docs/data/weekly_board_enriched.csv")
    print_player_debug(merged)
    return enriched


if __name__ == "__main__":
    main()
