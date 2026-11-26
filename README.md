This is a weekly analytics pipeline for Fantasy Premier League draft leagues. It pulls three sources together (player and match data from a public FPL repository, weekly team Elo ratings, and live league ownership from the FPL Draft API), then builds individual and team-level stats, forecasts each fixture with a goal model, and scores every player. Players are given two VORP (Value Over Replacement) rankings, one overall and one weekly, enabling better decision-making week to week on who to start, pick up off waivers, or trade for (wVORP), while maintaining an overall view free of recency bias (oVORP). There are 38 gameweeks in an English Premier League season, meaning 38 opportunities for analysis-backed decision-making.

Note that this is for draft leagues, and therefore assumes unique asset ownership, unlimited roster movement, and no player pricing or budget constraints that apply in the more common, classic FPL format. 

Live dashboard: https://tiags.github.io/fpl-draft-analytics/

## Data sources

- [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) by olbauday: player and match data, updated daily
- [ClubElo](http://clubelo.com): team Elo ratings
- FPL Draft API: league ownership and gameweek state

## How it works

Each week the pipeline:

1. **Pulls the latest data:** player and match stats (FPL-Core-Insights), team Elo ratings (ClubElo) and league ownership (FPL Draft API).
2. **Forecasts each fixture:** a Poisson regression predicts the expected goals (xG) each team will create, using home advantage and the Elo gap between the two teams.
3. **Scores every player:** expected FPL points from underlying stats, blended with actual points, form, fixture difficulty and team strength.
4. **Ranks by value over replacement (VORP):** how much better each player is than the benchmark player at the same position.
5. **Publishes the board:** a static dashboard on GitHub Pages, with filters for my personal squads, waiver wire standouts and buy-low / sell-high targets.

```
FPL-Core-Insights─┐
ClubElo (Elo)    ─┼─► pipeline/ ─► Poisson goal model ─► player scoring ─► VORP ─► docs/data/*.csv ─► dashboard
FPL Draft API    ─┘
```

Using a rudimentary version of this tool, I finished first in my 2025–26 league, winning by 11 points.

## Form score

Measures how a player has performed over the last three gameweeks, with the most recent counting most (50%, 30%, 20%). Everything is converted to a per-90-minutes rate, so a player who came off the bench isn't necessarily punished for playing fewer minutes.

What counts depends on position:

- Forwards: points and goal involvements (xGI), plus bonus
- Midfielders: the same, plus defensive contributions
- Defenders: points, defensive contributions and a little attacking threat, minus goals conceded
- Goalkeepers: points and bonus, minus goals conceded

## Team score

Rates each team's attack and defence from their recent matches, with greater weight on latest matches.

- Attack: mostly open-play xG, then set-play xG, plus a small bonus for finishing (shots on target vs xG)
- Defence: the inverse, using xG conceded, plus a small bonus for shot-stopping

A score is then mapped to each position (scaled from 0–100). Defenders and keepers are given a score matching their team's relative defensive strength, forwards are given their team's relative attacking strength, and midfielders are given a mix of the two, with more weight towards attack.

## The goal model

The model uses a single Poisson regression to predict each team's non-penalty xG in a match, which I've found is a cleaner outcome to train on than actual goals, which are more randomly influenced and therefore less stable. Currently, I use the Elo gap between the teams and home advantage as features. 

I've tested the model by training on last season and blindly predicting this season, and compared it against guessing the average each time. It beat that guess, clearly for home teams and slightly for away teams (the away side originally lost to the guess, which is what led me to add `is_home`). 

The actual production model trains on both seasons (all played EPL matches) and forecasts only the gameweek ahead.

| Feature | Meaning |
|---|---|
| `elo_context` | (team Elo − opponent Elo) / 250 |
| `is_home` | 1 for the home side, 0 for the away side |

The validation is in `exploration.ipynb` and can be rerun at any time.

I've tested out various other features like a computed open_play_attack_strength vs defence, and set_piece_attack vs set_piece_defence, other Elo x interactions (as Elo always showed to be a worthwhile feature) as well as rest_days/rest_diff, but they all seemed to only introduce more noise/not worthwhile.

## Fixture scoring

The goal model directly funnels into each player's perceived fixture favourability for that gameweek, scaled from 0–100. Forwards are boosted when their team is predicted to create more xG than average, defenders and keepers when the opponent is predicted to create little (a higher chance of a clean sheet), and midfielders a mix of both, weighted towards attack.

## Player scoring

In addition to actual data like points per game (PPG), bonus point system score (BPS), and defensive contributions (DEFCON), the pipeline attempts to compute an expected FPL points per 90 metric from underlying stats (xG, xA, clean sheets, goals conceded, saves, defensive contributions), using FPL's position-specific scoring rules.

- Small-sample guard: low-minute players' expected points are pulled toward the median, scaled by √(share of available minutes played)
- Robust scaling: each input is scaled by median and interquartile range before blending

## Last season

I've included last season's PPG (in addition to training on last season's matches in the goal model above) to guard against noise from the current, incomplete season. Currently the final scores take in 10% of last season's PPG.

## Two final rankings

  - wVORP (weekly): underlying stats, actual points, this week's fixture, recent form and team strength, adjusted for recent minutes and injury/suspension risk
  - oVORP (overall): the same building blocks minus fixture and form, adjusted for minutes played this season

Replacement level assumes an 8-team league: the 16th-best goalkeeper, 40th defender, 40th midfielder and 24th forward.

Weights are currently hand-set; see `pipeline/scoring.py`.

## Buy Low/Sell High signals

Used to identify potential value cases.
```
buy_low  = 80% × (underlying stats − actual points)        → playing better than their points show
         + 20% × (last season's PPG − this season's PPG)   → below their usual level
           (last-season part only for players with 900+ minutes last season)```

sell_high = the exact opposite of buy_low
```

## Project structure
```
main.py                  runs the full pipeline
config.py                league IDs, data paths, season
pipeline/
  context.py             gameweek state, raw data loads
  player_base.py         player / team merge
  form.py                recent-form signal
  team_strength.py       Elo history, team attack/defence strength
  predictions.py         train/test split, Poisson goal model
  team_scores.py         fixture and team scores onto the player board
  last_season.py         last season's stats
  scoring.py             expected points, VORP, buy-low / sell-high
  draft_ownership.py     league ownership flags
  board_output.py        export to CSV
fixture_loader.py        fixture loading, rest days
get_elo.py               appends weekly Elo snapshots
exploration.ipynb        experiments and model validation
docs/                    GitHub Pages dashboard (index.html + data/)
```

## Limitations
- Ballooning per-90 rates for low-minute players can still be outliers, and though I've taken steps to compress low-minute player signals (while still being able to react to breakouts), there's more that can be done.
- One replacement level (8 teams) is used for both leagues, though one has 15 teams. Should be modular VORP calculation, with configurable number of teams.
- Ranking weights are hand-set, not fitted.
- Elo ratings are entered weekly by hand (original database Elo link has broken, and ClubElo has no reliable public API).
- Due to changing player IDs, last season's stats are joined by player name, which misses some renamed players. To avoid doubles, in some cases values are hardcoded
- Some new players don't respond to the draft ownership API's filter, and ownership must be hardcoded

## Roadmap
- Replace historical clean-sheet rate with goal model's clean-sheet probability during xFPL calculation.
- Replace hand-set ranking weights with weights fitted by regression on past gameweeks
- Per-league replacement level from each league's size, making it modular
- Automated weekday runs, skipped until the gameweek has finished