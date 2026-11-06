from pathlib import Path
from io import StringIO
import argparse
import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
from playwright.sync_api import sync_playwright

HISTORY = Path("data/elo_history.csv")
HISTORY_COLUMNS = ["date", "gw", "code", "elo"]
DRAFT_GAME_URL = "https://draft.premierleague.com/api/game"

NAME_MAP = {"Man United": "Man Utd", "Forest": "Nott'm Forest", "Tottenham": "Spurs",
            "Coventry": "Coventry City", "Ipswich": "Ipswich Town", "Hull": "Hull City"}

FPL_DATA_FOLDER = os.path.expanduser('~/MyProjects/FPL/FPL-Elo-Insights/data/2026-2027')
teams = pd.read_csv(f"{FPL_DATA_FOLDER}/teams.csv",encoding="utf-8")

def scrape_elo():
    with sync_playwright() as p:
        browser = p.firefox()  # Or p.chromium()
        page = browser.new_page()

        # Bypass Cloudflare by rendering the page
        page.goto("https://clubelo.com/ENG", timeout=60000)

        # Wait for the table to load (adjust timeout if needed)
        page.wait_for_selector("table.sidebar-table tbody", timeout=10000)

        # Extract data from the table
        table = page.query_selector("table.sidebar-table tbody")
        rows = table.query_selector_all("tr")
        data = []
        for row in rows:
            cells = row.query_selector_all("td")
            rank = cells[0].inner_text().strip()
            team = cells[1].inner_text().strip()
            elo = cells[2].inner_text().strip()
            data.append({"rank": rank, "team": team, "elo": elo})

        browser.close()

    # Process data into DataFrame
    scraped_table = pd.DataFrame(data)
    scraped_table = scraped_table.replace({"team": NAME_MAP})
    scraped_table = scraped_table[["team", "elo"]].rename(columns={"team": "name", "elo": "elo"})
    elo_by_team = teams[["code", "name"]].merge(scraped_table, on="name", how="left")
    assert elo_by_team["elo"].notna().all(), elo_by_team.loc[elo_by_team["elo"].isna(), "name"].tolist()
    return elo_by_team[["code", "elo"]]


def load_elo_history():
    if HISTORY.exists():
        return pd.read_csv(HISTORY, parse_dates=["date"])
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_elo_snapshot(elo_snapshot):
    """Append rows to HISTORY. Never rewrites existing rows."""
    elo_snapshot = elo_snapshot[HISTORY_COLUMNS].assign(
        date=pd.to_datetime(elo_snapshot["date"]).dt.strftime("%Y-%m-%d"))
    HISTORY.parent.mkdir(exist_ok=True)
    if not HISTORY.exists() or HISTORY.stat().st_size == 0:
        elo_snapshot.to_csv(HISTORY, index=False)
        return
    existing_bytes = HISTORY.read_bytes()
    line_ending = "\r\n" if b"\r\n" in existing_bytes else "\n"
    if not existing_bytes.endswith(b"\n"):          # e.g. last row saved by a spreadsheet without a newline
        with open(HISTORY, "a", newline="") as history_file:
            history_file.write(line_ending)
    elo_snapshot.to_csv(HISTORY, mode="a", header=False, index=False, lineterminator=line_ending)


def update_elo_history(gw, max_age_days=6):
    elo_history = load_elo_history()
    if len(elo_history) and (pd.Timestamp.today() - elo_history["date"].max()).days < max_age_days:
        return elo_history
    try:
        elo_snapshot = scrape_elo().assign(date=pd.Timestamp.today().normalize(), gw=gw)
    except Exception as e:
        print(f"Elo scrape failed ({type(e).__name__}: {e}); using last saved snapshot")
        print("  -> add this week's Elo: clubelo.com/ENG bookmarklet, then `python get_elo.py`")
        if len(elo_history):
            return elo_history
        raise
    append_elo_snapshot(elo_snapshot)
    return load_elo_history()


# ---- Manual entry from the clubelo bookmarklet (scripts/clubelo_bookmarklet.js) ----

def match_team_codes(clubelo_table):
    """clubelo rows (name, elo) -> (code, elo) for this season's 20 FPL teams."""
    clubelo_table = clubelo_table.replace({"name": NAME_MAP}).drop_duplicates("name")
    elo_by_team = teams[["code", "name"]].merge(clubelo_table, on="name", how="left")
    unmatched_teams = elo_by_team.loc[elo_by_team["elo"].isna(), "name"].tolist()
    if unmatched_teams:
        raise ValueError(f"No clubelo row for {unmatched_teams}. Add clubelo's spelling to NAME_MAP. "
                         f"clubelo names pasted: {sorted(clubelo_table['name'])}")
    return elo_by_team[["code", "elo"]]


def parse_bookmarklet_rows(text):
    """Bookmarklet text ('date,name,elo' CSV) -> (code, elo, date) rows."""
    pasted_rows = pd.read_csv(StringIO(text.strip()))
    missing_columns = {"date", "name", "elo"} - set(pasted_rows.columns)
    if missing_columns:
        raise ValueError(f"Expected the bookmarklet's 'date,name,elo' header; missing {sorted(missing_columns)}")
    elo_snapshot = match_team_codes(pasted_rows[["name"]].assign(elo=pd.to_numeric(pasted_rows["elo"])))
    snapshot_date = pd.to_datetime(pasted_rows["date"].iloc[0]).normalize()
    return elo_snapshot.assign(date=snapshot_date)


def read_pasted_rows():
    print("Paste the bookmarklet rows, then press Enter on an empty line (or Ctrl-D):")
    pasted_lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            if pasted_lines:
                break
            continue
        pasted_lines.append(line)
    return "\n".join(pasted_lines)


def ask_yes_no(question):
    try:
        return input(question).strip().lower() == "y"
    except EOFError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Append a clubelo snapshot (copied by the bookmarklet) to data/elo_history.csv. Run from repo root.")
    parser.add_argument("--file", type=Path, help="file holding the bookmarklet's rows, instead of pasting")
    parser.add_argument("--gw", type=int, help="GW label = the GW this Elo goes INTO (default: draft API next_event)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    bookmarklet_text = args.file.read_text(encoding="utf-8-sig") if args.file else read_pasted_rows()
    elo_snapshot = parse_bookmarklet_rows(bookmarklet_text)
    gw = args.gw or requests.get(DRAFT_GAME_URL).json()["next_event"]
    snapshot_date = elo_snapshot["date"].iloc[0]
    elo_history = load_elo_history()

    preview = (elo_snapshot.merge(teams[["code", "short_name"]], on="code")
               .sort_values("elo", ascending=False))
    print(preview[["short_name", "code", "elo"]].to_string(index=False))
    if gw in set(elo_history["gw"]):
        print(f"Warning: GW{gw} is already in {HISTORY}; the newer rows win (load_elo_all keeps last).")
    if snapshot_date in set(elo_history["date"]):
        print(f"Warning: a snapshot dated {snapshot_date.date()} is already in {HISTORY}.")

    if not args.yes and not ask_yes_no(f"Append {len(elo_snapshot)} teams as GW{gw} ({snapshot_date.date()})? [y/N] "):
        print("Nothing written.")
        return

    append_elo_snapshot(elo_snapshot.assign(gw=gw))
    print(f"Appended GW{gw} Elo to {HISTORY}")


if __name__ == "__main__":
    main()
