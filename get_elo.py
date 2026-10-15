from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import requests
import os

HISTORY = Path("data/elo_history.csv")

NAME_MAP = {"Man United": "Man Utd", "Forest": "Nott'm Forest", "Tottenham": "Spurs",
            "Coventry": "Coventry City", "Ipswich": "Ipswich Town", "Hull": "Hull City"}

FPL_DATA_FOLDER = os.path.expanduser('~/MyProjects/FPL/FPL-Elo-Insights/data/2026-2027')
teams = pd.read_csv(f"{FPL_DATA_FOLDER}/teams.csv",encoding="utf-8")

def scrape_elo():
    soup = BeautifulSoup(requests.get("https://clubelo.com/ENG", headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text, "html.parser")
    tab = next(t for t in soup.select("table.ast") if t.select_one('img[alt="ENG"]'))
    rows = [(s.text.strip(), int(s.find_next("td", class_="r").text)) for s in tab.select("span.Ast")[:20]]
    df = pd.DataFrame(rows, columns=["name", "elo"]).replace({"name": NAME_MAP})
    out = teams[["code", "name"]].merge(df, on="name", how="left")
    assert out["elo"].notna().all(), out.loc[out["elo"].isna(), "name"].tolist()
    return out[["code", "elo"]]

elo_df = scrape_elo()

def update_elo_history(gw, max_age_days=6):
    hist = pd.read_csv(HISTORY, parse_dates=["date"]) if HISTORY.exists() else pd.DataFrame(columns=["date", "gw", "code", "elo"])
    if len(hist) and (pd.Timestamp.today() - hist["date"].max()).days < max_age_days:
        return hist
    snap = scrape_elo().assign(date=pd.Timestamp.today().normalize(), gw=gw)
    HISTORY.parent.mkdir(exist_ok=True)
    pd.concat([hist, snap[["date", "gw", "code", "elo"]]]).to_csv(HISTORY, index=False)
    return pd.read_csv(HISTORY, parse_dates=["date"])