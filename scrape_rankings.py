"""
Scrape HLTV weekly ranking history using Playwright.

HLTV publishes rankings weekly on Mondays at:
  https://www.hltv.org/ranking/teams/YYYY/month_name/DD

Scrapes from 2025-07-21 through 2026-02-09 (~32 weeks).
Saves to data/rankings_history.csv with resume support.

Usage:
  python scrape_rankings.py
"""
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_CSV = os.path.join(DATA_DIR, "rankings_history.csv")

# Reuse the same JS from scrape_data.py
EXTRACT_RANKINGS_JS = """
() => {
    const teams = [];
    document.querySelectorAll('.ranked-team').forEach(el => {
        const name = el.querySelector('.name');
        const pos = el.querySelector('.position');
        const pts = el.querySelector('.points');
        if (name && pos) {
            const rank = parseInt(pos.textContent.trim().replace('#', ''));
            let points = 0;
            if (pts) {
                const m = pts.textContent.match(/\\d+/);
                if (m) points = parseInt(m[0]);
            }
            teams.push({
                team: name.textContent.trim(),
                rank: rank,
                points: points
            });
        }
    });
    return JSON.stringify(teams);
}
"""


def generate_monday_dates(start_date_str="2025-07-21", end_date_str="2026-02-09"):
    """Generate Monday dates for HLTV ranking URLs."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=7)
    return dates


def get_already_scraped():
    """Return set of date strings already in the CSV."""
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(df["date"].unique())


def scrape_rankings():
    mondays = generate_monday_dates()
    already_done = get_already_scraped()
    to_scrape = [d for d in mondays if d.strftime("%Y-%m-%d") not in already_done]

    if not to_scrape:
        print("All ranking dates already scraped. Nothing to do.")
        return

    print(f"Total Monday dates: {len(mondays)}")
    print(f"Already scraped: {len(already_done)}")
    print(f"Remaining: {len(to_scrape)}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
        })

        for i, date in enumerate(to_scrape):
            month_name = date.strftime("%B").lower()
            url = f"https://www.hltv.org/ranking/teams/{date.year}/{month_name}/{date.day}"
            date_str = date.strftime("%Y-%m-%d")
            print(f"  [{i+1}/{len(to_scrape)}] {date_str} ...", end=" ", flush=True)

            retries = 0
            while retries < 3:
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_selector(".ranked-team", timeout=15000)

                    data = json.loads(page.evaluate(EXTRACT_RANKINGS_JS))
                    if not data:
                        print("no teams found, skipping")
                        break

                    rows = []
                    for entry in data:
                        rows.append({
                            "date": date_str,
                            "team": entry["team"],
                            "rank": entry["rank"],
                            "points": entry["points"],
                        })

                    df_new = pd.DataFrame(rows)
                    if os.path.exists(OUT_CSV):
                        df_new.to_csv(OUT_CSV, mode="a", header=False, index=False)
                    else:
                        df_new.to_csv(OUT_CSV, index=False)

                    print(f"{len(data)} teams")
                    page.wait_for_timeout(3000)
                    break

                except Exception as e:
                    retries += 1
                    print(f"retry {retries}/3: {e}")
                    page.wait_for_timeout(10000)
                    if retries >= 3:
                        print(f"    Skipping {date_str} after 3 retries")

        browser.close()

    # Summary
    if os.path.exists(OUT_CSV):
        df = pd.read_csv(OUT_CSV)
        print(f"\nTotal rows: {len(df)}")
        print(f"Unique dates: {df['date'].nunique()}")
        print(f"Unique teams: {df['team'].nunique()}")


if __name__ == "__main__":
    scrape_rankings()
