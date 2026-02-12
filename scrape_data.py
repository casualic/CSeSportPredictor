"""
Scrape HLTV match results using Playwright (handles Cloudflare).
Captures match detail URLs for subsequent player stats scraping.

Usage:
  python scrape_data.py [max_pages] [start_offset]

Examples:
  python scrape_data.py 60        # scrape 60 pages from offset 0
  python scrape_data.py 60 1800   # resume from offset 1800

Saves incrementally after every page to data/all_results_with_urls.csv.
On resume, appends to existing CSV.
"""
import json
import sys
import os
import pandas as pd
from playwright.sync_api import sync_playwright

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PARTIAL_CSV = os.path.join(DATA_DIR, "all_results_with_urls.csv")

EXTRACT_JS = """
() => {
    const results = [];
    document.querySelectorAll('.result-con').forEach(el => {
        const teams = el.querySelectorAll('.team');
        const score = el.querySelector('.result-score');
        const event = el.querySelector('.event-name');
        const link = el.querySelector('a.a-reset');

        if (teams.length >= 2 && score) {
            const scoreText = score.textContent.trim();
            const parts = scoreText.split('-').map(s => parseInt(s.trim()));
            if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
                const dateHolder = el.closest('.results-sublist');
                const dateEl = dateHolder ? dateHolder.querySelector('.standard-headline') : null;

                results.push({
                    team1: teams[0].textContent.trim(),
                    team2: teams[1].textContent.trim(),
                    score1: parts[0],
                    score2: parts[1],
                    event: event ? event.textContent.trim() : '',
                    date: dateEl ? dateEl.textContent.trim() : '',
                    match_url: link ? link.href : ''
                });
            }
        }
    });
    return JSON.stringify(results);
}
"""

EXTRACT_RANKINGS_JS = """
() => {
    const teams = {};
    document.querySelectorAll('.ranked-team').forEach(el => {
        const name = el.querySelector('.name');
        const pos = el.querySelector('.position');
        if (name && pos) {
            const rank = parseInt(pos.textContent.trim().replace('#', ''));
            teams[name.textContent.trim()] = rank;
        }
    });
    return JSON.stringify(teams);
}
"""


def append_results_to_csv(results, path):
    """Append results to CSV, creating it if needed."""
    df = pd.DataFrame(results)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def scrape_results_with_urls(max_pages=60, start_offset=0):
    total_new = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
        })

        # Step 1: Get top 100 teams (always refresh)
        print("Fetching top 100 teams from HLTV rankings...")
        page.goto("https://www.hltv.org/ranking/teams/", timeout=30000, wait_until="domcontentloaded")
        page.wait_for_selector(".ranked-team", timeout=15000)
        top_teams = json.loads(page.evaluate(EXTRACT_RANKINGS_JS))
        print(f"  Found {len(top_teams)} teams")

        with open(os.path.join(DATA_DIR, "top_teams.json"), "w") as f:
            json.dump(top_teams, f, indent=2)

        page.wait_for_timeout(2000)

        # Step 2: Scrape results pages with URLs
        start_page = start_offset // 100
        print(f"\nScraping pages {start_page+1} to {max_pages} (offset {start_offset} to {(max_pages-1)*100})...")

        for pg in range(start_page, max_pages):
            offset = pg * 100
            url = f"https://www.hltv.org/results?offset={offset}"
            print(f"  Page {pg+1}/{max_pages} (offset={offset})...", end=" ", flush=True)

            retries = 0
            while retries < 3:
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    # Wait longer for Cloudflare - user may need to solve challenge
                    page.wait_for_selector(".result-con", timeout=30000)

                    data = page.evaluate(EXTRACT_JS)
                    results = json.loads(data)

                    if not results:
                        print("No results, stopping.")
                        browser.close()
                        return total_new, top_teams

                    for r in results:
                        r["winner"] = r["team1"] if r["score1"] > r["score2"] else r["team2"]

                    # Save incrementally after each page
                    append_results_to_csv(results, PARTIAL_CSV)
                    total_new += len(results)
                    print(f"{len(results)} results (total new: {total_new})")

                    page.wait_for_timeout(3000)
                    break  # success, move to next page

                except Exception as e:
                    retries += 1
                    print(f"retry {retries}/3: {e}")
                    # Longer wait for Cloudflare resolution
                    page.wait_for_timeout(10000)
                    if retries >= 3:
                        print(f"    Skipping page {pg+1} after 3 retries")

        browser.close()

    return total_new, top_teams


def build_filtered_csv():
    """Build the top100 filtered CSV from the full results CSV."""
    with open(os.path.join(DATA_DIR, "top_teams.json")) as f:
        top_teams = json.load(f)

    df_all = pd.read_csv(PARTIAL_CSV)
    print(f"\nTotal results in CSV: {len(df_all)}")

    # Deduplicate (in case of overlapping resume runs)
    df_all = df_all.drop_duplicates(subset=["team1", "team2", "score1", "score2", "event", "date"])
    print(f"After dedup: {len(df_all)}")

    # Filter to top 100 matches
    top_names = set(top_teams.keys())
    mask = df_all["team1"].isin(top_names) & df_all["team2"].isin(top_names)
    df_top = df_all[mask].copy()
    df_top["rank1"] = df_top["team1"].map(top_teams)
    df_top["rank2"] = df_top["team2"].map(top_teams)

    df_top.to_csv(os.path.join(DATA_DIR, "top100_matches_with_urls.csv"), index=False)
    print(f"Top 100 matches: {len(df_top)}")
    n_urls = (df_top["match_url"].notna() & (df_top["match_url"] != "")).sum()
    print(f"Matches with URLs: {n_urls}")
    print(f"Unique teams: {len(set(df_top['team1']) | set(df_top['team2']))}")


def main():
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    start_offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if start_offset == 0 and os.path.exists(PARTIAL_CSV):
        # Starting fresh - remove old partial
        os.remove(PARTIAL_CSV)
        print("Removed old partial CSV, starting fresh.")

    total_new, top_teams = scrape_results_with_urls(max_pages, start_offset)
    print(f"\nScraped {total_new} new results this run.")

    # Build filtered CSV from all accumulated data
    build_filtered_csv()


if __name__ == "__main__":
    main()
