"""
Scrape HLTV match results using Playwright (handles Cloudflare).
Run from command line: python scrape_playwright.py [max_pages]
This script outputs JSON lines to stdout for each page of results.
"""
import asyncio
import json
import sys
import os

# We'll use playwright's sync API
from playwright.sync_api import sync_playwright


EXTRACT_JS = """
() => {
    const results = [];
    document.querySelectorAll('.result-con').forEach(el => {
        const teams = el.querySelectorAll('.team');
        const score = el.querySelector('.result-score');
        const event = el.querySelector('.event-name');

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
                    date: dateEl ? dateEl.textContent.trim() : ''
                });
            }
        }
    });
    return JSON.stringify(results);
}
"""


def scrape_results(max_pages=80):
    all_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9",
        })

        for pg in range(max_pages):
            offset = pg * 100
            url = f"https://www.hltv.org/results?offset={offset}"
            print(f"Page {pg+1}/{max_pages} (offset={offset})...", file=sys.stderr)

            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_selector(".result-con", timeout=10000)

                data = page.evaluate(EXTRACT_JS)
                results = json.loads(data)

                if not results:
                    print(f"  No results found, stopping.", file=sys.stderr)
                    break

                for r in results:
                    winner = r["team1"] if r["score1"] > r["score2"] else r["team2"]
                    r["winner"] = winner

                all_results.extend(results)
                print(f"  Got {len(results)} results (total: {len(all_results)})", file=sys.stderr)

                # Rate limit
                page.wait_for_timeout(3000)

            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                page.wait_for_timeout(5000)
                continue

        browser.close()

    return all_results


def main():
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 80

    print(f"Scraping {max_pages} pages of HLTV results...", file=sys.stderr)
    results = scrape_results(max_pages)

    # Save all results
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(data_dir, "all_results.csv"), index=False)
    print(f"\nTotal: {len(df)} results saved to data/all_results.csv", file=sys.stderr)

    # Filter for top 100 teams
    with open(os.path.join(data_dir, "top_teams.json")) as f:
        top_teams = json.load(f)

    top_names = set(top_teams.keys())
    mask = df["team1"].isin(top_names) & df["team2"].isin(top_names)
    df_top = df[mask].copy()
    df_top["rank1"] = df_top["team1"].map(top_teams)
    df_top["rank2"] = df_top["team2"].map(top_teams)
    df_top.to_csv(os.path.join(data_dir, "top100_matches.csv"), index=False)
    print(f"Top 100 matches: {len(df_top)}", file=sys.stderr)
    print(f"Unique teams: {len(set(df_top['team1']) | set(df_top['team2']))}", file=sys.stderr)


if __name__ == "__main__":
    main()
