"""
Scrape HLTV match detail pages for player stats.
Uses async Playwright with multiple concurrent pages for speed.

Usage:
  python scrape_match_details.py               # scrape all, 5 workers
  python scrape_match_details.py --workers 3   # use 3 workers
  python scrape_match_details.py --resume      # skip already-scraped matches

Reads:  data/top100_matches_with_urls.csv
Writes: data/match_details.csv, data/player_stats.csv
"""
import asyncio
import argparse
import json
import os
import csv
import time
from datetime import datetime
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MATCH_DETAILS_CSV = os.path.join(DATA_DIR, "match_details.csv")
PLAYER_STATS_CSV = os.path.join(DATA_DIR, "player_stats.csv")
PARTIAL_JSON = os.path.join(DATA_DIR, "player_stats_partial.json")

MATCH_DETAILS_FIELDS = [
    "match_id", "match_url", "team1", "team2", "score1", "score2",
    "maps_played", "map_names", "bo_format", "event", "date", "winner"
]
PLAYER_STATS_FIELDS = [
    "match_id", "match_url", "team", "player", "kills", "deaths",
    "adr", "kast", "rating", "date", "event"
]

# JS to extract all match data from a detail page
EXTRACT_MATCH_JS = """
() => {
    const data = {players: [], maps: []};

    // Team names
    const teamNames = document.querySelectorAll('.teamName');
    data.team1 = teamNames[0]?.textContent?.trim() || '';
    data.team2 = teamNames[1]?.textContent?.trim() || '';

    // Event
    data.event = document.querySelector('.event a')?.textContent?.trim() || '';

    // Date
    data.date = document.querySelector('.timeAndEvent .date')?.textContent?.trim() || '';

    // Maps
    document.querySelectorAll('.mapholder').forEach(m => {
        const mapName = m.querySelector('.mapname')?.textContent?.trim();
        const scores = m.querySelectorAll('.results-team-score');
        if (mapName && scores.length >= 2) {
            data.maps.push({
                map: mapName,
                score1: scores[0]?.textContent?.trim() || '',
                score2: scores[1]?.textContent?.trim() || ''
            });
        }
    });

    // Player stats from the FIRST .stats-content (all-maps overview)
    const statsContent = document.querySelector('.stats-content');
    if (!statsContent) return JSON.stringify(data);

    const tables = statsContent.querySelectorAll('table.totalstats');
    tables.forEach((table, tableIdx) => {
        const rows = table.querySelectorAll('tr');
        // First row is header with team name
        const teamName = rows[0]?.querySelector('.players')?.textContent?.trim() || '';

        for (let i = 1; i < rows.length; i++) {
            const row = rows[i];
            const playerCell = row.querySelector('.players');
            if (!playerCell) continue;

            // Extract in-game name (it's the last part after real name)
            const playerText = playerCell.textContent.trim();
            // Format: "Real Name\\n    alias" - get the alias
            const parts = playerText.split('\\n').map(s => s.trim()).filter(s => s);
            const alias = parts.length > 1 ? parts[parts.length - 1] : parts[0];

            // K-D (traditional, not eco-adjusted)
            const kdCell = row.querySelector('.kd.traditional-data');
            const kdText = kdCell?.textContent?.trim() || '0-0';
            const kdParts = kdText.split('-').map(s => parseInt(s.trim()));

            // ADR
            const adrCell = row.querySelector('.adr.traditional-data');
            const adr = parseFloat(adrCell?.textContent?.trim()) || 0;

            // KAST
            const kastCell = row.querySelector('.kast.traditional-data');
            const kastText = kastCell?.textContent?.trim() || '0%';
            const kast = parseFloat(kastText.replace('%', '')) || 0;

            // Rating
            const ratingCell = row.querySelector('.rating');
            const rating = parseFloat(ratingCell?.textContent?.trim()) || 0;

            data.players.push({
                team: teamName,
                player: alias,
                kills: kdParts[0] || 0,
                deaths: kdParts[1] || 0,
                adr: adr,
                kast: kast,
                rating: rating
            });
        }
    });

    return JSON.stringify(data);
}
"""


def extract_match_id(url):
    """Extract numeric match ID from URL like /matches/2390129/..."""
    try:
        parts = url.split("/matches/")[1].split("/")
        return parts[0]
    except (IndexError, AttributeError):
        return ""


def init_csv(path, fieldnames):
    """Create CSV with headers if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_csv(path, rows, fieldnames):
    """Append rows to CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


async def scrape_worker(context, urls, worker_id, scraped_ids, stats):
    """Worker that processes a list of match URLs using one browser page."""
    page = await context.new_page()
    await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    for i, (url, orig_row) in enumerate(urls):
        match_id = extract_match_id(url)
        if match_id in scraped_ids:
            stats["skipped"] += 1
            continue

        retries = 0
        success = False
        while retries < 3 and not success:
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                # Wait for stats table to load
                await page.wait_for_selector(".stats-content .totalstats", timeout=20000)

                raw = await page.evaluate(EXTRACT_MATCH_JS)
                data = json.loads(raw)

                if not data.get("players"):
                    retries += 1
                    print(f"  [W{worker_id}] No players found at {url}, retry {retries}")
                    await asyncio.sleep(5)
                    continue

                # Build match details row
                map_names = [m["map"] for m in data["maps"]]
                n_maps = len(map_names)
                bo = f"BO{max(n_maps, 1)}" if n_maps <= 1 else f"BO{2*n_maps - 1}" if n_maps <= 2 else f"BO{2*n_maps - 1}"
                # Simpler: infer from max possible maps
                if n_maps == 1:
                    bo = "BO1"
                elif n_maps == 2:
                    bo = "BO3"
                elif n_maps == 3:
                    bo = "BO5" if int(orig_row.get("score1", 0)) + int(orig_row.get("score2", 0)) > 3 else "BO3"
                elif n_maps >= 4:
                    bo = "BO5"

                match_detail = {
                    "match_id": match_id,
                    "match_url": url,
                    "team1": data["team1"],
                    "team2": data["team2"],
                    "score1": orig_row.get("score1", ""),
                    "score2": orig_row.get("score2", ""),
                    "maps_played": n_maps,
                    "map_names": "|".join(map_names),
                    "bo_format": bo,
                    "event": data["event"],
                    "date": data["date"],
                    "winner": orig_row.get("winner", ""),
                }

                # Build player stats rows
                player_rows = []
                for p in data["players"]:
                    player_rows.append({
                        "match_id": match_id,
                        "match_url": url,
                        "team": p["team"],
                        "player": p["player"],
                        "kills": p["kills"],
                        "deaths": p["deaths"],
                        "adr": p["adr"],
                        "kast": p["kast"],
                        "rating": p["rating"],
                        "date": data["date"],
                        "event": data["event"],
                    })

                # Append to CSVs
                append_csv(MATCH_DETAILS_CSV, [match_detail], MATCH_DETAILS_FIELDS)
                append_csv(PLAYER_STATS_CSV, player_rows, PLAYER_STATS_FIELDS)

                stats["done"] += 1
                scraped_ids.add(match_id)
                success = True

                if stats["done"] % 25 == 0:
                    elapsed = time.time() - stats["start_time"]
                    rate = stats["done"] / elapsed * 60
                    print(f"  [Progress] {stats['done']}/{stats['total']} done, "
                          f"{stats['skipped']} skipped, {stats['errors']} errors "
                          f"({rate:.0f}/min, {elapsed/60:.1f}min elapsed)")

            except Exception as e:
                retries += 1
                if retries < 3:
                    await asyncio.sleep(8)
                else:
                    stats["errors"] += 1
                    if stats["errors"] <= 20:
                        print(f"  [W{worker_id}] Failed after 3 retries: {url}: {e}")

        # Rate limit
        await asyncio.sleep(2)

    await page.close()


async def main_async(args):
    import pandas as pd

    # Load match URLs
    urls_path = os.path.join(DATA_DIR, "top100_matches_with_urls.csv")
    if not os.path.exists(urls_path):
        print(f"ERROR: {urls_path} not found. Run scrape_data.py first.")
        return

    df = pd.read_csv(urls_path)
    print(f"Loaded {len(df)} matches from top100_matches_with_urls.csv")

    # Filter to matches with valid URLs
    df = df[df["match_url"].notna() & (df["match_url"] != "")].copy()
    print(f"Matches with URLs: {len(df)}")

    if df.empty:
        print("No match URLs found.")
        return

    # Check for already-scraped matches (resume mode)
    scraped_ids = set()
    if args.resume and os.path.exists(MATCH_DETAILS_CSV):
        existing = pd.read_csv(MATCH_DETAILS_CSV)
        scraped_ids = set(existing["match_id"].astype(str).values)
        print(f"Resume mode: {len(scraped_ids)} matches already scraped")
    else:
        # Fresh start
        init_csv(MATCH_DETAILS_CSV, MATCH_DETAILS_FIELDS)
        init_csv(PLAYER_STATS_CSV, PLAYER_STATS_FIELDS)

    # Build (url, row) pairs
    url_pairs = []
    for _, row in df.iterrows():
        url = row["match_url"]
        mid = extract_match_id(url)
        if mid not in scraped_ids:
            url_pairs.append((url, row.to_dict()))

    print(f"Matches to scrape: {len(url_pairs)}")
    if not url_pairs:
        print("All matches already scraped!")
        return

    # Split across workers
    n_workers = min(args.workers, len(url_pairs))
    chunks = [[] for _ in range(n_workers)]
    for i, pair in enumerate(url_pairs):
        chunks[i % n_workers].append(pair)

    stats = {
        "done": 0, "errors": 0, "skipped": 0,
        "total": len(url_pairs),
        "start_time": time.time()
    }

    print(f"\nStarting {n_workers} workers...")
    print(f"Estimated time: {len(url_pairs) / n_workers * 3 / 60:.0f}-{len(url_pairs) / n_workers * 5 / 60:.0f} min\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        tasks = []
        for wid, chunk in enumerate(chunks):
            tasks.append(scrape_worker(context, chunk, wid, scraped_ids, stats))

        await asyncio.gather(*tasks)
        await browser.close()

    elapsed = time.time() - stats["start_time"]
    print(f"\nDone! {stats['done']} scraped, {stats['errors']} errors, "
          f"{stats['skipped']} skipped in {elapsed/60:.1f} min")

    # Print summary
    if os.path.exists(PLAYER_STATS_CSV):
        ps = pd.read_csv(PLAYER_STATS_CSV)
        print(f"\nPlayer stats: {len(ps)} rows")
        print(f"Unique players: {ps['player'].nunique()}")
        print(f"Unique matches: {ps['match_id'].nunique()}")
    if os.path.exists(MATCH_DETAILS_CSV):
        md = pd.read_csv(MATCH_DETAILS_CSV)
        print(f"Match details: {len(md)} rows")


def main():
    parser = argparse.ArgumentParser(description="Scrape HLTV match details")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel browser pages")
    parser.add_argument("--resume", action="store_true", help="Skip already-scraped matches")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
