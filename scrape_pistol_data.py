"""
Scrape HLTV per-map stats pages for pistol round data.
Uses async Playwright with multiple concurrent pages for speed.

Two phases:
  Phase A: Extract per-map stats URLs from main match pages
  Phase B: Extract round history (pistol winners) from per-map stats pages

Usage:
  python scrape_pistol_data.py                          # both phases, 5 workers
  python scrape_pistol_data.py --phase A --resume       # only phase A, resume
  python scrape_pistol_data.py --phase B --workers 3    # only phase B, 3 workers

Reads:  data/all_results_with_urls.csv, data/match_details.csv
Writes: data/map_stats_urls.csv, data/pistol_rounds.csv
"""
import asyncio
import argparse
import json
import os
import csv
import time
import re
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MAP_STATS_URLS_CSV = os.path.join(DATA_DIR, "map_stats_urls.csv")
PISTOL_ROUNDS_CSV = os.path.join(DATA_DIR, "pistol_rounds.csv")

MAP_STATS_FIELDS = ["match_id", "map_name", "map_number", "stats_url"]
PISTOL_FIELDS = ["match_id", "map_name", "map_number", "team1", "team2",
                 "pistol1_winner", "pistol2_winner"]


def extract_match_id(url):
    """Extract numeric match ID from URL like /matches/2390129/..."""
    try:
        parts = url.split("/matches/")[1].split("/")
        return parts[0]
    except (IndexError, AttributeError):
        return ""


def extract_mapstatsid(url):
    """Extract mapstatsid from URL like /stats/matches/mapstatsid/12345/..."""
    try:
        parts = url.split("/mapstatsid/")[1].split("/")
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


# ---------------------------------------------------------------------------
# Phase A: Extract per-map stats URLs from main match pages
# ---------------------------------------------------------------------------

EXTRACT_MAP_STATS_JS = """
() => {
    const links = document.querySelectorAll('.results-center-stats a');
    const results = [];
    links.forEach(a => {
        const href = a.getAttribute('href');
        if (href && href.includes('/mapstatsid/')) {
            results.push(href);
        }
    });
    return JSON.stringify(results);
}
"""


async def phase_a_worker(context, urls, worker_id, scraped_ids, stats):
    """Worker that extracts map stats URLs from main match pages."""
    page = await context.new_page()
    await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    for i, (url, match_id) in enumerate(urls):
        if match_id in scraped_ids:
            stats["skipped"] += 1
            continue

        retries = 0
        success = False
        while retries < 3 and not success:
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_selector(".mapholder", timeout=15000)

                raw = await page.evaluate(EXTRACT_MAP_STATS_JS)
                stat_urls = json.loads(raw)

                if not stat_urls:
                    retries += 1
                    if retries < 3:
                        await asyncio.sleep(5)
                    continue

                rows = []
                for idx, stat_url in enumerate(stat_urls):
                    # Extract map name from URL: /stats/matches/mapstatsid/ID/team1-vs-team2-mapname
                    map_name = ""
                    url_parts = stat_url.rstrip("/").split("-")
                    if url_parts:
                        map_name = url_parts[-1].capitalize()

                    full_url = stat_url if stat_url.startswith("http") else f"https://www.hltv.org{stat_url}"
                    rows.append({
                        "match_id": match_id,
                        "map_name": map_name,
                        "map_number": idx + 1,
                        "stats_url": full_url,
                    })

                append_csv(MAP_STATS_URLS_CSV, rows, MAP_STATS_FIELDS)
                stats["done"] += 1
                scraped_ids.add(match_id)
                success = True

                if stats["done"] % 25 == 0:
                    elapsed = time.time() - stats["start_time"]
                    rate = stats["done"] / elapsed * 60
                    print(f"  [A][Progress] {stats['done']}/{stats['total']} done, "
                          f"{stats['skipped']} skipped, {stats['errors']} errors "
                          f"({rate:.0f}/min, {elapsed/60:.1f}min elapsed)")

            except Exception as e:
                retries += 1
                if retries < 3:
                    await asyncio.sleep(8)
                else:
                    stats["errors"] += 1
                    if stats["errors"] <= 20:
                        print(f"  [A][W{worker_id}] Failed after 3 retries: {url}: {e}")

        await asyncio.sleep(2)

    await page.close()


async def run_phase_a(args):
    import pandas as pd

    print("=== Phase A: Extract per-map stats URLs ===\n")

    urls_path = os.path.join(DATA_DIR, "all_results_with_urls.csv")
    if not os.path.exists(urls_path):
        print(f"ERROR: {urls_path} not found.")
        return

    df = pd.read_csv(urls_path)
    df = df[df["match_url"].notna() & (df["match_url"] != "")].copy()
    df["match_id"] = df["match_url"].apply(extract_match_id)

    # Filter to matches that exist in match_details.csv (already scraped)
    details_path = os.path.join(DATA_DIR, "match_details.csv")
    if os.path.exists(details_path):
        details_df = pd.read_csv(details_path)
        detail_ids = set(details_df["match_id"].astype(str).values)
        df = df[df["match_id"].isin(detail_ids)].copy()
        print(f"Filtered to {len(df)} matches with player data")
    else:
        print("WARNING: match_details.csv not found, scraping all matches")

    # Resume support
    scraped_ids = set()
    if args.resume and os.path.exists(MAP_STATS_URLS_CSV):
        existing = pd.read_csv(MAP_STATS_URLS_CSV)
        scraped_ids = set(existing["match_id"].astype(str).values)
        print(f"Resume mode: {len(scraped_ids)} matches already scraped")
    else:
        init_csv(MAP_STATS_URLS_CSV, MAP_STATS_FIELDS)

    url_pairs = []
    for _, row in df.iterrows():
        mid = str(row["match_id"])
        if mid not in scraped_ids:
            url_pairs.append((row["match_url"], mid))

    print(f"Matches to scrape: {len(url_pairs)}")
    if not url_pairs:
        print("All matches already scraped!")
        return

    n_workers = min(args.workers, len(url_pairs))
    chunks = [[] for _ in range(n_workers)]
    for i, pair in enumerate(url_pairs):
        chunks[i % n_workers].append(pair)

    stats = {
        "done": 0, "errors": 0, "skipped": 0,
        "total": len(url_pairs),
        "start_time": time.time()
    }

    print(f"Starting {n_workers} workers...")
    print(f"Estimated time: {len(url_pairs) / n_workers * 3 / 60:.0f}-{len(url_pairs) / n_workers * 5 / 60:.0f} min\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        tasks = [phase_a_worker(context, chunk, wid, scraped_ids, stats)
                 for wid, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks)
        await browser.close()

    elapsed = time.time() - stats["start_time"]
    print(f"\nPhase A done! {stats['done']} scraped, {stats['errors']} errors, "
          f"{stats['skipped']} skipped in {elapsed/60:.1f} min")

    if os.path.exists(MAP_STATS_URLS_CSV):
        result = pd.read_csv(MAP_STATS_URLS_CSV)
        print(f"Map stats URLs: {len(result)} rows, {result['match_id'].nunique()} matches")


# ---------------------------------------------------------------------------
# Phase B: Extract round history from per-map stats pages
# ---------------------------------------------------------------------------

EXTRACT_ROUND_HISTORY_JS = """
() => {
    const rows = document.querySelectorAll('.round-history-team-row');
    if (rows.length < 2) return JSON.stringify({error: 'no round history rows'});

    // Team names come from the first img (class=round-history-team) in each row
    let team1 = '', team2 = '';
    const teamImg1 = rows[0].querySelector('img.round-history-team');
    const teamImg2 = rows[1].querySelector('img.round-history-team');
    if (teamImg1) team1 = (teamImg1.getAttribute('title') || teamImg1.getAttribute('alt') || '').trim();
    if (teamImg2) team2 = (teamImg2.getAttribute('title') || teamImg2.getAttribute('alt') || '').trim();

    const result = {team1, team2, rounds: [[], []]};

    // Only process first 2 rows (regulation), skip overtime rows
    rows.forEach((row, rowIdx) => {
        if (rowIdx > 1) return;
        const imgs = row.querySelectorAll('img.round-history-outcome');
        imgs.forEach(img => {
            const title = (img.getAttribute('title') || '').trim();
            // Non-empty title with score pattern (e.g. "1-0") = team won that round
            // Empty title or emptyHistory.svg = team lost that round
            const won = /^\\d+-\\d+$/.test(title);
            result.rounds[rowIdx].push({title, won});
        });
    });

    return JSON.stringify(result);
}
"""


def determine_pistol_winners(round_data):
    """Determine pistol round winners from round history data.

    Pistol rounds are round 1 (index 0) and round 13 (index 12).
    A team won a round if their entry has a non-empty alt (score text).
    """
    team1 = round_data.get("team1", "")
    team2 = round_data.get("team2", "")
    rounds = round_data.get("rounds", [[], []])

    if len(rounds) < 2:
        return team1, team2, "", ""

    r1 = rounds[0]  # team1's round entries
    r2 = rounds[1]  # team2's round entries

    # Pistol 1: round 1 (index 0)
    pistol1_winner = ""
    if len(r1) > 0 and len(r2) > 0:
        if r1[0].get("won"):
            pistol1_winner = team1
        elif r2[0].get("won"):
            pistol1_winner = team2

    # Pistol 2: round 13 (index 12)
    pistol2_winner = ""
    if len(r1) > 12 and len(r2) > 12:
        if r1[12].get("won"):
            pistol2_winner = team1
        elif r2[12].get("won"):
            pistol2_winner = team2

    return team1, team2, pistol1_winner, pistol2_winner


async def phase_b_worker(context, urls, worker_id, scraped_ids, stats):
    """Worker that extracts round history from per-map stats pages."""
    page = await context.new_page()
    await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    for i, row_data in enumerate(urls):
        mapstatsid = extract_mapstatsid(row_data["stats_url"])
        if mapstatsid in scraped_ids:
            stats["skipped"] += 1
            continue

        retries = 0
        success = False
        while retries < 3 and not success:
            try:
                await page.goto(row_data["stats_url"], timeout=30000,
                                wait_until="domcontentloaded")
                await page.wait_for_selector(".round-history-team-row", timeout=15000)

                raw = await page.evaluate(EXTRACT_ROUND_HISTORY_JS)
                data = json.loads(raw)

                if "error" in data:
                    retries += 1
                    if retries < 3:
                        await asyncio.sleep(5)
                    continue

                team1, team2, p1_winner, p2_winner = determine_pistol_winners(data)

                pistol_row = {
                    "match_id": row_data["match_id"],
                    "map_name": row_data["map_name"],
                    "map_number": row_data["map_number"],
                    "team1": team1,
                    "team2": team2,
                    "pistol1_winner": p1_winner,
                    "pistol2_winner": p2_winner,
                }

                append_csv(PISTOL_ROUNDS_CSV, [pistol_row], PISTOL_FIELDS)
                stats["done"] += 1
                scraped_ids.add(mapstatsid)
                success = True

                if stats["done"] % 25 == 0:
                    elapsed = time.time() - stats["start_time"]
                    rate = stats["done"] / elapsed * 60
                    print(f"  [B][Progress] {stats['done']}/{stats['total']} done, "
                          f"{stats['skipped']} skipped, {stats['errors']} errors "
                          f"({rate:.0f}/min, {elapsed/60:.1f}min elapsed)")

            except Exception as e:
                retries += 1
                if retries < 3:
                    await asyncio.sleep(8)
                else:
                    stats["errors"] += 1
                    if stats["errors"] <= 20:
                        print(f"  [B][W{worker_id}] Failed after 3 retries: "
                              f"{row_data['stats_url']}: {e}")

        await asyncio.sleep(2)

    await page.close()


async def run_phase_b(args):
    import pandas as pd

    print("=== Phase B: Extract pistol round data ===\n")

    if not os.path.exists(MAP_STATS_URLS_CSV):
        print(f"ERROR: {MAP_STATS_URLS_CSV} not found. Run phase A first.")
        return

    map_urls_df = pd.read_csv(MAP_STATS_URLS_CSV)
    print(f"Loaded {len(map_urls_df)} map stats URLs")

    # Resume support
    scraped_ids = set()
    if args.resume and os.path.exists(PISTOL_ROUNDS_CSV):
        existing = pd.read_csv(PISTOL_ROUNDS_CSV)
        # Build set of already-scraped mapstatsids
        if os.path.exists(MAP_STATS_URLS_CSV):
            for _, row in existing.iterrows():
                # Find the corresponding stats_url to get mapstatsid
                match = map_urls_df[
                    (map_urls_df["match_id"].astype(str) == str(row["match_id"])) &
                    (map_urls_df["map_number"].astype(str) == str(row["map_number"]))
                ]
                if not match.empty:
                    msid = extract_mapstatsid(match.iloc[0]["stats_url"])
                    if msid:
                        scraped_ids.add(msid)
        print(f"Resume mode: {len(scraped_ids)} maps already scraped")
    else:
        init_csv(PISTOL_ROUNDS_CSV, PISTOL_FIELDS)

    # Build work items
    work_items = []
    for _, row in map_urls_df.iterrows():
        msid = extract_mapstatsid(str(row["stats_url"]))
        if msid not in scraped_ids:
            work_items.append({
                "match_id": str(row["match_id"]),
                "map_name": row["map_name"],
                "map_number": row["map_number"],
                "stats_url": row["stats_url"],
            })

    print(f"Maps to scrape: {len(work_items)}")
    if not work_items:
        print("All maps already scraped!")
        return

    n_workers = min(args.workers, len(work_items))
    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(work_items):
        chunks[i % n_workers].append(item)

    stats = {
        "done": 0, "errors": 0, "skipped": 0,
        "total": len(work_items),
        "start_time": time.time()
    }

    print(f"Starting {n_workers} workers...")
    print(f"Estimated time: {len(work_items) / n_workers * 3 / 60:.0f}-"
          f"{len(work_items) / n_workers * 5 / 60:.0f} min\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        tasks = [phase_b_worker(context, chunk, wid, scraped_ids, stats)
                 for wid, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks)
        await browser.close()

    elapsed = time.time() - stats["start_time"]
    print(f"\nPhase B done! {stats['done']} scraped, {stats['errors']} errors, "
          f"{stats['skipped']} skipped in {elapsed/60:.1f} min")

    if os.path.exists(PISTOL_ROUNDS_CSV):
        result = pd.read_csv(PISTOL_ROUNDS_CSV)
        print(f"Pistol rounds: {len(result)} rows, {result['match_id'].nunique()} matches")
        # Quick stats
        has_p1 = (result["pistol1_winner"] != "").sum()
        has_p2 = (result["pistol2_winner"] != "").sum()
        print(f"  Pistol 1 winners found: {has_p1}/{len(result)}")
        print(f"  Pistol 2 winners found: {has_p2}/{len(result)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    if args.phase in ("A", "all"):
        await run_phase_a(args)
    if args.phase in ("B", "all"):
        await run_phase_b(args)


def main():
    parser = argparse.ArgumentParser(description="Scrape HLTV pistol round data")
    parser.add_argument("--phase", choices=["A", "B", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel browser pages")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scraped pages")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
