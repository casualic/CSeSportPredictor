"""
Scrape HLTV pistol round data by connecting to the MCP browser via CDP.
This reuses the MCP's Chrome session which bypasses Cloudflare.

Usage:
  python scrape_pistol_cdp.py --phase A          # Extract map stats URLs
  python scrape_pistol_cdp.py --phase B          # Extract pistol round data
  python scrape_pistol_cdp.py --phase all         # Both phases
  python scrape_pistol_cdp.py --phase A --port 51007  # Specify CDP port
"""
import asyncio
import argparse
import json
import os
import csv
import time
import subprocess
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
    try:
        return url.split("/matches/")[1].split("/")[0]
    except (IndexError, AttributeError):
        return ""


def extract_mapstatsid(url):
    try:
        return url.split("/mapstatsid/")[1].split("/")[0]
    except (IndexError, AttributeError):
        return ""


def init_csv(path, fieldnames):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv(path, rows, fieldnames):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def find_cdp_port():
    """Find the CDP port from running Chrome processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "remote-debugging-port=" in line and "Google Chrome" in line:
                for part in line.split():
                    if part.startswith("--remote-debugging-port="):
                        return int(part.split("=")[1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Phase A: Extract per-map stats URLs
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

    // Also extract map names from mapholders
    const mapNames = [];
    document.querySelectorAll('.mapholder .mapname').forEach(el => {
        mapNames.push(el.textContent.trim());
    });

    return JSON.stringify({urls: results, mapNames: mapNames});
}
"""


async def run_phase_a(page, args):
    import pandas as pd

    print("=== Phase A: Extract per-map stats URLs ===\n")

    urls_df = pd.read_csv(os.path.join(DATA_DIR, "all_results_with_urls.csv"))
    urls_df = urls_df[urls_df["match_url"].notna() & (urls_df["match_url"] != "")].copy()
    urls_df["match_id"] = urls_df["match_url"].apply(extract_match_id)

    details_path = os.path.join(DATA_DIR, "match_details.csv")
    if os.path.exists(details_path):
        details_df = pd.read_csv(details_path)
        detail_ids = set(details_df["match_id"].astype(str))
        urls_df = urls_df[urls_df["match_id"].isin(detail_ids)].copy()
        print(f"Filtered to {len(urls_df)} matches with player data")

    # Resume support
    scraped_ids = set()
    if os.path.exists(MAP_STATS_URLS_CSV):
        existing = pd.read_csv(MAP_STATS_URLS_CSV)
        scraped_ids = set(existing["match_id"].astype(str))
        print(f"Resume: {len(scraped_ids)} matches already scraped")
    else:
        init_csv(MAP_STATS_URLS_CSV, MAP_STATS_FIELDS)

    work = [(str(r["match_id"]), r["match_url"])
            for _, r in urls_df.iterrows()
            if str(r["match_id"]) not in scraped_ids]

    print(f"Matches to scrape: {len(work)}")
    if not work:
        print("All done!")
        return

    done = 0
    errors = 0
    start_time = time.time()

    for match_id, url in work:
        retries = 0
        success = False
        while retries < 3 and not success:
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_selector(".mapholder", timeout=15000)

                raw = await page.evaluate(EXTRACT_MAP_STATS_JS)
                data = json.loads(raw)
                stat_urls = data.get("urls", [])
                map_names = data.get("mapNames", [])

                if not stat_urls:
                    retries += 1
                    if retries < 3:
                        await asyncio.sleep(3)
                    continue

                rows = []
                for idx, stat_url in enumerate(stat_urls):
                    # Use extracted map name if available, otherwise parse from URL
                    if idx < len(map_names):
                        map_name = map_names[idx]
                    else:
                        url_parts = stat_url.rstrip("/").split("-")
                        map_name = url_parts[-1].capitalize() if url_parts else ""

                    full_url = stat_url if stat_url.startswith("http") else f"https://www.hltv.org{stat_url}"
                    rows.append({
                        "match_id": match_id,
                        "map_name": map_name,
                        "map_number": idx + 1,
                        "stats_url": full_url,
                    })

                append_csv(MAP_STATS_URLS_CSV, rows, MAP_STATS_FIELDS)
                done += 1
                success = True

                if done % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed * 60
                    print(f"  [A] {done}/{len(work)} done, {errors} errors "
                          f"({rate:.0f}/min, {elapsed/60:.1f}min elapsed)")

            except Exception as e:
                retries += 1
                if retries < 3:
                    await asyncio.sleep(5)
                else:
                    errors += 1
                    if errors <= 20:
                        print(f"  [A] Failed: {url}: {e}")

        await asyncio.sleep(1.5)

    elapsed = time.time() - start_time
    print(f"\nPhase A done! {done} scraped, {errors} errors in {elapsed/60:.1f} min")


# ---------------------------------------------------------------------------
# Phase B: Extract pistol round data
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
    team1 = round_data.get("team1", "")
    team2 = round_data.get("team2", "")
    rounds = round_data.get("rounds", [[], []])

    if len(rounds) < 2:
        return team1, team2, "", ""

    r1, r2 = rounds[0], rounds[1]

    # Pistol 1: round 1 (index 0)
    pistol1_winner = ""
    if len(r1) > 0 and len(r2) > 0:
        if r1[0].get("won"):
            pistol1_winner = team1
        elif r2[0].get("won"):
            pistol1_winner = team2

    # Pistol 2: round 13 (index 12) — CS2 MR12 format
    pistol2_winner = ""
    if len(r1) > 12 and len(r2) > 12:
        if r1[12].get("won"):
            pistol2_winner = team1
        elif r2[12].get("won"):
            pistol2_winner = team2

    return team1, team2, pistol1_winner, pistol2_winner


async def run_phase_b(page, args):
    import pandas as pd

    print("=== Phase B: Extract pistol round data ===\n")

    if not os.path.exists(MAP_STATS_URLS_CSV):
        print(f"ERROR: {MAP_STATS_URLS_CSV} not found. Run phase A first.")
        return

    map_urls_df = pd.read_csv(MAP_STATS_URLS_CSV)
    print(f"Loaded {len(map_urls_df)} map stats URLs")

    # Resume support
    scraped_ids = set()
    if os.path.exists(PISTOL_ROUNDS_CSV):
        existing = pd.read_csv(PISTOL_ROUNDS_CSV)
        for _, row in existing.iterrows():
            match = map_urls_df[
                (map_urls_df["match_id"].astype(str) == str(row["match_id"])) &
                (map_urls_df["map_number"].astype(str) == str(row["map_number"]))
            ]
            if not match.empty:
                msid = extract_mapstatsid(str(match.iloc[0]["stats_url"]))
                if msid:
                    scraped_ids.add(msid)
        print(f"Resume: {len(scraped_ids)} maps already scraped")
    else:
        init_csv(PISTOL_ROUNDS_CSV, PISTOL_FIELDS)

    work = []
    for _, row in map_urls_df.iterrows():
        msid = extract_mapstatsid(str(row["stats_url"]))
        if msid not in scraped_ids:
            work.append({
                "match_id": str(row["match_id"]),
                "map_name": row["map_name"],
                "map_number": row["map_number"],
                "stats_url": row["stats_url"],
                "mapstatsid": msid,
            })

    print(f"Maps to scrape: {len(work)}")
    if not work:
        print("All done!")
        return

    done = 0
    errors = 0
    start_time = time.time()

    for item in work:
        retries = 0
        success = False
        while retries < 3 and not success:
            try:
                await page.goto(item["stats_url"], timeout=30000,
                                wait_until="domcontentloaded")
                await page.wait_for_selector(".round-history-team-row", timeout=15000)

                raw = await page.evaluate(EXTRACT_ROUND_HISTORY_JS)
                data = json.loads(raw)

                if "error" in data:
                    retries += 1
                    if retries < 3:
                        await asyncio.sleep(3)
                    continue

                team1, team2, p1_winner, p2_winner = determine_pistol_winners(data)

                pistol_row = {
                    "match_id": item["match_id"],
                    "map_name": item["map_name"],
                    "map_number": item["map_number"],
                    "team1": team1,
                    "team2": team2,
                    "pistol1_winner": p1_winner,
                    "pistol2_winner": p2_winner,
                }

                append_csv(PISTOL_ROUNDS_CSV, [pistol_row], PISTOL_FIELDS)
                done += 1
                success = True

                if done % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed * 60
                    print(f"  [B] {done}/{len(work)} done, {errors} errors "
                          f"({rate:.0f}/min, {elapsed/60:.1f}min elapsed)")

            except Exception as e:
                retries += 1
                if retries < 3:
                    await asyncio.sleep(5)
                else:
                    errors += 1
                    if errors <= 20:
                        print(f"  [B] Failed: {item['stats_url']}: {e}")

        await asyncio.sleep(1.5)

    elapsed = time.time() - start_time
    print(f"\nPhase B done! {done} scraped, {errors} errors in {elapsed/60:.1f} min")

    if os.path.exists(PISTOL_ROUNDS_CSV):
        result = pd.read_csv(PISTOL_ROUNDS_CSV)
        has_p1 = (result["pistol1_winner"] != "").sum()
        has_p2 = (result["pistol2_winner"] != "").sum()
        print(f"Pistol rounds: {len(result)} rows")
        print(f"  Pistol 1 winners found: {has_p1}/{len(result)}")
        print(f"  Pistol 2 winners found: {has_p2}/{len(result)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    port = args.port
    if port is None:
        port = find_cdp_port()
        if port is None:
            print("ERROR: Could not find CDP port. Make sure MCP browser is running.")
            print("Specify manually with --port")
            return

    print(f"Connecting to Chrome CDP on port {port}...")

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(f"http://localhost:{port}")
        context = browser.contexts[0]
        # Reuse existing tab instead of creating a new one
        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        if args.phase in ("A", "all"):
            await run_phase_a(page, args)
        if args.phase in ("B", "all"):
            await run_phase_b(page, args)


def main():
    parser = argparse.ArgumentParser(description="Scrape HLTV pistol data via CDP")
    parser.add_argument("--phase", choices=["A", "B", "all"], default="all")
    parser.add_argument("--port", type=int, default=None, help="Chrome CDP port")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
