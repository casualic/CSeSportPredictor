"""
Scrape HLTV match detail pages via CDP (reuses MCP Chrome to bypass Cloudflare).

Extracts:
  - Match details (teams, scores, maps, event, date, BO format)
  - Player stats (kills, deaths, ADR, KAST, rating per player)
  - Map veto/pick-ban sequence
  - Half scores per map
  - Per-map player stats (individual map performance)

Usage:
  python scrape_details_cdp.py                    # scrape all remaining
  python scrape_details_cdp.py --port 51007       # specify CDP port
"""
import asyncio
import argparse
import json
import os
import csv
import time
import subprocess
import pandas as pd
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MATCH_DETAILS_CSV = os.path.join(DATA_DIR, "match_details.csv")
PLAYER_STATS_CSV = os.path.join(DATA_DIR, "player_stats.csv")
VETO_CSV = os.path.join(DATA_DIR, "veto_data.csv")
HALF_SCORES_CSV = os.path.join(DATA_DIR, "half_scores.csv")
MAP_PLAYER_STATS_CSV = os.path.join(DATA_DIR, "map_player_stats.csv")

MATCH_DETAILS_FIELDS = [
    "match_id", "match_url", "team1", "team2", "score1", "score2",
    "maps_played", "map_names", "bo_format", "event", "date", "winner"
]
PLAYER_STATS_FIELDS = [
    "match_id", "match_url", "team", "player", "kills", "deaths",
    "adr", "kast", "rating", "date", "event"
]
VETO_FIELDS = [
    "match_id", "veto_order", "team", "action", "map_name"
]
HALF_SCORES_FIELDS = [
    "match_id", "map_number", "map_name",
    "team1", "team1_ct_half", "team1_t_half", "team1_ot",
    "team2", "team2_ct_half", "team2_t_half", "team2_ot",
    "team1_total", "team2_total"
]
MAP_PLAYER_STATS_FIELDS = [
    "match_id", "map_number", "map_name", "team", "player",
    "kills", "deaths", "adr", "kast", "rating"
]


def extract_match_id(url):
    try:
        return url.split("/matches/")[1].split("/")[0]
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
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "remote-debugging-port=" in line and "Google Chrome" in line:
                for part in line.split():
                    if part.startswith("--remote-debugging-port="):
                        return int(part.split("=")[1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# JS extraction: pulls everything from a match detail page
# ---------------------------------------------------------------------------

EXTRACT_ALL_JS = """
() => {
    const data = {players: [], maps: [], vetos: [], halfScores: [], mapPlayerStats: []};

    // Team names
    const teamNames = document.querySelectorAll('.teamName');
    data.team1 = teamNames[0]?.textContent?.trim() || '';
    data.team2 = teamNames[1]?.textContent?.trim() || '';

    // Event
    data.event = document.querySelector('.event a')?.textContent?.trim() || '';

    // Date
    data.date = document.querySelector('.timeAndEvent .date')?.textContent?.trim() || '';

    // Maps (from mapholders)
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

    // Veto/pick-ban sequence
    document.querySelectorAll('.veto-box .padding div').forEach((el, idx) => {
        const text = el.textContent?.trim();
        if (!text) return;
        // Format: "1. TeamName removed MapName" or "1. TeamName picked MapName"
        // or "2. TeamName removed MapName" etc.
        const match = text.match(/^\\d+\\.\\s+(.+?)\\s+(removed|picked|left over)\\s+(.+)$/i);
        if (match) {
            data.vetos.push({
                order: idx + 1,
                team: match[1].trim(),
                action: match[2].trim().toLowerCase(),
                map: match[3].trim()
            });
        }
    });

    // Overall player stats from first .stats-content (all-maps overview)
    const statsContent = document.querySelector('.stats-content');
    if (statsContent) {
        const tables = statsContent.querySelectorAll('table.totalstats');
        tables.forEach((table) => {
            const rows = table.querySelectorAll('tr');
            const teamName = rows[0]?.querySelector('.players')?.textContent?.trim() || '';
            for (let i = 1; i < rows.length; i++) {
                const row = rows[i];
                const playerCell = row.querySelector('.players');
                if (!playerCell) continue;
                const playerText = playerCell.textContent.trim();
                const parts = playerText.split('\\n').map(s => s.trim()).filter(s => s);
                const alias = parts.length > 1 ? parts[parts.length - 1] : parts[0];
                const kdCell = row.querySelector('.kd.traditional-data');
                const kdText = kdCell?.textContent?.trim() || '0-0';
                const kdParts = kdText.split('-').map(s => parseInt(s.trim()));
                const adrCell = row.querySelector('.adr.traditional-data');
                const adr = parseFloat(adrCell?.textContent?.trim()) || 0;
                const kastCell = row.querySelector('.kast.traditional-data');
                const kastText = kastCell?.textContent?.trim() || '0%';
                const kast = parseFloat(kastText.replace('%', '')) || 0;
                const ratingCell = row.querySelector('.rating');
                const rating = parseFloat(ratingCell?.textContent?.trim()) || 0;
                data.players.push({
                    team: teamName, player: alias,
                    kills: kdParts[0] || 0, deaths: kdParts[1] || 0,
                    adr, kast, rating
                });
            }
        });
    }

    // Half scores per map
    const halfElements = document.querySelectorAll('.results-center-half-score');
    halfElements.forEach((halfEl, mapIdx) => {
        // Each half-score container has spans for CT, T, and potentially OT scores
        const spans = halfEl.querySelectorAll('span');
        // Structure: team1_ct, team1_t, [team1_ot], team2_ct, team2_t, [team2_ot]
        // But actual structure varies - let's grab all numbers
        const numbers = [];
        spans.forEach(span => {
            const text = span.textContent.trim();
            const cls = span.className || '';
            if (text && /^\\d+$/.test(text)) {
                numbers.push({val: parseInt(text), cls: cls});
            }
        });
        data.halfScores.push({mapIdx, numbers});
    });

    // Per-map player stats (from tabbed stats sections)
    const mapStatsTabs = document.querySelectorAll('.stats-content');
    mapStatsTabs.forEach((statsEl, statsIdx) => {
        if (statsIdx === 0) return; // skip overall stats (already captured above)
        const mapName = '';  // We'll correlate with map order
        const tables = statsEl.querySelectorAll('table.totalstats');
        tables.forEach((table) => {
            const rows = table.querySelectorAll('tr');
            const teamName = rows[0]?.querySelector('.players')?.textContent?.trim() || '';
            for (let i = 1; i < rows.length; i++) {
                const row = rows[i];
                const playerCell = row.querySelector('.players');
                if (!playerCell) continue;
                const playerText = playerCell.textContent.trim();
                const parts = playerText.split('\\n').map(s => s.trim()).filter(s => s);
                const alias = parts.length > 1 ? parts[parts.length - 1] : parts[0];
                const kdCell = row.querySelector('.kd.traditional-data');
                const kdText = kdCell?.textContent?.trim() || '0-0';
                const kdParts = kdText.split('-').map(s => parseInt(s.trim()));
                const adrCell = row.querySelector('.adr.traditional-data');
                const adr = parseFloat(adrCell?.textContent?.trim()) || 0;
                const kastCell = row.querySelector('.kast.traditional-data');
                const kastText = kastCell?.textContent?.trim() || '0%';
                const kast = parseFloat(kastText.replace('%', '')) || 0;
                const ratingCell = row.querySelector('.rating');
                const rating = parseFloat(ratingCell?.textContent?.trim()) || 0;
                data.mapPlayerStats.push({
                    mapIdx: statsIdx,  // 1-based (map 1, map 2, etc.)
                    team: teamName, player: alias,
                    kills: kdParts[0] || 0, deaths: kdParts[1] || 0,
                    adr, kast, rating
                });
            }
        });
    });

    return JSON.stringify(data);
}
"""


async def run_scraper(page, args):
    print("=== Scrape All Match Details via CDP ===\n")

    urls_df = pd.read_csv(os.path.join(DATA_DIR, "all_results_with_urls.csv"))
    urls_df = urls_df[urls_df["match_url"].notna() & (urls_df["match_url"] != "")].copy()
    urls_df["match_id"] = urls_df["match_url"].apply(extract_match_id)
    print(f"Total matches with URLs: {len(urls_df)}")

    # Resume: check already scraped
    scraped_ids = set()
    if os.path.exists(MATCH_DETAILS_CSV):
        existing = pd.read_csv(MATCH_DETAILS_CSV)
        scraped_ids = set(existing["match_id"].astype(str))
        print(f"Resume: {len(scraped_ids)} matches already scraped")

    # Init CSVs that don't exist yet
    init_csv(MATCH_DETAILS_CSV, MATCH_DETAILS_FIELDS)
    init_csv(PLAYER_STATS_CSV, PLAYER_STATS_FIELDS)
    init_csv(VETO_CSV, VETO_FIELDS)
    init_csv(HALF_SCORES_CSV, HALF_SCORES_FIELDS)
    init_csv(MAP_PLAYER_STATS_CSV, MAP_PLAYER_STATS_FIELDS)

    work = []
    for _, r in urls_df.iterrows():
        mid = str(r["match_id"])
        if mid not in scraped_ids:
            work.append({"match_id": mid, "match_url": r["match_url"], "row": r.to_dict()})

    print(f"Matches to scrape: {len(work)}")
    if not work:
        print("All done!")
        return

    done = 0
    errors = 0
    start_time = time.time()

    for item in work:
        match_id = item["match_id"]
        url = item["match_url"]
        orig_row = item["row"]
        retries = 0
        success = False

        while retries < 3 and not success:
            try:
                await page.goto(url, timeout=45000, wait_until="domcontentloaded")
                # Wait for page content to render
                await asyncio.sleep(2)
                # Try waiting for stats table, fall back to simple delay
                try:
                    await page.wait_for_selector(".stats-content .totalstats", timeout=10000)
                except Exception:
                    await asyncio.sleep(2)

                raw = await page.evaluate(EXTRACT_ALL_JS)
                data = json.loads(raw)

                if not data.get("team1") and not data.get("team2"):
                    retries += 1
                    if retries < 3:
                        await asyncio.sleep(3)
                    continue

                # --- Match details ---
                map_names = [m["map"] for m in data.get("maps", [])]
                n_maps = len(map_names)
                if n_maps == 1:
                    bo = "BO1"
                elif n_maps == 2:
                    bo = "BO3"
                elif n_maps == 3:
                    s1 = int(orig_row.get("score1", 0) or 0)
                    s2 = int(orig_row.get("score2", 0) or 0)
                    bo = "BO5" if s1 + s2 > 3 else "BO3"
                elif n_maps >= 4:
                    bo = "BO5"
                else:
                    bo = "BO1"

                match_detail = {
                    "match_id": match_id,
                    "match_url": url,
                    "team1": data.get("team1", ""),
                    "team2": data.get("team2", ""),
                    "score1": orig_row.get("score1", ""),
                    "score2": orig_row.get("score2", ""),
                    "maps_played": n_maps,
                    "map_names": "|".join(map_names),
                    "bo_format": bo,
                    "event": data.get("event", ""),
                    "date": data.get("date", ""),
                    "winner": orig_row.get("winner", ""),
                }
                append_csv(MATCH_DETAILS_CSV, [match_detail], MATCH_DETAILS_FIELDS)

                # --- Player stats (overall) ---
                player_rows = []
                for p in data.get("players", []):
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
                        "date": data.get("date", ""),
                        "event": data.get("event", ""),
                    })
                if player_rows:
                    append_csv(PLAYER_STATS_CSV, player_rows, PLAYER_STATS_FIELDS)

                # --- Veto data ---
                veto_rows = []
                for v in data.get("vetos", []):
                    veto_rows.append({
                        "match_id": match_id,
                        "veto_order": v["order"],
                        "team": v["team"],
                        "action": v["action"],
                        "map_name": v["map"],
                    })
                if veto_rows:
                    append_csv(VETO_CSV, veto_rows, VETO_FIELDS)

                # --- Half scores ---
                half_rows = []
                maps_list = data.get("maps", [])
                for hs in data.get("halfScores", []):
                    mi = hs.get("mapIdx", 0)
                    nums = hs.get("numbers", [])
                    map_name = maps_list[mi]["map"] if mi < len(maps_list) else ""
                    # Parse half score numbers
                    # Typical: [ct1, t1, ct2, t2] or [ct1, t1, ot1, ct2, t2, ot2]
                    vals = [n["val"] for n in nums]
                    if len(vals) >= 4:
                        if len(vals) == 4:
                            t1_ct, t1_t, t2_ct, t2_t = vals[0], vals[1], vals[2], vals[3]
                            t1_ot, t2_ot = 0, 0
                        elif len(vals) == 6:
                            t1_ct, t1_t, t1_ot = vals[0], vals[1], vals[2]
                            t2_ct, t2_t, t2_ot = vals[3], vals[4], vals[5]
                        else:
                            # Best effort
                            mid_pt = len(vals) // 2
                            t1_ct = vals[0] if len(vals) > 0 else 0
                            t1_t = vals[1] if len(vals) > 1 else 0
                            t1_ot = vals[2] if mid_pt > 2 else 0
                            t2_ct = vals[mid_pt] if len(vals) > mid_pt else 0
                            t2_t = vals[mid_pt + 1] if len(vals) > mid_pt + 1 else 0
                            t2_ot = vals[mid_pt + 2] if len(vals) > mid_pt + 2 else 0

                        t1_total = t1_ct + t1_t + t1_ot
                        t2_total = t2_ct + t2_t + t2_ot

                        half_rows.append({
                            "match_id": match_id,
                            "map_number": mi + 1,
                            "map_name": map_name,
                            "team1": data.get("team1", ""),
                            "team1_ct_half": t1_ct,
                            "team1_t_half": t1_t,
                            "team1_ot": t1_ot,
                            "team2": data.get("team2", ""),
                            "team2_ct_half": t2_ct,
                            "team2_t_half": t2_t,
                            "team2_ot": t2_ot,
                            "team1_total": t1_total,
                            "team2_total": t2_total,
                        })
                if half_rows:
                    append_csv(HALF_SCORES_CSV, half_rows, HALF_SCORES_FIELDS)

                # --- Per-map player stats ---
                map_p_rows = []
                for mps in data.get("mapPlayerStats", []):
                    mi = mps.get("mapIdx", 1)  # 1-based
                    map_name = maps_list[mi - 1]["map"] if (mi - 1) < len(maps_list) else ""
                    map_p_rows.append({
                        "match_id": match_id,
                        "map_number": mi,
                        "map_name": map_name,
                        "team": mps["team"],
                        "player": mps["player"],
                        "kills": mps["kills"],
                        "deaths": mps["deaths"],
                        "adr": mps["adr"],
                        "kast": mps["kast"],
                        "rating": mps["rating"],
                    })
                if map_p_rows:
                    append_csv(MAP_PLAYER_STATS_CSV, map_p_rows, MAP_PLAYER_STATS_FIELDS)

                done += 1
                success = True

                if done % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed * 60
                    print(f"  {done}/{len(work)} done, {errors} errors "
                          f"({rate:.0f}/min, {elapsed/60:.1f}min elapsed)")

            except Exception as e:
                retries += 1
                if retries < 3:
                    await asyncio.sleep(5)
                else:
                    errors += 1
                    if errors <= 30:
                        print(f"  Failed: {url}: {e}")

        await asyncio.sleep(1.5)

    elapsed = time.time() - start_time
    print(f"\nDone! {done} scraped, {errors} errors in {elapsed/60:.1f} min")

    # Summary
    for label, path in [("Match details", MATCH_DETAILS_CSV),
                        ("Player stats", PLAYER_STATS_CSV),
                        ("Veto data", VETO_CSV),
                        ("Half scores", HALF_SCORES_CSV),
                        ("Map player stats", MAP_PLAYER_STATS_CSV)]:
        if os.path.exists(path):
            with open(path) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {label}: {n} rows")


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
        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        await run_scraper(page, args)


def main():
    parser = argparse.ArgumentParser(description="Scrape all HLTV match data via CDP")
    parser.add_argument("--port", type=int, default=None, help="Chrome CDP port")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
