"""
Standalone HLTV scraper for the CS2 prediction website.

Connects to a running Chrome instance via CDP to bypass Cloudflare,
scrapes upcoming matches, runs predictions, and stores results in the
SQLite database.

Usage:
  # 1. Launch Chrome with remote debugging enabled:
  #    /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
  #        --remote-debugging-port=9222

  # 2. Scrape upcoming matches and add predictions:
  python -m website.scrape_hltv upcoming

  # 3. Resolve completed matches by scraping results:
  python -m website.scrape_hltv resolve

  # 4. Both in one run:
  python -m website.scrape_hltv all

  # Optional: specify CDP port (default: auto-detect)
  python -m website.scrape_hltv upcoming --port 9222
"""
import asyncio
import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.async_api import async_playwright

from website.config import TRACKER_STATE_PATH
from website.scraper import (
    EXTRACT_UPCOMING_JS, EXTRACT_MATCH_RESULT_JS,
    parse_odds, implied_probability, compute_edge, parse_bo_format,
)
from website import database as db


def _parse_hltv_date(date_str):
    """Parse HLTV date like '21st of February 2026' or 'February 21, 2026' to 'YYYY-MM-DD'."""
    if not date_str:
        return None
    import re
    # Remove ordinal suffixes (st, nd, rd, th)
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str).strip()
    for fmt in ("%d of %B %Y", "%B %d, %Y", "%B %d %Y", "%d %B %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def find_cdp_port():
    """Find Chrome's CDP debugging port from running processes."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "remote-debugging-port=" in line and ("Google Chrome" in line or "chrome" in line.lower()):
                for part in line.split():
                    if part.startswith("--remote-debugging-port="):
                        return int(part.split("=")[1])
    except Exception:
        pass
    return None


# -- Lazy globals for prediction pipeline --
_bundle = None
_predictor_loaded = False


def _get_bundle():
    global _bundle
    if _bundle is None:
        if os.path.exists(TRACKER_STATE_PATH):
            from website.tracker_state import TrackerBundle
            _bundle = TrackerBundle.load()
        else:
            print("ERROR: No tracker state found. Run: python -m website.tracker_state")
            sys.exit(1)
    return _bundle


def _make_prediction(team1, team2, event="", bo_format="BO3"):
    """Build features and run ensemble prediction."""
    bundle = _get_bundle()
    from website.feature_builder import build_match_features
    from website.predictor import predict

    features = build_match_features(
        bundle, team1, team2,
        event=event, bo_format=bo_format,
        match_date=datetime.now(),
    )
    result = predict(features)
    result["predicted_winner"] = team1 if result["predicted_winner"] == 1 else team2
    result["t1_rank"] = bundle.top_teams.get(team1, 101)
    result["t2_rank"] = bundle.top_teams.get(team2, 101)
    return result


async def scrape_upcoming(page):
    """Scrape upcoming matches from HLTV and add predictions to DB."""
    print("\n=== Scraping Upcoming Matches ===\n")

    await page.goto("https://www.hltv.org/matches", timeout=30000, wait_until="domcontentloaded")
    await asyncio.sleep(3)

    # Extract upcoming matches
    raw = await page.evaluate(EXTRACT_UPCOMING_JS)
    matches = json.loads(raw)
    print(f"Found {len(matches)} upcoming matches")

    if not matches:
        print("No upcoming matches found. The page structure may have changed.")
        return 0

    db.init_db()
    added = 0

    for m in matches:
        team1, team2 = m["team1"], m["team2"]
        event = m.get("event", "")
        match_url = m.get("url", f"{team1}-vs-{team2}-{datetime.now().isoformat()}")
        bo_format = parse_bo_format(m.get("meta", ""))
        match_time = m.get("time", "")
        match_date = _parse_hltv_date(m.get("matchDate", ""))

        # Skip if already in DB
        from website.database import _get_client
        existing = (
            _get_client()
            .table("predictions")
            .select("id")
            .eq("match_url", match_url)
            .execute()
        ).data
        if existing:
            continue

        # Run prediction
        try:
            pred = _make_prediction(team1, team2, event, bo_format)
        except Exception as e:
            print(f"  Prediction failed for {team1} vs {team2}: {e}")
            continue

        # Try to scrape odds from match page
        odds_t1, odds_t2 = None, None
        if match_url and match_url.startswith("http"):
            try:
                await page.goto(match_url, timeout=20000, wait_until="domcontentloaded")
                await asyncio.sleep(2)
                odds_raw = await page.evaluate("""
                    () => {
                        const odds = document.querySelectorAll('.provider-odds .odds-cell');
                        if (odds.length >= 2) {
                            return JSON.stringify({
                                t1: odds[0]?.textContent?.trim() || '',
                                t2: odds[1]?.textContent?.trim() || ''
                            });
                        }
                        return '{}';
                    }
                """)
                odds_data = json.loads(odds_raw)
                odds_t1 = parse_odds(odds_data.get("t1"))
                odds_t2 = parse_odds(odds_data.get("t2"))
            except Exception:
                pass

        # Compute edge
        implied_t1 = implied_probability(odds_t1)
        implied_t2 = implied_probability(odds_t2)
        model_prob_winner = pred["t1_win_prob"] if pred["predicted_winner"] == team1 else 1 - pred["t1_win_prob"]
        implied_winner = implied_t1 if pred["predicted_winner"] == team1 else implied_t2
        edge = compute_edge(model_prob_winner, implied_winner)

        data = {
            "match_url": match_url,
            "team1": team1,
            "team2": team2,
            "event": event,
            "bo_format": bo_format,
            "match_time": match_time,
            "match_date": match_date,
            "predicted_winner": pred["predicted_winner"],
            "t1_win_prob": pred["t1_win_prob"],
            "fsvm_prob": pred["fsvm_prob"],
            "xgb_prob": pred["xgb_prob"],
            "models_agree": pred["models_agree"],
            "confidence": pred["confidence"],
            "odds_t1": odds_t1,
            "odds_t2": odds_t2,
            "implied_prob_t1": implied_t1,
            "implied_prob_t2": implied_t2,
            "edge": edge,
            "t1_rank": pred.get("t1_rank"),
            "t2_rank": pred.get("t2_rank"),
        }
        db.insert_prediction(data)
        added += 1
        winner_prob = pred["t1_win_prob"] if pred["predicted_winner"] == team1 else 1 - pred["t1_win_prob"]
        print(f"  + {team1} vs {team2} -> {pred['predicted_winner']} ({winner_prob*100:.1f}%)"
              f"{'  edge: ' + f'{edge*100:.1f}%' if edge else ''}")

    print(f"\nAdded {added} new predictions")

    # Log scrape
    from website.database import _get_client
    _get_client().table("scrape_log").insert({
        "scrape_type": "upcoming",
        "matches_found": len(matches),
        "status": f"added {added}",
    }).execute()
    return added


async def resolve_matches(page):
    """Check unresolved predictions and scrape results for completed matches."""
    print("\n=== Resolving Completed Matches ===\n")

    db.init_db()
    upcoming = db.get_upcoming()
    if not upcoming:
        print("No unresolved predictions to check.")
        return 0

    print(f"Checking {len(upcoming)} unresolved predictions...")
    resolved = 0
    bundle = _get_bundle()

    today = datetime.now().strftime("%Y-%m-%d")
    for p in upcoming:
        match_url = p["match_url"]
        if not match_url or not match_url.startswith("http"):
            continue

        # Skip future matches
        match_date = p.get("match_date")
        if match_date and match_date > today:
            continue

        try:
            await page.goto(match_url, timeout=20000, wait_until="domcontentloaded")
            await asyncio.sleep(2)

            # Check if match is completed (has score elements)
            has_result = await page.evaluate("""
                () => {
                    const won = document.querySelector('.team .won');
                    return !!won;
                }
            """)
            if not has_result:
                continue

            # Extract result
            raw = await page.evaluate(EXTRACT_MATCH_RESULT_JS)
            data = json.loads(raw)

            winner = data.get("winner", "")
            if not winner:
                continue

            # Resolve in DB
            success = db.resolve_prediction(p["id"], winner)
            if success:
                # Update trackers
                match_dict = {
                    "team1": p["team1"],
                    "team2": p["team2"],
                    "winner": winner,
                    "event": p["event"] or "",
                    "bo_format": p["bo_format"] or "BO3",
                    "players": data.get("players", []),
                    "maps": data.get("maps", []),
                }
                bundle.update_with_result(match_dict)

                correct = "CORRECT" if winner == p["predicted_winner"] else "WRONG"
                print(f"  Resolved: {p['team1']} vs {p['team2']} -> {winner} [{correct}]")
                resolved += 1

        except Exception as e:
            print(f"  Failed to check {p['team1']} vs {p['team2']}: {e}")
            continue

        await asyncio.sleep(1.5)

    # Save updated tracker state
    if resolved > 0:
        bundle.save()
        print(f"\nResolved {resolved} matches, tracker state saved.")
    else:
        print("\nNo matches resolved (none completed yet).")

    return resolved


async def main_async(args):
    port = args.port
    if port is None:
        port = find_cdp_port()
        if port is None:
            print("ERROR: Could not find Chrome CDP port.")
            print("Launch Chrome with: --remote-debugging-port=9222")
            print("Or specify port with: --port <port>")
            return

    print(f"Connecting to Chrome CDP on port {port}...")

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        context = browser.contexts[0]
        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        if args.command in ("upcoming", "all"):
            await scrape_upcoming(page)

        if args.command in ("resolve", "all"):
            await resolve_matches(page)


def main():
    parser = argparse.ArgumentParser(description="HLTV scraper for CS2 prediction website")
    parser.add_argument("command", choices=["upcoming", "resolve", "all"],
                        help="upcoming: scrape new matches | resolve: check results | all: both")
    parser.add_argument("--port", type=int, default=None,
                        help="Chrome CDP port (default: auto-detect)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
