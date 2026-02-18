"""Process a batch of scraped match data from JSON and append to all 5 CSVs."""
import json
import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

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
VETO_FIELDS = ["match_id", "veto_order", "team", "action", "map_name"]
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


def append_csv(path, rows, fieldnames):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def load_orig_rows():
    """Load original match data for score/winner info."""
    rows = {}
    with open(os.path.join(DATA_DIR, "all_results_with_urls.csv")) as f:
        reader = csv.DictReader(f)
        for r in reader:
            url = r.get("match_url", "")
            if url and "/matches/" in url:
                mid = url.split("/matches/")[1].split("/")[0]
                rows[mid] = r
    return rows


def process_batch(json_path):
    with open(json_path) as f:
        batch = json.load(f)

    orig = load_orig_rows()
    counts = {"details": 0, "players": 0, "vetos": 0, "halfs": 0, "map_players": 0, "skipped": 0}

    for item in batch:
        match_id = str(item["match_id"])
        url = item["url"]
        data = item.get("data")

        if not data or (not data.get("team1") and not data.get("team2")):
            counts["skipped"] += 1
            continue

        orig_row = orig.get(match_id, {})
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

        detail = {
            "match_id": match_id, "match_url": url,
            "team1": data.get("team1", ""), "team2": data.get("team2", ""),
            "score1": orig_row.get("score1", ""), "score2": orig_row.get("score2", ""),
            "maps_played": n_maps, "map_names": "|".join(map_names),
            "bo_format": bo, "event": data.get("event", ""),
            "date": data.get("date", ""), "winner": orig_row.get("winner", ""),
        }
        append_csv(MATCH_DETAILS_CSV, [detail], MATCH_DETAILS_FIELDS)
        counts["details"] += 1

        for p in data.get("players", []):
            append_csv(PLAYER_STATS_CSV, [{
                "match_id": match_id, "match_url": url,
                "team": p["team"], "player": p["player"],
                "kills": p["kills"], "deaths": p["deaths"],
                "adr": p["adr"], "kast": p["kast"], "rating": p["rating"],
                "date": data.get("date", ""), "event": data.get("event", ""),
            }], PLAYER_STATS_FIELDS)
            counts["players"] += 1

        for v in data.get("vetos", []):
            append_csv(VETO_CSV, [{
                "match_id": match_id, "veto_order": v["order"],
                "team": v["team"], "action": v["action"], "map_name": v["map"],
            }], VETO_FIELDS)
            counts["vetos"] += 1

        maps_list = data.get("maps", [])
        for hs in data.get("halfScores", []):
            mi = hs.get("mapIdx", 0)
            nums = hs.get("numbers", [])
            map_name = maps_list[mi]["map"] if mi < len(maps_list) else ""
            vals = [n["val"] for n in nums]
            if len(vals) >= 4:
                if len(vals) == 4:
                    t1_ct, t1_t, t2_ct, t2_t = vals[0], vals[1], vals[2], vals[3]
                    t1_ot, t2_ot = 0, 0
                elif len(vals) == 6:
                    t1_ct, t1_t, t1_ot = vals[0], vals[1], vals[2]
                    t2_ct, t2_t, t2_ot = vals[3], vals[4], vals[5]
                else:
                    mid_pt = len(vals) // 2
                    t1_ct = vals[0]
                    t1_t = vals[1] if len(vals) > 1 else 0
                    t1_ot = vals[2] if mid_pt > 2 else 0
                    t2_ct = vals[mid_pt] if len(vals) > mid_pt else 0
                    t2_t = vals[mid_pt + 1] if len(vals) > mid_pt + 1 else 0
                    t2_ot = vals[mid_pt + 2] if len(vals) > mid_pt + 2 else 0
                t1_total = t1_ct + t1_t + t1_ot
                t2_total = t2_ct + t2_t + t2_ot
                append_csv(HALF_SCORES_CSV, [{
                    "match_id": match_id, "map_number": mi + 1, "map_name": map_name,
                    "team1": data.get("team1", ""),
                    "team1_ct_half": t1_ct, "team1_t_half": t1_t, "team1_ot": t1_ot,
                    "team2": data.get("team2", ""),
                    "team2_ct_half": t2_ct, "team2_t_half": t2_t, "team2_ot": t2_ot,
                    "team1_total": t1_total, "team2_total": t2_total,
                }], HALF_SCORES_FIELDS)
                counts["halfs"] += 1

        for mps in data.get("mapPlayerStats", []):
            mi = mps.get("mapIdx", 1)
            map_name = maps_list[mi - 1]["map"] if (mi - 1) < len(maps_list) else ""
            append_csv(MAP_PLAYER_STATS_CSV, [{
                "match_id": match_id, "map_number": mi, "map_name": map_name,
                "team": mps["team"], "player": mps["player"],
                "kills": mps["kills"], "deaths": mps["deaths"],
                "adr": mps["adr"], "kast": mps["kast"], "rating": mps["rating"],
            }], MAP_PLAYER_STATS_FIELDS)
            counts["map_players"] += 1

    print(f"Processed: {counts['details']} matches, {counts['players']} players, "
          f"{counts['vetos']} vetos, {counts['halfs']} halfs, {counts['map_players']} map_players, "
          f"{counts['skipped']} skipped")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_batch.py <batch_json>")
        sys.exit(1)
    process_batch(sys.argv[1])
