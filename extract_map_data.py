"""
Extract map-level results from batch JSON files into a CSV.

Reads all data/match_batch_*.json files and produces data/map_results.csv.
Skips maps with "-" scores (unplayed maps in BO3/BO5).

Usage:
  python extract_map_data.py
"""
import json
import glob
import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")


def extract_match_id(url):
    """Extract numeric match_id from HLTV URL."""
    if not url:
        return ""
    m = re.search(r"/matches/(\d+)/", url)
    return m.group(1) if m else ""


def extract_map_results():
    batch_files = sorted(glob.glob(os.path.join(DATA_DIR, "match_batch_*.json")))
    if not batch_files:
        print("No match_batch_*.json files found in data/")
        return

    print(f"Found {len(batch_files)} batch files")

    rows = []
    skipped_maps = 0
    skipped_matches = 0

    for fpath in batch_files:
        with open(fpath) as f:
            matches = json.load(f)

        for match in matches:
            url = match.get("url", "")
            match_id = extract_match_id(url)
            if not match_id:
                skipped_matches += 1
                continue

            team1 = match.get("team1", "")
            team2 = match.get("team2", "")
            date = match.get("date", "")
            maps = match.get("maps", [])

            for i, map_entry in enumerate(maps):
                map_name = map_entry.get("map", "")
                s1 = map_entry.get("s1", "-")
                s2 = map_entry.get("s2", "-")

                # Skip unplayed maps
                if s1 == "-" or s2 == "-":
                    skipped_maps += 1
                    continue

                try:
                    score_t1 = int(s1)
                    score_t2 = int(s2)
                except (ValueError, TypeError):
                    skipped_maps += 1
                    continue

                map_winner = team1 if score_t1 > score_t2 else team2

                rows.append({
                    "match_id": match_id,
                    "team1": team1,
                    "team2": team2,
                    "map_name": map_name,
                    "score_t1": score_t1,
                    "score_t2": score_t2,
                    "map_winner": map_winner,
                    "map_number": i + 1,
                    "date": date,
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "map_results.csv")
    df.to_csv(out_path, index=False)

    print(f"Extracted {len(df)} map results from {len(batch_files)} batch files")
    print(f"Skipped {skipped_maps} unplayed maps, {skipped_matches} matches without IDs")
    print(f"Unique maps: {df['map_name'].nunique()} — {df['map_name'].value_counts().to_dict()}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    extract_map_results()
