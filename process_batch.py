"""Process match detail batch JSON files into CSVs."""
import json
import csv
import sys
import os
import pandas as pd

MATCH_FIELDS = ["match_id","match_url","team1","team2","score1","score2","maps_played","map_names","bo_format","event","date","winner"]
PLAYER_FIELDS = ["match_id","match_url","team","player","kills","deaths","adr","kast","rating","date","event"]

def extract_match_id(url):
    try: return url.split("/matches/")[1].split("/")[0]
    except: return ""

def safe_int(s, default=0):
    try: return int(s)
    except: return default

def process_mcp_output(mcp_path, batch_json_path):
    """Extract match data from MCP output file and save as batch JSON."""
    with open(mcp_path) as f:
        raw = json.load(f)
    for item in raw:
        text = item.get('text', '')
        if '### Result' in text:
            lines = text.split('\n')
            for line in lines:
                if line.startswith('"') and 'matches' in line:
                    data = json.loads(json.loads(line))
                    with open(batch_json_path, 'w') as out:
                        json.dump(data['matches'], out)
                    return data['count'], data.get('errors', 0)
    return 0, 0

def process_match_batch(batch_path):
    """Append match batch JSON to CSVs."""
    with open(batch_path) as f:
        matches = json.load(f)

    n_m, n_p = 0, 0
    with open('data/match_details.csv', 'a', newline='') as mf, \
         open('data/player_stats.csv', 'a', newline='') as pf:
        mw = csv.DictWriter(mf, fieldnames=MATCH_FIELDS)
        pw = csv.DictWriter(pf, fieldnames=PLAYER_FIELDS)

        for m in matches:
            if not m.get('players'): continue
            mid = extract_match_id(m['url'])
            maps = m.get('maps', [])
            map_names = [mp['map'] for mp in maps]
            n_maps = len(map_names)
            if n_maps <= 1: bo = "BO1"
            elif n_maps == 2: bo = "BO3"
            elif n_maps == 3: bo = "BO3"
            else: bo = "BO5"

            t1_wins = sum(1 for mp in maps if safe_int(mp.get('s1',0)) > safe_int(mp.get('s2',0)))
            winner = m['team1'] if t1_wins > n_maps / 2 else m['team2']

            mw.writerow({"match_id": mid, "match_url": m['url'], "team1": m['team1'], "team2": m['team2'],
                "score1": t1_wins, "score2": n_maps - t1_wins, "maps_played": n_maps, "map_names": "|".join(map_names),
                "bo_format": bo, "event": m.get('event',''), "date": m.get('date',''), "winner": winner})
            n_m += 1
            for p in m['players']:
                pw.writerow({"match_id": mid, "match_url": m['url'], "team": p['team'], "player": p['player'],
                    "kills": p['kills'], "deaths": p['deaths'], "adr": p['adr'], "kast": p['kast'],
                    "rating": p['rating'], "date": m.get('date',''), "event": m.get('event','')})
                n_p += 1
    return n_m, n_p

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_batch.py <mcp_output_path> <batch_num>")
        sys.exit(1)
    mcp_path = sys.argv[1]
    batch_num = sys.argv[2] if len(sys.argv) > 2 else "X"
    batch_json = f'data/match_batch_{batch_num}.json'

    count, errors = process_mcp_output(mcp_path, batch_json)
    print(f"Extracted batch {batch_num}: {count} matches, {errors} errors")

    nm, np_ = process_match_batch(batch_json)
    print(f"Appended: {nm} matches, {np_} players")

    md = pd.read_csv('data/match_details.csv')
    ps = pd.read_csv('data/player_stats.csv')
    print(f"Total: {len(md)} matches, {len(ps)} player rows")
