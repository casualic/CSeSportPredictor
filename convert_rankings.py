"""
Extract rankings data from agent output files and existing rankings_raw.txt,
combine and create rankings_history.csv.
"""
import re
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "rankings_raw.txt")
OUT_CSV = os.path.join(DATA_DIR, "rankings_history.csv")

# Agent output files that contain the scraped data
AGENT_OUTPUT_DIR = "/private/tmp/claude-501/-Users-mateuszdelpercio-Code-Python-CSeSportPredictor/tasks"
AGENT_FILES = ["a7d16f3.output", "aed647d.output", "a81233a.output"]

# Pattern to match pipe-delimited ranking lines
LINE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(.+?)\|(\d+)\|(\d+)$")

def extract_from_file(filepath):
    """Extract pipe-delimited ranking lines from a file."""
    rows = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                # Lines in agent output files have \\n literal newlines in strings
                # Split on literal \n sequences
                parts = line.replace("\\n", "\n").split("\n")
                for part in parts:
                    part = part.strip()
                    m = LINE_PATTERN.match(part)
                    if m:
                        rows.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    except FileNotFoundError:
        print(f"  Not found: {filepath}")
    return rows

def main():
    all_rows = set()

    # 1. Read existing rankings_raw.txt
    print(f"Reading {RAW_FILE}...")
    raw_rows = extract_from_file(RAW_FILE)
    print(f"  Found {len(raw_rows)} rows")
    for r in raw_rows:
        all_rows.add(r)

    # 2. Read agent output files
    for agent_file in AGENT_FILES:
        filepath = os.path.join(AGENT_OUTPUT_DIR, agent_file)
        print(f"Reading {agent_file}...")
        agent_rows = extract_from_file(filepath)
        print(f"  Found {len(agent_rows)} rows")
        for r in agent_rows:
            all_rows.add(r)

    # 3. Sort by date, then rank
    sorted_rows = sorted(all_rows, key=lambda x: (x[0], x[2]))

    # 4. Write CSV
    print(f"\nTotal unique rows: {len(sorted_rows)}")
    dates = sorted(set(r[0] for r in sorted_rows))
    print(f"Unique dates: {len(dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    with open(OUT_CSV, "w") as f:
        f.write("date,team,rank,points\n")
        for date, team, rank, points in sorted_rows:
            # Escape commas in team names
            if "," in team:
                team = f'"{team}"'
            f.write(f"{date},{team},{rank},{points}\n")

    print(f"\nWritten to {OUT_CSV}")

if __name__ == "__main__":
    main()
