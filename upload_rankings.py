"""
Upload rankings from data/rankings_history.csv to Supabase.

Reads the CSV and upserts all rows in batches of 500.
Safe to run multiple times (upsert on date+team).

Usage:
  ./venv/bin/python upload_rankings.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from website.database import upsert_rankings

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "rankings_history.csv")
BATCH_SIZE = 500


def main():
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "date": r["date"],
                "team": r["team"],
                "rank": int(r["rank"]),
                "points": int(r["points"]),
            }
            for r in reader
        ]

    print(f"Total rows: {len(rows)}")

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        upsert_rankings(batch)
        print(f"  Upserted {i + len(batch)}/{len(rows)}")

    print("Done.")


if __name__ == "__main__":
    main()
