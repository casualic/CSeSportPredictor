"""One-time migration: SQLite predictions.db -> Supabase Postgres."""
import os
import sqlite3
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(os.path.join(os.path.dirname(__file__), "website", ".env"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
DB_PATH = os.path.join(os.path.dirname(__file__), "website", "predictions.db")

BOOL_COLS_PREDICTIONS = {"models_agree", "prediction_correct"}
BOOL_COLS_BETS = {"won"}


def dict_from_row(cursor, row):
    return {desc[0]: row[i] for i, desc in enumerate(cursor.description)}


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"SQLite DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = dict_from_row

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Migrate predictions
    rows = conn.execute("SELECT * FROM predictions ORDER BY id").fetchall()
    print(f"Migrating {len(rows)} predictions...")
    for row in rows:
        row.pop("id", None)  # Let Supabase assign new IDs
        for col in BOOL_COLS_PREDICTIONS:
            if row.get(col) is not None:
                row[col] = bool(row[col])
        client.table("predictions").upsert(row, on_conflict="match_url").execute()
    print(f"  Done: {len(rows)} predictions migrated")

    # Migrate bets
    rows = conn.execute("SELECT * FROM bets ORDER BY id").fetchall()
    print(f"Migrating {len(rows)} bets...")
    for row in rows:
        row.pop("id", None)
        for col in BOOL_COLS_BETS:
            if row.get(col) is not None:
                row[col] = bool(row[col])
        # prediction_id may not match Supabase IDs; skip if no match
        client.table("bets").insert(row).execute()
    print(f"  Done: {len(rows)} bets migrated")

    # Migrate scrape_log
    rows = conn.execute("SELECT * FROM scrape_log ORDER BY id").fetchall()
    print(f"Migrating {len(rows)} scrape log entries...")
    for row in rows:
        row.pop("id", None)
        client.table("scrape_log").insert(row).execute()
    print(f"  Done: {len(rows)} scrape log entries migrated")

    conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
