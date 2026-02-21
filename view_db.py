# view_db.py
import argparse
import sqlite3
from pathlib import Path
from textwrap import shorten

DB_PATH_DEFAULT = Path("backend/db/claims.sqlite")  # <-- adjust if your DB lives elsewhere

def connect(db_path: Path):
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con

def print_columns(cur):
    cur.execute("PRAGMA table_info(claims)")
    cols = [r["name"] for r in cur.fetchall()]
    print("Columns:", cols)

def show_latest(cur, limit: int):
    sql = """
    SELECT
      transaction_id,
      claim_id,
      status,
      datetime(updated_at) as updated_at,
      LENGTH(COALESCE(document_extracted_text,'')) AS ocr_len,
      SUBSTR(COALESCE(document_extracted_text,''), 1, 300) AS ocr_preview
    FROM claims
    ORDER BY datetime(updated_at) DESC, rowid DESC
    LIMIT ?
    """
    for row in cur.execute(sql, (limit,)):
        preview = row["ocr_preview"].replace("\n", " ")
        preview = shorten(preview, width=200, placeholder="...")
        print(f"- txn={row['transaction_id']} | claim={row['claim_id']} | status={row['status']} | "
              f"ocr_len={row['ocr_len']} | updated_at={row['updated_at']}\n  preview: {preview}")

def show_one(cur, txn: str):
    sql = """
    SELECT
      transaction_id, claim_id, status, datetime(updated_at) as updated_at,
      LENGTH(COALESCE(document_extracted_text,'')) AS ocr_len,
      SUBSTR(COALESCE(document_extracted_text,''), 1, 1000) AS ocr_preview,
      validation
    FROM claims
    WHERE transaction_id = ?
    """
    row = cur.execute(sql, (txn,)).fetchone()
    if not row:
        print("No row for transaction_id:", txn)
        return
    print(f"txn={row['transaction_id']} | claim={row['claim_id']} | status={row['status']} "
          f"| ocr_len={row['ocr_len']} | updated_at={row['updated_at']}")
    print("validation (raw JSON/string):", row["validation"])
    print("\n--- document_extracted_text (first 1000 chars) ---\n")
    print((row["ocr_preview"] or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=str(DB_PATH_DEFAULT), help="Path to sqlite db")
    ap.add_argument("--txn", type=str, help="Transaction ID to inspect")
    ap.add_argument("-n", "--limit", type=int, default=10, help="How many latest rows to show")
    args = ap.parse_args()

    con = connect(Path(args.db))
    cur = con.cursor()
    print_columns(cur)
    if args.txn:
        show_one(cur, args.txn)
    else:
        show_latest(cur, args.limit)

if __name__ == "__main__":
    main()