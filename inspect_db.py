# migrate_db.py
import sqlite3
import os

# -----------------------------
# DB path (adjust if needed)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "claims.db")

# -----------------------------
# Connect to DB
# -----------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# -----------------------------
# Helper to check existing columns
# -----------------------------
def get_columns(table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

# -----------------------------
# Columns we want to add
# -----------------------------
new_columns = {
    "final_decision": "TEXT",
    "updated_at": "TEXT",
    "fraud_score": "REAL",
    "fraud_decision": "TEXT",
    "claim_validated": "INTEGER",
    "manager_comment": "TEXT",
    "assignment": "TEXT",
    "manager_agent": "TEXT",
    "investigator_agent": "TEXT"
}

existing_cols = get_columns("claims")

for col, col_type in new_columns.items():
    if col not in existing_cols:
        print(f"Adding column: {col}")
        cursor.execute(f"ALTER TABLE claims ADD COLUMN {col} {col_type};")
    else:
        print(f"Column already exists: {col}")

conn.commit()
conn.close()
print("DB migration completed ✅")
