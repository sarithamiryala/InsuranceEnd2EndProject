# migrate_add_manager_decision.py
import sqlite3

# Absolute path to your DB
DB_PATH = r"C:\Users\SAMARTH\Desktop\End2EndInsuranceClaim\data\claims.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Column to add
column_name = "manager_decision"
column_type = "TEXT"

# Check if column already exists
cursor.execute("PRAGMA table_info(claims);")
existing_columns = [col[1] for col in cursor.fetchall()]

if column_name not in existing_columns:
    cursor.execute(f"ALTER TABLE claims ADD COLUMN {column_name} {column_type}")
    print(f"[DB] Added column: {column_name} ({column_type})")
else:
    print(f"[DB] Column already exists: {column_name}")

conn.commit()
conn.close()
print("[DB] Migration complete.")
