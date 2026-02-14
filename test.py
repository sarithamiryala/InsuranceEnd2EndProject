import sqlite3
import os

DB_PATH = os.path.join("data", "investigators.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Show tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:")
for row in cursor.fetchall():
    print(row)

# Show columns of investigators table
cursor.execute("PRAGMA table_info(investigators);")
print("\nColumns in investigators table:")
for col in cursor.fetchall():
    print(col)

conn.close()
