import sqlite3

DB_PATH = r"C:\Users\SAMARTH\Desktop\InsuranceEnd2EndProject\data\claims.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n[DB] Checking existing columns...\n")

cursor.execute("PRAGMA table_info(claims);")
columns = [col[1] for col in cursor.fetchall()]

if "document_extracted_text" not in columns:
    cursor.execute(
        "ALTER TABLE claims ADD COLUMN document_extracted_text TEXT"
    )
    print("[DB] document_extracted_text column added successfully ✅")
else:
    print("[DB] Column already exists ⚠️")

conn.commit()
conn.close()

print("\n[DB] Migration completed 🚀")
