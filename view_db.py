import sqlite3
import pandas as pd

DB_PATH = r"C:\Users\SAMARTH\Desktop\InsuranceEnd2EndProject\data\claims.db"

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("SELECT * FROM claims", conn)
print(df.columns)

conn.close()
