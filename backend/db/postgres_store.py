import os
import psycopg2
from contextlib import contextmanager
from dotenv import load_dotenv 
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# ============================================================
# CONNECTION
# ============================================================
@contextmanager
def db_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ============================================================
# INIT TABLES
# ============================================================
def init_db():
    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            transaction_id TEXT PRIMARY KEY,
            claim_id TEXT,
            customer_name TEXT,
            policy_number TEXT,
            amount FLOAT,
            claim_type TEXT,
            extracted_text TEXT,
            document_extracted_text TEXT,
            registered_at TEXT,
            status TEXT,
            updated_at TEXT,
            claim_registered BOOLEAN DEFAULT FALSE,
            claim_validated BOOLEAN DEFAULT FALSE,
            fraud_checked BOOLEAN DEFAULT FALSE,
            fraud_score FLOAT,
            fraud_decision TEXT,
            claim_decision_made BOOLEAN DEFAULT FALSE,
            claim_approved BOOLEAN DEFAULT FALSE,
            payment_processed BOOLEAN DEFAULT FALSE,
            claim_closed BOOLEAN DEFAULT FALSE,
            final_decision TEXT,
            validation JSONB,
            assignment JSONB,
            logs JSONB,
            investigator_id TEXT,
            manager_comment TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS claim_documents (
            id SERIAL PRIMARY KEY,
            transaction_id TEXT,
            filename TEXT,
            content_type TEXT,
            size_bytes INTEGER,
            doc_type TEXT,
            extracted_text TEXT
        );
        """)

# ============================================================
# CLAIM UPSERT
# ============================================================
def upsert_claim_registration(**kwargs):
    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO claims (
          transaction_id, claim_id, customer_name,
          policy_number, amount, claim_type,
          extracted_text, registered_at,
          claim_registered, status
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,TRUE,'REGISTERED')
        ON CONFLICT (transaction_id)
        DO UPDATE SET
          claim_id=EXCLUDED.claim_id,
          customer_name=EXCLUDED.customer_name,
          policy_number=EXCLUDED.policy_number,
          amount=EXCLUDED.amount,
          claim_type=EXCLUDED.claim_type,
          extracted_text=EXCLUDED.extracted_text;
        """, (
          kwargs["transaction_id"],
          kwargs["claim_id"],
          kwargs.get("customer_name"),
          kwargs.get("policy_number"),
          kwargs.get("amount"),
          kwargs.get("claim_type"),
          kwargs.get("extracted_text"),
          kwargs["registered_at"]
        ))

# ============================================================
# INSERT DOCUMENTS
# ============================================================
def insert_documents(transaction_id: str, docs: list[dict]):
    if not docs:
        return

    with db_conn() as conn:
        cur = conn.cursor()

        for d in docs:
            cur.execute("""
            INSERT INTO claim_documents (
              transaction_id, filename, content_type,
              size_bytes, doc_type, extracted_text
            ) VALUES (%s,%s,%s,%s,%s,%s)
            """, (
                transaction_id,
                d.get("filename"),
                d.get("content_type"),
                d.get("size_bytes"),
                d.get("doc_type"),
                d.get("extracted_text")
            ))

# ============================================================
# FETCH CLAIM
# ============================================================
def fetch_claim_and_docs(transaction_id: str):
    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT * FROM claims WHERE transaction_id=%s", (transaction_id,))
        claim = cur.fetchone()

        if not claim:
            return None, []

        cols = [desc[0] for desc in cur.description]
        claim_dict = dict(zip(cols, claim))

        cur.execute("""
            SELECT filename, content_type,
                   size_bytes, doc_type, extracted_text
            FROM claim_documents
            WHERE transaction_id=%s
        """, (transaction_id,))

        docs = cur.fetchall()
        doc_cols = [desc[0] for desc in cur.description]
        docs_list = [dict(zip(doc_cols, d)) for d in docs]

        return claim_dict, docs_list

# ============================================================
# UPDATE CLAIM
# ============================================================
def update_claim_fields(transaction_id: str, **fields):
    if not fields:
        return

    set_clause = ", ".join([f"{k}=%s" for k in fields.keys()])
    values = list(fields.values())
    values.append(transaction_id)

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE claims SET {set_clause} WHERE transaction_id=%s",
            values
        )