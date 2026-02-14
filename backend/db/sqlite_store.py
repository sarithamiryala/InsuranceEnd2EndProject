import os
import sqlite3
import shutil
from contextlib import contextmanager

# -----------------------------
# Stable absolute DB path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Default DB path in repo
REPO_DB_PATH = os.path.join(PROJECT_ROOT, "data", "claims.db")

# Render / env override
DB_PATH = os.getenv("CLAIMS_DB_PATH", REPO_DB_PATH)

# Ensure folder exists for repo path
os.makedirs(os.path.dirname(REPO_DB_PATH), exist_ok=True)

# -----------------------------
# Handle read-only DB (Render-safe)
# -----------------------------
if os.path.exists(DB_PATH) and not os.access(DB_PATH, os.W_OK):
    # Copy to writable temp folder
    tmp_db_path = "/tmp/claims.db"
    if not os.path.exists(tmp_db_path):
        shutil.copy(DB_PATH, tmp_db_path)
    DB_PATH = tmp_db_path
    print(f"[DB] Using temporary writable DB: {DB_PATH}")

else:
    # If DB doesn't exist, create parent folder
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print(f"[DB] Using DB path: {DB_PATH}")


# ============================================================
# CONNECTION
# ============================================================
def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# TABLE COLUMN HELPERS
# ============================================================
def _table_columns(conn, table: str) -> set[str]:
    cols = set()
    for row in conn.execute(f"PRAGMA table_info({table});").fetchall():
        cols.add(row[1])
    return cols


def _ensure_claims_extra_columns(conn):
    cols = _table_columns(conn, "claims")
    add = []

    # Existing extra columns
    if "final_decision" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN final_decision TEXT;")

    if "updated_at" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN updated_at TEXT;")

    if "fraud_score" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN fraud_score REAL;")

    if "fraud_decision" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN fraud_decision TEXT;")

    if "claim_validated" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN claim_validated INTEGER;")

    if "manager_comment" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN manager_comment TEXT;")

    # Investigator assignment columns
    if "investigator_id" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN investigator_id TEXT;")

    if "assignment_reason" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN assignment_reason TEXT;")

    if "assignment_status" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN assignment_status TEXT;")

    if "assigned_at" not in cols:
        add.append("ALTER TABLE claims ADD COLUMN assigned_at TEXT;")

    for stmt in add:
        conn.execute(stmt)


# ============================================================
# INIT DATABASE
# ============================================================
def init_db():
    print(f"[DB] Initializing database: {DB_PATH}")

    with db_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS claims (
            transaction_id TEXT PRIMARY KEY,
            claim_id TEXT,
            customer_name TEXT,
            policy_number TEXT,
            amount REAL,
            claim_type TEXT,
            extracted_text TEXT,
            registered_at TEXT,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS claim_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            filename TEXT,
            content_type TEXT,
            size_bytes INTEGER,
            doc_type TEXT,
            extracted_text TEXT,
            FOREIGN KEY(transaction_id)
                REFERENCES claims(transaction_id)
                ON DELETE CASCADE
        );
        """)

        # Auto-migrate missing columns safely
        _ensure_claims_extra_columns(conn)


# ============================================================
# CLAIM OPERATIONS
# ============================================================
def upsert_claim_registration(**kwargs):
    with db_conn() as conn:
        conn.execute("""
        INSERT INTO claims (
          transaction_id, claim_id, customer_name, policy_number,
          amount, claim_type, extracted_text, registered_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(transaction_id) DO UPDATE SET
          claim_id=excluded.claim_id,
          customer_name=excluded.customer_name,
          policy_number=excluded.policy_number,
          amount=excluded.amount,
          claim_type=excluded.claim_type,
          extracted_text=excluded.extracted_text,
          registered_at=excluded.registered_at,
          status=excluded.status
        """, (
          kwargs["transaction_id"],
          kwargs["claim_id"],
          kwargs.get("customer_name"),
          kwargs.get("policy_number"),
          kwargs.get("amount"),
          kwargs.get("claim_type"),
          kwargs.get("extracted_text"),
          kwargs["registered_at"],
          kwargs.get("status", "REGISTERED")
        ))


def insert_documents(transaction_id: str, docs: list[dict]):
    if not docs:
        return

    rows = []
    for d in docs:
        rows.append((
            transaction_id,
            d.get("filename"),
            d.get("content_type"),
            d.get("size_bytes", 0),
            d.get("doc_type"),
            d.get("extracted_text")
        ))

    with db_conn() as conn:
        conn.executemany("""
        INSERT INTO claim_documents (
          transaction_id, filename, content_type,
          size_bytes, doc_type, extracted_text
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, rows)


def fetch_claim_and_docs(transaction_id: str):
    with db_conn() as conn:
        c = conn.execute(
            "SELECT * FROM claims WHERE transaction_id=?",
            (transaction_id,)
        ).fetchone()

        if not c:
            return None, []

        docs = conn.execute("""
            SELECT id, filename, content_type,
                   size_bytes, doc_type, extracted_text
            FROM claim_documents
            WHERE transaction_id=?
            ORDER BY id ASC
        """, (transaction_id,)).fetchall()

        return dict(c), [dict(d) for d in docs]


def update_claim_fields(transaction_id: str, **fields):
    if not fields:
        return

    cols = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [transaction_id]

    with db_conn() as conn:
        conn.execute(
            f"UPDATE claims SET {cols} WHERE transaction_id=?",
            vals
        )
