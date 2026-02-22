import os
import time
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Optional

import psycopg2
from psycopg2 import sql

# Optional: load .env in local/dev. In cloud, env is injected by platform.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────

# Global flags/state
_TABLES_READY = False

def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    return v if v is not None else (default if default is not None else "")

def _build_dsn(base_url: Optional[str] = None, connect_timeout_s: Optional[int] = None) -> str:
    """
    Build a safe PostgreSQL DSN for psycopg2, enforcing:
      - sslmode=require (needed for Supabase)
      - connect_timeout (small by default to avoid long hangs)
    """
    url = base_url or _get_env("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")

    # Force sslmode=require if missing
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    # Ensure a connect timeout (seconds)
    if connect_timeout_s is None:
        # Default small timeout; can be overridden by DB_CONNECT_TIMEOUT
        connect_timeout_s = int(_get_env("DB_CONNECT_TIMEOUT", "5"))

    if "connect_timeout=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}connect_timeout={connect_timeout_s}"

    return url

def _connect_with_retries(
    retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    connect_timeout_s: Optional[int] = None
):
    """
    Create a psycopg2 connection with retry + backoff.
    """
    max_retries = int(_get_env("DB_INIT_RETRIES", "5")) if retries is None else retries
    delay = float(_get_env("DB_INIT_DELAY", "0.5")) if initial_delay is None else initial_delay

    dsn = _build_dsn(connect_timeout_s=connect_timeout_s)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # Note: libpq honors PGHOSTADDR env (if you set it at runtime)
            # so you can prefer IPv4 by exporting PGHOSTADDR=0.0.0.0 or the server's A record.
            conn = psycopg2.connect(dsn)
            return conn
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 5.0)
    # Exhausted retries
    raise last_err

# ─────────────────────────────────────────────────────────────
# Lazy connection manager
# ─────────────────────────────────────────────────────────────

@contextmanager
def db_conn():
    """
    Get a short-lived connection with retries/backoff.
    Commits if the block succeeds; rolls back on error.
    """
    conn = _connect_with_retries()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
# Schema management (lazy)
# ─────────────────────────────────────────────────────────────

def _ensure_tables(conn) -> None:
    """
    Create tables if they don't exist. Idempotent; safe to call more than once.
    """
    global _TABLES_READY
    if _TABLES_READY:
        return

    with conn.cursor() as cur:
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

    _TABLES_READY = True

def init_db(non_blocking: bool = False) -> None:
    """
    Prepare DB layer.
    - If non_blocking=True: do NOT attempt any network connection; if DATABASE_URL is missing, return quietly.
    - If non_blocking=False: connect once and ensure tables (with retries).
    """
    if non_blocking:
        # In non-blocking mode, we must never raise if DATABASE_URL is absent.
        try:
            _ = _build_dsn()
        except Exception:
            # No DATABASE_URL set → skip silently so startup/inspection won't fail.
            return
        # DSN is fine; still skip real connection in non-blocking mode.
        return

    # Blocking path: require a valid DSN and connect
    _ = _build_dsn()                 # will raise if DATABASE_URL missing/malformed
    conn = _connect_with_retries()
    try:
        _ensure_tables(conn)
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _ensure_tables_if_needed(conn) -> None:
    """
    Lazily ensure tables exist on first real DB use.
    """
    if not _TABLES_READY:
        _ensure_tables(conn)

# ─────────────────────────────────────────────────────────────
# Health ping (for /readyz)
# ─────────────────────────────────────────────────────────────

def ping_db(timeout_seconds: float = 2.0) -> bool:
    """
    Quick readiness probe:
      - Tries to connect with a small connect_timeout
      - Executes SELECT 1
    Returns True if reachable; False otherwise.
    """
    try:
        conn = _connect_with_retries(retries=1, initial_delay=0.1, connect_timeout_s=int(timeout_seconds))
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────
# Public CRUD API (unchanged signatures)
# ─────────────────────────────────────────────────────────────

def upsert_claim_registration(**kwargs):
    """
    Upsert minimal registration row.
    Expects keys:
      - transaction_id, claim_id, registered_at
      - customer_name, policy_number, amount, claim_type
      - extracted_text
    """
    with db_conn() as conn:
        _ensure_tables_if_needed(conn)
        with conn.cursor() as cur:
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

def insert_documents(transaction_id: str, docs: List[Dict[str, Any]]):
    if not docs:
        return
    with db_conn() as conn:
        _ensure_tables_if_needed(conn)
        with conn.cursor() as cur:
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
                    d.get("extracted_text"),
                ))

def fetch_claim_and_docs(transaction_id: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    with db_conn() as conn:
        _ensure_tables_if_needed(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM claims WHERE transaction_id=%s", (transaction_id,))
            row = cur.fetchone()
            if not row:
                return None, []

            cols = [desc[0] for desc in cur.description]
            claim_dict = dict(zip(cols, row))

        with conn.cursor() as cur:
            cur.execute("""
                SELECT filename, content_type, size_bytes, doc_type, extracted_text
                FROM claim_documents
                WHERE transaction_id=%s
            """, (transaction_id,))
            docs_rows = cur.fetchall()
            doc_cols = [desc[0] for desc in cur.description]
            docs_list = [dict(zip(doc_cols, r)) for r in docs_rows]

    return claim_dict, docs_list

def update_claim_fields(transaction_id: str, **fields):
    if not fields:
        return
    with db_conn() as conn:
        _ensure_tables_if_needed(conn)
        with conn.cursor() as cur:
            set_cols = []
            values = []
            for k, v in fields.items():
                set_cols.append(sql.Identifier(k))
                values.append(v)

            # Build: UPDATE claims SET col1=%s, col2=%s ... WHERE transaction_id=%s
            assignments = sql.SQL(", ").join(
                sql.Composed([col, sql.SQL("=%s")]) for col in set_cols
            )
            query = sql.SQL("UPDATE claims SET {assignments} WHERE transaction_id=%s").format(
                assignments=assignments
            )
            cur.execute(query, (*values, transaction_id))