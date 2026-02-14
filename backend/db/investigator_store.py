import os, sqlite3
from contextlib import contextmanager

# -----------------------------
# Stable absolute DB path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

DB_PATH = os.getenv(
    "INVESTIGATOR_DB_PATH",
    os.path.join(PROJECT_ROOT, "data", "investigators.db")
)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ============================================================
# CONNECTION
# ============================================================

def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
# TABLE HELPERS
# ============================================================

def _table_columns(conn, table: str) -> set[str]:
    cols = set()
    for row in conn.execute(f"PRAGMA table_info({table});").fetchall():
        cols.add(row[1])
    return cols


def _ensure_extra_columns(conn):
    cols = _table_columns(conn, "investigators")
    add = []

    if "active_cases" not in cols:
        add.append("ALTER TABLE investigators ADD COLUMN active_cases INTEGER DEFAULT 0;")

    if "max_cases" not in cols:
        add.append("ALTER TABLE investigators ADD COLUMN max_cases INTEGER DEFAULT 5;")

    if "status" not in cols:
        add.append("ALTER TABLE investigators ADD COLUMN status TEXT DEFAULT 'ACTIVE';")

    for stmt in add:
        conn.execute(stmt)


# ============================================================
# INIT DB
# ============================================================

def init_investigator_db():
    print(f"[INVESTIGATOR DB] Using: {DB_PATH}")

    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS investigators (
            investigator_id TEXT PRIMARY KEY,
            name TEXT,
            specialization TEXT,
            active_cases INTEGER DEFAULT 0,
            max_cases INTEGER DEFAULT 5,
            status TEXT DEFAULT 'ACTIVE'
        );
        """)

        _ensure_extra_columns(conn)

        seed_investigators(conn)


# ============================================================
# SEED DATA
# ============================================================

def seed_investigators(conn):
    investigators = [
        ("INV001", "Ravi Kumar", "motor", 1, 5, "ACTIVE"),
        ("INV002", "Sneha Reddy", "health", 2, 5, "ACTIVE"),
        ("INV003", "Arjun Mehta", "motor", 0, 3, "ACTIVE"),
        ("INV004", "Priya Sharma", "health", 3, 4, "ACTIVE"),
        ("INV005", "Suresh Iyer", "fraud", 1, 2, "ACTIVE"),
        ("INV006", "Kiran Rao", "motor", 2, 5, "ACTIVE"),
        ("INV007", "Meena Das", "health", 0, 5, "ACTIVE"),
        ("INV008", "Rahul Verma", "fraud", 0, 3, "ACTIVE"),
        ("INV009", "Anita Singh", "motor", 4, 5, "ACTIVE"),
        ("INV010", "Vikram Patel", "health", 1, 4, "INACTIVE"),
    ]

    for inv in investigators:
        conn.execute("""
        INSERT OR IGNORE INTO investigators
        (investigator_id, name, specialization, active_cases, max_cases, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """, inv)


# ============================================================
# ASSIGNMENT LOGIC
# ============================================================

def get_available_investigator(claim_type: str):
    with db_conn() as conn:
        row = conn.execute("""
            SELECT investigator_id
            FROM investigators
            WHERE specialization = ?
            AND status = 'ACTIVE'
            AND active_cases < max_cases
            ORDER BY active_cases ASC
            LIMIT 1
        """, (claim_type,)).fetchone()

        return row["investigator_id"] if row else None


def increment_investigator_load(investigator_id: str):
    with db_conn() as conn:
        conn.execute("""
            UPDATE investigators
            SET active_cases = active_cases + 1
            WHERE investigator_id = ?
        """, (investigator_id,))


def decrement_investigator_load(investigator_id: str):
    with db_conn() as conn:
        conn.execute("""
            UPDATE investigators
            SET active_cases =
                CASE
                    WHEN active_cases > 0 THEN active_cases - 1
                    ELSE 0
                END
            WHERE investigator_id = ?
        """, (investigator_id,))
