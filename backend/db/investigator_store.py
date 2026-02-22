# backend/db/investigator_store.py
from backend.db.postgres_store import db_conn
from datetime import datetime, timezone

def get_available_investigator(claim_type: str):
    """
    Pick the active investigator with least current load for this claim_type
    """ 
    claim_type = (claim_type or "").upper()
    query = """
        SELECT investigator_id, name, current_load, max_cases
        FROM investigators
        WHERE active = TRUE AND claim_type = %s AND current_load < max_cases
        ORDER BY current_load ASC, created_at ASC
        LIMIT 1;
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (claim_type,))
            row = cur.fetchone()
            if row:
                return {"investigator_id": row[0], "name": row[1]}
    return None

def increment_investigator_load(investigator_id: int):
    """
    Increment the current_load of the investigator
    """
    query = """
        UPDATE investigators
        SET current_load = current_load + 1, updated_at = NOW()
        WHERE investigator_id = %s;
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (investigator_id,))

from backend.state.claim_state import Assignment
import json


def record_assignment(transaction_id: str, investigator_id: int, reason: str):
    """
    Record assignment in history table
    AND update claims.assignment JSONB (UI reads from here)
    """

    assignment = Assignment(
        investigator_id=str(investigator_id),
        sla_days=3,
        reason=reason
    )

    insert_query = """
        INSERT INTO investigator_assignments
        (transaction_id, investigator_id, reason, assigned_at)
        VALUES (%s, %s, %s, NOW());
    """

    update_claim_query = """
        UPDATE claims
        SET investigator_id = %s,
            assignment = %s,
            updated_at = NOW()
        WHERE transaction_id = %s;
    """

    with db_conn() as conn:
        with conn.cursor() as cur:

            # 1️⃣ History table
            cur.execute(insert_query, (
                transaction_id,
                investigator_id,
                reason
            ))

            # 2️⃣ MAIN CLAIMS JSONB (THIS WAS MISSING ❌)
            cur.execute(update_claim_query, (
                investigator_id,
                json.dumps(assignment.model_dump()),   # ✅ JSONB safe
                transaction_id
            ))

        conn.commit()