from backend.utils.logger import logger
from backend.state.claim_state import ClaimState
from datetime import datetime, timezone
import uuid

# NOTE:
# Do NOT import or initialize the DB here at module import time.
# This keeps MCP inspection and app startup from failing when DATABASE_URL is absent.
# We will lazy-import the required DB functions inside the agent.

MAX_TEXT_LEN = 90000

def _aggregate_extracted_text(state: ClaimState) -> str:
    parts = []
    if getattr(state, "extracted_text", None):
        parts.append(state.extracted_text)
    for d in getattr(state, "documents", []) or []:
        if getattr(d, "extracted_text", None):
            parts.append(d.extracted_text)
    combined = "\n\n".join([p.strip() for p in parts if p and p.strip()])
    return combined[:MAX_TEXT_LEN] if combined else ""

def registration_agent(state: ClaimState):
    """
    Registers a claim:
    - Ensures transaction_id and registered_at
    - Aggregates OCR text into state.extracted_text (bounded by MAX_TEXT_LEN)
    - Persists claim and documents to PostgreSQL
    - Updates state flags and logs
    """
    # Generate transaction id if missing
    if not getattr(state, "transaction_id", None):
        state.transaction_id = str(uuid.uuid4())

    # Mark and timestamp registration
    state.claim_registered = True
    if not getattr(state, "registered_at", None):
        state.registered_at = datetime.now(timezone.utc).isoformat()

    # Aggregate OCR into extracted_text
    agg = _aggregate_extracted_text(state)
    if agg:
        state.extracted_text = agg

    # ⬇️ Lazy-import DB operations ONLY when this function runs.
    # This prevents build/inspection/startup from touching the DB.
    try:
        from backend.db.postgres_store import upsert_claim_registration, insert_documents

        # Persist claim
        upsert_claim_registration(
            transaction_id=state.transaction_id,
            claim_id=state.claim_id,
            customer_name=state.customer_name,
            policy_number=state.policy_number,
            amount=state.amount,
            claim_type=state.claim_type,
            extracted_text=state.extracted_text,
            registered_at=state.registered_at,
            status="REGISTERED",
        )

        # Persist attached documents (if any)
        docs_payload = []
        for d in getattr(state, "documents", []) or []:
            docs_payload.append({
                "filename": getattr(d, "filename", None),
                "content_type": getattr(d, "content_type", None),
                "size_bytes": getattr(d, "size_bytes", None),
                "doc_type": getattr(d, "doc_type", None),
                "extracted_text": (getattr(d, "extracted_text", "") or "")[:MAX_TEXT_LEN],
            })
        if docs_payload:
            insert_documents(state.transaction_id, docs_payload)

        logger.info(f"[RegistrationAgent] Claim registered & saved: {state.claim_id} tx={state.transaction_id}")
        if not hasattr(state, "logs") or state.logs is None:
            state.logs = []
        state.logs.append(f"[registration] saved tx={state.transaction_id}")

    except Exception as e:
        # Never crash the agent on DB failure—log and append to state.logs
        logger.error(f"[RegistrationAgent] DB error: {type(e).__name__}: {e}")
        if not hasattr(state, "logs") or state.logs is None:
            state.logs = []
        state.logs.append(f"[registration] db_error={type(e).__name__}")

    return state