from backend.utils.logger import logger
from backend.state.claim_state import ClaimState
from datetime import datetime, timezone
import uuid
from backend.db.sqlite_store import init_db, upsert_claim_registration, insert_documents
init_db()

MAX_TEXT_LEN = 90000

def _aggregate_extracted_text(state: ClaimState) -> str:
    parts = []
    if state.extracted_text:
        parts.append(state.extracted_text)
    for d in state.documents:
        if d.extracted_text:
            parts.append(d.extracted_text)
    combined = "\n\n".join([p.strip() for p in parts if p and p.strip()])
    return combined[:MAX_TEXT_LEN] if combined else ""

def registration_agent(state: ClaimState):
    if not state.transaction_id:
        state.transaction_id = str(uuid.uuid4())
    state.claim_registered = True
    if not state.registered_at:
        state.registered_at = datetime.now(timezone.utc).isoformat()

    # Aggregate OCR into extracted_text
    agg = _aggregate_extracted_text(state)
    if agg:
        state.extracted_text = agg

    # Persist
    try:
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
        insert_documents(state.transaction_id, [
            {
              "filename": d.filename, "content_type": d.content_type, "size_bytes": d.size_bytes,
              "doc_type": d.doc_type, "extracted_text": (d.extracted_text or "")[:MAX_TEXT_LEN]
            } for d in state.documents
        ])
        logger.info(f"[RegistrationAgent] Claim registered & saved: {state.claim_id} tx={state.transaction_id}")
        state.logs.append(f"[registration] saved tx={state.transaction_id}")
    except Exception as e:
        logger.error(f"[RegistrationAgent] DB error: {type(e).__name__}: {e}")
        state.logs.append(f"[registration] db_error={type(e).__name__}")
    return state