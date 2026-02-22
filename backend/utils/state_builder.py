# backend/utils/state_builder.py

import json
from backend.state.claim_state import ClaimState, ValidationResult


# ============================================================
# SAFE JSON HANDLER (SQLITE + POSTGRES SUPPORT)
# ============================================================

def _safe_json_obj(val):

    # PostgreSQL JSONB already returns dict
    if isinstance(val, dict):
        return val

    # SQLite TEXT needs parsing
    if isinstance(val, str):
        try:
            data = json.loads(val)
            return data if isinstance(data, dict) else {}
        except Exception:
            s, e = val.find("{"), val.rfind("}")
            if s != -1 and e != -1 and e > s:
                try:
                    data = json.loads(val[s:e+1])
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}

    return {}


# ============================================================
# STATE BUILDER
# ============================================================

def build_state_from_db(claim, docs):

    state = ClaimState(
        transaction_id=claim["transaction_id"],
        claim_id=(claim.get("claim_id") or "").strip(),
        customer_name=(claim.get("customer_name") or "").strip(),
        policy_number=(claim.get("policy_number") or "").strip(),
        amount=claim.get("amount"),
        claim_type=(claim.get("claim_type") or "").strip().lower(),
        extracted_text=(claim.get("extracted_text") or "").strip(),
        document_extracted_text=(claim.get("document_extracted_text") or "").strip(),
    )

    # -------------------------
    # Validation Restore
    # -------------------------
    val_json = _safe_json_obj(claim.get("validation"))

    if val_json:
        state.validation = ValidationResult(
            required_missing=val_json.get("required_missing", []),
            warnings=val_json.get("warnings", []),
            errors=val_json.get("errors", []),
            docs_ok=bool(val_json.get("docs_ok", False)),
            note=val_json.get("note", ""),
            recommendation=(val_json.get("recommendation", "") or "").upper(),
        )
        state.claim_validated = state.validation.docs_ok

    # -------------------------
    # 🔥 MCP CLOUD RESUME RESTORE
    # -------------------------

    state.fraud_checked = bool(claim.get("fraud_checked", False))
    state.fraud_score = claim.get("fraud_score")
    state.fraud_decision = claim.get("fraud_decision")

    state.final_decision = claim.get("final_decision")

    if state.final_decision:
        state.claim_decision_made = True

    state.payment_processed = bool(claim.get("payment_processed", False))
    state.claim_closed = bool(claim.get("claim_closed", False))

    state.claim_registered = True
    state.claim_validated = bool(claim.get("claim_validated", state.claim_validated))

    return state