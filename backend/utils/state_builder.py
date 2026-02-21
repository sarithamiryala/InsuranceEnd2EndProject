# backend/utils/state_builder.py
import json
from backend.state.claim_state import ClaimState, ValidationResult

def _safe_json_obj(text: str) -> dict:
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        # Attempt brace-slice rescue
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(text[s:e+1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

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

    val_json = _safe_json_obj(claim.get("validation") or "")
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

    return state