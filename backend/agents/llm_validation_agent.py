from backend.state.claim_state import ClaimState, ValidationResult
from backend.services.llm_client import llm_response
import json


def llm_validation_agent(state: ClaimState) -> ClaimState:

    # Only apply AI validation for Motor Claims
    if state.claim_type.lower() != "motor":
        return state

    prompt = f"""
You are a Senior Motor Insurance Claim Validation Officer.

Below is the extracted content from all uploaded
documents and accident description related to
a Motor Insurance Claim.

Motor Claim Content:
---------------------
{state.extracted_text}

Mandatory Documents Required:
-----------------------------
FIR
DRIVING_LICENSE
RC_BOOK
POLICY_COPY
REPAIR_ESTIMATE
ACCIDENT_PHOTOS

Your Responsibilities:
----------------------
1. Check if all mandatory documents exist.
2. Analyse:
   - Accident description consistency
   - Claimed damage vs accident narrative
   - Policy validity
   - Repair estimate inflation
   - Suspicious behaviour or exaggeration

Respond ONLY in JSON format:

{{
  "required_missing": [],
  "warnings": [],
  "errors": [],
  "docs_ok": true,
  "note": "",
  "recommendation": ""
}}

Rules:
------
Add missing mandatory documents into required_missing.

Add warnings for:
- unusually high repair estimate
- vague FIR description
- mismatch in damage description

Add errors for:
- policy expired
- no FIR
- invalid Driving License
- RC mismatch

docs_ok should be TRUE only if:
All mandatory documents are present AND
No critical errors found.

Recommendation Rules:
---------------------
If all documents valid → APPROVE
If mandatory docs missing → NEED_MORE_DOCUMENTS
If fraud suspicion / policy expired → REJECT

Note must explain reasoning for manager review.

Return ONLY JSON.
"""

    response = llm_response(prompt)

    try:
        parsed = json.loads(response)

    except Exception as e:
        parsed = {
            "required_missing": ["LLM_PARSE_ERROR"],
            "warnings": [],
            "errors": ["AI_VALIDATION_FAILED"],
            "docs_ok": False,
            "note": "AI validation could not process the claim content.",
            "recommendation": "NEED_MORE_DOCUMENTS"
        }

    # Populate validation result into state
    state.validation = ValidationResult(
        required_missing=parsed.get("required_missing", []),
        warnings=parsed.get("warnings", []),
        errors=parsed.get("errors", []),
        docs_ok=parsed.get("docs_ok", False),
        note=parsed.get("note", ""),
        recommendation=parsed.get("recommendation", "")
    )

    # Update claim validated flag
    state.claim_validated = state.validation.docs_ok

    return state
