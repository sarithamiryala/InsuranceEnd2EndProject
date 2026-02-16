from backend.state.claim_state import ClaimState, ValidationResult
from backend.services.llm_client import llm_response
import json


def llm_validation_agent(state: ClaimState) -> ClaimState:

    # Apply only for Motor Claims
    if not state.claim_type or state.claim_type.lower() != "motor":
        return state

    claim_description = state.extracted_text or "No accident description provided."
    uploaded_docs = state.document_extracted_text or "No supporting documents uploaded."

    prompt = f"""
You are a Senior Motor Insurance Claim Validation Officer.

Below are:

1. Accident Description provided by customer
2. OCR extracted content from all uploaded claim documents
   such as FIR, Driving License, RC Book, Policy Copy,
   Repair Estimate and Accident Photos.

---------------------------------------------------------
ACCIDENT DESCRIPTION
---------------------------------------------------------
{claim_description}

---------------------------------------------------------
UPLOADED DOCUMENT OCR CONTENT
---------------------------------------------------------
{uploaded_docs}

---------------------------------------------------------
MANDATORY MOTOR CLAIM DOCUMENTS REQUIRED
---------------------------------------------------------
FIR
DRIVING_LICENSE
RC_BOOK
POLICY_COPY
REPAIR_ESTIMATE
ACCIDENT_PHOTOS

---------------------------------------------------------
YOUR VALIDATION TASK
---------------------------------------------------------
1. Check whether all mandatory documents are present.
2. Verify accident description matches FIR narrative.
3. Check claimed damage vs repair estimate consistency.
4. Validate policy validity.
5. Identify inflated repair estimates.
6. Detect suspicious or exaggerated claims.

---------------------------------------------------------
CRITICAL ERRORS IF:
---------------------------------------------------------
- FIR not found
- Driving License invalid
- RC Book mismatch
- Policy expired
- Claim narrative inconsistent with FIR
- Fake or missing repair estimate

---------------------------------------------------------
WARNINGS IF:
---------------------------------------------------------
- Unusually high repair estimate
- Vague FIR description
- Partial damage mismatch
- Incomplete invoice details

---------------------------------------------------------
OUTPUT STRICTLY JSON:
---------------------------------------------------------
{{
  "required_missing": [],
  "warnings": [],
  "errors": [],
  "docs_ok": true,
  "note": "",
  "recommendation": ""
}}

---------------------------------------------------------
RECOMMENDATION RULES
---------------------------------------------------------
IF:
All documents present AND No critical errors → APPROVE

IF:
Mandatory documents missing → NEED_MORE_DOCUMENTS

IF:
Fraud suspicion OR Policy expired → REJECT

Note must explain reasoning clearly for Manager review.

RETURN ONLY JSON.
"""

    response = llm_response(prompt)

    try:
        parsed = json.loads(response)
    except Exception:
        parsed = {
            "required_missing": ["AI_PARSE_ERROR"],
            "warnings": [],
            "errors": ["AI_VALIDATION_FAILED"],
            "docs_ok": False,
            "note": "AI validation could not process uploaded claim documents.",
            "recommendation": "NEED_MORE_DOCUMENTS"
        }

    state.validation = ValidationResult(
        required_missing=parsed.get("required_missing", []),
        warnings=parsed.get("warnings", []),
        errors=parsed.get("errors", []),
        docs_ok=parsed.get("docs_ok", False),
        note=parsed.get("note", ""),
        recommendation=parsed.get("recommendation", "")
    )

    state.claim_validated = state.validation.docs_ok

    return state
