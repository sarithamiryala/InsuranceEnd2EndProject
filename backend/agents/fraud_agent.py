from backend.services.llm_client import llm_response
from backend.utils.safe_json import safe_json_parse
import os


def _sanitize_result(data: dict) -> dict:
    out = {}

    try:
        out["fraud_score"] = float(data.get("fraud_score", 0.0))
        if out["fraud_score"] < 0:
            out["fraud_score"] = 0.0
        if out["fraud_score"] > 1:
            out["fraud_score"] = 1.0
    except:
        out["fraud_score"] = 0.0

    decision = str(data.get("fraud_decision", "SAFE")).strip().upper()
    out["fraud_decision"] = "SUSPECT" if decision == "SUSPECT" else "SAFE"

    return out


def fraud_agent(state):

    if state.claim_type.lower() != "motor":
        return state

    validation = state.validation

    prompt = f"""
You are a Motor Insurance Fraud Risk Assessment Officer.

Assess fraud risk for the following claim:

Claim Amount:
{state.amount}

Claim Content:
{state.extracted_text}

Validation Findings:
--------------------
Missing Documents: {validation.required_missing}
Warnings: {validation.warnings}
Errors: {validation.errors}
Officer Note: {validation.note}
Validation Recommendation: {validation.recommendation}

Risk Guidelines:
---------------
Increase fraud risk if:
- FIR missing
- Driving License missing
- RC Book missing
- Policy expired
- Damage inconsistency warning
- High repair estimate
- Validation recommendation is REJECT

Medium risk if:
- NEED_MORE_DOCUMENTS
- Minor inconsistencies

Low risk if:
- APPROVE
- No warnings/errors

Respond ONLY in JSON:

{{
  "fraud_score": 0.0,
  "fraud_decision": "SAFE"
}}

Rules:
------
fraud_score:
0.0 – 0.3 → SAFE
0.3 – 0.7 → MODERATE
0.7 – 1.0 → SUSPECT

fraud_decision:
Return SUSPECT if score > 0.6
Else SAFE

Return ONLY JSON.
"""

    if not os.getenv("GOOGLE_API_KEY"):
        print("[fraud_agent] WARNING: GOOGLE_API_KEY not set. Using fallback.")
        raw_result = '{"fraud_score": 0.0, "fraud_decision": "SAFE"}'
    else:
        try:
            raw_result = llm_response(prompt)
        except Exception as e:
            print(f"[fraud_agent] ERROR calling LLM: {e}")
            raw_result = '{"fraud_score": 0.0, "fraud_decision": "SAFE"}'

    print("Raw LLM response (fraud_agent):", repr(raw_result))

    fallback = {"fraud_score": 0.0, "fraud_decision": "SAFE"}
    parsed = safe_json_parse(raw_result, fallback)
    cleaned = _sanitize_result(parsed)

    state.fraud_checked = True
    state.fraud_score = cleaned["fraud_score"]
    state.fraud_decision = cleaned["fraud_decision"]

    return state
