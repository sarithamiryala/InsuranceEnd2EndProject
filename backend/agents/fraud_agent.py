# backend/agents/fraud_agent.py

from __future__ import annotations
import re
from typing import Any, Dict

from backend.state.claim_state import ClaimState
from backend.services.llm_client import llm_response
from backend.utils.safe_json import safe_json_parse


# -----------------------------
# Utilities / Normalization
# -----------------------------

def _normalize(s: str | None) -> str:
    return (s or "").strip()

def _lower(s: str | None) -> str:
    return _normalize(s).lower()

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _finalize_decision(score: float) -> str:
    """
    Thresholds:
      0.0 – 0.3  -> SAFE
      0.3 – 0.6  -> MODERATE
      > 0.6      -> SUSPECT
    """
    if score > 0.6:
        return "SUSPECT"
    if score >= 0.3:
        return "MODERATE"
    return "SAFE"


# -----------------------------
# Deterministic Risk Heuristics
# -----------------------------

def _name_mismatch(state: ClaimState) -> bool:
    """
    If customer name is not found in OCR docs, consider a mismatch.
    """
    docs = (_normalize(state.document_extracted_text)).lower()
    if not state.customer_name:
        return False
    return state.customer_name.lower() not in docs

def _vehicle_reg_set(text: str) -> set[str]:
    """
    Extract likely Indian vehicle registration numbers (e.g., KA03MN4567).
    """
    t = re.sub(r"[^A-Za-z0-9]", "", (text or "").upper())
    return set(re.findall(r"[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}", t))

def _vehicle_mismatch(state: ClaimState) -> bool:
    """
    If multiple distinct regs appear across the docs block, flag as vehicle mismatch.
    """
    regs = _vehicle_reg_set(state.document_extracted_text or "")
    return len(regs) >= 2

def _narrative_contradiction(state: ClaimState) -> bool:
    """
    Basic contradiction: description says 'divider' but docs say rear-end (or vice versa).
    """
    desc = _normalize(state.extracted_text).lower()
    docs = _normalize(state.document_extracted_text).lower()
    divider = "divider" in desc
    rearish = any(k in docs for k in ["rear", "rear-ended", "hit from rear"])
    return divider and rearish


def _risk_bump_from_validation(state: ClaimState) -> float:
    """
    Deterministic score floor/bumps from validation findings (if present).
    Returns a floor (0..1).
    """
    v = getattr(state, "validation", None)
    if not v:
        return 0.0

    bump = 0.0
    errors = set(v.errors or [])
    warnings = set(v.warnings or [])
    missing = set(v.required_missing or [])

    # Missing mandatory docs => at least moderate floor
    if missing:
        bump = max(bump, 0.35)

    # Critical consistency/errors => high floor
    critical_errors = {
        "Policy expired or not covering OD",
        "Driving License invalid/expired",
        "RC/Policy/FIR vehicle mismatch",
        "Fake or unverifiable repair estimate",
        "FIR not found",
    }
    if errors & critical_errors:
        bump = max(bump, 0.70)

    # Narrative contradictions => moderate/high floor
    if "Claim narrative inconsistent with FIR" in errors or \
       "Claim narrative inconsistent with FIR" in warnings:
        bump = max(bump, 0.60)

    # If validator recommended REJECT, ensure high floor
    rec = (getattr(v, "recommendation", "") or "").upper()
    if rec == "REJECT":
        bump = max(bump, 0.70)

    # If docs not ok and there are findings, ensure at least moderate
    if not getattr(v, "docs_ok", False) and (errors or missing):
        bump = max(bump, 0.35)

    return _clamp01(bump)


def _risk_bump_from_heuristics(state: ClaimState) -> float:
    """
    Deterministic floors independent of validation object (in case parsing fails upstream).
    """
    score = 0.0

    if _name_mismatch(state):
        score = max(score, 0.45)

    if _vehicle_mismatch(state):
        score = max(score, 0.60)

    if _narrative_contradiction(state):
        score = max(score, 0.55)

    return _clamp01(score)


# -----------------------------
# LLM Prompt
# -----------------------------

def _build_fraud_prompt(state: ClaimState) -> str:
    v = getattr(state, "validation", None)
    return f"""
You are a Motor Insurance Fraud Risk Assessment Officer.

Assess fraud risk for the claim below and respond ONLY in JSON.

Claim Details from customer:
- Customer: {state.customer_name}
- Claimed Amount: {state.amount}
- Description: {state.extracted_text}

OCR Documents (verbatim):
{_normalize(state.document_extracted_text)}

Validation Findings:
- Missing Documents: {(v.required_missing if v else [])}
- Warnings: {(v.warnings if v else [])}
- Errors: {(v.errors if v else [])}
- Officer Note: {(v.note if v else "")}
- Validation Recommendation: {((v.recommendation or "") if v else "")}

Guidelines:
- Increase fraud risk if: FIR missing, DL missing/expired, RC/Policy mismatch,
  Policy expired/no OD cover, narrative inconsistency, unusually high/inflated estimate,
  or validation recommendation is REJECT.
- Medium risk if: NEED_MORE_DOCUMENTS or minor inconsistencies.
- Low risk if: APPROVE and no warnings/errors.

Return ONLY this JSON (no extra text):
{{
  "fraud_score": 0.0,
  "fraud_decision": "SAFE"
}}

Rules:
- fraud_score in [0..1]
- 0.0–0.3 -> SAFE
- 0.3–0.7 -> MODERATE
- >0.6 -> SUSPECT
- fraud_decision must be one of: SAFE, MODERATE, SUSPECT (based on score)
""".strip()


# -----------------------------
# Main Agent
# -----------------------------

def fraud_agent(state: ClaimState) -> ClaimState:
    """
    Hybrid fraud scorer:
    1) Normalize inputs
    2) Call LLM (safe JSON parse)
    3) Apply deterministic floors from validation + heuristics
    4) Produce final score & decision
    """
    # Normalize
    claim_type = _lower(state.claim_type)
    state.customer_name = _normalize(state.customer_name)
    state.extracted_text = _normalize(state.extracted_text)
    state.document_extracted_text = _normalize(state.document_extracted_text)

    # Only for motor claims per your current design
    if claim_type != "motor":
        return state

    # ---- LLM call (best effort) ----
    prompt = _build_fraud_prompt(state)
    try:
        raw = llm_response(prompt)
    except Exception as e:
        # Fallback if LLM fails
        raw = '{"fraud_score": 0.0, "fraud_decision": "SAFE"}'

    # ---- Parse LLM output robustly ----
    llm_obj = safe_json_parse(
        raw,
        fallback={"fraud_score": 0.0, "fraud_decision": "SAFE"},
        expect="object"
    )

    # Sanitize keys
    try:
        score_llm = float(llm_obj.get("fraud_score", 0.0))
    except Exception:
        score_llm = 0.0
    score_llm = _clamp01(score_llm)

    decision_llm = str(llm_obj.get("fraud_decision", "SAFE")).strip().upper()
    if decision_llm not in {"SAFE", "MODERATE", "SUSPECT"}:
        decision_llm = "SAFE"

    # ---- Deterministic floors ----
    floor_val = _risk_bump_from_validation(state)     # from validation findings
    floor_hx  = _risk_bump_from_heuristics(state)     # direct OCR/name/vehicle/narrative

    # Final score = max(LLM score, deterministic floors)
    final_score = max(score_llm, floor_val, floor_hx)
    final_score = round(_clamp01(final_score), 2)

    # Map to final decision by thresholds (LLM decision is informational)
    final_decision = _finalize_decision(final_score)

    # ---- Save to state ----
    state.fraud_checked = True
    state.fraud_score = final_score
    state.fraud_decision = final_decision

    return state