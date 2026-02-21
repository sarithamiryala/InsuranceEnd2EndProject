# backend/agents/llm_validation_agent.py

from __future__ import annotations
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from backend.state.claim_state import ClaimState, ValidationResult
from backend.services.llm_client import llm_response


# -----------------------------
# Config & Utilities
# -----------------------------

DATE_FORMATS = [
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
    "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
]

MANDATORY_DOC_MARKERS = [
    ("=== FIR ===", "FIR"),
    ("=== DRIVING_LICENSE ===", "DRIVING_LICENSE"),
    ("=== RC_BOOK ===", "RC_BOOK"),
    ("=== POLICY_COPY ===", "POLICY_COPY"),
    ("=== REPAIR_ESTIMATE ===", "REPAIR_ESTIMATE"),
    ("=== ACCIDENT_PHOTOS ===", "ACCIDENT_PHOTOS"),
]

RECOMMENDATION_ORDER = {
    "REJECT": 3,
    "NEED_MORE_DOCUMENTS": 2,
    "APPROVE": 1
}


def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


def _lower(s: Optional[str]) -> str:
    return _normalize(s).lower()

def _parse_date_with_formats(token: str):
    for fmt in [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
        "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
    ]:
        try:
            return datetime.strptime(token, fmt)
        except Exception:
            continue
    return None

def _extract_valid_to(dl_text: str):
    # Strictly pick the 'Valid To:' date
    m = re.search(
        r"Valid\s*To\s*:\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}(?: [0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
        dl_text, flags=re.IGNORECASE
    )
    if not m:
        return None
    return _parse_date_with_formats(m.group(1))


def _safe_json_loads(s: str, fallback: dict) -> dict:
    try:
        data = json.loads(s)
        if not isinstance(data, dict):
            return fallback
        return data
    except Exception:
        brace_start = s.find("{")
        brace_end = s.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                return json.loads(s[brace_start:brace_end + 1])
            except Exception:
                return fallback
        return fallback


def _detect_vehicle_reg(text: str) -> List[str]:
    """
    Extract likely Indian registration numbers like KA03MN4567, KA-03-MN-4567, etc.
    """
    t = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    matches = re.findall(r"[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}", t)
    return list(set(matches))


def _extract_name(text: str) -> Optional[str]:
    """
    Heuristic: pull 'Name:' or 'Insured:' or 'Owner:' fields.
    """
    for key in ["Name", "Insured", "Owner", "Complainant"]:
        m = re.search(rf"{key}\s*:\s*([A-Za-z][A-Za-z\s\.\-']+)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _extract_policy_period(text: str) -> Tuple[Optional[datetime], Optional[datetime], str]:
    """
    Attempts to parse policy validity period and coverage type (OD/TP/Comprehensive).
    """
    coverage = "UNKNOWN"
    t = text.lower()

    if "comprehensive" in t or "od + tp" in t or "od+tp" in t:
        coverage = "COMPREHENSIVE"
    elif "tp only" in t or "third party only" in t or "third-party only" in t:
        coverage = "TP_ONLY"
    elif "own damage" in t or "od cover" in t:
        coverage = "OD"

    period_line = None
    for line in text.splitlines():
        if any(k in line.lower() for k in ["period", "valid", "validity"]):
            period_line = line
            break

    start = end = None
    if period_line:
        dates = []
        for token in re.findall(r"(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}(?: \d{1,2}:\d{2}(?::\d{2})?)?)", period_line):
            for fmt in DATE_FORMATS:
                try:
                    dates.append(datetime.strptime(token, fmt))
                    break
                except Exception:
                    continue
        if len(dates) >= 2:
            start, end = dates[0], dates[1]

    if not (start and end):
        all_dates = []
        for token in re.findall(r"(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{2,4}(?: \d{1,2}:\d{2}(?::\d{2})?)?)", text):
            for fmt in DATE_FORMATS:
                try:
                    all_dates.append(datetime.strptime(token, fmt))
                    break
                except Exception:
                    continue
        if len(all_dates) >= 2:
            start, end = all_dates[0], all_dates[1]

    return start, end, coverage


def _parse_estimate_total(text: str) -> Optional[float]:
    """
    Try to compute a numeric total from the repair estimate block. Looks for 'Total' first,
    otherwise sums line items with numbers.
    """
    m = re.search(r"Total[^0-9]*([\d,]+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))

    nums = re.findall(r"([\d,]+(?:\.\d+)?)", text)
    if nums:
        s = sum(float(n.replace(",", "")) for n in nums)
        return s if s > 0 else None

    return None


def _infer_damage_severity(description: str, fir_text: str) -> str:
    """
    Heuristic severity: MINOR/MODERATE/MAJOR based on keywords across description & FIR narrative.
    """
    t = f"{description}\n{fir_text}".lower()
    minor_kw = ["minor", "scratch", "scratches", "grazed", "light", "superficial"]
    moderate_kw = ["rear-ended", "rear ended", "hit from rear", "bumper", "tail lamp", "fender", "dent"]
    major_kw = ["head-on", "hit divider", "rollover", "totaled", "airbag deployed", "radiator", "chassis"]

    score = 0
    if any(k in t for k in minor_kw):
        score -= 1
    if any(k in t for k in moderate_kw):
        score += 1
    if any(k in t for k in major_kw):
        score += 2

    if score <= 0:
        return "MINOR"
    if score == 1:
        return "MODERATE"
    return "MAJOR"


def _recommendation_from(findings: Dict[str, List[str]]) -> str:
    """
    Deterministic recommendation from local findings.
    """
    errors = findings["errors"]
    missing = findings["required_missing"]

    critical_error_keys = {
        "FIR not found",
        "Driving License invalid/expired",
        "Policy expired or not covering OD",
        "RC/Policy/FIR vehicle mismatch",
        "Claim narrative inconsistent with FIR",
        "Fake or unverifiable repair estimate",
    }

    if missing:
        return "NEED_MORE_DOCUMENTS"

    if any(e in critical_error_keys for e in errors):
        return "REJECT"

    return "APPROVE"


def _merge_recommendation(local_rec: str, llm_rec: str) -> str:
    RECOMMENDATION_ORDER = {"REJECT": 3, "NEED_MORE_DOCUMENTS": 2, "APPROVE": 1}
    a = RECOMMENDATION_ORDER.get((local_rec or "APPROVE").upper(), 1)
    b = RECOMMENDATION_ORDER.get((llm_rec or "APPROVE").upper(), 1)
    return local_rec if a >= b else llm_rec


# -----------------------------
# Deterministic Pre-Validation
# -----------------------------

def _prevalidate(state: ClaimState) -> Dict[str, List[str] | bool | str]:
    """
    Deterministic checks to ensure we never return empty findings.
    Returns dict with required_missing, warnings, errors, docs_ok, note, recommendation.
    """
    required_missing: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []

    desc = _normalize(state.extracted_text)
    docs_raw = _normalize(state.document_extracted_text)
    docs_lower = docs_raw.lower()

    # 1) Presence of mandatory documents
    for marker, label in MANDATORY_DOC_MARKERS:
        if marker.lower() not in docs_lower:
            required_missing.append(label)

    # 2) Extract sub-blocks for more precise checks
    def _extract_block(title: str) -> str:
        pattern = rf"===\s*{re.escape(title)}\s*===\s*(.*?)(?:(?:\n===)|\Z)"
        m = re.search(pattern, docs_raw, flags=re.IGNORECASE | re.DOTALL)
        return (m.group(1).strip() if m else "")

    fir_text = _extract_block("FIR")
    dl_text = _extract_block("DRIVING_LICENSE")
    rc_text = _extract_block("RC_BOOK")
    policy_text = _extract_block("POLICY_COPY")
    estimate_text = _extract_block("REPAIR_ESTIMATE")
    photos_text = _extract_block("ACCIDENT_PHOTOS")

    # 3) Incident date (from description or FIR)
    incident_date = _parse_date_with_formats(desc) or _parse_date_with_formats(fir_text)

    # 4) DL validity
    if dl_text:
        dl_valid_to = _parse_date_with_formats(dl_text)
        if incident_date and dl_valid_to and dl_valid_to < incident_date:
            errors.append("Driving License invalid/expired")
    else:
        if "DRIVING_LICENSE" not in required_missing:
            warnings.append("Driving License details unclear")

    # 5) Policy period & coverage
    if policy_text:
        p_start, p_end, coverage = _extract_policy_period(policy_text)
        if incident_date and (not p_start or not p_end):
            warnings.append("Policy validity period not clearly parsed")
        if incident_date and p_end and p_end < incident_date:
            errors.append("Policy expired or not covering OD")
        if coverage in ("TP_ONLY", "UNKNOWN"):
            errors.append("Policy expired or not covering OD")
    else:
        if "POLICY_COPY" not in required_missing:
            warnings.append("Policy details unclear")

    # 6) Vehicle consistency across FIR, RC, Policy
    regs = set()
    for t in [fir_text, rc_text, policy_text]:
        regs.update(_detect_vehicle_reg(t))
    if len(regs) >= 2:
        errors.append("RC/Policy/FIR vehicle mismatch")

    # 7) Customer name presence
    registered_name = _normalize(state.customer_name)
    doc_name = _extract_name("\n".join([dl_text, policy_text, fir_text]))
    if registered_name and doc_name and registered_name.lower() not in doc_name.lower():
        warnings.append("Customer name mismatch between registration and documents")

    # 8) Narrative consistency: description vs FIR
    if desc and fir_text:
        d = desc.lower()
        f = fir_text.lower()
        contradiction_pairs = [
            (("divider",), ("rear", "rear-ended", "hit from rear")),
            (("rear", "rear-ended", "hit from rear"), ("divider",)),
            (("minor", "scratch", "scratches", "grazed"), ("bumper", "headlamp", "radiator", "chassis", "bonnet", "fender")),
        ]
        for lhs, rhs in contradiction_pairs:
            if any(x in d for x in lhs) and any(y in f for y in rhs):
                errors.append("Claim narrative inconsistent with FIR")
                break

    # 9) Estimate plausibility vs severity & photos
    if estimate_text:
        total = _parse_estimate_total(estimate_text) or 0.0
        severity = _infer_damage_severity(desc, fir_text)
        if severity == "MINOR" and total > 50000:
            warnings.append("Unusually high repair estimate for minor damage")
        if severity == "MODERATE" and total > 150000:
            warnings.append("Repair estimate appears inflated for moderate impact")
        if photos_text:
            pt = photos_text.lower()
            if any(k in pt for k in ["minor", "hairline", "scratch"]) and re.search(r"(bumper|headlamp|door|fender).*(replace|assembly|assy)", estimate_text, flags=re.IGNORECASE | re.DOTALL):
                warnings.append("Photos suggest minor damage but estimate lists major replacements")
    else:
        warnings.append("Repair estimate not provided or unclear")

    # 10) FIR presence is critical
    if "FIR" in required_missing:
        errors.append("FIR not found")

    # 11) Unverifiable estimate (non-network / missing identifiers)
    if estimate_text and re.search(r"(handwritten|non[- ]network|no gst|no gstin|no part|no part number)", estimate_text, flags=re.IGNORECASE):
        errors.append("Fake or unverifiable repair estimate")

    # Aggregate
    recommendation = _recommendation_from({
        "required_missing": required_missing,
        "warnings": warnings,
        "errors": errors
    })
    docs_ok = (len(errors) == 0 and len(required_missing) == 0)

    reasons = []
    if required_missing:
        reasons.append(f"Missing: {', '.join(required_missing)}")
    if errors:
        reasons.append(f"Errors: {', '.join(errors)}")
    if warnings:
        reasons.append(f"Warnings: {', '.join(warnings)}")
    note = "; ".join(reasons) or "Deterministic pre-validation found no critical issues."

    return {
        "required_missing": required_missing,
        "warnings": warnings,
        "errors": errors,
        "docs_ok": docs_ok,
        "note": note,
        "recommendation": recommendation
    }


# -----------------------------
# LLM Prompt (Refined) & Merge
# -----------------------------

def _build_llm_prompt(state: ClaimState, pre: dict) -> str:
    claim_description = _normalize(state.extracted_text) or "No accident description provided."
    uploaded_docs = _normalize(state.document_extracted_text) or "No supporting documents uploaded."

    return f"""
You are a Senior Motor Insurance Claim Validation Officer.
You must act as a strict auditor.

INPUTS
------
Claim Metadata:
- Transaction ID: {state.transaction_id}
- Claim ID: {state.claim_id}
- Customer Name: {state.customer_name}
- Policy Number: {state.policy_number}
- Claim Type: {state.claim_type}
- Claimed Amount: {state.amount}

Accident Description (from customer):
{claim_description}

Uploaded Document OCR (verbatim):
{uploaded_docs}

MANDATORY DOCUMENTS (Must be present)
- FIR
- DRIVING_LICENSE
- RC_BOOK
- POLICY_COPY
- REPAIR_ESTIMATE
- ACCIDENT_PHOTOS

LOCAL DETERMINISTIC FINDINGS (from pre-validation)
- required_missing: {pre.get('required_missing')}
- warnings: {pre.get('warnings')}
- errors: {pre.get('errors')}
- docs_ok: {pre.get('docs_ok')}
- local_recommendation: {pre.get('recommendation')}

YOUR TASK
---------
1) Verify presence of all mandatory documents (do not rely only on headings; check content).
2) Check narrative consistency between Description and FIR.
3) Check DL validity on incident date.
4) Check Policy validity and whether OD/Comprehensive coverage applies.
5) Verify vehicle consistency (RC/Policy/FIR registration numbers).
6) Evaluate repair estimate plausibility (inflation vs severity & photos).
7) Identify fake/unverifiable estimates (non-network, no GSTIN, no part numbers).
8) Provide actionable notes for a human manager.

STRICT OUTPUT (Return ONLY valid JSON, no extra text):
{{
  "required_missing": ["FIR","DRIVING_LICENSE","RC_BOOK","POLICY_COPY","REPAIR_ESTIMATE","ACCIDENT_PHOTOS"],
  "warnings": ["..."],
  "errors": ["..."],
  "docs_ok": true,
  "note": "Short, specific reasoning referencing which checks passed/failed.",
  "recommendation": "APPROVE | NEED_MORE_DOCUMENTS | REJECT"
}}

RULES
-----
- If any mandatory document is missing -> recommendation = "NEED_MORE_DOCUMENTS".
- If policy expired or DL invalid -> recommendation = "REJECT".
- If narrative mismatch or photo/estimate contradiction and strong -> prefer "NEED_MORE_DOCUMENTS" unless clearly fraudulent -> "REJECT".
- If all documents present and no critical errors -> "APPROVE".
- Keep note concise (<= 5 lines) but specific.
"""


def _sanitize_llm_dict(data: dict) -> dict:
    return {
        "required_missing": list(data.get("required_missing") or []),
        "warnings": list(data.get("warnings") or []),
        "errors": list(data.get("errors") or []),
        "docs_ok": bool(data.get("docs_ok")) if data.get("docs_ok") is not None else False,
        "note": (_normalize(data.get("note")) or ""),
        "recommendation": _normalize(data.get("recommendation")).upper() or "",
    }


def llm_validation_agent(state: ClaimState) -> ClaimState:
    """
    Hybrid validator:
    1) Normalize & deterministic pre-validation
    2) LLM validation with refined prompt
    3) Merge results with clear precedence
    4) Always return populated ValidationResult & claim_validated flag
    """
    # ---- Normalize early
    state.claim_type = _lower(state.claim_type)
    state.customer_name = _normalize(state.customer_name)
    state.policy_number = _normalize(state.policy_number)
    state.extracted_text = _normalize(state.extracted_text)
    state.document_extracted_text = _normalize(state.document_extracted_text)

    # Apply only for Motor Claims (but never return empty)
    if not state.claim_type or state.claim_type != "motor":
        state.validation = ValidationResult(
            required_missing=[],
            warnings=[],
            errors=["Non-motor claim type or missing claim_type"],
            docs_ok=False,
            note="Validation skipped: claim_type is not 'motor'.",
            recommendation="NEED_MORE_DOCUMENTS"
        )
        state.claim_validated = False
        return state

    # ---- Deterministic pre-validation
    pre = _prevalidate(state)

    # ---- LLM validation
    prompt = _build_llm_prompt(state, pre)
    try:
        raw = llm_response(prompt)
    except Exception:
        raw = '{"required_missing":[],"warnings":[],"errors":["LLM_CALL_FAILED"],"docs_ok":false,"note":"LLM call failed","recommendation":"NEED_MORE_DOCUMENTS"}'

    llm = _safe_json_loads(raw, fallback={
        "required_missing": [],
        "warnings": [],
        "errors": ["AI_VALIDATION_FAILED"],
        "docs_ok": False,
        "note": "AI validation could not process uploaded claim documents.",
        "recommendation": "NEED_MORE_DOCUMENTS"
    })
    llm = _sanitize_llm_dict(llm)

    # ---- Merge results
    required_missing = sorted(list(set(pre["required_missing"] + llm["required_missing"])))
    warnings = sorted(list(set(pre["warnings"] + llm["warnings"])))
    errors = sorted(list(set(pre["errors"] + llm["errors"])))

    # docs_ok is true only if both agree
    docs_ok = bool(pre["docs_ok"] and llm["docs_ok"])

    # recommendation precedence
    final_rec = _merge_recommendation(pre.get("recommendation", "APPROVE"), llm.get("recommendation", "APPROVE"))

    # Ensure critical rules override
    if required_missing:
        final_rec = "NEED_MORE_DOCUMENTS"
        docs_ok = False
    if any(e in errors for e in ["Policy expired or not covering OD", "Driving License invalid/expired"]):
        final_rec = "REJECT"
        docs_ok = False

    # Compose concise note (always populated)
    notes = []
    if required_missing:
        notes.append(f"Missing: {', '.join(required_missing)}")
    if errors:
        notes.append(f"Errors: {', '.join(errors)}")
    if warnings:
        notes.append(f"Warnings: {', '.join(warnings)}")
    if not notes:
        notes.append("All mandatory checks passed; no critical findings.")
    final_note = " | ".join(notes)

    # ---- Save into state
    state.validation = ValidationResult(
        required_missing=required_missing,
        warnings=warnings,
        errors=errors,
        docs_ok=docs_ok,
        note=final_note,
        recommendation=final_rec
    )
    state.claim_validated = docs_ok

    return state