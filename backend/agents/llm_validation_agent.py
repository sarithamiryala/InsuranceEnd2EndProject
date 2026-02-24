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
    "%d-%b-%Y", "%d/%b/%Y", "%d-%B-%Y", "%d/%B/%Y",
    "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%d-%b-%Y %H:%M", "%d/%b/%Y %H:%M",
]

MANDATORY_DOC_MARKERS = [
    ("=== FIR ===", "FIR"),
    ("=== DRIVING_LICENSE ===", "DRIVING_LICENSE"),
    ("=== RC_BOOK ===", "RC_BOOK"),
    ("=== POLICY_COPY ===", "POLICY_COPY"),
    ("=== REPAIR_ESTIMATE ===", "REPAIR_ESTIMATE"),
    ("=== ACCIDENT_PHOTOS ===", "ACCIDENT_PHOTOS"),
]

# OCR fallback keyword hints when markers are absent (used by the text-only LLM and pre checks if needed)
DOC_HINTS: Dict[str, List[str]] = {
    "FIR": [r"\bFIR\b", r"First\s+Information\s+Report", r"\bFIR\s*No\b"],
    "DRIVING_LICENSE": [r"\bDriving\s+License\b", r"\bDL\s*No\b", r"\bDL\b"],
    "RC_BOOK": [r"\bRC\s*Book\b", r"Registration\s+Certificate", r"\bReg\s*No\b"],
    "POLICY_COPY": [r"\bInsurance\s+Policy\b", r"\bPolicy\b", r"\bPolicy\s*Number\b"],
    "REPAIR_ESTIMATE": [r"\bRepair\s+Estimate\b", r"\bEstimate\b", r"\bQuotation\b"],
    "ACCIDENT_PHOTOS": [r"\bAccident\s+Photo", r"\bPhotos?\b", r"\.jpe?g\b", r"\.png\b"],
}

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
        "%d-%b-%Y", "%d/%b/%Y", "%d-%B-%Y", "%d/%B/%Y",
        "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M",
        "%d-%b-%Y %H:%M", "%d/%b/%Y %H:%M",
    ]:
        try:
            return datetime.strptime(token, fmt)
        except Exception:
            continue
    return None


def _extract_valid_to(dl_text: str):
    # Pick the 'Valid To' or 'Valid Until' date
    m = re.search(
        r"Valid\s*(?:To|Until)\s*:\s*("
        r"[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4}"
        r"(?: [0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?|"
        r"[0-9]{1,2}[-/][A-Za-z]{3,9}[-/][0-9]{2,4}"
        r")",
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
    NOTE: This keeps the same flow but reduces false negatives by better date parsing and
    avoids treating UNKNOWN coverage as a hard error.
    """
    required_missing: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []

    desc = _normalize(state.extracted_text)
    docs_raw = _normalize(state.document_extracted_text)
    docs_lower = docs_raw.lower()

    # 1) Presence of mandatory documents (marker-only in pre; LLM text-only will complement later)
    for marker, label in MANDATORY_DOC_MARKERS:
        if marker.lower() not in docs_lower:
            required_missing.append(label)

    # 2) Extract sub-blocks for more precise checks (markers if present, else fallback headings)
    def _extract_block(title: str) -> str:
        # 1) Marker-based block extraction
        pattern = rf"===\s*{re.escape(title)}\s*===\s*(.*?)(?:(?:\n===)|\Z)"
        m = re.search(pattern, docs_raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return (m.group(1).strip() if m else "")

        # 2) Fallback: recognize common heading variants
        title_alias: Dict[str, List[str]] = {
            "FIR": [r"^FIR(?:\s*Copy)?\s*:", r"^First\s+Information\s+Report\s*:"],
            "DRIVING_LICENSE": [r"^Driving\s*License\s*:", r"^DL\s*:"],
            "RC_BOOK": [r"^RC\s*Book\s*:", r"^Registration\s*Certificate\s*:"],
            "POLICY_COPY": [r"^Insurance\s*Policy\s*:", r"^Policy\s*(?:Copy|Number)?\s*:"],
            "REPAIR_ESTIMATE": [r"^Repair\s*Estimate\s*:", r"^Estimate\s*:"],
            "ACCIDENT_PHOTOS": [r"^Accident\s*Photo\s*:", r"^Photos?\s*:"],
        }
        aliases = title_alias.get(title, [rf"^{re.escape(title)}\s*:"])

        lines = docs_raw.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            if any(re.search(p, line, flags=re.IGNORECASE) for p in aliases):
                start_idx = i + 1
                break
        if start_idx is None:
            return ""

        # Stop at the next known heading
        all_heading_patterns = []
        for pats in title_alias.values():
            all_heading_patterns.extend(pats)

        buf: List[str] = []
        for j in range(start_idx, len(lines)):
            if any(re.search(p, lines[j], flags=re.IGNORECASE) for p in all_heading_patterns):
                break
            buf.append(lines[j])
        return "\n".join(buf).strip()

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
        dl_valid_to = _extract_valid_to(dl_text) or _parse_date_with_formats(dl_text)
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
        # Only error if definitely TP_ONLY; UNKNOWN -> warning
        if coverage == "TP_ONLY":
            errors.append("Policy expired or not covering OD")
        elif coverage == "UNKNOWN":
            warnings.append("Policy coverage type not clearly identified (OD/Comprehensive)")
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
            if any(k in pt for k in ["minor", "hairline", "scratch"]) and re.search(
                r"(bumper|headlamp|door|fender).*(replace|assembly|assy)",
                estimate_text, flags=re.IGNORECASE | re.DOTALL
            ):
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
# LLM Prompts (Existing + Text-only Inference)
# -----------------------------

def _build_llm_prompt(state: ClaimState, pre: dict) -> str:
    claim_description = _normalize(state.extracted_text) or "No accident description provided."
    uploaded_docs = _normalize(state.document_extracted_text) or "No supporting documents uploaded."

    pre_required = json.dumps(pre.get('required_missing'), ensure_ascii=False)
    pre_warnings = json.dumps(pre.get('warnings'), ensure_ascii=False)
    pre_errors = json.dumps(pre.get('errors'), ensure_ascii=False)
    pre_docs_ok = json.dumps(pre.get('docs_ok'))
    pre_rec = pre.get('recommendation')

    template = (
        "You are a Senior Motor Insurance Claim Validation Officer.\n"
        "You must act as a strict auditor.\n\n"
        "INPUTS\n"
        "------\n"
        "Claim Metadata:\n"
        "- Transaction ID: {transaction_id}\n"
        "- Claim ID: {claim_id}\n"
        "- Customer Name: {customer_name}\n"
        "- Policy Number: {policy_number}\n"
        "- Claim Type: {claim_type}\n"
        "- Claimed Amount: {amount}\n\n"
        "Accident Description (from customer):\n"
        "{claim_description}\n\n"
        "Uploaded Document OCR (verbatim):\n"
        "{uploaded_docs}\n\n"
        "MANDATORY DOCUMENTS (Must be present)\n"
        "- FIR\n- DRIVING_LICENSE\n- RC_BOOK\n- POLICY_COPY\n- REPAIR_ESTIMATE\n- ACCIDENT_PHOTOS\n\n"
        "LOCAL DETERMINISTIC FINDINGS (from pre-validation)\n"
        f"- required_missing: {pre_required}\n"
        f"- warnings: {pre_warnings}\n"
        f"- errors: {pre_errors}\n"
        f"- docs_ok: {pre_docs_ok}\n"
        f"- local_recommendation: {pre_rec}\n\n"
        "YOUR TASK\n"
        "---------\n"
        "1) Verify presence of all mandatory documents (do not rely only on headings; check content).\n"
        "2) Check narrative consistency between Description and FIR.\n"
        "3) Check DL validity on incident date.\n"
        "4) Check Policy validity and whether OD/Comprehensive coverage applies.\n"
        "5) Verify vehicle consistency (RC/Policy/FIR registration numbers).\n"
        "6) Evaluate repair estimate plausibility (inflation vs severity & photos).\n"
        "7) Identify fake/unverifiable estimates (non-network, no GSTIN, no part numbers).\n"
        "8) Provide actionable notes for a human manager.\n\n"
        "STRICT OUTPUT (Return ONLY valid JSON, no extra text):\n"
        "{{\n"
        '  "required_missing": ["FIR","DRIVING_LICENSE","RC_BOOK","POLICY_COPY","REPAIR_ESTIMATE","ACCIDENT_PHOTOS"],\n'
        '  "warnings": ["..."],\n'
        '  "errors": ["..."],\n'
        '  "docs_ok": true,\n'
        '  "note": "Short, specific reasoning referencing which checks passed/failed.",\n'
        '  "recommendation": "APPROVE | NEED_MORE_DOCUMENTS | REJECT"\n'
        "}}\n\n"
        "RULES\n"
        "-----\n"
        '- If any mandatory document is missing -> recommendation = "NEED_MORE_DOCUMENTS".\n'
        '- If policy expired or DL invalid -> recommendation = "REJECT".\n'
        '- If narrative mismatch or photo/estimate contradiction and strong -> prefer "NEED_MORE_DOCUMENTS" unless clearly fraudulent -> "REJECT".\n'
        '- If all documents present and no critical errors -> "APPROVE".\n'
        "- Keep note concise (<= 5 lines) but specific.\n"
    )

    return template.format(
        transaction_id=state.transaction_id,
        claim_id=state.claim_id,
        customer_name=state.customer_name,
        policy_number=state.policy_number,
        claim_type=state.claim_type,
        amount=state.amount,
        claim_description=claim_description,
        uploaded_docs=uploaded_docs,
    )


def _build_llm_text_only_prompt(state: ClaimState) -> str:
    """
    Text-only document inference from OCR to reduce false 'missing' when markers/structured docs are absent.
    """
    claim_description = (_normalize(state.extracted_text) or "No accident description provided.")
    uploaded_docs_ocr = (_normalize(state.document_extracted_text) or "No OCR available.")

    template = (
        "You are a Senior Motor Insurance Claim Validation Officer.\n"
        "Work strictly from the OCR text. Do not assume any files exist beyond the OCR content below.\n\n"
        "OCR TEXT (verbatim)\n"
        "-------------------\n"
        "{uploaded_docs}\n\n"
        "ACCIDENT DESCRIPTION\n"
        "--------------------\n"
        "{claim_description}\n\n"
        "TASKS\n"
        "-----\n"
        "1) Determine presence of these documents from OCR text only (true/false) and extract key fields:\n"
        "   - FIR (number, date, vehicle reg, complainant)\n"
        "   - DRIVING_LICENSE (DL no, Valid Until/To)\n"
        "   - RC_BOOK (reg, engine, chassis, owner)\n"
        "   - POLICY_COPY (policy number, period start/end, coverage = COMPREHENSIVE | OD | TP_ONLY | UNKNOWN)\n"
        "   - REPAIR_ESTIMATE (total amount numeric, presence of GSTIN, presence of part numbers)\n"
        "   - ACCIDENT_PHOTOS (describe references if any)\n"
        "2) Validate:\n"
        "   - Policy period covers the incident date (if an incident date is present or inferable).\n"
        "   - DL validity on incident date.\n"
        "   - Vehicle registration consistency across FIR/RC/Policy.\n"
        "3) Provide confidence [0..1] per document presence and citations: short substrings from OCR that justify your decision.\n"
        "4) Produce a strict JSON with this schema ONLY (no extra keys, no extra text):\n"
        "{{\n"
        '  "docs": {{\n'
        '    "FIR": {{"present": true/false, "confidence": 0.0-1.0, "citations": ["..."]}},\n'
        '    "DRIVING_LICENSE": {{"present": true/false, "confidence": 0.0-1.0, "citations": ["..."]}},\n'
        '    "RC_BOOK": {{"present": true/false, "confidence": 0.0-1.0, "citations": ["..."]}},\n'
        '    "POLICY_COPY": {{"present": true/false, "confidence": 0.0-1.0, "citations": ["..."]}},\n'
        '    "REPAIR_ESTIMATE": {{"present": true/false, "confidence": 0.0-1.0, "citations": ["..."]}},\n'
        '    "ACCIDENT_PHOTOS": {{"present": true/false, "confidence": 0.0-1.0, "citations": ["..."]}}\n'
        "  }},\n"
        '  "fields": {{\n'
        '    "policy": {{"number": "string|null", "start": "string|null", "end": "string|null", "coverage": "COMPREHENSIVE|OD|TP_ONLY|UNKNOWN"}},\n'
        '    "dl": {{"number": "string|null", "valid_until": "string|null"}},\n'
        '    "fir": {{"number": "string|null", "date": "string|null", "vehicle_reg": "string|null"}},\n'
        '    "rc": {{"reg": "string|null", "engine": "string|null", "chassis": "string|null", "owner": "string|null"}},\n'
        '    "estimate": {{"total": "number|null", "has_gstin": true/false, "has_part_numbers": true/false}}\n'
        "  }},\n"
        '  "required_missing": ["..."],\n'
        '  "warnings": ["..."],\n'
        '  "errors": ["..."],\n'
        '  "docs_ok": true/false,\n'
        '  "recommendation": "APPROVE|NEED_MORE_DOCUMENTS|REJECT",\n'
        '  "note": "Short, <=5 lines, cite which checks passed/failed."\n'
        "}}\n\n"
        "RULES\n"
        "-----\n"
        "- If any mandatory document is absent in OCR -> include it in required_missing and set docs_ok=false unless other docs decisively support approval.\n"
        "- If policy is TP_ONLY or expired for incident date -> errors must include this; recommendation should be REJECT.\n"
        "- If DL expired on incident date -> REJECT.\n"
        "- If presence is asserted, include at least one citation snippet that appears in the OCR text.\n"
        "- Be conservative if evidence is weak; lower confidence instead of guessing.\n"
    )

    return template.format(
        claim_description=claim_description,
        uploaded_docs=uploaded_docs_ocr,
    )


def _sanitize_llm_dict(data: dict) -> dict:
    return {
        "required_missing": list(data.get("required_missing") or []),
        "warnings": list(data.get("warnings") or []),
        "errors": list(data.get("errors") or []),
        "docs_ok": bool(data.get("docs_ok")) if data.get("docs_ok") is not None else False,
        "note": (_normalize(data.get("note")) or ""),
        "recommendation": _normalize(data.get("recommendation")).upper() or "",
    }


def _safe_llm_text_only_loads(raw: str) -> dict:
    """
    Parse the text-only LLM JSON safely with strict defaults.
    """
    fallback = {
        "docs": {
            "FIR": {"present": False, "confidence": 0.0, "citations": []},
            "DRIVING_LICENSE": {"present": False, "confidence": 0.0, "citations": []},
            "RC_BOOK": {"present": False, "confidence": 0.0, "citations": []},
            "POLICY_COPY": {"present": False, "confidence": 0.0, "citations": []},
            "REPAIR_ESTIMATE": {"present": False, "confidence": 0.0, "citations": []},
            "ACCIDENT_PHOTOS": {"present": False, "confidence": 0.0, "citations": []},
        },
        "fields": {
            "policy": {"number": None, "start": None, "end": None, "coverage": "UNKNOWN"},
            "dl": {"number": None, "valid_until": None},
            "fir": {"number": None, "date": None, "vehicle_reg": None},
            "rc": {"reg": None, "engine": None, "chassis": None, "owner": None},
            "estimate": {"total": None, "has_gstin": False, "has_part_numbers": False},
        },
        "required_missing": [],
        "warnings": [],
        "errors": [],
        "docs_ok": False,
        "recommendation": "NEED_MORE_DOCUMENTS",
        "note": "",
    }
    data = _safe_json_loads(raw, fallback=fallback)
    # Ensure `docs` structure exists for all labels
    docs = data.get("docs") or {}
    for _, label in MANDATORY_DOC_MARKERS:
        if label not in docs:
            docs[label] = {"present": False, "confidence": 0.0, "citations": []}
    data["docs"] = docs
    # Ensure `fields` exists
    if "fields" not in data or not isinstance(data["fields"], dict):
        data["fields"] = fallback["fields"]
    return data


def _merge_text_only_presence(pre: dict, llm_text_only: dict, fraud_score: float) -> Dict[str, bool]:
    """
    Decide final 'present' flags per document using:
      - pre (marker-based) presence
      - llm_text_only presence + confidence + citations
    Fraud-aware thresholds applied.
    """
    tau = 0.75 if (fraud_score or 0.0) < 0.6 else 0.85
    present: Dict[str, bool] = {}

    pre_missing = set(pre.get("required_missing") or [])
    for _, label in MANDATORY_DOC_MARKERS:
        pre_present = (label not in pre_missing)

        d = (llm_text_only.get("docs") or {}).get(label, {}) or {}
        llm_present = bool(d.get("present"))
        llm_conf = float(d.get("confidence") or 0.0)
        llm_cites = d.get("citations") or []
        llm_ok = (llm_present and llm_conf >= tau and len(llm_cites) > 0)

        present[label] = pre_present or llm_ok

    return present


# -----------------------------
# Main Agent
# -----------------------------

def llm_validation_agent(state: ClaimState) -> ClaimState:
    """
    Hybrid validator:
    1) Normalize & deterministic pre-validation
    2) LLM text-only inference to fix OCR-without-markers cases
    3) LLM validation with refined prompt (existing)
    4) Merge results with clear precedence
    5) Always return populated ValidationResult & claim_validated flag
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

    # ---- LLM text-only presence inference (NEW)
    text_only_prompt = _build_llm_text_only_prompt(state)
    try:
        text_only_raw = llm_response(text_only_prompt)
    except Exception:
        text_only_raw = json.dumps({
            "docs": {label: {"present": False, "confidence": 0.0, "citations": []} for _, label in MANDATORY_DOC_MARKERS},
            "fields": {
                "policy": {"number": None, "start": None, "end": None, "coverage": "UNKNOWN"},
                "dl": {"number": None, "valid_until": None},
                "fir": {"number": None, "date": None, "vehicle_reg": None},
                "rc": {"reg": None, "engine": None, "chassis": None, "owner": None},
                "estimate": {"total": None, "has_gstin": False, "has_part_numbers": False},
            },
            "required_missing": [],
            "warnings": [],
            "errors": ["LLM_TEXT_ONLY_CALL_FAILED"],
            "docs_ok": False,
            "recommendation": "NEED_MORE_DOCUMENTS",
            "note": "Text-only LLM call failed."
        })
    llm_text_only = _safe_llm_text_only_loads(text_only_raw)

    # Fraud-aware presence merge (pre markers + LLM OCR inference)
    fraud_score = getattr(state, "fraud_score", 0.0) or 0.0
    present_map = _merge_text_only_presence(pre, llm_text_only, fraud_score)

    # Adjust pre-missing using present_map (so later LLM can't be overruled by pre union)
    adjusted_pre_missing = [label for _, label in MANDATORY_DOC_MARKERS if not present_map.get(label, False)]
    pre_adjusted = dict(pre)
    pre_adjusted["required_missing"] = adjusted_pre_missing

    # ---- Existing LLM validation (kept for compatibility)
    prompt = _build_llm_prompt(state, pre_adjusted)
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

    # ---- Final Merge

    # Presence by each source
    pre_present = {label: (label not in pre_adjusted["required_missing"]) for _, label in MANDATORY_DOC_MARKERS}
    llm_text_present = {label: bool((llm_text_only.get("docs") or {}).get(label, {}).get("present")) for _, label in MANDATORY_DOC_MARKERS}
    llm_missing_set = set(llm.get("required_missing") or [])
    llm_present = {label: (label not in llm_missing_set) for _, label in MANDATORY_DOC_MARKERS}

    # Final presence: if ANY source says present (with the earlier fraud-aware filter applied already to pre_adjusted)
    final_present = {}
    for _, label in MANDATORY_DOC_MARKERS:
        final_present[label] = pre_present[label] or llm_present[label] or llm_text_present[label]

    # Final missing computed from final_present
    required_missing = [label for _, label in MANDATORY_DOC_MARKERS if not final_present[label]]

    # Combine warnings & errors
    warnings = sorted(list(set((pre.get("warnings") or []) + (llm.get("warnings") or []) + (llm_text_only.get("warnings") or []))))
    errors = sorted(list(set((pre.get("errors") or []) + (llm.get("errors") or []) + (llm_text_only.get("errors") or []))))

    # --- Presence-aware cleanup: remove stale findings once presence is confirmed ---
    if final_present.get("FIR", False):
        errors = [e for e in errors if e != "FIR not found"]
    if final_present.get("REPAIR_ESTIMATE", False):
        warnings = [w for w in warnings if w.strip().lower() != "repair estimate not provided or unclear"]

    # docs_ok is true only if no missing and no critical errors
    docs_ok = (len(required_missing) == 0 and not any(e in errors for e in ["Policy expired or not covering OD", "Driving License invalid/expired"]))

    # recommendation precedence
    final_rec = _merge_recommendation(pre_adjusted.get("recommendation", "APPROVE"), llm.get("recommendation", "APPROVE"))

    # Ensure critical rules override
    if required_missing:
        final_rec = "NEED_MORE_DOCUMENTS"
        docs_ok = False
    if any(e in errors for e in ["Policy expired or not covering OD", "Driving License invalid/expired"]):
        final_rec = "REJECT"
        docs_ok = False
    # If nothing missing and no critical errors -> APPROVE
    if docs_ok and not required_missing:
        final_rec = "APPROVE"

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