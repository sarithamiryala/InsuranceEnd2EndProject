
from fastapi import FastAPI
from fastmcp import FastMCP
from typing import Optional
from datetime import datetime, timezone 
import json
from backend.utils.state_builder import build_state_from_db 
from fastapi.encoders import jsonable_encoder
# ----------------------
# Backend modules
# ----------------------
from backend.state.claim_state import ClaimState
from backend.db.sqlite_store import (
    init_db,
    fetch_claim_and_docs,
    update_claim_fields
)
from backend.agents.registration_agent import registration_agent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent
from backend.agents.manager_agent import ManagerAgent
from backend.graph.claim_graph_v3 import claim_graph_v3
from backend.graph.claim_graph_v3 import claim_graph_v3_postreg 

# ----------------------
# FastAPI app init
# ----------------------
app = FastAPI(title="Enterprise MCP Insurance Server")

@app.on_event("startup")
def startup():
    init_db()
    print("[DB] Initialized")

# ----------------------
# Convert FastAPI → MCP
# ----------------------
mcp = FastMCP.from_fastapi(app=app)

# ============================================================
# 1️⃣ CLAIM REGISTRATION TOOL
# ============================================================

@mcp.tool
async def ClaimRegistrationTool(
    claim_id: str,
    customer_name: str,
    policy_number: str,
    description: str,
    amount: float,
    claim_type: str
): 
    """
    Registers a new insurance claim in the system.

    Use this tool when a customer wants to create or submit a new insurance claim.
    It stores claim details such as claim ID, customer name, policy number,
    claim amount, claim type (motor or health), and claim description.

    This is the FIRST step in the insurance claim lifecycle.
    Returns a transaction ID used for further claim processing.
    """
    
# Normalize at the boundary
    claim_id = (claim_id or "").strip()
    customer_name = (customer_name or "").strip()
    policy_number = (policy_number or "").strip()
    description = (description or "").strip()
    claim_type = (claim_type or "").strip().lower()

    state = ClaimState(
        claim_id=claim_id,
        customer_name=customer_name,
        policy_number=policy_number,
        amount=amount,
        claim_type=claim_type,
        extracted_text=description,
    )

    state = registration_agent(state)

    update_claim_fields(
        state.transaction_id,
        extracted_text=description,
        status="REGISTERED",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": state.transaction_id,
        "registered_at": state.registered_at,
        "claim_id": state.claim_id
    }


# ============================================================
# 2️⃣ DOCUMENT OCR TEXT UPDATE TOOL (FRONTEND → DB)
# ============================================================
from datetime import datetime, timezone
from typing import Optional, Dict, List
import re

from backend.db.sqlite_store import (
    fetch_claim_and_docs,
    update_claim_fields,
    insert_documents,
)

# --------------------------------------------------------------------
# Helpers: section rendering & merging
# --------------------------------------------------------------------
_DOC_TITLES = [
    "FIR",
    "DRIVING_LICENSE",
    "RC_BOOK",
    "POLICY_COPY",
    "REPAIR_ESTIMATE",
    "ACCIDENT_PHOTOS",
    "MISC",  # used only when we need a catch‑all
]

def _render_section(title: str, body: str) -> str:
    body = (body or "").strip()
    return f"=== {title} ===\n{body}\n" if body else ""

def _split_sections(text: str) -> Dict[str, str]:
    """
    Split a canonical OCR block into {SECTION: body}.
    Accepts text with '=== SECTION ===' headers. If no headers found, returns {"MISC": text}.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return {}

    sections: Dict[str, str] = {}
    # Split by header lines, keep titles
    parts = re.split(r"(?m)^\s*===\s*([A-Z_]+)\s*===\s*$", text)
    # parts -> ["before?", TITLE1, BODY1, TITLE2, BODY2, ...]
    if len(parts) >= 3:
        it = iter(parts[1:])
        for title, body in zip(it, it):
            title = title.strip().upper()
            sections[title] = (body or "").strip()
        # If the very first chunk before the first header had content, push to MISC
        head = (parts[0] or "").strip()
        if head:
            sections.setdefault("MISC", head)
    else:
        # No headers at all; put entire text under MISC
        sections["MISC"] = text

    return sections

def _merge_section_maps(old_map: Dict[str, str], new_map: Dict[str, str]) -> Dict[str, str]:
    """
    Merge section dicts; new non‑empty content replaces old. Missing sections keep old.
    """
    if not old_map:
        return dict(new_map)
    if not new_map:
        return dict(old_map)

    merged = dict(old_map)
    for k, v in new_map.items():
        if (v or "").strip():
            merged[k] = v.strip()
    return merged

def _render_canonical_block(sections: Dict[str, str]) -> str:
    """
    Render sections dict into canonical OCR block in a fixed order.
    """
    lines = []
    for t in _DOC_TITLES:
        if t in sections and sections[t]:
            lines.append(f"=== {t} ===")
            lines.append(sections[t].strip())
    # Add any unknown keys not in the ordered list
    for k in sections:
        if k not in _DOC_TITLES and sections[k]:
            lines.append(f"=== {k} ===")
            lines.append(sections[k].strip())

    return ("\n".join(lines)).strip()

def _sections_present(text: str) -> List[str]:
    present = []
    for t in _DOC_TITLES:
        if f"=== {t} ===" in (text or ""):
            present.append(t)
    return present


# --------------------------------------------------------------------
# The tool: UpdateDocumentExtractedTextTool
# --------------------------------------------------------------------
from datetime import datetime, timezone
from typing import Optional

from backend.db.sqlite_store import fetch_claim_and_docs, update_claim_fields

@mcp.tool
async def UpdateDocumentExtractedTextTool(
    transaction_id: str,
    extracted_text: str,
    overwrite: bool = False  # default: merge/append
):
    """
    Store OCR extracted text for a claim (single blob), with optional overwrite.
    - If overwrite=False (default), merges by appending with a separator line.
    - If overwrite=True, replaces existing content.
    - Never wipes with empty input.
    """

    new_text = (extracted_text or "").replace("\r\n", "\n").strip()

    # 1) Ensure txn exists
    claim, _ = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Transaction ID not found. Please register claim first."}

    existing = (claim.get("document_extracted_text") or "").strip()

    # 2) Decide final text
    if overwrite:
        final_text = new_text if new_text else existing  # don't wipe with empty
    else:
        if not new_text:
            # merge mode + empty input => keep existing
            final_text = existing
        elif not existing:
            final_text = new_text
        else:
            # Simple merge strategy: append a separator (you can make this smarter later)
            sep = "\n\n---\n\n"
            final_text = existing + sep + new_text

    # 3) Persist (only set column if something to save)
    updates = {
        "status": "DOCUMENTS_UPLOADED",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    if final_text:
        updates["document_extracted_text"] = final_text

    update_claim_fields(transaction_id, **updates)

    # 4) Read-back confirmation
    saved, _ = fetch_claim_and_docs(transaction_id)
    saved_text = (saved.get("document_extracted_text") or "")
    return {
        "transaction_id": transaction_id,
        "status": saved.get("status"),
        "saved_len": len(saved_text),
        "message": "Merged with existing content." if (existing and not overwrite and new_text) else "Saved.",
        "preview": saved_text[:800]
    }
# @mcp.tool
# async def UpdateDocumentExtractedTextTool(
#     transaction_id: str,
#     extracted_text: str
# ):
#     """
#     Updates OCR extracted text of uploaded documents
#     for a registered insurance claim.

#     Use this tool AFTER customer uploads FIR / DL / RC /
#     repair estimate etc.

#     This text will later be used by AI Validation Agent
#     during claim validation.
#     """

#     claim, docs = fetch_claim_and_docs(transaction_id)

#     if not claim:
#         return {"error": "Transaction ID not found. Please register claim first."}

#     update_claim_fields(
#         transaction_id,
#         document_extracted_text=extracted_text,
#         status="DOCUMENTS_UPLOADED",
#         updated_at=datetime.now(timezone.utc).isoformat()
#     )

#     return {
#         "transaction_id": transaction_id,
#         "message": "Uploaded documents processed successfully.",
#         "status": "DOCUMENTS_UPLOADED"
#     }


# ============================================================
# 2 LLM VALIDATION TOOL
# ============================================================

@mcp.tool
async def ClaimLLMValidationTool(transaction_id: str):
    """
    Performs AI-based validation of a claim using LLM reasoning.

    Use this tool when rule-based validation is insufficient and
    intelligent document analysis is required.

    It evaluates claim legitimacy and document completeness using AI.
    Updates claim status as:
    AI_VALIDATED or PENDING_DOCUMENTS.
    """
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

        # inside ClaimLLMValidationTool
    state = build_state_from_db(claim, docs)
    state = llm_validation_agent(state)

    # Serialize proper JSON (not str(dict))
    validation_json = json.dumps(state.validation.model_dump(), ensure_ascii=False)

    update_claim_fields(
        transaction_id,
        validation=validation_json,
        status="AI_VALIDATED" if state.claim_validated else "PENDING_DOCUMENTS",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return state.model_dump()

# ============================================================
# 4️⃣ FRAUD CHECK TOOL
# ============================================================

@mcp.tool
async def FraudCheckTool(transaction_id: str):
    """
    Performs fraud risk assessment for an insurance claim.

    Use this tool after validation to determine the fraud probability
    of the submitted claim.

    It generates:
    - Fraud score
    - Fraud decision (LOW, MEDIUM, HIGH risk)

    Updates claim status to FRAUD_CHECKED.
    """
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    state = build_state_from_db(claim, docs)
    state = fraud_agent(state)

    update_claim_fields(
        transaction_id,
        fraud_score=state.fraud_score,
        fraud_decision=state.fraud_decision,
        status="FRAUD_CHECKED",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return state.model_dump()

# ============================================================
# 5️⃣ INVESTIGATOR ASSIGNMENT TOOL
# ============================================================

@mcp.tool
async def InvestigatorAssignmentTool(transaction_id: str):
    """
    Assigns an investigator to a claim if fraud risk is detected.

    Use this tool when fraud score is high or manual verification
    of claim documents is required.

    Updates claim status to:
    UNDER_INVESTIGATION or NO_INVESTIGATION_REQUIRED.
    """
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    state = build_state_from_db(claim, docs)
    state = investigator_agent(state)

    update_claim_fields(
        transaction_id,
        investigator_id=state.assignment.investigator_id,
        status="UNDER_INVESTIGATION"
        if state.assignment.investigator_id else "NO_INVESTIGATION_REQUIRED",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return state.model_dump()

# ============================================================
# 6️⃣ FULL AI GRAPH PROCESSING
# ============================================================


@mcp.tool
async def ManagerProcessingTool(transaction_id: str):
    """
    Runs the complete post‑registration AI claim processing workflow.

    This tool executes the claim flow starting from validation, and then
    automatically routes through fraud scoring (when documents are complete)
    and finally to the manager node, which produces the final claim decision.

    Workflow (matches business rule diagram):
    1. LLM Validation
    2. If documents are NOT OK → Manager → PENDING_DOCUMENTS / REJECTED
    3. If documents are OK → Fraud Scoring
    4. If fraud_score ≥ 0.7 → ESCALATED_TO_SIU
    5. Otherwise → Manager decision using validation recommendation
    (APPROVE, REJECT, or NEED_MORE_DOCUMENTS)

    Returns:
    - final_decision      → APPROVED | REJECTED | PENDING_DOCUMENTS | ESCALATED_TO_SIU
    - fraud_score         → float or null (if fraud not executed)
    - fraud_decision      → SAFE | MODERATE | SUSPECT or null
    - validation          → full validation result
    """
    # 1) Load state from DB
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    state = build_state_from_db(claim, docs)

    # 2) Run graph (validate -> fraud (optional by routing) -> manager(finalize) -> END)
    final_state = await claim_graph_v3_postreg.ainvoke(state)

    # 3) Convert the ENTIRE result to JSON-safe structure
    #    This flattens any Pydantic models (including ValidationResult), datetimes, etc.
    returned = jsonable_encoder(
        final_state,
        custom_encoder={datetime: lambda v: v.isoformat()}
    )
    # `returned` is now a plain dict of primitives (JSON serializable)

    # 4) Persist: store `validation` as proper JSON string (never str(dict), never a Pydantic object)
    validation_json = None
    if "validation" in returned and returned["validation"] is not None:
        # returned["validation"] is a plain dict now
        validation_json = json.dumps(returned["validation"], ensure_ascii=False)

    update_claim_fields(
        transaction_id,
        final_decision=returned.get("final_decision"),
        status=returned.get("final_decision") or claim.get("status") or "UNDER_REVIEW",
        fraud_score=returned.get("fraud_score"),
        fraud_decision=returned.get("fraud_decision"),
        validation=validation_json,  # <-- JSON string (or None)
        manager_decision="Finalized by manager node",
        updated_at=datetime.now().isoformat()
    )

    # 5) Return JSON-safe response (FastAPI will json.dumps this without error)
    return {
        "transaction_id": transaction_id,
        "final_decision": returned.get("final_decision"),
        "manager_decision": "Finalized by manager node",
        "fraud_score": returned.get("fraud_score"),
        "fraud_decision": returned.get("fraud_decision"),
        "validation": returned.get("validation"),
    }
# ============================================================
# 7️⃣ MANUAL MANAGER OVERRIDE
# ============================================================

@mcp.tool
def ManagerDecisionTool(transaction_id: str, decision: str, comment: Optional[str] = None):
    """
    Allows manual override of claim decision by a manager.

    Use this tool when a human decision is required to approve,
    reject, or request additional documents for a claim.

    Valid decisions:
    APPROVED, REJECTED, PENDING_DOCUMENTS
    """
    decision = decision.upper()
    valid = ["APPROVED", "REJECTED", "PENDING_DOCUMENTS"]

    if decision not in valid:
        return {"error": "Invalid decision"}

    update_claim_fields(
        transaction_id,
        final_decision=decision,
        status=decision,
        manager_comment=comment,
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": transaction_id,
        "status": decision
    }

# ============================================================
# 8️⃣ STATUS CHECK TOOL
# ============================================================

@mcp.tool
def ClaimStatusTool(transaction_id: str):
    """
    Retrieves the current processing status of an insurance claim.

    Use this tool when a user wants to check the progress or
    final decision of a submitted claim.

    Returns claim status and final approval or rejection decision.
    """
    claim, _ = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Transaction not found"}

    return {
        "transaction_id": claim["transaction_id"],
        "claim_id": claim["claim_id"],
        "status": claim["status"],
        "final_decision": claim.get("final_decision")
    }

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
 