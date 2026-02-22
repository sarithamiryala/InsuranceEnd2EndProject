
from fastapi import FastAPI
from fastmcp import FastMCP
from typing import Optional
from datetime import datetime, timezone 
import json
import os
from backend.utils.state_builder import build_state_from_db 
from fastapi.encoders import jsonable_encoder
from backend.state.claim_state import ClaimState
from backend.db.postgres_store import (
    init_db,
    fetch_claim_and_docs,
    update_claim_fields,
    upsert_claim_registration,
    insert_documents
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
    init_db(non_blocking=True)  # no network connection here 
@app.get("/livez")
def livez():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    from fastapi.responses import JSONResponse
    from backend.db.postgres_store import ping_db
    return {"status": "ready"} if ping_db(2) else JSONResponse({"status": "not_ready"}, 503)


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


@mcp.tool
async def UpdateDocumentExtractedTextTool(
    transaction_id: str,
    extracted_text: str,
    overwrite: bool = False  # default: merge/append
        ):
    """
    Stores OCR extracted text for a claim, with optional merge/overwrite.

    Parameters:
    - transaction_id: str
    - extracted_text: str
    - overwrite: bool (default False)

    Behavior:
    - overwrite=False: merges new text with existing OCR content
    - overwrite=True: replaces existing OCR content
    - Never wipes existing content with empty input

    Returns:
    - transaction_id, status, preview of saved text, message
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

    This tool analyzes uploaded claim documents using OCR extracted text
    and evaluates document completeness and legitimacy.

    Updates:
    - validation
    - claim_validated
    - status

    Status becomes:
    - AI_VALIDATED → if all required docs are present
    - PENDING_DOCUMENTS → if documents are incomplete

    Returns updated claim validation result.
    """

    # 1️⃣ Load claim from PostgreSQL
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    # 2️⃣ Build ClaimState
    state = build_state_from_db(claim, docs)

    # 3️⃣ Run AI validation
    state = llm_validation_agent(state)

    # 4️⃣ Persist VALIDATION as JSONB (NOT STRING)
    update_claim_fields(
        transaction_id,
        validation=json.dumps(state.validation.model_dump()),  # ✅ JSONB SAFE
        claim_validated=state.claim_validated,
        status="AI_VALIDATED" if state.claim_validated else "PENDING_DOCUMENTS",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    # 5️⃣ Return response
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

# ============================================================
# 6️⃣ FULL AI GRAPH PROCESSING
# ============================================================

# @mcp.tool
# async def ManagerProcessingTool(transaction_id: str):
#     """
#     Runs the complete post-registration AI claim processing workflow.

#     This tool executes the AI claim lifecycle AFTER claim registration
#     and document upload have been completed.

#     It automatically performs:
#     1. AI-based Claim Validation using uploaded document OCR text
#     2. Fraud Risk Scoring (if documents are complete)
#     3. Investigator escalation (if fraud risk is high)
#     4. Final Manager Decision

#     Workflow:
#     - If documents are incomplete → Claim marked as PENDING_DOCUMENTS
#     - If documents are complete → Fraud score is calculated
#     - If fraud_score ≥ 0.7 → Claim escalated for investigation
#     - Otherwise → Final decision made based on validation recommendation

#     Updates claim fields:
#     - validation
#     - fraud_score
#     - fraud_decision
#     - final_decision
#     - manager_decision
#     - status

#     Returns:
#     {
#         transaction_id: str,
#         final_decision: APPROVED | REJECTED | PENDING_DOCUMENTS | ESCALATED_TO_SIU,
#         fraud_score: float,
#         fraud_decision: SAFE | MODERATE | SUSPECT,
#         validation: object
#     }
#     """

#     from fastapi.encoders import jsonable_encoder
#     from backend.agents.llm_validation_agent import llm_validation_agent
#     from backend.graph.claim_graph_v3 import manager_node

#     # 1️⃣ Load claim
#     claim, docs = fetch_claim_and_docs(transaction_id)
#     if not claim:
#         return {"error": "Claim not found"}

#     # 2️⃣ Rebuild state
#     state = build_state_from_db(claim, docs)

#     # 3️⃣ AI-based validation (if not already validated)
#     if not getattr(state, "claim_validated", False):
#         state = llm_validation_agent(state)

#     # Persist validation safely as JSON
#     validation_json = jsonable_encoder(getattr(state, "validation", {}))

#     update_claim_fields(
#         transaction_id,
#         validation=validation_json,
#         claim_validated=state.claim_validated,
#         status="AI_VALIDATED" if state.claim_validated else "PENDING_DOCUMENTS",
#         updated_at=datetime.now(timezone.utc).isoformat()
#     )

#     # 4️⃣ Fraud scoring (if docs ok)
#     if getattr(state.validation, "docs_ok", False):
#         state = fraud_agent(state)

#     # 5️⃣ Investigator assignment for high fraud
#     if getattr(state, "fraud_score", 0.0) >= 0.7 and getattr(state, "fraud_checked", False):
#         state = investigator_agent(state)

#     # 6️⃣ Manager node final decision
#     state = manager_node(state)

#     # 7️⃣ Ensure status aligns with final_decision
#     final_decision = getattr(state, "final_decision", None) or claim.get("status") or "UNDER_REVIEW"
#     state.status = final_decision

#     # 8️⃣ Persist final state safely
#     assignment_json = jsonable_encoder(getattr(state, "assignment", {}))
#     validation_json = jsonable_encoder(getattr(state, "validation", {}))

#     update_claim_fields(
#         transaction_id,
#         final_decision=final_decision,
#         status=state.status,
#         fraud_score=getattr(state, "fraud_score", None),
#         fraud_decision=getattr(state, "fraud_decision", None),
#         investigator_id=assignment_json.get("investigator_id") if assignment_json else None,
#         assignment=json.dumps(assignment_json or {}),
#         validation=validation_json,
#         manager_decision="Finalized by manager node",
#         updated_at=datetime.now(timezone.utc).isoformat()
#     )

#     # 9️⃣ Return results
#     return {
#         "transaction_id": transaction_id,
#         "final_decision": final_decision,
#         "manager_decision": "Finalized by manager node",
#         "fraud_score": getattr(state, "fraud_score", None),
#         "fraud_decision": getattr(state, "fraud_decision", None),
#         "validation": validation_json,
#         "assignment": assignment_json
#     }

#     # 9️⃣ Return results
#     return {
#         "transaction_id": transaction_id,
#         "final_decision": final_decision,
#         "manager_decision": "Finalized by manager node",
#         "fraud_score": getattr(state, "fraud_score", None),
#         "fraud_decision": getattr(state, "fraud_decision", None),
#         "validation": getattr(state, "validation", {}),
#         "assignment": getattr(state, "assignment", {})
#     } 
@mcp.tool
async def ManagerProcessingTool(transaction_id: str):
    """
    Runs the complete post-registration AI claim processing workflow.

    This tool executes the AI claim lifecycle AFTER claim registration
    and document upload have been completed.

    It automatically performs:
    1. AI-based Claim Validation using uploaded document OCR text
    2. Fraud Risk Scoring (if documents are complete)
    3. Investigator escalation (if fraud risk is high)
    4. Final Manager Decision

    Workflow:
    - If documents are incomplete → Claim marked as PENDING_DOCUMENTS
    - If documents are complete → Fraud score is calculated
    - If fraud_score ≥ 0.7 → Claim escalated for investigation
    - Otherwise → Final decision made based on validation recommendation

    Updates claim fields:
    - validation
    - fraud_score
    - fraud_decision
    - final_decision
    - manager_decision
    - status

    Returns:
    {
        transaction_id: str,
        final_decision: APPROVED | REJECTED | PENDING_DOCUMENTS | ESCALATED_TO_SIU,
        fraud_score: float,
        fraud_decision: SAFE | MODERATE | SUSPECT,
        validation: object
    }
    """
    from fastapi.encoders import jsonable_encoder
    from backend.agents.llm_validation_agent import llm_validation_agent
    from backend.graph.claim_graph_v3 import manager_node

    # 1️⃣ Load claim
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    # 2️⃣ Rebuild state
    state = build_state_from_db(claim, docs)

    # 3️⃣ AI-based validation (if not already validated)
    if not getattr(state, "claim_validated", False):
        state = llm_validation_agent(state)

    # 4️⃣ Fraud scoring (if docs ok)
    if getattr(state.validation, "docs_ok", False):
        state = fraud_agent(state)
        state.fraud_checked = True  # mark fraud check done

    # 5️⃣ Investigator assignment for high fraud
    if getattr(state, "fraud_score", 0.0) >= 0.7 and getattr(state, "fraud_checked", False):
        state = investigator_agent(state)

    # 6️⃣ Manager node final decision
    state = manager_node(state)

    # 7️⃣ Ensure final decision is set
    final_decision = getattr(state, "final_decision", None) or claim.get("status") or "UNDER_REVIEW"

    # 8️⃣ Convert complex objects to JSON strings for DB
    validation_json_str = json.dumps(jsonable_encoder(getattr(state, "validation", {})))
    assignment_json_str = json.dumps(jsonable_encoder(getattr(state, "assignment", {})))

    # 9️⃣ Persist final state safely
    update_claim_fields(
        transaction_id,
        final_decision=final_decision,
        status=final_decision,  # DB column
        fraud_score=getattr(state, "fraud_score", None),
        fraud_decision=getattr(state, "fraud_decision", None),
        fraud_checked=getattr(state, "fraud_checked", False),
        investigator_id=state.assignment.investigator_id if state.assignment else None,
        assignment=assignment_json_str,  # JSON string for DB
        validation=validation_json_str,  # JSON string for DB
        manager_decision="Finalized by manager node",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    # 10️⃣ Return JSON-safe dict for API response
    return {
        "transaction_id": transaction_id,
        "final_decision": final_decision,
        "manager_decision": "Finalized by manager node",
        "fraud_score": getattr(state, "fraud_score", None),
        "fraud_decision": getattr(state, "fraud_decision", None),
        "validation": jsonable_encoder(getattr(state, "validation", {})),
        "assignment": jsonable_encoder(getattr(state, "assignment", {}))
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
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )