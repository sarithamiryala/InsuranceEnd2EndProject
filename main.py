
from fastapi import FastAPI
from fastmcp import FastMCP
from typing import Optional
from datetime import datetime, timezone 
from backend.utils.state_builder import build_state_from_db

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
from backend.agents.validation_agent import validation_agent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent
from backend.agents.manager_agent import ManagerAgent
from backend.graph.claim_graph_v3 import claim_graph_v3

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

@mcp.tool
async def UpdateDocumentExtractedTextTool(
    transaction_id: str,
    extracted_text: str
):
    """
    Updates OCR extracted text of uploaded documents
    for a registered insurance claim.

    Use this tool AFTER customer uploads FIR / DL / RC /
    repair estimate etc.

    This text will later be used by AI Validation Agent
    during claim validation.
    """

    claim, docs = fetch_claim_and_docs(transaction_id)

    if not claim:
        return {"error": "Transaction ID not found. Please register claim first."}

    update_claim_fields(
        transaction_id,
        document_extracted_text=extracted_text,
        status="DOCUMENTS_UPLOADED",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": transaction_id,
        "message": "Uploaded documents processed successfully.",
        "status": "DOCUMENTS_UPLOADED"
    }


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

    state = build_state_from_db(claim, docs)
    state = llm_validation_agent(state)

    update_claim_fields(
        transaction_id,
        validation=str(state.validation.model_dump()),
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
    Executes the full AI-driven claim processing workflow.

    Use this tool to automatically run the complete claim lifecycle
    including validation, fraud detection, investigation assignment,
    and decision making.

    Produces final claim decision such as:
    APPROVED, REJECTED, or PENDING_DOCUMENTS.
    """
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    state = build_state_from_db(claim, docs)

    # Run graph workflow
    final_state = await claim_graph_v3.ainvoke(state)
    if not isinstance(final_state, dict):
        final_state = final_state.model_dump()

    # Run Manager Agent
    manager = ManagerAgent()
    manager_result = manager.run(state)

    update_claim_fields(
        transaction_id,
        final_decision=manager_result.get("final_decision"),
        status=manager_result.get("final_decision") or "UNDER_REVIEW",
        fraud_score=final_state.get("fraud_score"),
        fraud_decision=final_state.get("fraud_decision"),
        validation=str(final_state.get("validation")),
        manager_decision=manager_result.get("manager_decision"),
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": transaction_id,
        "final_decision": manager_result.get("final_decision"),
        "manager_decision": manager_result.get("manager_decision"),
        "fraud_score": final_state.get("fraud_score"),
        "fraud_decision": final_state.get("fraud_decision"),
        "validation": final_state.get("validation")
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
