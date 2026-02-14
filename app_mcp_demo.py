# server/app_mcp_demo.py

from fastapi import FastAPI
from fastmcp import FastMCP
from typing import Optional
from datetime import datetime, timezone
import json

# ----------------------
# Backend modules
# ----------------------
from backend.state.claim_state import ClaimState
from backend.db.sqlite_store import init_db, fetch_claim_and_docs, update_claim_fields
from backend.agents.registration_agent import registration_agent
from backend.graph.claim_graph_v3 import claim_graph_v3

# ----------------------
# FastAPI app init
# ----------------------
app = FastAPI(title="Simple MCP Insurance Server")

@app.on_event("startup")
def startup():
    init_db()
    print("[DB] Initialized")

# ----------------------
# Convert FastAPI app into MCP
# ----------------------
mcp = FastMCP.from_fastapi(app=app)  # no 'title' argument

# ----------------------
# MCP Tool: Register Claim
# ----------------------
@mcp.tool
async def ClaimRegistrationTool(
    claim_id: str,
    customer_name: str,
    policy_number: str,
    description: str,
    amount: float,
    claim_type: str
):
    # Build initial state
    state = ClaimState(
        claim_id=claim_id,
        customer_name=customer_name,
        policy_number=policy_number,
        amount=amount,
        claim_type=claim_type,
        extracted_text=description,
    )

    # Run registration agent
    state = registration_agent(state)

    # Persist claim
    update_claim_fields(
        state.transaction_id,
        extracted_text=description,
        status="REGISTERED",
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": state.transaction_id,
        "registered_at": state.registered_at,
        "claim_id": state.claim_id,
        "policy_number": state.policy_number
    }

# ----------------------
# MCP Tool: Check Status
# ----------------------
@mcp.tool
def ClaimStatusTool(transaction_id: str):
    claim, _ = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Transaction not found"}

    return {
        "transaction_id": claim["transaction_id"],
        "claim_id": claim["claim_id"],
        "policy_number": claim["policy_number"],
        "status": claim["status"],
        "final_decision": claim.get("final_decision")
    }

# ----------------------
# MCP Tool: AI Processing (Manager)
# ----------------------
from backend.agents.manager_agent import ManagerAgent

@mcp.tool
async def ManagerProcessingTool(transaction_id: str):
    # 1️⃣ Fetch claim from DB
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    # 2️⃣ Build ClaimState
    state = ClaimState(
        transaction_id=claim["transaction_id"],
        claim_id=claim["claim_id"],
        customer_name=claim["customer_name"],
        policy_number=claim["policy_number"],
        amount=claim["amount"],
        claim_type=claim["claim_type"],
        extracted_text=claim["extracted_text"],
        claim_registered=True,
        registered_at=claim["registered_at"],
        documents=docs
    )

    # 3️⃣ Run AI workflow (claim_graph_v3)
    final_state = await claim_graph_v3.ainvoke(state)
    if not isinstance(final_state, dict):
        final_state = final_state.model_dump()

    # 4️⃣ Run Manager Agent
    manager = ManagerAgent()
    manager_result = manager.run(state)

    # 5️⃣ Persist only JSON-serializable fields to DB
    update_claim_fields(
        transaction_id,
        final_decision=manager_result.get("final_decision"),
        status=final_state.get("final_decision") or "UNDER_REVIEW",
        fraud_score=final_state.get("fraud_score"),
        fraud_decision=final_state.get("fraud_decision"),
        validation=str(final_state.get("validation")),  # convert to string if complex
        manager_decision=manager_result.get("manager_decision"),
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    # 6️⃣ Return clean JSON for UI
    return {
        "transaction_id": transaction_id,
        "claim_id": claim["claim_id"],
        "policy_number": claim["policy_number"],
        "customer_name": claim["customer_name"],
        "description": claim["extracted_text"],
        "amount": claim["amount"],
        "claim_type": claim["claim_type"],
        "documents": docs,
        "status": manager_result.get("final_decision") or "UNDER_REVIEW",
        "final_decision": manager_result.get("final_decision"),
        "validation": str(final_state.get("validation")),  # safe for JSON
        "fraud_score": final_state.get("fraud_score"),
        "fraud_decision": final_state.get("fraud_decision"),
        "manager_decision": manager_result.get("manager_decision"),
        "ai_raw_state": final_state  # optional, for UI/debug
    }

# ----------------------
# MCP Tool: Manager Decision (Override)
# ----------------------
@mcp.tool
def ManagerDecisionTool(transaction_id: str, decision: str, comment: Optional[str] = None):
    claim, _ = fetch_claim_and_docs(transaction_id)
    if not claim:
        return {"error": "Claim not found"}

    decision = decision.upper()
    valid_decisions = ["APPROVED", "REJECTED", "PENDING_DOCUMENTS"]
    if decision not in valid_decisions:
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
        "status": decision,
        "message": f"Claim has been {decision.replace('_', ' ').lower()} successfully."
    }

# ----------------------
# Run MCP server (HTTP)
# ----------------------
if __name__ == "__main__":
    # HTTP transport exposes tools at /tools/<ToolName>
    mcp.run(transport="http", host="0.0.0.0", port=8000)
