# server/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional

# Graph
from backend.graph.claim_graph_v3 import claim_graph_v3

# State
from backend.state.claim_state import ClaimState

# DB
from backend.db.sqlite_store import (
    init_db,
    fetch_claim_and_docs,
    update_claim_fields,
)

# --------------------------------------------------
# App Init
# --------------------------------------------------

app = FastAPI(title="Enterprise Insurance Claim System")

@app.on_event("startup")
def startup():
    init_db()
    print("Database initialized.")


# ==================================================
# REQUEST MODELS
# ==================================================

class ClaimRegistrationRequest(BaseModel):
    claim_id: str
    customer_name: str
    policy_number: str
    description: str
    amount: float
    claim_type: str


class ClaimStatusRequest(BaseModel):
    transaction_id: str


class ManagerDecisionRequest(BaseModel):
    decision: str  # APPROVED / REJECTED


# ==================================================
# HELPER - Confirmation Message
# ==================================================

def generate_confirmation_message(
    claim_id: str,
    policy_number: str,
    transaction_id: str,
    registered_at: str
) -> str:

    dt = datetime.fromisoformat(registered_at)
    formatted_date = dt.strftime("%B %d, %Y")
    formatted_time = dt.strftime("%I:%M %p UTC")

    return (
        f"Thank you for registering your claim.\n\n"
        f"Your claim '{claim_id}' under policy '{policy_number}' "
        f"was successfully registered on {formatted_date} at {formatted_time}.\n\n"
        f"Your reference number for this transaction is:\n"
        f"{transaction_id}\n\n"
        f"Our team will now validate and review your claim. "
        f"You can use this reference number to track the status anytime.\n\n"
        f"Thank you for choosing our insurance services."
    )


# ==================================================
# CUSTOMER: REGISTER CLAIM
# ==================================================

@app.post("/claims/register")
async def register_claim(request: ClaimRegistrationRequest):

    # Build initial state
    state = ClaimState(
        claim_id=request.claim_id,
        customer_name=request.customer_name,
        policy_number=request.policy_number,
        amount=request.amount,
        claim_type=request.claim_type,
        extracted_text=request.description,
    )

    # Run only registration node (not full graph)
    from backend.agents.registration_agent import registration_agent
    state = registration_agent(state)

    message = generate_confirmation_message(
        state.claim_id,
        state.policy_number,
        state.transaction_id,
        state.registered_at
    )

    return {
        "transaction_id": state.transaction_id,
        "registered_at": state.registered_at,
        "message": message,
        "claim_id": state.claim_id,
        "policy_number": state.policy_number
    }


# ==================================================
# CUSTOMER: CHECK STATUS
# ==================================================

@app.post("/claims/status")
def check_status(request: ClaimStatusRequest):

    claim, _ = fetch_claim_and_docs(request.transaction_id)

    if not claim:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found. Please check your reference number."
        )

    return {
        "transaction_id": claim["transaction_id"],
        "claim_id": claim["claim_id"],
        "policy_number": claim["policy_number"],
        "status": claim["status"],
        "registered_at": claim["registered_at"],
        "final_decision": claim.get("final_decision")
    }


# ==================================================
# MANAGER: RUN FULL AI FLOW
# ==================================================

@app.post("/claims/manager/process/{transaction_id}")
async def process_claim(transaction_id: str):

    claim, docs = fetch_claim_and_docs(transaction_id)

    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Rebuild state
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
    )

    final_state = await claim_graph_v3.ainvoke(state)

    # Ensure dict
    if not isinstance(final_state, dict):
        final_state = final_state.model_dump()  
    
    
   
    

    return {
        "transaction_id": final_state.get("transaction_id"),
        "final_decision": final_state.get("final_decision"),
        "fraud_score": final_state.get("fraud_score"),
        "fraud_decision": final_state.get("fraud_decision"),
        "validation": final_state.get("validation"),
        "assignment": final_state.get("assignment"),
    }



# ==================================================
# MANAGER: FINAL DECISION (Human Override)
# ==================================================

@app.post("/claims/manager/decision/{transaction_id}")
def manager_decision(transaction_id: str, request: ManagerDecisionRequest):

    claim, _ = fetch_claim_and_docs(transaction_id)

    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    decision = request.decision.upper()

    if decision not in ["APPROVED", "REJECTED", "PENDING_DOCUMENTS"]:
        raise HTTPException(status_code=400, detail="Invalid decision")

    update_claim_fields(
        transaction_id,
        final_decision=decision,
        status=decision,
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    return {
        "transaction_id": transaction_id,
        "status": decision,
        "message": f"Claim has been {decision.replace('_', ' ').lower()} successfully."
    }

