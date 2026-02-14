from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict

app = FastAPI(title="Enterprise Insurance Claim System")

# -----------------------------
# In-memory database
# -----------------------------
claims_db: Dict[str, dict] = {}


# -----------------------------
# Request Models
# -----------------------------
class ClaimRegistrationRequest(BaseModel):
    claim_id: str
    policy_number: str
    description: str
    amount: float


class ClaimStatusRequest(BaseModel):
    transaction_id: str


# -----------------------------
# Helper Function
# -----------------------------
def generate_confirmation_message(
    claim_id: str,
    policy_number: str,
    transaction_id: str,
    registered_at: datetime
) -> str:

    formatted_date = registered_at.strftime("%B %d, %Y")
    formatted_time = registered_at.strftime("%I:%M %p UTC")

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


# -----------------------------
# Register Claim Endpoint
# -----------------------------
@app.post("/claims/register")
def register_claim(request: ClaimRegistrationRequest):

    transaction_id = str(uuid4())
    registered_at = datetime.now(timezone.utc)

    # Store claim
    claims_db[transaction_id] = {
        "claim_id": request.claim_id,
        "policy_number": request.policy_number,
        "description": request.description,
        "amount": request.amount,
        "status": "Registered",
        "registered_at": registered_at.isoformat()
    }

    message = generate_confirmation_message(
        request.claim_id,
        request.policy_number,
        transaction_id,
        registered_at
    )

    return {
        "transaction_id": transaction_id,
        "registered_at": registered_at.isoformat(),
        "message": message,
        "claim_id": request.claim_id,
        "policy_number": request.policy_number
    }


# -----------------------------
# Check Claim Status
# -----------------------------
@app.post("/claims/status")
def check_status(request: ClaimStatusRequest):

    claim = claims_db.get(request.transaction_id)

    if not claim:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found. Please check your reference number."
        )

    return {
        "transaction_id": request.transaction_id,
        "claim_id": claim["claim_id"],
        "policy_number": claim["policy_number"],
        "status": claim["status"],
        "registered_at": claim["registered_at"]
    }


# -----------------------------
# Manager Decision Endpoint
# -----------------------------
@app.post("/claims/manager/approve/{transaction_id}")
def approve_claim(transaction_id: str):

    claim = claims_db.get(transaction_id)

    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim["status"] = "Approved"

    return {
        "transaction_id": transaction_id,
        "status": "Approved",
        "message": "Claim has been approved successfully."
    }


@app.post("/claims/manager/reject/{transaction_id}")
def reject_claim(transaction_id: str):

    claim = claims_db.get(transaction_id)

    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim["status"] = "Rejected"

    return {
        "transaction_id": transaction_id,
        "status": "Rejected",
        "message": "Claim has been rejected."
    }
