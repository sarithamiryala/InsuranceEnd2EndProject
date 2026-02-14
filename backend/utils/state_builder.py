from backend.state.claim_state import ClaimState


def build_state_from_db(claim: dict, docs: list) -> ClaimState:
    """
    Safely reconstruct ClaimState from DB row.
    Prevents None → Pydantic validation errors.
    """

    return ClaimState(
        transaction_id=claim.get("transaction_id"),
        claim_id=claim.get("claim_id"),
        customer_name=claim.get("customer_name"),
        policy_number=claim.get("policy_number"),
        amount=claim.get("amount"),
        claim_type=claim.get("claim_type"),
        extracted_text=claim.get("extracted_text"),

        claim_registered=bool(claim.get("claim_registered", False)),
        registered_at=claim.get("registered_at"),

        claim_validated=bool(claim.get("claim_validated", False)),

        fraud_checked=bool(claim.get("fraud_checked", False)),
        fraud_score=claim.get("fraud_score"),
        fraud_decision=claim.get("fraud_decision"),

        claim_decision_made=bool(claim.get("claim_decision_made", False)),
        claim_approved=bool(claim.get("claim_approved", False)),
        payment_processed=bool(claim.get("payment_processed", False)),
        claim_closed=bool(claim.get("claim_closed", False)),

        final_decision=claim.get("final_decision"),

        # DO NOT pass validation or assignment from DB
        # Let Pydantic create default objects

        documents=docs or []
    )
