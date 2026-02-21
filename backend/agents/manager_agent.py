# backend/agents/manager_agent.py
from typing import Dict, Any
from datetime import datetime, timezone

from backend.state.claim_state import ClaimState
from backend.db.sqlite_store import update_claim_fields

class ManagerAgent:
    """
    Final decision agent (terminal node in the graph).
    """

    def __init__(self):
        pass

    def finalize_claim(self, state: ClaimState) -> ClaimState:
        """
        Applies the diagram's decision table:
          - docs_ok == False         -> PENDING_DOCUMENTS
          - fraud_score >= 0.7       -> ESCALATED_TO_SIU
          - else by validation.recommendation:
                APPROVE              -> APPROVED
                REJECT               -> REJECTED
                NEED_MORE_DOCUMENTS  -> PENDING_DOCUMENTS
        """
        validation = getattr(state, "validation", None)
        docs_ok = bool(getattr(validation, "docs_ok", False))
        rec = (getattr(validation, "recommendation", None) or "").upper()
        fraud_score = state.fraud_score if state.fraud_score is not None else None

        if not docs_ok:
            final_decision = "PENDING_DOCUMENTS"
        elif fraud_score is not None and fraud_score >= 0.7:
            final_decision = "ESCALATED_TO_SIU"
        else:
            if rec == "APPROVE":
                final_decision = "APPROVED"
            elif rec == "REJECT":
                final_decision = "REJECTED"
            elif rec == "NEED_MORE_DOCUMENTS":
                final_decision = "PENDING_DOCUMENTS"
            else:
                # Safety default if recommendation is empty/unknown
                final_decision = "PENDING_DOCUMENTS"

        # Update state
        state.final_decision = final_decision
        state.claim_decision_made = True

        # Persist
        try:
            update_claim_fields(
                state.transaction_id,
                final_decision=final_decision,
                status=final_decision,  # keep status same as decision for clarity
                updated_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            print(f"[Manager] DB update failed: {e}")

        return state

    # Optional: keep run/decide_next_step for non-graph usage
    def decide_next_step(self, state: ClaimState) -> str:
        if not state.claim_registered:
            return "registration_agent"
        if not state.claim_validated:
            return "validation_agent"
        if not state.fraud_checked:
            return "fraud_agent"
        if not state.claim_decision_made:
            return "decision_agent"
        if state.claim_approved and not state.payment_processed:
            return "payment_agent"
        if state.payment_processed and not state.claim_closed:
            return "closure_agent"
        return "end"

    def run(self, state: ClaimState) -> Dict[str, Any]:
        # In the graph we call finalize_claim() directly via manager_node.
        # This run() remains for any external orchestration you might have.
        next_step = self.decide_next_step(state)
        if next_step == "end":
            state = self.finalize_claim(state)
        return {
            "next_step": next_step,
            "final_decision": getattr(state, "final_decision", None),
            "manager_decision": f"Routing to {next_step}"
        }