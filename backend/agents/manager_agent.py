from typing import Dict, Any
from datetime import datetime, timezone

from backend.state.claim_state import ClaimState
from backend.db.sqlite_store import update_claim_fields


class ManagerAgent:
    """
    Manager Agent
    -------------
    Responsible for:
    - Orchestrating claim workflow
    - Deciding next step
    - Finalizing claim decision
    """

    def __init__(self):
        pass

    # -----------------------------------
    # Routing Logic
    # -----------------------------------
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

    # -----------------------------------
    # Final Decision Logic
    # -----------------------------------
    def finalize_claim(self, state: ClaimState):

        if not state.validation or not getattr(state.validation, "docs_ok", False):
            state.final_decision = "PENDING_DOCUMENTS"
            status = "PENDING_DOCUMENTS"

        elif state.fraud_score is not None and state.fraud_score >= 0.7:
            state.final_decision = "ESCALATED_TO_SIU"
            status = "UNDER_INVESTIGATION"

        elif state.claim_approved:
            state.final_decision = "APPROVED"
            status = "APPROVED"

        else:
            state.final_decision = "REJECTED"
            status = "REJECTED"

        # Persist in DB
        try:
            update_claim_fields(
                state.transaction_id,
                final_decision=state.final_decision,
                status=status,
                updated_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            print(f"[Manager] DB update failed: {e}")

        return state

    # -----------------------------------
    # Graph Entry
    # -----------------------------------
    def run(self, state: ClaimState) -> Dict[str, Any]:

        next_step = self.decide_next_step(state)

        # 🔥 If workflow finished → finalize
        if next_step == "end":
            state = self.finalize_claim(state)

        return {
            "next_step": next_step,
            "final_decision": getattr(state, "final_decision", None),
            "manager_decision": f"Routing to {next_step}"
        }
