from typing import Dict, Any
from backend.state.claim_state import ClaimState


class ManagerAgent:
    """
    Manager Agent
    -------------
    Responsible for:
    - Orchestrating claim workflow
    - Deciding next step based on claim state
    - Acting as central router in graph execution
    """

    def __init__(self):
        pass

    # ----------------------------
    # Main Decision Function
    # ----------------------------
    def decide_next_step(self, state: ClaimState) -> str:
        """
        Decide which agent should run next based on claim state
        Returns:
            str -> next node name in graph
        """

        # Step 1 — Registration
        if not state.claim_registered:
            return "registration_agent"

        # Step 2 — Validation
        if not state.claim_validated:
            return "validation_agent"

        # Step 3 — Fraud Check
        if not state.fraud_checked:
            return "fraud_agent"

        # Step 4 — Decision
        if not state.claim_decision_made:
            return "decision_agent"

        # Step 5 — Payment
        if state.claim_approved and not state.payment_processed:
            return "payment_agent"

        # Step 6 — Close Claim
        if state.payment_processed and not state.claim_closed:
            return "closure_agent"

        return "end"

    # ----------------------------
    # Graph Entry Function
    # ----------------------------
    def run(self, state: ClaimState) -> Dict[str, Any]:
        """
        Called by graph node
        """
        next_step = self.decide_next_step(state)

        return {
            "next_step": next_step,
            "manager_decision": f"Routing to {next_step}"
        }
