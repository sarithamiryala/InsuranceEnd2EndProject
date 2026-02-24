from typing import Dict, Any
from datetime import datetime, timezone

from backend.state.claim_state import ClaimState
# from backend.db.postgres_store import update_claim_fields


class ManagerAgent:
    """
    Manager Final Decision Agent (FastMCP Cloud Resume Safe)

    This node is the terminal decision-making authority in the claim graph.

    Decision Table:
        - docs_ok == False         -> PENDING_DOCUMENTS
        - fraud_score >= 0.7       -> ESCALATED_TO_SIU
        - else by validation.recommendation:
              APPROVE              -> APPROVED
              REJECT               -> REJECTED
              NEED_MORE_DOCUMENTS  -> PENDING_DOCUMENTS

    Cloud Resume Requirement:
    When FastMCP reconstructs ClaimState from PostgreSQL JSONB
    during a new Copilot session, this node must not re-run
    decision logic if a final_decision already exists in DB.

    Therefore:
    - final_decision
    - claim_decision_made

    must be restored from persistent storage to prevent
    duplicate execution and incorrect routing into payment node.

    Persists:
        state.final_decision
        state.claim_decision_made
    """

    def __init__(self):
        pass

    def finalize_claim(self, state: ClaimState) -> ClaimState:
        from backend.db.postgres_store import update_claim_fields

        # -------------------------
        # Resume Guard
        # -------------------------
        # Resume-safe but allow recalculation if fraud or validation changed
        if state.claim_decision_made and state.final_decision in [
            "APPROVED",
            "REJECTED",
            "ESCALATED_TO_SIU"
        ]:
            # Re-evaluate only if fraud or validation now indicates risk
            validation = getattr(state, "validation", None)
            rec = (getattr(validation, "recommendation", "") or "").upper()
            fraud_score = state.fraud_score or 0.0

            if fraud_score < 0.7 and rec not in ["SUSPECT", "REJECT", "NEED_MORE_DOCUMENTS"]:
                return state

        validation = getattr(state, "validation", None)

        docs_ok = bool(getattr(validation, "docs_ok", False))
        warnings = getattr(validation, "warnings", []) or []
        rec = (getattr(validation, "recommendation", "") or "").upper()

        fraud_score = state.fraud_score if state.fraud_score is not None else 0.0
        fraud_decision = (state.fraud_decision or "").upper()

        # -------------------------
        # Decision Hierarchy
        # -------------------------

        # 1️⃣ Missing documents
        if not docs_ok or rec == "NEED_MORE_DOCUMENTS":
            final_decision = "PENDING_DOCUMENTS"

        # 2️⃣ Hard reject
        elif rec == "REJECT":
            final_decision = "REJECTED"

        # 3️⃣ High fraud risk OR SUSPECT → Escalate
        elif fraud_score >= 0.7 or fraud_decision == "SUSPECT" or rec == "SUSPECT":
            final_decision = "ESCALATED_TO_SIU"

        # 4️⃣ Warnings present → Escalate (even if fraud low)
        elif warnings:
            final_decision = "ESCALATED_TO_SIU"

        # 5️⃣ Only clean case should approve
        else:
            final_decision = "APPROVED"

        state.final_decision = final_decision
        state.claim_decision_made = True

        try:
            update_claim_fields(
                state.transaction_id,
                final_decision=final_decision,
                status=final_decision,
                updated_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            print(f"[Manager] DB update failed: {e}")

        return state

    # Resume-safe routing
    def decide_next_step(self, state: ClaimState) -> str:

        if not state.claim_registered:
            return "registration_agent"

        if not state.claim_validated:
            return "validation_agent"

        if not state.fraud_checked:
            return "fraud_agent"

        # Cloud Resume Safe Check
        if not state.claim_decision_made and not state.final_decision:
            return "decision_agent"

        if state.final_decision == "APPROVED" and not state.payment_processed:
            return "payment_agent"

        if state.payment_processed and not state.claim_closed:
            return "closure_agent"

        return "end"

    def run(self, state: ClaimState) -> Dict[str, Any]:

        next_step = self.decide_next_step(state)

        if next_step == "end":
            state = self.finalize_claim(state)

        return {
            "next_step": next_step,
            "final_decision": getattr(state, "final_decision", None),
            "manager_decision": f"Routing to {next_step}"
        }