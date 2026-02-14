# backend/agents/investigator_agent.py
"""
Investigator Agent
==================

Assigns an investigator to a claim when it meets escalation criteria, e.g.:
- High fraud risk (fraud_score ≥ FRAUD_ESCALATION_THRESHOLD)
- High claim amount (amount > HIGH_AMOUNT_THRESHOLD)

It updates:
- state.assignment.investigator_id
- state.assignment.sla_days
- state.assignment.reason
- state.logs (for audit trail)

This agent is typically invoked after the fraud agent in v3 flow:
    register → validate → fraud → (investigator?) → manager
"""

from datetime import datetime, timezone

from backend.db.investigator_store import (
    get_available_investigator,
    increment_investigator_load
)

from backend.db.sqlite_store import update_claim_fields


def investigator_agent(state):

    # -----------------------------------
    # 1️⃣ Only assign if fraud checked
    # -----------------------------------
    if not getattr(state, "fraud_checked", False):
        state.logs.append("[investigator] Fraud not checked")
        return state

    # -----------------------------------
    # 2️⃣ Only escalate if high risk
    # -----------------------------------
    if state.fraud_score is None or state.fraud_score < 0.7:
        state.logs.append("[investigator] No escalation required")
        return state

    # -----------------------------------
    # 3️⃣ Fetch available investigator
    # -----------------------------------
    investigator_id = get_available_investigator(state.claim_type)

    if not investigator_id:
        state.logs.append("[investigator] No available investigator")
        return state

    # -----------------------------------
    # 4️⃣ Increment workload
    # -----------------------------------
    increment_investigator_load(investigator_id)

    # -----------------------------------
    # 5️⃣ Update claims DB (VERY IMPORTANT)
    # -----------------------------------
    update_claim_fields(
        state.transaction_id,
        investigator_id=investigator_id,
        assignment_reason="High fraud risk",
        assignment_status="ASSIGNED",
        assigned_at=datetime.now(timezone.utc).isoformat()
    )

    # -----------------------------------
    # 6️⃣ Update state (optional but recommended)
    # -----------------------------------
    state.logs.append(f"[investigator] Assigned {investigator_id}")
    state.assignment_done = True

    return state
