from datetime import datetime, timezone
from backend.db.investigator_store import (
    get_available_investigator,
    increment_investigator_load,
    record_assignment
)
from backend.db.postgres_store import update_claim_fields  # use your PostgreSQL updater

def investigator_agent(state):
    # 1️⃣ Only assign if fraud checked
    if not getattr(state, "fraud_checked", False):
        state.logs.append("[investigator] Fraud not checked")
        return state

    # 2️⃣ Only escalate if high risk
    if state.fraud_score is None or state.fraud_score < 0.7:
        state.logs.append("[investigator] No escalation required")
        return state

    # 3️⃣ Fetch available investigator
    investigator = get_available_investigator(state.claim_type)
    if not investigator:
        state.logs.append("[investigator] No available investigator")
        return state

    # 4️⃣ Increment workload
    increment_investigator_load(investigator["investigator_id"])

    # 5️⃣ Record assignment in history
    record_assignment(
        transaction_id=state.transaction_id,
        investigator_id=investigator["investigator_id"],
        reason="High fraud risk"
    )



    # 7️⃣ Update state
    state.logs.append(f"[investigator] Assigned {investigator['name']} ({investigator['investigator_id']})")
    state.assignment.investigator_id = str(investigator["investigator_id"])
    state.assignment.reason = "High fraud risk"
    state.assignment.sla_days = 3

    return state