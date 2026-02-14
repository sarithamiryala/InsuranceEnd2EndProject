"""
Claim Processing Graph Flow
---------------------------
register → validate → fraud → investigator → manager → END
"""

from backend.state.claim_state import ClaimState
from backend.agents.registration_agent import registration_agent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent
from backend.agents.manager_agent import ManagerAgent


def run_claim_flow(initial_state: ClaimState) -> ClaimState:
    """
    Executes full claim lifecycle sequentially
    Safe for demo (no async, no recursion)
    """

    state = initial_state
    manager = ManagerAgent()

    while True:

        decision = manager.decide_next_step(state)

        state.logs.append(f"[manager] routing → {decision}")

        if decision == "registration_agent":
            state = registration_agent(state)

        elif decision == "validation_agent":
            state = llm_validation_agent(state)

        elif decision == "fraud_agent":
            state = fraud_agent(state)

        elif decision == "investigator_agent":
            state = investigator_agent(state)

        elif decision == "decision_agent":
            # Simple auto decision logic
            if state.claim_validated and state.fraud_decision == "SAFE":
                state.claim_approved = True
            else:
                state.claim_approved = False

            state.claim_decision_made = True
            state.logs.append(
                f"[decision] approved={state.claim_approved}"
            )

        elif decision == "payment_agent":
            state.payment_processed = True
            state.logs.append("[payment] processed")

        elif decision == "closure_agent":
            state.claim_closed = True
            state.logs.append("[closure] claim closed")

        elif decision == "end":
            state.logs.append("[flow] completed")
            break

        else:
            state.logs.append(f"[flow] unknown step {decision}")
            break

    return state
