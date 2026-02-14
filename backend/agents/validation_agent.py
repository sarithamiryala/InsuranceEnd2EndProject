from backend.utils.logger import logger
from backend.state.claim_state import ClaimState


def validation_agent(state: ClaimState):
    logger.info(f"[ValidationAgent] Validating claim {state.claim_id}")

    if state.amount is None:
        logger.error("[ValidationAgent] Claim amount missing")
        state.claim_validated = False
        state.final_decision = "REJECTED: Missing amount"
        return state

    if state.amount > 1_000_000:
        logger.info("[ValidationAgent] High amount — manual review needed")
        state.claim_validated = True
    else:
        state.claim_validated = True

    logger.info("[ValidationAgent] Result: VALID")
    return state
