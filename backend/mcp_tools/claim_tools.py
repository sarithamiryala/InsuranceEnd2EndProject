from backend.state.claim_state import ClaimState

from backend.agents.registration_agent import registration_agent
from backend.agents.validation_agent import validation_agent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent
from backend.agents.manager_agent import ManagerAgent


# ---------------------------------------------------
# Helper
# ---------------------------------------------------

def _serialize(state: ClaimState):
    return state.model_dump()


# ===================================================
# 1️⃣ REGISTRATION (DB Persisted)
# ===================================================

def registration_tool(input_data: dict):
    state = ClaimState(**input_data)
    state = registration_agent(state)
    return _serialize(state)


# ===================================================
# 2️⃣ RULE VALIDATION
# ===================================================

def validation_tool(input_data: dict):
    state = ClaimState(**input_data)
    state = validation_agent(state)
    return _serialize(state)


# ===================================================
# 3️⃣ LLM VALIDATION
# ===================================================

def llm_validation_tool(input_data: dict):
    state = ClaimState(**input_data)
    state = llm_validation_agent(state)
    return _serialize(state)


# ===================================================
# 4️⃣ FRAUD
# ===================================================

def fraud_tool(input_data: dict):
    state = ClaimState(**input_data)
    state = fraud_agent(state)
    return _serialize(state)


# ===================================================
# 5️⃣ INVESTIGATOR
# ===================================================

def investigator_tool(input_data: dict):
    state = ClaimState(**input_data)
    state = investigator_agent(state)
    return _serialize(state)


# ===================================================
# 6️⃣ MANAGER ROUTER TOOL
# ===================================================

def manager_tool(input_data: dict):
    state = ClaimState(**input_data)

    manager = ManagerAgent()
    result = manager.run(state)

    return result
