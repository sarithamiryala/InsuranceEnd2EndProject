# backend/graph/claim_graph_v3.py

from __future__ import annotations
from typing import Literal

try:
    from langgraph.graph import StateGraph, END
except Exception as e:
    raise RuntimeError("LangGraph is required. Install with: pip install langgraph") from e

from backend.state.claim_state import ClaimState
from backend.agents.registration_agent import registration_agent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent  # (kept; not used by this flow)
from backend.agents.manager_agent import ManagerAgent

# Logger (fallback)
try:
    from backend.utils.logger import logger
except Exception:
    import logging
    logger = logging.getLogger("claim_graph_v3")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FRAUD_ESCALATION_THRESHOLD: float = 0.70

_manager = ManagerAgent()

def manager_node(state: ClaimState) -> ClaimState:
    """
    Terminal sink. Finalizes decision & returns updated state.
    """
    # If fraud wasn't run (e.g., docs not OK), just finalize from validation.
    state = _manager.finalize_claim(state)
    return state

def route_after_validation(state: ClaimState) -> Literal["manager", "fraud"]:
    """
    Diagram: if Docs OK? No → Manager; Yes → Fraud
    """
    if not state.validation or not state.validation.docs_ok:
        logger.info("[Router] Docs NOT OK → Manager")
        return "manager"
    logger.info("[Router] Docs OK → Fraud")
    return "fraud"

def route_after_fraud(state: ClaimState) -> Literal["manager"]:
    """
    Diagram: Fraud always followed by Manager for final decision.
    (We keep single route for clarity; threshold handled in ManagerAgent.)
    """
    return "manager"

def build_claim_graph_v3(start_from: Literal["register","validate"] = "register", return_uncompiled: bool = False):
    """
    Build the claim processing graph aligned to the diagram.
    You can start from 'register' (full flow) or 'validate' (post-registration).
    """
    graph = StateGraph(ClaimState)

    graph.add_node("register", registration_agent)
    graph.add_node("validate", llm_validation_agent)
    graph.add_node("fraud", fraud_agent)
    graph.add_node("manager", manager_node)

    if start_from == "register":
        graph.set_entry_point("register")
        graph.add_edge("register", "validate")
    else:
        graph.set_entry_point("validate")

    # Validation → (Manager | Fraud)
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"manager": "manager", "fraud": "fraud"},
    )

    # Fraud → Manager
    graph.add_conditional_edges(
        "fraud",
        route_after_fraud,
        {"manager": "manager"},
    )

    # Manager → END
    graph.add_edge("manager", END)

    if return_uncompiled:
        return graph
    return graph.compile()

# Export both variants
claim_graph_v3 = build_claim_graph_v3(start_from="register")
claim_graph_v3_postreg = build_claim_graph_v3(start_from="validate")