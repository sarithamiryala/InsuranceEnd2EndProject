# backend/graph/claim_graph_v3.py
"""
Claim Graph v3
==============
Real-world claims flow using LangGraph:

    register → validate (LLM) → fraud (LLM + fallback) → (investigator?) → manager → END

Nodes:
- registration_agent:
    • Generates transaction_id & registered_at
    • Aggregates OCR texts into `state.extracted_text`
    • Persists registration to SQLite (if configured in registration_agent)

- llm_validation_agent:
    • LLM-powered validation of documents & content
    • Sets state.validation.{required_missing, warnings, errors, docs_ok}
    • Sets state.claim_validated

- fraud_agent:
    • Computes fraud_score via LLM, with rule-based fallback on quota/errors
    • Sets fraud_checked, fraud_score [0..1], fraud_decision in {"SAFE","SUSPECT"}

- investigator_agent:
    • Assigns investigator on high risk (fraud_score) or large amount
    • Sets state.assignment.{investigator_id, sla_days, reason}

- manager_agent:
    • Final decision in {"APPROVED","REJECTED","PENDING_DOCUMENTS","ESCALATED_TO_SIU","MANUAL_REVIEW"}

Exports:
    • build_claim_graph_v3(return_uncompiled: bool = False) -> StateGraph | CompiledGraph
    • claim_graph_v3 (compiled)
"""

from __future__ import annotations

from typing import Literal

# LangGraph core
try:
    from langgraph.graph import StateGraph, END
except Exception as e:
    raise RuntimeError(
        "LangGraph is required. Install with: pip install langgraph"
    ) from e

# State model
from backend.state.claim_state import ClaimState

# Logger (fallback to std logging if your project logger isn't available)
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

# Agents
from backend.agents.registration_agent import registration_agent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent
from backend.agents.manager_agent import ManagerAgent


# -----------------------------------------------------------------------------
# Tunable thresholds (keep in sync with investigator_agent if you changed it)
# -----------------------------------------------------------------------------
FRAUD_ESCALATION_THRESHOLD: float = 0.70   # ≥0.70 -> investigator
HIGH_AMOUNT_THRESHOLD: float = 300_000.0   # amount > 300k -> investigator


# -----------------------------------------------------------------------------
# Manager node adapter (your ManagerAgent likely exposes .run(state))
# -----------------------------------------------------------------------------
_manager = ManagerAgent()

def manager_node(state: ClaimState) -> ClaimState:
    return _manager.run(state)


# -----------------------------------------------------------------------------
# Routing functions
# -----------------------------------------------------------------------------
def route_after_validation(state: ClaimState) -> Literal["manager", "fraud"]:
    """
    If documents are missing or there are validation errors → Manager (likely PENDING/REJECTED).
    Otherwise → Fraud.
    """
    if not state.validation.docs_ok or state.validation.errors:
        logger.info("[Router] Validation NOT OK → Manager")
        return "manager"
    logger.info("[Router] Validation OK → Fraud")
    return "fraud"


def route_after_fraud(state: ClaimState) -> Literal["investigator", "manager"]:
    """
    After fraud scoring: high-risk or high-amount → Investigator; otherwise → Manager.
    """
    amt = float(state.amount or 0.0)
    if state.fraud_score >= FRAUD_ESCALATION_THRESHOLD or amt > HIGH_AMOUNT_THRESHOLD:
        logger.info("[Router] High risk/amount → Investigator")
        return "investigator"
    logger.info("[Router] Acceptable risk → Manager")
    return "manager"


# -----------------------------------------------------------------------------
# Graph builder
# -----------------------------------------------------------------------------
def build_claim_graph_v3(return_uncompiled: bool = False):
    """
    Build the v3 claim processing graph.

    Args:
        return_uncompiled: If True, return an uncompiled StateGraph (useful
            for instrumentation/astream_events). If False (default), return the
            compiled graph ready to invoke/ainvoke.

    Returns:
        StateGraph (uncompiled) or compiled graph, depending on the flag.
    """
    graph = StateGraph(ClaimState)

    # Nodes
    graph.add_node("register", registration_agent)
    graph.add_node("validate", llm_validation_agent)
    graph.add_node("fraud", fraud_agent)
    graph.add_node("investigator", investigator_agent)
    graph.add_node("manager", manager_node)

    # Entry
    graph.set_entry_point("register")

    # Edges
    graph.add_edge("register", "validate")

    # Validation -> (Manager | Fraud)
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "manager": "manager",
            "fraud": "fraud",
        },
    )

    # Fraud -> (Investigator | Manager)
    graph.add_conditional_edges(
        "fraud",
        route_after_fraud,
        {
            "investigator": "investigator",
            "manager": "manager",
        },
    )

    # Investigator -> Manager
    graph.add_edge("investigator", "manager")

    # Manager -> END
    graph.add_edge("manager", END)

    if return_uncompiled:
        return graph
    return graph.compile()


# Default compiled export
claim_graph_v3 = build_claim_graph_v3()