from langgraph.graph import StateGraph, END
from backend.state.claim_state import ClaimState
from backend.utils.logger import logger
from backend.agents.registration_agent import registration_agent
from backend.agents.validation_agent import validation_agent
from backend.agents.llm_router_agent import llm_router_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.manager_agent import ManagerAgent

manager_agent = ManagerAgent()
def manager_node(state: ClaimState):
    return manager_agent.run(state)

def route_after_router(state: ClaimState):
    decision = state.router_decision
    if decision is None:
        logger.warning("[Router] No decision found — sending to manager")
        return "manager"
    if decision.need_documents:
        logger.info("[Router] Missing documents → Manager")
        return "manager"
    if decision.fraud_check:
        logger.info("[Router] Fraud check required")
        return "fraud"
    if decision.manual_review:
        logger.info("[Router] Manual review required")
        return "manager"
    return "manager"

def build_claim_graph(return_uncompiled: bool = False):
    graph = StateGraph(ClaimState)

    graph.add_node("register", registration_agent)
    graph.add_node("validate", validation_agent)
    graph.add_node("router", llm_router_agent)
    graph.add_node("fraud", fraud_agent)
    graph.add_node("manager", manager_node)

    graph.set_entry_point("register")

    graph.add_edge("register", "validate")
    graph.add_edge("validate", "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "fraud": "fraud",
            "manager": "manager",
        }
    )

    graph.add_edge("fraud", "manager")
    graph.add_edge("manager", END)

    if return_uncompiled:
        return graph

    return graph.compile()


claim_graph = build_claim_graph()