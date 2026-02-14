# backend/graph/instrumentor.py
import asyncio
from typing import Any, Dict, List, Tuple
from langgraph.graph import StateGraph
from backend.state.claim_state import ClaimState

RESERVED_PREFIX = "__"  # internal nodes like __start__, __end__

def _is_reserved(name: str) -> bool:
    return isinstance(name, str) and name.startswith(RESERVED_PREFIX)

def instrument_graph(original_graph, events: List[Dict[str, Any]]) -> StateGraph:
    """
    Works with BOTH uncompiled and compiled graphs.

    - Wraps each non-reserved node and records node_start/node_end/error
    - Rebuilds a new StateGraph with only non-reserved nodes
    - Filters edges/conditional edges that reference reserved nodes
    """
    g = StateGraph(ClaimState)

    # 1) Collect nodes safely
    try:
        nodes_items = list(original_graph.nodes.items())
    except Exception:
        # Some versions use ._nodes
        nodes_items = list(getattr(original_graph, "_nodes", {}).items())

    added = set()
    for name, node_fn in nodes_items:
        if _is_reserved(name):
            continue

        async def wrapped(state, _fn=node_fn, _name=name):
            events.append({"type": "node_start", "node": _name, "snapshot": _safe_snapshot(state)})
            try:
                res = _fn(state)
                if asyncio.iscoroutine(res):
                    res = await res
                events.append({"type": "node_end", "node": _name, "snapshot": _safe_snapshot(res)})
                return res
            except Exception as e:
                events.append({"type": "error", "node": _name, "error": f"{type(e).__name__}: {e}"})
                raise

        g.add_node(name, wrapped)
        added.add(name)

    # 2) Entry point (only if not reserved)
    try:
        entry = original_graph.entry_point
        if not _is_reserved(entry):
            g.set_entry_point(entry)
        else:
            # Fallback: pick a known node (e.g., 'register') if exists
            if "register" in added:
                g.set_entry_point("register")
            else:
                # Last resort: pick any added node
                g.set_entry_point(next(iter(added)))
    except Exception:
        # If the compiled graph hides entry_point, we try a sane default
        if "register" in added:
            g.set_entry_point("register")
        else:
            g.set_entry_point(next(iter(added)))

    # 3) Simple edges
    edges: List[Tuple[str, str]] = []
    try:
        edges = list(original_graph.edges)
    except Exception:
        pass
    for src, dst in edges:
        if _is_reserved(src) or _is_reserved(dst):
            continue
        if src in added and dst in added:
            g.add_edge(src, dst)

    # 4) Conditional edges
    conds = []
    try:
        conds = list(original_graph.conditional_edges)
    except Exception:
        pass
    for src, fn, branches in conds:
        if _is_reserved(src):
            continue
        # branches is dict[str, str]; filter reserved targets
        filtered = {k: v for k, v in branches.items() if not _is_reserved(v) and v in added}
        if src in added and filtered:
            g.add_conditional_edges(src, fn, filtered)

    return g

def _safe_snapshot(state: Any) -> Dict[str, Any]:
    try:
        d = state.model_dump()  # pydantic v2
    except Exception:
        d = getattr(state, "dict", lambda: state.__dict__)()
    if isinstance(d.get("extracted_text"), str) and len(d["extracted_text"]) > 240:
        d["extracted_text"] = d["extracted_text"][:240] + "…"
    keep = [
        "claim_id", "customer_name", "policy_number",
        "amount", "extracted_text",
        "claim_registered", "claim_validated",
        "router_decision", "fraud_checked", "fraud_score", "fraud_decision",
        "final_decision"
    ]
    return {k: d.get(k) for k in keep if k in d}