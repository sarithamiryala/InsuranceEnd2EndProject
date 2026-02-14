# backend/agents/llm_router_agent.py
from backend.services.llm_client import llm_response
from backend.state.claim_state import RouterDecision
from backend.utils.safe_json import safe_json_parse

def _to_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true","yes","y","1"}: return True
        if s in {"false","no","n","0"}: return False
    if isinstance(v, (int, float)): return bool(v)
    return False

def llm_router_agent(state):
    prompt = f"""
    You are an insurance claim routing AI.

    Claim Details:
    Claim ID: {state.claim_id}
    Amount: {state.amount}
    Extracted Text: {state.extracted_text}

    Return ONLY a minified JSON object with keys:
    - "fraud_check": boolean
    - "manual_review": boolean
    - "need_documents": boolean

    Example: {{"fraud_check": false, "manual_review": false, "need_documents": false}}
    """

    result = llm_response(prompt)
    print("Raw LLM response (router):", repr(result))

    fallback = {"fraud_check": False, "manual_review": True, "need_documents": True}
    parsed = safe_json_parse(result, fallback)

    decision = {
        "fraud_check": _to_bool(parsed.get("fraud_check", False)),
        "manual_review": _to_bool(parsed.get("manual_review", True)),
        "need_documents": _to_bool(parsed.get("need_documents", True)),
    }

    # Optional: log the sanitized decision
    print("[router] decision:", decision)

    state.router_decision = RouterDecision(**decision)
    return state