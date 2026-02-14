# server/app.py
import os
import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel

# ---- Project imports ----
from backend.state.claim_state import ClaimState, DocumentRecord
from backend.utils.ocr import ocr_any
from backend.graph.claim_graph_v3 import claim_graph_v3
from backend.agents.manager_agent import ManagerAgent
from backend.agents.llm_validation_agent import llm_validation_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.investigator_agent import investigator_agent
from backend.db.sqlite_store import init_db, DB_PATH, fetch_claim_and_docs, update_claim_fields

# Optional LLM client for registration confirmation
try:
    from backend.services.llm_client import llm_response
except Exception:
    llm_response = None  # fallback used if missing


# -----------------------------
# FastAPI app & CORS
# -----------------------------
app = FastAPI(title="Insurance Claims API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Startup: ensure DB exists
# -----------------------------
@app.on_event("startup")
def _startup():
    init_db()
    print(f"[Startup] DB at: {DB_PATH}")


# -----------------------------
# Utilities
# -----------------------------
def _safe_dump(obj) -> Dict[str, Any]:
    """Return dict for Pydantic v2/v1 or plain dict."""
    try:
        return obj.model_dump()
    except Exception:
        try:
            return obj.dict()
        except Exception:
            return obj

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _llm_registration_message(state: ClaimState) -> str:
    """
    Generate a friendly registration confirmation using LLM (if available),
    with a safe deterministic fallback.
    """
    base = (
        f"Registration successful for policy {state.policy_number or '(unknown)'}.\n"
        f"Claim ID: {state.claim_id}\n"
        f"Reference (Transaction ID): {state.transaction_id}\n"
        f"Registered At: {state.registered_at}\n"
        "Thank you. We will contact you if any additional documents are required."
    )

    # Skip LLM when disabled (e.g., demos)
    if os.getenv("LLM_DISABLE", "false").lower() in ("1", "true", "yes"):
        return base

    if not llm_response:
        return base

    try:
        prompt = f"""
You are an insurance assistant.
Create a short, friendly confirmation (3-4 lines) for a claim registration.

Include:
- Policy number
- Claim ID
- Reference number (transaction_id)
- Registered date/time

Tone: concise, professional, reassuring.
Plain text only (no markdown).

Data:
- Policy Number: {state.policy_number}
- Claim ID: {state.claim_id}
- Transaction ID: {state.transaction_id}
- Registered At: {state.registered_at}
"""
        text = llm_response(prompt)
        # Quota / content guard
        if not text or "RESOURCE_EXHAUSTED" in str(text) or "quota" in str(text).lower():
            return base
        return text.strip()
    except Exception:
        return base


def _build_state_from_db(transaction_id: str) -> Optional[ClaimState]:
    """
    Reconstruct ClaimState from DB (claim + documents) so manager can process without re-upload.
    """
    claim, docs = fetch_claim_and_docs(transaction_id)
    if not claim:
        return None

    # Convert docs
    doc_models: List[DocumentRecord] = []
    for d in docs:
        doc_models.append(
            DocumentRecord(
                filename=d.get("filename") or "unnamed",
                content_type=d.get("content_type") or "application/octet-stream",
                size_bytes=int(d.get("size_bytes") or 0),
                doc_type=d.get("doc_type"),
                extracted_text=d.get("extracted_text") or "",
            )
        )

    st = ClaimState(
        transaction_id=claim.get("transaction_id"),
        claim_id=claim.get("claim_id"),
        customer_name=claim.get("customer_name"),
        policy_number=claim.get("policy_number"),
        amount=claim.get("amount"),
        claim_type=claim.get("claim_type"),
        extracted_text=claim.get("extracted_text") or "",
        documents=doc_models,
        claim_registered=True,
        registered_at=claim.get("registered_at"),
        logs=[],
    )
    return st


def _summarize_for_manager(state: ClaimState) -> Dict[str, Any]:
    """
    Run validation → fraud → investigator → manager on DB-backed state, and return a manager-friendly summary.
    This does NOT persist results to DB automatically.
    """
    # 1) Validation (LLM + fallback)
    state = llm_validation_agent(state)

    # 2) Fraud (LLM + fallback).
    # If you prefer to skip fraud until docs_ok, add: if state.validation.docs_ok: ...
    state = fraud_agent(state)

    # 3) Investigator assignment (based on fraud/amount)
    state = investigator_agent(state)

    # 4) Manager computes suggested decision
    mgr = ManagerAgent()
    state = mgr.run(state)

    summary = {
        "transaction_id": state.transaction_id,
        "claim_id": state.claim_id,
        "customer_name": state.customer_name,
        "policy_number": state.policy_number,
        "amount": state.amount,
        "claim_type": state.claim_type,
        "registered_at": state.registered_at,
        "validation": {
            "docs_ok": state.validation.docs_ok,
            "required_missing": state.validation.required_missing,
            "warnings": state.validation.warnings,
            "errors": state.validation.errors,
        },
        "fraud": {
            "checked": state.fraud_checked,
            "score": state.fraud_score,
            "decision": state.fraud_decision,
        },
        "assignment": {
            "investigator_id": state.assignment.investigator_id,
            "sla_days": state.assignment.sla_days,
            "reason": state.assignment.reason,
        },
        "suggested_final_decision": state.final_decision,
        "logs": state.logs[-10:],  # tail logs for context
    }
    return summary


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse(url="/docs")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "db": DB_PATH}


# 1) Register claim (form + files) → OCR → registration agent → confirmation (LLM text or fallback)
@app.post("/claims/register")
async def register_claim(
    claim_id: str = Form(...),
    customer_name: str = Form(""),
    policy_number: str = Form(""),
    amount: float = Form(...),
    claim_type: str = Form(""),
    description: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    """
    Accepts claim details + document uploads.
    - OCRs each doc.
    - registration_agent aggregates OCR into extracted_text and persists claim + docs into SQLite.
    - Returns a friendly confirmation message containing transaction_id and registered_at.
    """
    # OCR doc uploads
    docs: List[DocumentRecord] = []
    for f in files:
        content = await f.read()
        try:
            text = ocr_any(content, filename=f.filename, content_type=f.content_type or "")
        except Exception:
            text = ""
        docs.append(
            DocumentRecord(
                filename=f.filename or "unnamed",
                content_type=f.content_type or "application/octet-stream",
                size_bytes=len(content or b""),
                doc_type=None,  # let validation classify later
                extracted_text=text,
            )
        )

    # Build initial state with user description; registration will aggregate OCR into extracted_text
    st = ClaimState(
        claim_id=claim_id,
        customer_name=customer_name,
        policy_number=policy_number,
        amount=amount,
        claim_type=claim_type,
        extracted_text=description,
        documents=docs,
    )

    # Run registration node logic (outside of graph) to avoid full flow here
    from backend.agents.registration_agent import registration_agent
    st = registration_agent(st)

    # Confirmation message
    confirmation = _llm_registration_message(st)

    return {
        "transaction_id": st.transaction_id,
        "registered_at": st.registered_at,
        "message": confirmation,
        "claim_id": st.claim_id,
        "policy_number": st.policy_number,
    }


# 2) Full E2E processing (Register → Validate → Fraud → Investigator → Manager)
@app.post("/claims/submit")
async def submit_claim_full(
    claim_id: str = Form(...),
    customer_name: str = Form(""),
    policy_number: str = Form(""),
    amount: float = Form(...),
    claim_type: str = Form(""),
    description: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    """
    Runs the full v3 graph with event streaming to capture timeline:
    Register → Validate (LLM) → Fraud (LLM+fallback) → Investigator → Manager.
    - Do NOT call registration_agent separately here (graph starts at register).
    - Confirmation message is generated AFTER the graph using the final state (which includes transaction_id).
    """
    # OCR uploads; DO NOT concatenate here to avoid duplication — registration will aggregate
    docs: List[DocumentRecord] = []
    for f in files:
        content = await f.read()
        try:
            text = ocr_any(content, filename=f.filename, content_type=f.content_type or "")
        except Exception:
            text = ""
        docs.append(
            DocumentRecord(
                filename=f.filename or "unnamed",
                content_type=f.content_type or "application/octet-stream",
                size_bytes=len(content or b""),
                doc_type=None,
                extracted_text=text,
            )
        )

    st = ClaimState(
        claim_id=claim_id,
        customer_name=customer_name,
        policy_number=policy_number,
        amount=amount,
        claim_type=claim_type,
        extracted_text=description,  # registration will aggregate OCR texts
        documents=docs,
    )

    timeline: List[Dict[str, Any]] = []
    final_obj: Any = None

    # Stream events to build a timeline
    try:
        async for event in claim_graph_v3.astream_events(st, version="v1"):
            ev = event.get("event") or event.get("type")
            name = event.get("name") or (event.get("data") or {}).get("name")
            if ev in ("on_node_start", "node_start"):
                timeline.append({"type": "node_start", "node": name})
            elif ev in ("on_node_end", "node_end"):
                timeline.append({"type": "node_end", "node": name})

            # Try to capture the final state from the stream
            if ev in ("on_graph_end", "on_chain_end", "end"):
                output = event.get("output") or (event.get("data") or {}).get("output")
                if output is not None:
                    final_obj = output
    except Exception:
        # If streaming isn't supported, fallback to ainvoke
        pass

    if final_obj is None:
        final_obj = await claim_graph_v3.ainvoke(st)

    final_state = _safe_dump(final_obj)

    # Build confirmation AFTER the graph (now we have transaction_id & registered_at set by registration node)
    try:
        # Rebuild Pydantic object if needed for the helper
        s = final_obj if isinstance(final_obj, ClaimState) else None
        if s is None:
            # Make a shallow compatible ClaimState for message
            s = ClaimState(**final_state)
        confirmation = _llm_registration_message(s)
    except Exception:
        confirmation = "Registration complete."

    return {
        "transaction_id": final_state.get("transaction_id"),
        "registered_at": final_state.get("registered_at"),
        "confirmation_message": confirmation,
        "timeline": timeline,
        "final_state": final_state,
    }


# 3) Manager: list recent claims (SQLite)
@app.get("/manager/claims")
async def manager_list_claims(limit: int = 50):
    with _db() as conn:
        rows = conn.execute("""
            SELECT transaction_id, claim_id, customer_name, policy_number,
                   amount, claim_type, registered_at, status, final_decision
            FROM claims
            ORDER BY datetime(registered_at) DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return {"claims": [dict(r) for r in rows]}


# 4) Manager: claim + documents (toggle OCR text with include_text)
@app.get("/manager/claims/{transaction_id}")
async def manager_get_claim(transaction_id: str, include_text: bool = False, text_limit: int = 1500):
    with _db() as conn:
        c = conn.execute("SELECT * FROM claims WHERE transaction_id=?", (transaction_id,)).fetchone()
        if not c:
            return JSONResponse(status_code=404, content={"error": "not found", "transaction_id": transaction_id})

        cols = "id, filename, content_type, size_bytes, doc_type"
        if include_text:
            cols += ", extracted_text"
        rows = conn.execute(
            f"SELECT {cols} FROM claim_documents WHERE transaction_id=? ORDER BY id ASC",
            (transaction_id,)
        ).fetchall()

        docs = []
        for r in rows:
            d = dict(r)
            if include_text and d.get("extracted_text"):
                t = d["extracted_text"]
                if len(t) > text_limit:
                    d["extracted_text"] = t[:text_limit] + "…"
            docs.append(d)

        return {"claim": dict(c), "documents": docs}


# 5) Manager: computed summary (validation, missing docs, fraud results, investigator, suggested decision)
@app.get("/manager/claims/{transaction_id}/summary")
async def manager_claim_summary(transaction_id: str):
    st = _build_state_from_db(transaction_id)
    if not st:
        return JSONResponse(status_code=404, content={"error": "not found", "transaction_id": transaction_id})

    # If you want demos to never use LLM, set env LLM_DISABLE=true (agents have fallbacks)
    summary = _summarize_for_manager(st)
    return summary


# 6) Manager: record human decision into DB
class ManagerDecisionPayload(BaseModel):
    decision: str                    # "APPROVED" | "REJECTED" | "PENDING_DOCUMENTS" | "ESCALATED_TO_SIU" | "MANUAL_REVIEW"
    comment: Optional[str] = None    # Optional note (store if you add a column for it)

@app.patch("/manager/claims/{transaction_id}/decision")
async def manager_set_decision(transaction_id: str, payload: ManagerDecisionPayload):
    decision = payload.decision.strip().upper()
    if decision not in {"APPROVED", "REJECTED", "PENDING_DOCUMENTS", "ESCALATED_TO_SIU", "MANUAL_REVIEW"}:
        return JSONResponse(status_code=400, content={"error": "invalid decision"})

    now = datetime.now(timezone.utc).isoformat()

    # NOTE: Ensure your `claims` table has these columns:
    #   ALTER TABLE claims ADD COLUMN final_decision TEXT;
    #   ALTER TABLE claims ADD COLUMN updated_at TEXT;
    fields = {
        "final_decision": decision,
        "status": decision,
        "updated_at": now,
        # "manager_comment": payload.comment,  # if you add this column
    }

    try:
        update_claim_fields(transaction_id, **fields)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DB error: {type(e).__name__}: {e}"})

    return {"transaction_id": transaction_id, "final_decision": decision, "updated_at": now}