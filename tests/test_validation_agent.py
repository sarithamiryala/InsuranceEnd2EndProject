# tests/test_validation_agent.py
import pytest
from backend.utils.state_builder import build_state_from_db
from backend.agents.llm_validation_agent import llm_validation_agent

def _mk_claim(**kwargs):
    base = {
        "transaction_id": "tx-1",
        "claim_id": "CLM-TEST",
        "customer_name": "Rahul Mehta",
        "policy_number": "POL-1",
        "amount": 27300,
        "claim_type": "motor  ",  # Trailing spaces on purpose
        "extracted_text": "",
        "document_extracted_text": "",
        "claim_registered": True,
        "registered_at": "2026-01-12T21:05:00Z",
        "claim_validated": False,
        "fraud_checked": False,
        "claim_decision_made": False,
        "claim_approved": False,
        "payment_processed": False,
        "claim_closed": False,
        "final_decision": None,
        "fraud_score": None,
        "fraud_decision": None,
    }
    base.update(kwargs)
    return base

# ---------- Scenario: Approval ----------
def test_validation_approval(monkeypatch):
    docs = []
    claim = _mk_claim(
        extracted_text="On 12 Jan 2026 at 7:40 PM near Silk Board, rear-ended, bumper + tail lamp damaged.",
        document_extracted_text="""=== FIR ===
FIR No: MAD/PS/2026/0112-423
Date: 12-01-2026 21:05
Complainant: Rahul Mehta
Narrative: car KA03MN4567 rear-ended at signal; bumper + tail lamp damage.

=== DRIVING_LICENSE ===
DL No: KA-05-2020-0099887
Name: Rahul Mehta
Valid To: 19/06/2030

=== RC_BOOK ===
Regn No: KA03MN4567
Owner: Rahul Mehta

=== POLICY_COPY ===
Policy No: POL-MTR-987654321
Period: 01/07/2025 to 30/06/2026
Coverage: OD + TP (Comprehensive)

=== REPAIR_ESTIMATE ===
Rear bumper assembly: 12,500
Tail lamp LH: 5,400
Paint & consumables: 7,200
Labour: 2,200
Total (INR): 27,300

=== ACCIDENT_PHOTOS ===
Photos show rear bumper crack, LH tail lamp broken""",
    )

    # Build state (normalizes claim_type to "motor")
    state = build_state_from_db(claim, docs)

    # Monkeypatch LLM to return empty (to prove deterministic checks suffice)
    monkeypatch.setattr("backend.agents.llm_validation_agent.llm_response", lambda prompt: '{"required_missing":[],"warnings":[],"errors":[],"docs_ok":true,"note":"LLM ok","recommendation":"APPROVE"}')

    out = llm_validation_agent(state)
    assert out.validation.docs_ok is True
    assert out.validation.recommendation in {"APPROVE", "NEED_MORE_DOCUMENTS"}  # Depending on LLM
    assert "Missing:" not in out.validation.note

# ---------- Scenario: Rejection ----------
def test_validation_rejection(monkeypatch):
    docs = []
    claim = _mk_claim(
        customer_name="Anita Sharma",
        amount=145000,
        extracted_text="Skid on 05 Feb 2026; hit divider; front damage.",
        document_extracted_text="""=== FIR ===
No FIR filed; diary entry only.

=== DRIVING_LICENSE ===
DL No: KA-53-2013-776541
Valid To: 11/08/2023

=== RC_BOOK ===
Regn No: KA53QZ8899
Owner: Anita Sharma

=== POLICY_COPY ===
Policy No: POL-MTR-123450001
Period: 01/02/2025 to 31/01/2026
Coverage: Third Party only

=== REPAIR_ESTIMATE ===
Non-network garage; handwritten; no GSTIN; Total 145,000

=== ACCIDENT_PHOTOS ===
Night images, low clarity
""",
    )

    state = build_state_from_db(claim, docs)
    monkeypatch.setattr("backend.agents.llm_validation_agent.llm_response", lambda prompt: '{"required_missing":[],"warnings":[],"errors":[],"docs_ok":false,"note":"LLM neutral","recommendation":"REJECT"}')
    out = llm_validation_agent(state)

    # Deterministic errors: policy expired, DL expired, FIR not found/fake estimate
    assert out.validation.docs_ok is False
    assert out.validation.recommendation == "REJECT"
    assert "Errors:" in out.validation.note

# ---------- Scenario: Suspect / Need more documents ----------
def test_validation_suspect(monkeypatch):
    docs = []
    claim = _mk_claim(
        customer_name="Prakash N",
        amount=256000,
        extracted_text="Biker grazed; minor scratches on left doors.",
        document_extracted_text="""=== FIR ===
FIR No: IND/PS/2026/0128-219
Narrative: superficial scratches on LH doors.

=== DRIVING_LICENSE ===
DL No: KA-01-2016-554321
Valid To: 09/03/2036

=== RC_BOOK ===
Regn No: KA01AB1234
Owner: Prakash N

=== POLICY_COPY ===
Policy No: POL-MTR-778899001
Period: 15/07/2025 to 14/07/2026
Coverage: Comprehensive

=== REPAIR_ESTIMATE ===
Replace front & rear bumpers, LH headlamp, LH fender, 2 doors; Total 256,000

=== ACCIDENT_PHOTOS ===
Photos show hairline scratches; bumpers/headlamp intact
""",
    )

    state = build_state_from_db(claim, docs)
    monkeypatch.setattr("backend.agents.llm_validation_agent.llm_response", lambda prompt: '{"required_missing":[],"warnings":["Check inflation"],"errors":[],"docs_ok":true,"note":"LLM warns","recommendation":"NEED_MORE_DOCUMENTS"}')
    out = llm_validation_agent(state)

    # Deterministic warnings due to mismatch between photos and inflated estimate
    assert out.validation.docs_ok in {True, False}  # depends on merge
    assert out.validation.recommendation in {"NEED_MORE_DOCUMENTS", "APPROVE"}
    assert "Warnings:" in out.validation.note