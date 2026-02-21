import pytest

from types import SimpleNamespace

# Import the improved fraud_agent (ensure your import path matches your project)
from backend.agents.fraud_agent import fraud_agent

# Create a minimal ValidationResult-like object
class _Validation:
    def __init__(self, missing=None, warnings=None, errors=None, note="", rec=""):
        self.required_missing = missing or []
        self.warnings = warnings or []
        self.errors = errors or []
        self.note = note
        self.recommendation = rec

def _mk_state(
    claim_type="motor",
    customer_name="Test User",
    amount=10000,
    extracted_text="Minor incident",
    document_extracted_text="=== FIR ===\n...ok...",
    validation=None,
):
    return SimpleNamespace(
        claim_type=claim_type,
        customer_name=customer_name,
        amount=amount,
        extracted_text=extracted_text,
        document_extracted_text=document_extracted_text,
        validation=validation or _Validation()
    )


def test_fraud_safe(monkeypatch):
    # LLM returns SAFE
    from backend.agents import fraud_agent as mod
    monkeypatch.setenv("groq_api_key", "dummy")
    monkeypatch.setattr(mod, "llm_response",
        lambda prompt: '{"fraud_score": 0.15, "fraud_decision": "SAFE"}'
    )

    st = _mk_state()
    out = fraud_agent(st)

    assert out.fraud_checked is True
    assert 0.0 <= out.fraud_score <= 1.0
    assert out.fraud_decision == "SAFE"


def test_fraud_suspect(monkeypatch):
    # LLM returns SUSPECT by decision
    from backend.agents import fraud_agent as mod
    monkeypatch.setenv("groq_api_key", "dummy")
    monkeypatch.setattr(mod, "llm_response",
        lambda prompt: '{"fraud_score": 0.62, "fraud_decision": "SUSPECT"}'
    )

    st = _mk_state(validation=_Validation(errors=["Policy expired"]))
    out = fraud_agent(st)

    assert out.fraud_checked is True
    assert out.fraud_decision == "SUSPECT"
    assert out.fraud_score == pytest.approx(0.62, abs=1e-6)


def test_score_sanitization(monkeypatch):
    # LLM returns out-of-bound score; we clip to [0,1]
    from backend.agents import fraud_agent as mod
    monkeypatch.setenv("groq_api_key", "dummy")
    monkeypatch.setattr(mod, "llm_response",
        lambda prompt: '{"fraud_score": 1.7, "fraud_decision": "SUSPECT"}'
    )

    st = _mk_state()
    out = fraud_agent(st)

    assert out.fraud_checked is True
    assert out.fraud_score == 1.0
    assert out.fraud_decision == "SUSPECT"