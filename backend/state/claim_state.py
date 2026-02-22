from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================
# Document Model
# ============================================================

class DocumentRecord(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    doc_type: Optional[str] = None
    extracted_text: Optional[str] = None


# ============================================================
# Validation Result
# ============================================================

class ValidationResult(BaseModel):
    required_missing: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    docs_ok: bool = False

    note: str = ""                 # Officer Observation
    recommendation: str = ""       # APPROVE / REJECT / NEED_MORE_DOCS



# ============================================================
# Investigator Assignment
# ============================================================

class Assignment(BaseModel):
    investigator_id: Optional[str] = None
    sla_days: Optional[int] = None
    reason: Optional[str] = None 
    


# ============================================================
# MAIN CLAIM STATE (ENTERPRISE SAFE)
# ============================================================

class ClaimState(BaseModel):

    # -------------------------
    # Identifiers
    # -------------------------
    transaction_id: Optional[str] = None
    claim_id: Optional[str] = None
    customer_name: Optional[str] = None
    policy_number: Optional[str] = None

    # -------------------------
    # Claim Data
    # -------------------------
    amount: Optional[float] = None
    claim_type: Optional[str] = None
    extracted_text: Optional[str] = None
    document_extracted_text: Optional[str] = None


    # -------------------------
    # Documents
    # -------------------------
    documents: List[DocumentRecord] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)

    # -------------------------
    # Registration Lifecycle
    # -------------------------
    claim_registered: bool = False
    registered_at: Optional[str] = None

    # -------------------------
    # Validation
    # -------------------------
    claim_validated: bool = False

    # -------------------------
    # Fraud
    # -------------------------
    fraud_checked: bool = False
    fraud_score: Optional[float] = None
    fraud_decision: Optional[str] = None

    # -------------------------
    # Investigator
    # -------------------------
    assignment: Assignment = Field(default_factory=Assignment)
    investigator_assigned: bool = False

    # -------------------------
    # Decision Stage
    # -------------------------
    claim_decision_made: bool = False
    claim_approved: bool = False

    # -------------------------
    # Payment Stage
    # -------------------------
    payment_processed: bool = False

    # -------------------------
    # Closure Stage
    # -------------------------
    claim_closed: bool = False

    # -------------------------
    # Final Output
    # -------------------------
    final_decision: Optional[str] = None

    # -------------------------
    # Audit Logs
    # -------------------------
    logs: List[str] = Field(default_factory=list)
