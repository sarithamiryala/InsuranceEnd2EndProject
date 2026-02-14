# mcp_server.py

from fastmcp import FastMCP
from server.app_v3 import app  # your existing FastAPI app
from server.app_v3 import register_claim, check_status, process_claim, manager_decision

# ----------------------
# Convert existing FastAPI app into MCP
# ----------------------
mcp = FastMCP.from_fastapi(app=app)

# ----------------------
# Expose endpoints as MCP tools
# ----------------------

@mcp.tool
async def ClaimRegistrationTool(
    claim_id: str,
    customer_name: str,
    policy_number: str,
    description: str,
    amount: float,
    claim_type: str,
    documents: list = []
):
    """Register a new claim with documents"""
    return await register_claim(
        claim_id=claim_id,
        customer_name=customer_name,
        policy_number=policy_number,
        description=description,
        amount=amount,
        claim_type=claim_type,
        documents=documents
    )

@mcp.tool
def ClaimStatusTool(transaction_id: str):
    """Check claim status"""
    from pydantic import BaseModel
    class Request(BaseModel):
        transaction_id: str
    return check_status(Request(transaction_id=transaction_id))

@mcp.tool
async def ManagerProcessingTool(transaction_id: str):
    """Run full AI processing for a claim"""
    return await process_claim(transaction_id)

@mcp.tool
def ManagerDecisionTool(transaction_id: str, decision: str):
    """Manager override decision"""
    from pydantic import BaseModel
    class Request(BaseModel):
        decision: str
    return manager_decision(transaction_id, Request(decision=decision))

# ----------------------
# Run MCP server
# ----------------------
if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=8000)
