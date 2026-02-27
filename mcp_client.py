# mcp_client.py
# MCP Client — used by Streamlit (app.py) to call MCP tools
#
# Streamlit imports this instead of db.py directly.
# It sends JSON-RPC requests to the MCP server over HTTP.
#
# Usage in app.py:
#   from mcp_client import mcp
#
#   summary  = mcp.get_portfolio_summary()
#   loans    = mcp.list_loans()
#   result   = mcp.get_compliance_result("LN-10001")
#   next_id  = mcp.get_next_loan_id()
#   inserted = mcp.insert_loan({...})

import os
import json
import logging
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("mcp_client")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp")
REQUEST_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "10"))


# ─────────────────────────────────────────────────────────────
# Low-level JSON-RPC caller
# ─────────────────────────────────────────────────────────────

def _call_tool(tool_name: str, arguments: Dict = None) -> Any:
    """
    Sends a tools/call JSON-RPC request to the MCP server.
    Returns the parsed result or raises RuntimeError on failure.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params": {
            "name":      tool_name,
            "arguments": arguments or {},
        },
    }

    try:
        resp = requests.post(
            MCP_SERVER_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach MCP server at {MCP_SERVER_URL}. "
            "Run: python mcp_server.py --http --port 3000"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"MCP server request timed out after {REQUEST_TIMEOUT}s")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"MCP server HTTP error: {e}")

    # JSON-RPC error response
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"MCP tool error [{err.get('code')}]: {err.get('message')}")

    # Extract the text content from MCP response
    content = data.get("result", {}).get("content", [])
    if not content:
        return None

    raw = content[0].get("text", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


# ─────────────────────────────────────────────────────────────
# MCP Client — high-level API (mirrors db.py interface exactly)
# ─────────────────────────────────────────────────────────────

class MCPClient:
    """
    High-level client for calling MCP tools from Streamlit.
    Methods mirror the original db.py interface so app.py changes are minimal.
    """

    # ── Health ────────────────────────────────────────────────
    def health_check(self) -> Dict:
        """Check if the backend service and database are reachable."""
        return _call_tool("health_check")

    def is_available(self) -> tuple[bool, str]:
        """
        Returns (True, "") if MCP + backend are reachable,
        or (False, error_message) if not.
        """
        try:
            result = self.health_check()
            db_ok = result.get("database") == "connected"
            if not db_ok:
                return False, "Database unavailable — check PostgreSQL is running"
            return True, ""
        except RuntimeError as e:
            return False, str(e)

    # ── Loan ID generation ────────────────────────────────────
    def get_next_loan_id(self) -> str:
        """Returns the next auto-generated loan ID (e.g. 'LN-10051')."""
        result = _call_tool("get_next_loan_id")
        return result["next_loan_id"]

    # ── Loan reads ────────────────────────────────────────────
    def get_all_loans(self) -> List[Dict]:
        """Returns all loans as a list of dicts (summary fields only)."""
        result = _call_tool("list_loans")
        return result.get("loans", [])

    def search_loans(self, q: str, limit: int = 25) -> List[Dict]:
        """Free-text search across loans. Returns loan rows with latest compliance summary."""
        result = _call_tool("search_loans", {"q": q, "limit": int(limit)})
        return result.get("loans", []) if isinstance(result, dict) else (result or [])

    def filter_loans(self, filters: Dict) -> List[Dict]:
        """
        Structured filtering across loans. Provide filter keys accepted by the MCP tool,
        e.g. {"dti_gt": 50, "borrower_name": "Zili"}.
        """
        result = _call_tool("filter_loans", filters or {})
        return result.get("loans", []) if isinstance(result, dict) else (result or [])

    def get_loan_by_id(self, loan_id: str) -> Optional[Dict]:
        """Returns a single loan dict by loan_id, or None if not found."""
        result = _call_tool("get_loan", {"loan_id": loan_id})
        if result and "error" in result:
            return None
        return result

    def get_all_loans_with_compliance(self) -> List[Dict]:
        """Returns all loans joined with their latest compliance result."""
        result = _call_tool("list_loans_with_compliance")
        return result.get("loans", [])

    # ── Loan write ────────────────────────────────────────────
    def insert_loan(self, loan_data: Dict) -> str:
        """
        Inserts a new loan via MCP → FastAPI → PostgreSQL.
        Returns the inserted loan_id string.
        Note: loan_id in loan_data is ignored — backend auto-generates it.
        """
        # Remove loan_id if present (backend generates it)
        payload = {k: v for k, v in loan_data.items() if k != "loan_id"}

        # Convert date objects to ISO strings for JSON serialisation
        for key, val in payload.items():
            if hasattr(val, "isoformat"):
                payload[key] = val.isoformat()

        result = _call_tool("insert_loan", payload)
        return result["loan_id"]

    # ── Compliance reads ──────────────────────────────────────
    def get_compliance_result_for_loan(self, loan_id: str) -> Optional[Dict]:
        """Returns the latest compliance result for a loan, or None if not found."""
        result = _call_tool("get_compliance_result", {"loan_id": loan_id})
        if result and "error" in result:
            return None
        # Ensure control_results is a dict (may arrive as JSON string via MCP transport)
        if result and isinstance(result.get("control_results"), str):
            import json as _json
            try:
                result["control_results"] = _json.loads(result["control_results"])
            except Exception:
                pass
        return result

    def get_portfolio_summary(self) -> Optional[Dict]:
        """Returns aggregate compliance metrics across all loans."""
        return _call_tool("get_portfolio_summary")


# ─────────────────────────────────────────────────────────────
# Singleton instance — import this in app.py
# ─────────────────────────────────────────────────────────────
mcp = MCPClient()


# ─────────────────────────────────────────────────────────────
# Quick test — run directly to verify connectivity
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MCP client connectivity...\n")

    ok, err = mcp.is_available()
    if not ok:
        print(f"❌ MCP/backend unavailable: {err}")
        print("\nMake sure both services are running:")
        print("  1. uvicorn loan_service:app --port 8000 --reload")
        print("  2. python mcp_server.py --http --port 3000")
        exit(1)

    print("✅ MCP server reachable")
    print("✅ Database connected\n")

    print("📋 Next Loan ID:", mcp.get_next_loan_id())

    loans = mcp.get_all_loans()
    print(f"📦 Total loans in DB: {len(loans)}")
    if loans:
        print(f"   First loan: {loans[0]['loan_id']} — {loans[0]['borrower_name']}")

    summary = mcp.get_portfolio_summary()
    if summary:
        print(f"\n📊 Portfolio Summary:")
        print(f"   Total checked : {summary['total_loans']}")
        print(f"   Compliant     : {summary['compliant_count']}")
        print(f"   Breaches      : {summary['breach_count']}")
        print(f"   High severity : {summary['high_severity']}")

    print("\n✅ All MCP client tests passed.")
