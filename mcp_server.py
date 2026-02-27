# mcp_server.py
# MCP (Model Context Protocol) Server — Control Monitor
#
# Exposes loan & compliance operations as MCP tools.
# Both Streamlit (via mcp_client.py) and Claude agents can call these tools.
#
# Run:
#   python mcp_server.py
#   (listens on stdio by default — standard MCP transport)
#
# Or as HTTP server (for Streamlit):
#   python mcp_server.py --http --port 3000
#
# Architecture:
#   [Streamlit / Claude Agent]
#        │  MCP tool calls (JSON-RPC)
#        ▼
#   [MCP Server]  ← this file
#        │  HTTP calls
#        ▼
#   [FastAPI loan_service.py :8000]
#        │  psycopg2
#        ▼
#   [PostgreSQL]

import os
import json
import sys
import argparse
import logging
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP] %(levelname)s — %(message)s",
)
logger = logging.getLogger("mcp_server")

# Base URL of the FastAPI backend service
BACKEND_URL = os.getenv("LOAN_SERVICE_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────
# Backend HTTP helper
# ─────────────────────────────────────────────────────────────

def _call_backend(method: str, path: str, json_body: Optional[Dict] = None) -> Any:
    """
    Makes an HTTP call to the FastAPI loan_service.
    Raises RuntimeError with a human-readable message on failure.
    """
    url = f"{BACKEND_URL}{path}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            resp = requests.post(url, json=json_body, timeout=10)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        if resp.status_code == 404:
            return None  # Callers handle None as "not found"

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach loan_service at {BACKEND_URL}. "
            "Is 'uvicorn loan_service:app' running?"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request to {url} timed out.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"Backend error {e.response.status_code}: {detail or str(e)}")


# ─────────────────────────────────────────────────────────────
# MCP TOOL DEFINITIONS
# Each function = one MCP tool exposed to Claude / Streamlit
# ─────────────────────────────────────────────────────────────

# Tool registry — maps tool name → (function, description, input_schema)
TOOLS: Dict[str, Dict] = {}


def tool(name: str, description: str, input_schema: Dict):
    """Decorator to register a function as an MCP tool."""
    def decorator(fn):
        TOOLS[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                **input_schema,
            },
            "fn": fn,
        }
        return fn
    return decorator


# ── Tool 1: Health Check ─────────────────────────────────────
@tool(
    name="health_check",
    description="Check if the loan service backend and database are healthy.",
    input_schema={"properties": {}, "required": []},
)
def tool_health_check(args: Dict) -> Dict:
    result = _call_backend("GET", "/health")
    return result


# ── Tool 2: Get Next Loan ID ─────────────────────────────────
@tool(
    name="get_next_loan_id",
    description=(
        "Get the next auto-generated loan ID in LN-XXXXX format. "
        "Call this before creating a new loan to display the ID to the user."
    ),
    input_schema={"properties": {}, "required": []},
)
def tool_get_next_loan_id(args: Dict) -> Dict:
    result = _call_backend("GET", "/loans/next-id")
    return result


# ── Tool 3: List All Loans ───────────────────────────────────
@tool(
    name="list_loans",
    description=(
        "List all loans in the system with summary information "
        "(loan_id, borrower_name, branch_id, loan_amount, loan_type, funding_date). "
        "Use this to populate dropdowns or loan lists in the UI."
    ),
    input_schema={"properties": {}, "required": []},
)
def tool_list_loans(args: Dict) -> Dict:
    loans = _call_backend("GET", "/loans")
    return {"loans": loans, "count": len(loans) if loans else 0}


# ── Tool: Search Loans ───────────────────────────────────────
@tool(
    name="search_loans",
    description=(
        "Free-text search across loan columns (borrower name, branch, dates, etc.). "
        "Returns matching loans with their latest compliance summary."
    ),
    input_schema={
        "properties": {
            "q": {"type": "string", "description": "Search text (e.g., 'Sunita', 'BR-001', 'insured')"},
            "limit": {"type": "integer", "description": "Max rows to return (default 25)"},
        },
        "required": ["q"],
    },
)
def tool_search_loans(args: Dict) -> Dict:
    q = (args.get("q") or "").strip()
    if not q:
        return {"loans": [], "count": 0}
    limit = int(args.get("limit") or 25)
    limit = max(1, min(limit, 200))

    path = f"/loans/search?q={quote_plus(q)}&limit={limit}"
    loans = _call_backend("GET", path)
    return {"loans": loans or [], "count": len(loans) if loans else 0}


# ── Tool: Filter Loans (Structured) ──────────────────────────
@tool(
    name="filter_loans",
    description=(
        "Structured filtering for loans (DTI/loan amount/credit score/etc.) with optional free-text. "
        "Use this for queries like 'DTI > 50 and name contains Zili'."
    ),
    input_schema={
        "properties": {
            "q":               {"type": "string",  "description": "Optional free-text match across all columns"},
            "borrower_name":   {"type": "string",  "description": "Borrower name contains"},
            "branch_id":       {"type": "string",  "description": "Branch ID contains"},
            "loan_type":       {"type": "string",  "description": "Loan type contains"},
            "stress_test_passed": {"type": "boolean", "description": "Stress test pass/fail"},
            "dti_gt":          {"type": "number",  "description": "DTI > value"},
            "dti_gte":         {"type": "number",  "description": "DTI >= value"},
            "dti_lt":          {"type": "number",  "description": "DTI < value"},
            "dti_lte":         {"type": "number",  "description": "DTI <= value"},
            "loan_amount_gt":  {"type": "number",  "description": "Loan amount > value"},
            "loan_amount_gte": {"type": "number",  "description": "Loan amount >= value"},
            "loan_amount_lt":  {"type": "number",  "description": "Loan amount < value"},
            "loan_amount_lte": {"type": "number",  "description": "Loan amount <= value"},
            "credit_score_gt": {"type": "integer", "description": "Credit score > value"},
            "credit_score_gte":{"type": "integer", "description": "Credit score >= value"},
            "credit_score_lt": {"type": "integer", "description": "Credit score < value"},
            "credit_score_lte":{"type": "integer", "description": "Credit score <= value"},
            "limit":           {"type": "integer", "description": "Max rows to return (default 50)"},
        },
        "required": [],
    },
)
def tool_filter_loans(args: Dict) -> Dict:
    params = {}
    for key in [
        "q", "borrower_name", "branch_id", "loan_type", "stress_test_passed",
        "dti_gt", "dti_gte", "dti_lt", "dti_lte",
        "loan_amount_gt", "loan_amount_gte", "loan_amount_lt", "loan_amount_lte",
        "credit_score_gt", "credit_score_gte", "credit_score_lt", "credit_score_lte",
        "limit",
    ]:
        if key in args and args.get(key) is not None and str(args.get(key)).strip() != "":
            params[key] = args.get(key)

    limit = int(params.get("limit") or 50)
    limit = max(1, min(limit, 200))
    params["limit"] = limit

    qs = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
    path = f"/loans/filter?{qs}" if qs else "/loans/filter"
    loans = _call_backend("GET", path)
    return {"loans": loans or [], "count": len(loans) if loans else 0}


# ── Tool 4: Insert New Loan ──────────────────────────────────
@tool(
    name="insert_loan",
    description=(
        "Insert a new mortgage loan into the system. "
        "The loan_id is auto-generated — do NOT include it. "
        "Once inserted, the Kafka CDC pipeline automatically runs all 10 OSFI compliance controls "
        "and generates an AI breach explanation if any controls fail. "
        "Refresh the dashboard in ~10 seconds to see the compliance result."
    ),
    input_schema={
        "properties": {
            "borrower_name":            {"type": "string",  "description": "Full legal name of the borrower"},
            "branch_id":                {"type": "string",  "description": "Branch ID, e.g. BR-001"},
            "loan_amount":              {"type": "number",  "description": "Loan amount in CAD dollars"},
            "loan_type":                {"type": "string",  "description": "'conventional' or 'insured'"},
            "borrower_dti":             {"type": "number",  "description": "Debt-to-income ratio as % (e.g. 38.5)"},
            "borrower_credit_score":    {"type": "integer", "description": "Credit score 300–900"},
            "stress_test_passed":       {"type": "boolean", "description": "Did borrower pass OSFI stress test?"},
            "income_docs_count":        {"type": "integer", "description": "Number of income documents (min 2)"},
            "employment_verified_date": {"type": "string",  "description": "Date verified (YYYY-MM-DD)"},
            "closing_date":             {"type": "string",  "description": "Closing date (YYYY-MM-DD)"},
            "fair_lending_reviewed":    {"type": "boolean", "description": "Was fair lending review completed?"},
            "aml_completed":            {"type": "boolean", "description": "Was AML check completed?"},
            "disclosure_sent_date":     {"type": "string",  "description": "Disclosure sent date (YYYY-MM-DD)"},
            "underwriter_count":        {"type": "integer", "description": "Number of underwriters (min 1)"},
            "funding_date":             {"type": "string",  "description": "Funding date (YYYY-MM-DD)"},
            "breach_identified_date":   {"type": "string",  "description": "Breach date if any (YYYY-MM-DD), optional"},
            "escalation_date":          {"type": "string",  "description": "Escalation date if any (YYYY-MM-DD), optional"},
        },
        "required": [
            "borrower_name", "branch_id", "loan_amount", "loan_type",
            "borrower_dti", "borrower_credit_score", "stress_test_passed",
            "income_docs_count", "employment_verified_date", "closing_date",
            "fair_lending_reviewed", "aml_completed", "disclosure_sent_date",
            "underwriter_count", "funding_date",
        ],
    },
)
def tool_insert_loan(args: Dict) -> Dict:
    result = _call_backend("POST", "/loans", json_body=args)
    return result


# ── Tool 5: Get Single Loan ──────────────────────────────────
@tool(
    name="get_loan",
    description="Get the full details of a single loan by its loan_id (e.g. 'LN-10001').",
    input_schema={
        "properties": {
            "loan_id": {"type": "string", "description": "Loan ID, e.g. LN-10001"},
        },
        "required": ["loan_id"],
    },
)
def tool_get_loan(args: Dict) -> Dict:
    loan_id = args["loan_id"]
    result = _call_backend("GET", f"/loans/{loan_id}")
    if result is None:
        return {"error": f"Loan '{loan_id}' not found"}
    return result


# ── Tool 6: Get Compliance Result ───────────────────────────
@tool(
    name="get_compliance_result",
    description=(
        "Get the latest OSFI compliance result for a specific loan. "
        "Returns compliance score, overall status (COMPLIANT/BREACH), severity, "
        "control-by-control pass/fail results, and the AI-generated breach explanation."
    ),
    input_schema={
        "properties": {
            "loan_id": {"type": "string", "description": "Loan ID, e.g. LN-10001"},
        },
        "required": ["loan_id"],
    },
)
def tool_get_compliance_result(args: Dict) -> Dict:
    loan_id = args["loan_id"]
    result = _call_backend("GET", f"/loans/{loan_id}/compliance")
    if result is None:
        return {"error": f"No compliance result found for loan '{loan_id}'"}
    return result


# ── Tool 7: All Loans With Compliance ───────────────────────
@tool(
    name="list_loans_with_compliance",
    description=(
        "Get all loans joined with their latest compliance result. "
        "Returns loan details plus compliance_score, overall_status, severity, and checked_at. "
        "Use this to power the portfolio overview table."
    ),
    input_schema={"properties": {}, "required": []},
)
def tool_list_loans_with_compliance(args: Dict) -> Dict:
    loans = _call_backend("GET", "/loans/with-compliance")
    return {"loans": loans, "count": len(loans) if loans else 0}


# ── Tool 8: Portfolio Summary ────────────────────────────────
@tool(
    name="get_portfolio_summary",
    description=(
        "Get aggregate compliance statistics across the entire loan portfolio. "
        "Returns: total_loans, compliant_count, breach_count, "
        "high_severity, medium_severity, low_severity. "
        "Use this to power the metrics cards at the top of the dashboard."
    ),
    input_schema={"properties": {}, "required": []},
)
def tool_get_portfolio_summary(args: Dict) -> Dict:
    result = _call_backend("GET", "/portfolio/summary")
    return result


# ─────────────────────────────────────────────────────────────
# MCP JSON-RPC Protocol Handler (stdio transport)
# ─────────────────────────────────────────────────────────────

def handle_request(request: Dict) -> Dict:
    """
    Handles a single MCP JSON-RPC 2.0 request.
    Supports: initialize, tools/list, tools/call
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    try:
        # ── Initialise handshake ──
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name":    "control-monitor-mcp",
                        "version": "1.0.0",
                    },
                },
            }

        # ── List available tools ──
        elif method == "tools/list":
            tool_list = [
                {
                    "name":        t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                }
                for t in TOOLS.values()
            ]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tool_list},
            }

        # ── Execute a tool ──
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            if tool_name not in TOOLS:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found. "
                                   f"Available: {list(TOOLS.keys())}",
                    },
                }

            logger.info(f"Tool call: {tool_name}({json.dumps(tool_args)[:200]})")
            result = TOOLS[tool_name]["fn"](tool_args)
            logger.info(f"Tool '{tool_name}' completed successfully")

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, default=str),
                        }
                    ]
                },
            }

        # ── Notifications (no response needed) ──
        elif method in ("notifications/initialized",):
            return None

        # ── Unknown method ──
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    except RuntimeError as e:
        # Backend connectivity / business logic errors
        logger.error(f"Tool error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(e)},
        }
    except Exception as e:
        logger.error(f"Unexpected error handling '{method}': {e}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
        }


# ─────────────────────────────────────────────────────────────
# HTTP wrapper mode (for Streamlit → MCP calls over HTTP)
# ─────────────────────────────────────────────────────────────

def run_http_server(port: int = 3000):
    """
    Runs the MCP server as a simple HTTP endpoint.
    POST /mcp  → accepts JSON-RPC request body, returns JSON-RPC response.
    This lets Streamlit call MCP tools via plain HTTP instead of stdio.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class MCPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/mcp":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                request = json.loads(body)
                response = handle_request(request)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response, default=str).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        def log_message(self, format, *args):
            logger.info(f"HTTP {args[0]} {args[1]}")

    logger.info(f"MCP HTTP server starting on http://localhost:{port}/mcp")
    logger.info(f"Tools registered: {list(TOOLS.keys())}")
    server = HTTPServer(("0.0.0.0", port), MCPHandler)
    server.serve_forever()


# ─────────────────────────────────────────────────────────────
# Stdio transport mode (standard MCP — for Claude Desktop / CLI)
# ─────────────────────────────────────────────────────────────

def run_stdio_server():
    """
    Standard MCP stdio transport.
    Reads JSON-RPC requests from stdin, writes responses to stdout.
    Used by Claude Desktop and Claude Code when configured as an MCP server.
    """
    logger.info("MCP stdio server starting...")
    logger.info(f"Tools registered: {list(TOOLS.keys())}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                print(json.dumps(response, default=str), flush=True)
        except json.JSONDecodeError as e:
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            print(json.dumps(error_resp), flush=True)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control Monitor MCP Server")
    parser.add_argument(
        "--http", action="store_true",
        help="Run as HTTP server (for Streamlit) instead of stdio"
    )
    parser.add_argument(
        "--port", type=int, default=3000,
        help="HTTP port (default: 3000, only used with --http)"
    )
    args = parser.parse_args()

    if args.http:
        run_http_server(port=args.port)
    else:
        run_stdio_server()
