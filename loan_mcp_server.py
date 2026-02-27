#!/usr/bin/env python3
"""
MCP Server for OSFI Retail Lending Database.

Exposes PostgreSQL retail_lending table as MCP tools so the chatbot
can query loan data using natural language.
"""

import json
import datetime
from decimal import Decimal
from typing import Optional

import psycopg2
import psycopg2.extras
from mcp.server.fastmcp import FastMCP

# ──────────────────────────────────────────────
# Server setup
# ──────────────────────────────────────────────
mcp = FastMCP("OSFI Loan Database")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "osfi_compliance",
    "user": "vikrantmandalia",
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def serialize_row(row: dict) -> dict:
    """Convert PostgreSQL-specific types to JSON-serializable Python types."""
    result = {}
    for key, val in row.items():
        if isinstance(val, Decimal):
            result[key] = float(val)
        elif isinstance(val, (datetime.date, datetime.datetime)):
            result[key] = val.isoformat()
        else:
            result[key] = val
    return result


# ──────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────

@mcp.tool()
def query_loans(
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    loan_type: Optional[str] = None,
    property_state: Optional[str] = None,
    branch_id: Optional[str] = None,
    stress_test_passed: Optional[bool] = None,
    breach_identified: Optional[bool] = None,
    min_credit_score: Optional[int] = None,
    max_credit_score: Optional[int] = None,
    min_dti: Optional[float] = None,
    max_dti: Optional[float] = None,
    limit: int = 20,
) -> str:
    """
    Query retail loans with optional filters. Results sorted by loan amount descending.

    Args:
        min_amount: Minimum loan amount in dollars (e.g. 300000 for loans >= $300K)
        max_amount: Maximum loan amount in dollars
        loan_type: Loan type filter (e.g. 'Jumbo', 'Conventional')
        property_state: Province filter (e.g. 'Ontario', 'Alberta', 'Manitoba')
        branch_id: Branch identifier filter (e.g. 'BR-001', 'BR-002')
        stress_test_passed: True = passed, False = failed
        breach_identified: True = has breach, False = no breach
        min_credit_score: Minimum borrower credit score
        max_credit_score: Maximum borrower credit score
        min_dti: Minimum borrower DTI ratio as a percentage (e.g. 40 for DTI >= 40%)
        max_dti: Maximum borrower DTI ratio as a percentage (e.g. 44 for DTI <= 44%)
        limit: Max rows to return (default 20, capped at 100)
    """
    conditions: list[str] = []
    params: list = []

    if min_amount is not None:
        conditions.append("loan_amount >= %s")
        params.append(min_amount)
    if max_amount is not None:
        conditions.append("loan_amount <= %s")
        params.append(max_amount)
    if loan_type:
        conditions.append("loan_type ILIKE %s")
        params.append(f"%{loan_type}%")
    if property_state:
        conditions.append("property_state ILIKE %s")
        params.append(f"%{property_state}%")
    if branch_id:
        conditions.append("branch_id ILIKE %s")
        params.append(f"%{branch_id}%")
    if stress_test_passed is not None:
        conditions.append("stress_test_passed = %s")
        params.append(stress_test_passed)
    if breach_identified is not None:
        conditions.append("breach_identified = %s")
        params.append(breach_identified)
    if min_credit_score is not None:
        conditions.append("borrower_credit_score >= %s")
        params.append(min_credit_score)
    if max_credit_score is not None:
        conditions.append("borrower_credit_score <= %s")
        params.append(max_credit_score)
    if min_dti is not None:
        conditions.append("borrower_dti >= %s")
        params.append(min_dti)
    if max_dti is not None:
        conditions.append("borrower_dti <= %s")
        params.append(max_dti)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit = min(int(limit), 100)

    # When filtering by DTI, sort by DTI descending so highest-risk loans appear first
    order_by = "borrower_dti DESC" if (min_dti is not None or max_dti is not None) else "loan_amount DESC"

    sql = f"""
        SELECT
            loan_id, loan_type, loan_amount, borrower_name,
            branch_id, borrower_credit_score, borrower_dti,
            property_value, property_state,
            contract_rate, stress_test_passed,
            breach_identified, funding_date
        FROM public.retail_lending
        {where_clause}
        ORDER BY {order_by}
        LIMIT %s
    """
    params.append(limit)

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return json.dumps({
            "count": 0,
            "loans": [],
            "message": "No loans found matching the criteria.",
        })

    loans = [serialize_row(dict(row)) for row in rows]
    return json.dumps({"count": len(loans), "loans": loans})


@mcp.tool()
def get_loan_by_id(loan_id: str) -> str:
    """
    Get full details of a single loan by its ID.

    Args:
        loan_id: The loan identifier (e.g. 'LN-10001')
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT * FROM public.retail_lending WHERE loan_id = %s", (loan_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return json.dumps({"error": f"Loan '{loan_id}' not found."})

    return json.dumps(serialize_row(dict(row)))


@mcp.tool()
def get_loan_stats() -> str:
    """
    Get portfolio-level statistics: totals, averages, breakdowns by loan type and province.
    No parameters required.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT
            COUNT(*)                                              AS total_loans,
            ROUND(AVG(loan_amount)::numeric, 2)                  AS avg_loan_amount,
            ROUND(MIN(loan_amount)::numeric, 2)                  AS min_loan_amount,
            ROUND(MAX(loan_amount)::numeric, 2)                  AS max_loan_amount,
            ROUND(SUM(loan_amount)::numeric, 2)                  AS total_portfolio_value,
            COUNT(*) FILTER (WHERE stress_test_passed = TRUE)    AS stress_test_passed_count,
            COUNT(*) FILTER (WHERE breach_identified  = TRUE)    AS breach_count,
            ROUND(AVG(borrower_credit_score)::numeric, 0)        AS avg_credit_score,
            ROUND(AVG(borrower_dti)::numeric, 2)                 AS avg_dti
        FROM public.retail_lending
    """)
    summary = serialize_row(dict(cur.fetchone()))

    cur.execute("""
        SELECT loan_type,
               COUNT(*) AS count,
               ROUND(AVG(loan_amount)::numeric, 2) AS avg_amount
        FROM public.retail_lending
        GROUP BY loan_type
        ORDER BY count DESC
    """)
    by_type = [serialize_row(dict(r)) for r in cur.fetchall()]

    cur.execute("""
        SELECT property_state,
               COUNT(*) AS count,
               ROUND(AVG(loan_amount)::numeric, 2) AS avg_amount
        FROM public.retail_lending
        GROUP BY property_state
        ORDER BY count DESC
        LIMIT 10
    """)
    by_province = [serialize_row(dict(r)) for r in cur.fetchall()]

    cur.close()
    conn.close()

    return json.dumps({
        "summary": summary,
        "by_loan_type": by_type,
        "by_province": by_province,
    })


@mcp.tool()
def search_borrower(name: str, limit: int = 10) -> str:
    """
    Search for loans by borrower name (partial match supported).

    Args:
        name: Borrower name or partial name (e.g. 'Jones', 'Daniel')
        limit: Max results to return (default 10)
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT loan_id, borrower_name, loan_type, loan_amount,
               property_state, funding_date,
               stress_test_passed, breach_identified
        FROM public.retail_lending
        WHERE borrower_name ILIKE %s
        ORDER BY funding_date DESC
        LIMIT %s
        """,
        (f"%{name}%", limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return json.dumps({
            "count": 0,
            "loans": [],
            "message": f"No borrower found matching '{name}'.",
        })

    loans = [serialize_row(dict(r)) for r in rows]
    return json.dumps({"count": len(loans), "loans": loans})


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
