"""
db.py — Direct PostgreSQL access for the Compliance Controls Dashboard.
Provides the same interface that vibrant-wozniak's mcp_client.py exposed,
but queries the database directly via psycopg2 (no HTTP MCP layer needed).
"""

import json
import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "osfi_compliance",
    "user": "vikrantmandalia",
}


def _conn():
    return psycopg2.connect(**DB_CONFIG)


def _serialize(row: dict) -> dict:
    result = {}
    for k, v in row.items():
        if isinstance(v, Decimal):
            result[k] = float(v)
        elif isinstance(v, (datetime.date, datetime.datetime)):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


# ──────────────────────────────────────────────
# Loan ID generation
# ──────────────────────────────────────────────

def get_next_loan_id() -> str:
    """Return the next available loan ID (e.g. 'LN-10056')."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT MAX(CAST(SUBSTRING(loan_id FROM 4) AS INTEGER)) FROM public.retail_lending"
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    max_num = row[0] if row and row[0] else 10000
    return f"LN-{max_num + 1}"


# ──────────────────────────────────────────────
# Loan reads
# ──────────────────────────────────────────────

def get_all_loans() -> List[Dict]:
    """Return summary fields for all loans, ordered by loan_id."""
    conn = _conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT loan_id, borrower_name, branch_id, loan_amount, loan_type
        FROM public.retail_lending
        ORDER BY loan_id
    """)
    rows = [_serialize(dict(r)) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def get_all_loans_with_compliance() -> List[Dict]:
    """Return all loans joined with their latest compliance result."""
    conn = _conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            rl.loan_id,
            rl.borrower_name,
            rl.branch_id,
            rl.loan_amount,
            rl.loan_type,
            cr.compliance_score,
            cr.overall_status,
            cr.severity,
            cr.checked_at
        FROM public.retail_lending rl
        LEFT JOIN (
            SELECT DISTINCT ON (loan_id) *
            FROM public.compliance_results
            ORDER BY loan_id, checked_at DESC
        ) cr ON rl.loan_id = cr.loan_id
        ORDER BY rl.loan_id
    """)
    rows = [_serialize(dict(r)) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def get_compliance_result_for_loan(loan_id: str) -> Optional[Dict]:
    """Return the latest compliance result for a given loan_id, or None."""
    conn = _conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT *
        FROM public.compliance_results
        WHERE loan_id = %s
        ORDER BY checked_at DESC
        LIMIT 1
    """, (loan_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None
    result = _serialize(dict(row))
    # Ensure control_results is a dict (may be stored as JSON string)
    if isinstance(result.get("control_results"), str):
        try:
            result["control_results"] = json.loads(result["control_results"])
        except Exception:
            pass
    return result


def get_portfolio_summary() -> Optional[Dict]:
    """Return aggregate compliance metrics across all loans."""
    conn = _conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            COUNT(*)                                               AS total_loans,
            COUNT(*) FILTER (WHERE overall_status = 'COMPLIANT')  AS compliant_count,
            COUNT(*) FILTER (WHERE overall_status = 'BREACH')     AS breach_count,
            COUNT(*) FILTER (WHERE severity = 'High')             AS high_severity,
            COUNT(*) FILTER (WHERE severity = 'Medium')           AS medium_severity,
            COUNT(*) FILTER (WHERE severity = 'Low')              AS low_severity
        FROM (
            SELECT DISTINCT ON (loan_id) *
            FROM public.compliance_results
            ORDER BY loan_id, checked_at DESC
        ) latest
    """)
    row = cur.fetchone()
    cur.close()
    conn.close()
    return _serialize(dict(row)) if row else None


# ──────────────────────────────────────────────
# Loan write
# ──────────────────────────────────────────────

def insert_loan(loan_data: Dict) -> str:
    """
    Insert a new loan into public.retail_lending.
    Returns the inserted loan_id.
    """
    loan_id = loan_data.get("loan_id") or get_next_loan_id()

    closing_date = loan_data.get("closing_date")
    emp_verified = loan_data.get("employment_verified_date")
    disclosure_date = loan_data.get("disclosure_sent_date")

    # Compute derived day-count fields
    if closing_date and emp_verified:
        if isinstance(closing_date, str):
            closing_date = datetime.date.fromisoformat(closing_date)
        if isinstance(emp_verified, str):
            emp_verified = datetime.date.fromisoformat(emp_verified)
        days_emp_to_closing = (closing_date - emp_verified).days
    else:
        days_emp_to_closing = None

    if closing_date and disclosure_date:
        if isinstance(disclosure_date, str):
            disclosure_date = datetime.date.fromisoformat(disclosure_date)
        days_before_closing = (closing_date - disclosure_date).days
    else:
        days_before_closing = None

    breach_date = loan_data.get("breach_identified_date")
    escalation_date = loan_data.get("escalation_date")
    breach_identified = bool(breach_date)

    # Compute hours_to_escalate
    hours_to_escalate = None
    if breach_date and escalation_date:
        if isinstance(breach_date, str):
            breach_date = datetime.date.fromisoformat(breach_date)
        if isinstance(escalation_date, str):
            escalation_date = datetime.date.fromisoformat(escalation_date)
        hours_to_escalate = float((escalation_date - breach_date).days * 24)

    funding_date = loan_data.get("funding_date")
    if isinstance(funding_date, str):
        funding_date = datetime.date.fromisoformat(funding_date)

    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO public.retail_lending (
            loan_id, loan_type, loan_amount, funding_date,
            branch_id, borrower_name,
            borrower_dti, borrower_credit_score,
            stress_test_passed, income_docs_count,
            employment_verified_date, closing_date, days_emp_to_closing,
            fair_lending_reviewed, aml_completed,
            disclosure_sent_date, days_before_closing,
            underwriter_count,
            breach_identified, breach_identified_date,
            escalation_date, hours_to_escalate
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s,
            %s, %s,
            %s, %s
        )
    """, (
        loan_id,
        loan_data.get("loan_type"),
        float(loan_data.get("loan_amount", 0)),
        funding_date,
        loan_data.get("branch_id"),
        loan_data.get("borrower_name"),
        float(loan_data.get("borrower_dti", 0)),
        int(loan_data.get("borrower_credit_score", 0)),
        bool(loan_data.get("stress_test_passed", False)),
        int(loan_data.get("income_docs_count", 0)),
        emp_verified,
        closing_date,
        days_emp_to_closing,
        bool(loan_data.get("fair_lending_reviewed", False)),
        bool(loan_data.get("aml_completed", False)),
        disclosure_date,
        days_before_closing,
        int(loan_data.get("underwriter_count", 1)),
        breach_identified,
        breach_date if breach_identified else None,
        escalation_date if breach_identified else None,
        hours_to_escalate,
    ))
    conn.commit()
    cur.close()
    conn.close()
    return loan_id
