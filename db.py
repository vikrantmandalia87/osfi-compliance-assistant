# db.py
# PostgreSQL connection layer for OSFI Compliance AI
# Imported by: kafka_consumer.py, app.py, run_seed_compliance.py
#
# Schema notes:
# - public.retail_lending: loan_id is VARCHAR(20) (e.g. 'LN-10001'), already has 50 real loans
# - public.compliance_results: loan_id is VARCHAR(20) FK, result_id is VARCHAR(36) UUID string
# - No 'loan_officer' or 'origination_date' columns — uses 'branch_id' and 'funding_date'

import os
import json
import uuid
from datetime import datetime

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """
    Returns a new psycopg2 connection using RealDictCursor so all rows
    come back as dict-like objects (accessed by column name, not index).
    Caller is responsible for closing the connection.
    """
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5432)),
        dbname=os.getenv("PG_DB", "osfi_compliance"),
        user=os.getenv("PG_USER", "vikrantmandalia"),
        password=os.getenv("PG_PASSWORD", ""),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def get_all_loans():
    """
    Returns a list of all loans ordered by loan_id.
    Used by the Streamlit dropdown selector.

    Returns: list of dicts with keys:
        loan_id, borrower_name, branch_id, loan_amount, loan_type, funding_date
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    loan_id,
                    borrower_name,
                    branch_id,
                    loan_amount,
                    loan_type,
                    funding_date
                FROM public.retail_lending
                ORDER BY loan_id ASC
            """)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()


def get_loan_by_id(loan_id: str):
    """
    Returns a single loan row as a dict, or None if not found.
    Used by kafka_consumer.py to re-fetch after CDC event
    (avoids Debezium type mapping issues with dates/decimals).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM public.retail_lending WHERE loan_id = %s",
                (str(loan_id),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_compliance_result_for_loan(loan_id: str):
    """
    Returns the most recent compliance_results row for a given loan, or None.
    Used by Streamlit Tab 1 to display per-loan compliance status.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM public.compliance_results
                WHERE loan_id = %s
                ORDER BY checked_at DESC
                LIMIT 1
            """, (str(loan_id),))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def write_compliance_result(
    loan_id: str,
    control_results: dict,
    compliance_score: int,
    overall_status: str,
    breach_explanations: str,
    severity: str,
):
    """
    Inserts a new row into public.compliance_results.

    Args:
        loan_id:             VARCHAR(20) loan ID (e.g. 'LN-10001')
        control_results:     dict mapping control name → bool (True=PASS, False=FAIL)
        compliance_score:    integer 0-10
        overall_status:      'COMPLIANT' or 'BREACH'
        breach_explanations: GPT-4o-mini plain English explanation (empty string if no breaches)
        severity:            'High', 'Medium', 'Low', or 'Clean'
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.compliance_results
                    (result_id, loan_id, checked_at, control_results,
                     compliance_score, overall_status, breach_explanations, severity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(uuid.uuid4()),
                str(loan_id),
                datetime.utcnow(),
                json.dumps(control_results),
                compliance_score,
                overall_status,
                breach_explanations,
                severity,
            ))
        conn.commit()
    finally:
        conn.close()


def get_portfolio_summary():
    """
    Returns aggregate compliance stats across all loans with results.
    Uses DISTINCT ON to pick only the latest result per loan before aggregating.

    Returns dict with keys:
        total_loans, compliant_count, breach_count,
        high_severity, medium_severity, low_severity
    Returns None if no results exist yet.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                                       AS total_loans,
                    SUM(CASE WHEN overall_status = 'COMPLIANT' THEN 1 ELSE 0 END) AS compliant_count,
                    SUM(CASE WHEN overall_status = 'BREACH'    THEN 1 ELSE 0 END) AS breach_count,
                    SUM(CASE WHEN severity = 'High'   THEN 1 ELSE 0 END)           AS high_severity,
                    SUM(CASE WHEN severity = 'Medium' THEN 1 ELSE 0 END)           AS medium_severity,
                    SUM(CASE WHEN severity = 'Low'    THEN 1 ELSE 0 END)           AS low_severity
                FROM (
                    -- Get only the latest compliance result per loan
                    SELECT DISTINCT ON (loan_id)
                        loan_id,
                        overall_status,
                        severity
                    FROM public.compliance_results
                    ORDER BY loan_id, checked_at DESC
                ) latest
            """)
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_next_loan_id() -> str:
    """
    Returns the next available loan_id in the format LN-XXXXX.
    Finds the highest existing numeric suffix and increments by 1.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT MAX(CAST(SUBSTRING(loan_id FROM 4) AS INTEGER)) AS max_num
                FROM public.retail_lending
                WHERE loan_id ~ '^LN-[0-9]+$'
            """)
            row = cur.fetchone()
            max_num = int(row["max_num"]) if row and row["max_num"] else 10000
            return f"LN-{max_num + 1:05d}"
    finally:
        conn.close()


def insert_loan(loan: dict) -> str:
    """
    Inserts a new loan row into public.retail_lending.

    Args:
        loan: dict with loan fields. loan_id must be pre-generated.

    Returns:
        The loan_id that was inserted.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.retail_lending (
                    loan_id, borrower_name, branch_id, loan_amount, loan_type,
                    borrower_dti, borrower_credit_score, stress_test_passed,
                    income_docs_count, employment_verified_date, closing_date,
                    fair_lending_reviewed, aml_completed, disclosure_sent_date,
                    underwriter_count, breach_identified_date, escalation_date,
                    funding_date
                ) VALUES (
                    %(loan_id)s, %(borrower_name)s, %(branch_id)s, %(loan_amount)s, %(loan_type)s,
                    %(borrower_dti)s, %(borrower_credit_score)s, %(stress_test_passed)s,
                    %(income_docs_count)s, %(employment_verified_date)s, %(closing_date)s,
                    %(fair_lending_reviewed)s, %(aml_completed)s, %(disclosure_sent_date)s,
                    %(underwriter_count)s, %(breach_identified_date)s, %(escalation_date)s,
                    %(funding_date)s
                )
            """, loan)
        conn.commit()
        return loan["loan_id"]
    finally:
        conn.close()


def get_all_loans_with_compliance():
    """
    Returns all loans joined with their latest compliance result.
    Used by the portfolio overview table in Tab 1.

    Returns list of dicts with combined loan + compliance fields.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    rl.loan_id,
                    rl.borrower_name,
                    rl.branch_id,
                    rl.loan_amount,
                    rl.loan_type,
                    rl.funding_date,
                    cr.compliance_score,
                    cr.overall_status,
                    cr.severity,
                    cr.checked_at
                FROM public.retail_lending rl
                LEFT JOIN LATERAL (
                    SELECT compliance_score, overall_status, severity, checked_at
                    FROM public.compliance_results
                    WHERE loan_id = rl.loan_id
                    ORDER BY checked_at DESC
                    LIMIT 1
                ) cr ON true
                ORDER BY rl.loan_id ASC
            """)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()
