# loan_service.py
# FastAPI Backend Service — Control Monitor
#
# Owns ALL database operations. Streamlit and the MCP server
# never touch PostgreSQL directly — they go through this service.
#
# Run:
#   uvicorn loan_service:app --host 0.0.0.0 --port 8000 --reload
#
# Endpoints:
#   GET  /health                          → health check
#   GET  /loans/next-id                   → get next auto loan ID
#   GET  /loans                           → list all loans (summary)
#   POST /loans                           → insert a new loan
#   GET  /loans/{loan_id}                 → get single loan
#   GET  /loans/{loan_id}/compliance      → get compliance result
#   GET  /loans/with-compliance           → all loans + latest compliance
#   GET  /portfolio/summary               → aggregate compliance stats

import os
import json
import uuid
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware

from models import (
    LoanCreateRequest,
    LoanCreateResponse,
    LoanSummary,
    LoanWithCompliance,
    ComplianceResult,
    PortfolioSummary,
    NextLoanIdResponse,
    HealthResponse,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("loan_service")

# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Control Monitor — Loan Service",
    description=(
        "Backend REST API for the OSFI Compliance Control Monitor. "
        "Owns all PostgreSQL operations for loans and compliance results. "
        "Called by the MCP server and (optionally) directly by Streamlit."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc at http://localhost:8000/redoc
)

# Allow Streamlit (localhost:8501) and MCP server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# DB connection helper
# ─────────────────────────────────────────────────────────────
def get_connection():
    """Returns a psycopg2 connection with RealDictCursor."""
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5432)),
        dbname=os.getenv("PG_DB", "osfi_compliance"),
        user=os.getenv("PG_USER", "vikrantmandalia"),
        password=os.getenv("PG_PASSWORD", ""),
        cursor_factory=psycopg2.extras.RealDictCursor,
        connect_timeout=5,
    )


def db_available() -> bool:
    """Quick DB ping — used in health check."""
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Health Check
# ─────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check — confirms service and DB are up",
)
def health_check():
    db_status = "connected" if db_available() else "unavailable"
    return HealthResponse(
        status="ok",
        service="Control Monitor Loan Service",
        database=db_status,
        version="1.0.0",
    )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Next Loan ID
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans/next-id",
    response_model=NextLoanIdResponse,
    tags=["Loans"],
    summary="Get the next auto-generated loan ID (LN-XXXXX format)",
)
def get_next_loan_id():
    """
    Calculates the next available LN-XXXXX loan ID by reading the
    highest existing numeric suffix from the database.
    """
    try:
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
                next_id = f"LN-{max_num + 1:05d}"
                logger.info(f"Next loan ID: {next_id}")
                return NextLoanIdResponse(next_loan_id=next_id)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to get next loan ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: List All Loans
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans",
    response_model=List[LoanSummary],
    tags=["Loans"],
    summary="List all loans (summary view for dropdown/table)",
)
def list_loans():
    """Returns all loans ordered by loan_id. Used by the dashboard dropdown."""
    try:
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
    except Exception as e:
        logger.error(f"Failed to list loans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: All Loans With Compliance (Portfolio Table)
# NOTE: Must be defined BEFORE /loans/{loan_id} so FastAPI doesn't
# match "with-compliance" as a loan_id parameter.
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans/with-compliance",
    response_model=List[LoanWithCompliance],
    tags=["Compliance"],
    summary="All loans joined with their latest compliance result",
)
def list_loans_with_compliance():
    """
    Returns all loans left-joined with their most recent compliance result.
    Powers the portfolio overview table in the dashboard.
    """
    try:
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
    except Exception as e:
        logger.error(f"Failed to list loans with compliance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Search Loans (Free-text)
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans/search",
    response_model=List[LoanWithCompliance],
    tags=["Loans"],
    summary="Free-text search across loan columns (and latest compliance fields)",
)
def search_loans(
    q: str = Query(..., min_length=1, description="Free-text search query"),
    limit: int = Query(25, ge=1, le=200, description="Max rows to return"),
):
    """
    Searches public.retail_lending across all columns using a conservative
    free-text match (ILIKE). Also matches against the latest compliance
    result fields (status/severity) for convenience.
    """
    q = (q or "").strip()
    pattern = f"%{q}%"
    tokens = [t for t in q.lower().split() if t][:8]
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                params = {"pattern": pattern, "limit": int(limit)}
                token_ors = []
                for i, t in enumerate(tokens):
                    params[f"t{i}"] = f"%{t}%"
                    token_ors.append(
                        f"regexp_replace(lower(to_jsonb(rl)::text), '[^a-z0-9]+', ' ', 'g') ILIKE %(t{i})s"
                    )

                where_clause = """
                        (to_jsonb(rl)::text ILIKE %(pattern)s)
                        OR (COALESCE(cr.overall_status, '') ILIKE %(pattern)s)
                        OR (COALESCE(cr.severity, '') ILIKE %(pattern)s)
                """
                if token_ors:
                    where_clause = f"({where_clause}) OR (" + " OR ".join(token_ors) + ")"

                cur.execute(f"""
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
                    WHERE
                        {where_clause}
                    ORDER BY rl.loan_id ASC
                    LIMIT %(limit)s
                """, params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to search loans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Filter Loans (Structured)
# NOTE: Must be defined BEFORE /loans/{loan_id}
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans/filter",
    response_model=List[LoanWithCompliance],
    tags=["Loans"],
    summary="Structured filtering (DTI/amount/credit score/etc.) with optional free-text",
)
def filter_loans(
    q: Optional[str] = Query(None, description="Optional free-text match across loan columns"),
    borrower_name: Optional[str] = Query(None, description="Borrower name contains (case-insensitive)"),
    branch_id: Optional[str] = Query(None, description="Branch ID contains (case-insensitive)"),
    loan_type: Optional[str] = Query(None, description="Loan type contains (case-insensitive)"),
    stress_test_passed: Optional[bool] = Query(None, description="Stress test pass/fail"),
    dti_gt: Optional[float] = Query(None, description="Borrower DTI > value"),
    dti_gte: Optional[float] = Query(None, description="Borrower DTI >= value"),
    dti_lt: Optional[float] = Query(None, description="Borrower DTI < value"),
    dti_lte: Optional[float] = Query(None, description="Borrower DTI <= value"),
    loan_amount_gt: Optional[float] = Query(None, description="Loan amount > value"),
    loan_amount_gte: Optional[float] = Query(None, description="Loan amount >= value"),
    loan_amount_lt: Optional[float] = Query(None, description="Loan amount < value"),
    loan_amount_lte: Optional[float] = Query(None, description="Loan amount <= value"),
    credit_score_gt: Optional[int] = Query(None, description="Credit score > value"),
    credit_score_gte: Optional[int] = Query(None, description="Credit score >= value"),
    credit_score_lt: Optional[int] = Query(None, description="Credit score < value"),
    credit_score_lte: Optional[int] = Query(None, description="Credit score <= value"),
    limit: int = Query(50, ge=1, le=200, description="Max rows to return"),
):
    """
    Structured filtering endpoint for common dashboard queries like:
      - DTI > 50 and name contains 'Zili'
      - loan_amount >= 500000
      - credit_score < 620
    """
    where = []
    params: Dict[str, Any] = {"limit": int(limit)}

    if borrower_name:
        where.append("rl.borrower_name ILIKE %(borrower_name)s")
        params["borrower_name"] = f"%{borrower_name.strip()}%"
    if branch_id:
        where.append("rl.branch_id ILIKE %(branch_id)s")
        params["branch_id"] = f"%{branch_id.strip()}%"
    if loan_type:
        where.append("rl.loan_type ILIKE %(loan_type)s")
        params["loan_type"] = f"%{loan_type.strip()}%"
    if stress_test_passed is not None:
        where.append("rl.stress_test_passed = %(stress_test_passed)s")
        params["stress_test_passed"] = bool(stress_test_passed)

    def _add_num(field: str, op: str, key: str, value: Any):
        if value is None:
            return
        where.append(f"{field} {op} %({key})s")
        params[key] = value

    _add_num("rl.borrower_dti", ">",  "dti_gt",  dti_gt)
    _add_num("rl.borrower_dti", ">=", "dti_gte", dti_gte)
    _add_num("rl.borrower_dti", "<",  "dti_lt",  dti_lt)
    _add_num("rl.borrower_dti", "<=", "dti_lte", dti_lte)

    _add_num("rl.loan_amount", ">",  "loan_amount_gt",  loan_amount_gt)
    _add_num("rl.loan_amount", ">=", "loan_amount_gte", loan_amount_gte)
    _add_num("rl.loan_amount", "<",  "loan_amount_lt",  loan_amount_lt)
    _add_num("rl.loan_amount", "<=", "loan_amount_lte", loan_amount_lte)

    _add_num("rl.borrower_credit_score", ">",  "credit_score_gt",  credit_score_gt)
    _add_num("rl.borrower_credit_score", ">=", "credit_score_gte", credit_score_gte)
    _add_num("rl.borrower_credit_score", "<",  "credit_score_lt",  credit_score_lt)
    _add_num("rl.borrower_credit_score", "<=", "credit_score_lte", credit_score_lte)

    if q:
        params["q_pattern"] = f"%{q.strip()}%"
        where.append(
            "(to_jsonb(rl)::text ILIKE %(q_pattern)s)"
        )

    where_sql = " AND ".join(where) if where else "TRUE"

    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
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
                    WHERE {where_sql}
                    ORDER BY rl.loan_id ASC
                    LIMIT %(limit)s
                """, params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to filter loans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Get Single Loan
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans/{loan_id}",
    tags=["Loans"],
    summary="Get a single loan by ID (full record)",
)
def get_loan(loan_id: str):
    """Returns the full loan record. Used by kafka_consumer to re-fetch after CDC."""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM public.retail_lending WHERE loan_id = %s",
                    (loan_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Loan '{loan_id}' not found",
                    )
                return dict(row)
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get loan {loan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Insert New Loan
# ─────────────────────────────────────────────────────────────
@app.post(
    "/loans",
    response_model=LoanCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Loans"],
    summary="Insert a new loan — triggers the Kafka CDC compliance pipeline",
)
def create_loan(loan: LoanCreateRequest):
    """
    Inserts a new loan into public.retail_lending.
    Debezium CDC automatically detects the INSERT and fires a Kafka event,
    which triggers the compliance pipeline (run_all_controls → GPT explanation).

    The loan_id is auto-generated by the service — do not send it in the payload.
    """
    try:
        # Auto-generate loan ID
        next_id_resp = get_next_loan_id()
        loan_id = next_id_resp.next_loan_id

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
                        %(loan_id)s, %(borrower_name)s, %(branch_id)s,
                        %(loan_amount)s, %(loan_type)s, %(borrower_dti)s,
                        %(borrower_credit_score)s, %(stress_test_passed)s,
                        %(income_docs_count)s, %(employment_verified_date)s,
                        %(closing_date)s, %(fair_lending_reviewed)s,
                        %(aml_completed)s, %(disclosure_sent_date)s,
                        %(underwriter_count)s, %(breach_identified_date)s,
                        %(escalation_date)s, %(funding_date)s
                    )
                """, {
                    "loan_id":                  loan_id,
                    "borrower_name":            loan.borrower_name,
                    "branch_id":                loan.branch_id,
                    "loan_amount":              loan.loan_amount,
                    "loan_type":                loan.loan_type,
                    "borrower_dti":             loan.borrower_dti,
                    "borrower_credit_score":    loan.borrower_credit_score,
                    "stress_test_passed":       loan.stress_test_passed,
                    "income_docs_count":        loan.income_docs_count,
                    "employment_verified_date": loan.employment_verified_date,
                    "closing_date":             loan.closing_date,
                    "fair_lending_reviewed":    loan.fair_lending_reviewed,
                    "aml_completed":            loan.aml_completed,
                    "disclosure_sent_date":     loan.disclosure_sent_date,
                    "underwriter_count":        loan.underwriter_count,
                    "breach_identified_date":   loan.breach_identified_date,
                    "escalation_date":          loan.escalation_date,
                    "funding_date":             loan.funding_date,
                })
            conn.commit()
            logger.info(f"Loan {loan_id} inserted for borrower '{loan.borrower_name}'")
            return LoanCreateResponse(
                loan_id=loan_id,
                message=(
                    f"Loan {loan_id} inserted successfully. "
                    "The Kafka CDC pipeline will automatically run compliance checks."
                ),
                status="inserted",
            )
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to insert loan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Get Compliance Result for a Loan
# ─────────────────────────────────────────────────────────────
@app.get(
    "/loans/{loan_id}/compliance",
    tags=["Compliance"],
    summary="Get the latest compliance result for a specific loan",
)
def get_compliance_result(loan_id: str):
    """
    Returns the most recent compliance result for a given loan_id.
    Returns 404 if no result exists yet (loan may be pending pipeline processing).
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT *
                    FROM public.compliance_results
                    WHERE loan_id = %s
                    ORDER BY checked_at DESC
                    LIMIT 1
                """, (loan_id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No compliance result found for loan '{loan_id}'",
                    )
                result = dict(row)
                # Deserialise JSONB control_results if it came back as string
                if isinstance(result.get("control_results"), str):
                    result["control_results"] = json.loads(result["control_results"])
                return result
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get compliance result for {loan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Portfolio Summary (Metrics)
# ─────────────────────────────────────────────────────────────
@app.get(
    "/portfolio/summary",
    response_model=PortfolioSummary,
    tags=["Portfolio"],
    summary="Aggregate compliance metrics across all loans",
)
def get_portfolio_summary():
    """
    Returns total loans, compliant count, breach count, and severity breakdown.
    Uses DISTINCT ON to count only the latest result per loan.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)                                                       AS total_loans,
                        SUM(CASE WHEN overall_status = 'COMPLIANT' THEN 1 ELSE 0 END) AS compliant_count,
                        SUM(CASE WHEN overall_status = 'BREACH'    THEN 1 ELSE 0 END) AS breach_count,
                        SUM(CASE WHEN severity = 'High'   THEN 1 ELSE 0 END)          AS high_severity,
                        SUM(CASE WHEN severity = 'Medium' THEN 1 ELSE 0 END)          AS medium_severity,
                        SUM(CASE WHEN severity = 'Low'    THEN 1 ELSE 0 END)          AS low_severity
                    FROM (
                        SELECT DISTINCT ON (loan_id)
                            loan_id, overall_status, severity
                        FROM public.compliance_results
                        ORDER BY loan_id, checked_at DESC
                    ) latest
                """)
                row = cur.fetchone()
                if not row or row["total_loans"] == 0:
                    return PortfolioSummary(
                        total_loans=0, compliant_count=0, breach_count=0,
                        high_severity=0, medium_severity=0, low_severity=0,
                    )
                return PortfolioSummary(
                    total_loans=int(row["total_loans"]),
                    compliant_count=int(row["compliant_count"] or 0),
                    breach_count=int(row["breach_count"] or 0),
                    high_severity=int(row["high_severity"] or 0),
                    medium_severity=int(row["medium_severity"] or 0),
                    low_severity=int(row["low_severity"] or 0),
                )
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────
# Run directly for dev
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("loan_service:app", host="0.0.0.0", port=8000, reload=True)
