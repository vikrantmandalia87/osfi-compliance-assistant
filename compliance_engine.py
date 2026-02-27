# compliance_engine.py
# Deterministic compliance control engine for OSFI Retail Lending
# Imported by: kafka_consumer.py, run_seed_compliance.py
#
# Design principles:
# - Pass/fail decisions are 100% deterministic Python logic (no LLM)
# - GPT-4o-mini is ONLY used to generate plain-English breach explanations
# - Missing data (None) is treated as a conservative FAIL
# - _date_or_none() handles both psycopg2 date objects AND Debezium epoch-day integers

import os
from datetime import date, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# Severity mapping per control
# -------------------------------------------------------
CONTROL_SEVERITY = {
    "control_1":  "Medium",   # DTI Ratio
    "control_2":  "High",     # Credit Score
    "control_3":  "High",     # Stress Test
    "control_4":  "Medium",   # Income Verification
    "control_5":  "Low",      # Employment Verification Timeliness
    "control_6":  "Medium",   # Fair Lending
    "control_7":  "High",     # AML Check
    "control_8":  "Low",      # Closing Disclosure Timing
    "control_9":  "Low",      # Dual Underwriter
    "control_10": "Low",      # Exception Escalation
}

SEVERITY_RANK = {"High": 3, "Medium": 2, "Low": 1, "Clean": 0}

CONTROL_LABELS = {
    "control_1":  "DTI Ratio Check",
    "control_2":  "Minimum Credit Score",
    "control_3":  "Stress Test (OSFI B-20)",
    "control_4":  "Income Verification",
    "control_5":  "Employment Verification Timeliness",
    "control_6":  "Fair Lending Check",
    "control_7":  "AML Check Completion",
    "control_8":  "Closing Disclosure Timing",
    "control_9":  "Dual Underwriter Review (High Value Loans)",
    "control_10": "Control Exception Escalation",
}

CONTROL_DESCRIPTIONS = {
    "control_1":  ("DTI Ratio Check",
                   "Borrower DTI must be at or below 44%",
                   "OSFI B-20 credit risk guidelines"),
    "control_2":  ("Credit Score",
                   "Borrower credit score must be >= 620 for conventional loans",
                   "OSFI B-20 Section 4 — Borrower Assessment"),
    "control_3":  ("Stress Test",
                   "Borrower must pass the minimum qualifying rate stress test (contract rate + 2%)",
                   "OSFI B-20 Section 2 — Minimum Qualifying Rate"),
    "control_4":  ("Income Verification",
                   "At least 2 independent income documents must be on file at funding",
                   "OSFI B-20 Section 3 — Income Verification"),
    "control_5":  ("Employment Verification Timeliness",
                   "Employment must be verified within 10 days of closing",
                   "OSFI B-20 Section 3 — Documentation"),
    "control_6":  ("Fair Lending Review",
                   "A documented fair lending review must be completed before funding",
                   "OSFI B-20 Section 6 — Fair Lending"),
    "control_7":  ("AML Check",
                   "A completed Anti-Money Laundering check is required before funding",
                   "FINTRAC / PCMLTFA — AML compliance"),
    "control_8":  ("Closing Disclosure Timing",
                   "Closing disclosure must be sent at least 3 business days before closing",
                   "Federal mortgage disclosure rules"),
    "control_9":  ("Dual Underwriter Review",
                   "Loans above $750,000 must be reviewed and signed off by two underwriters",
                   "OSFI B-20 Section 5 — High-Value Loan Oversight"),
    "control_10": ("Exception Escalation",
                   "Any identified control breach must be escalated to compliance within 24 hours",
                   "OSFI B-20 Section 7 — Compliance Escalation"),
}


# -------------------------------------------------------
# Date helper
# -------------------------------------------------------

def _date_or_none(value):
    """
    Safely coerces a value to a Python date object.

    Handles three cases:
    - Python date object (psycopg2 direct DB reads) → return as-is
    - Integer (Debezium CDC events encode DATE as days since 1970-01-01) → convert
    - String (fallback) → parse with dateutil
    - None → return None
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, int):
        return date(1970, 1, 1) + timedelta(days=value)
    # Fallback: string parsing
    try:
        from dateutil.parser import parse
        return parse(str(value)).date()
    except Exception:
        return None


# -------------------------------------------------------
# Individual control functions
# Each takes the full loan dict and returns True (PASS) or False (FAIL)
# -------------------------------------------------------

def check_control_1(loan: dict) -> bool:
    """Control 1 — DTI Ratio: must be <= 44%."""
    dti = loan.get("borrower_dti")
    if dti is None:
        return False
    return float(dti) <= 44.0


def check_control_2(loan: dict) -> bool:
    """Control 2 — Credit Score: >= 620 for conventional loans. Insured loans auto-pass."""
    loan_type = (loan.get("loan_type") or "").lower().strip()
    score = loan.get("borrower_credit_score")
    if loan_type == "insured":
        return True  # Insured loans have separate underwriting standards
    if score is None:
        return False
    return int(score) >= 620


def check_control_3(loan: dict) -> bool:
    """Control 3 — Stress Test: must have been passed."""
    return loan.get("stress_test_passed") is True


def check_control_4(loan: dict) -> bool:
    """Control 4 — Income Verification: at least 2 independent source documents."""
    count = loan.get("income_docs_count")
    if count is None:
        return False
    return int(count) >= 2


def check_control_5(loan: dict) -> bool:
    """
    Control 5 — Employment Verification Timeliness:
    Employment must be verified within 10 days BEFORE closing.
    Verified after closing = fail. Verified more than 10 days before closing = fail.
    """
    closing = _date_or_none(loan.get("closing_date"))
    verified = _date_or_none(loan.get("employment_verified_date"))
    if closing is None or verified is None:
        return False
    delta = (closing - verified).days
    # delta must be between 0 and 10 (verified before or on closing day, within 10 days)
    return 0 <= delta <= 10


def check_control_6(loan: dict) -> bool:
    """Control 6 — Fair Lending Check: must be reviewed and documented."""
    return loan.get("fair_lending_reviewed") is True


def check_control_7(loan: dict) -> bool:
    """Control 7 — AML Check: must be completed before funding."""
    return loan.get("aml_completed") is True


def check_control_8(loan: dict) -> bool:
    """
    Control 8 — Closing Disclosure Timing:
    Disclosure must be sent at least 3 BUSINESS DAYS before closing.
    Business days = Mon-Fri (no holiday calendar for now).
    """
    closing = _date_or_none(loan.get("closing_date"))
    sent = _date_or_none(loan.get("disclosure_sent_date"))
    if closing is None or sent is None:
        return False
    if sent >= closing:
        return False  # Sent on or after closing day → immediate fail
    # Count business days from sent+1 up to and including closing
    delta_days = (closing - sent).days
    business_days = sum(
        1 for i in range(1, delta_days + 1)
        if (sent + timedelta(days=i)).weekday() < 5  # 0=Mon ... 4=Fri
    )
    return business_days >= 3


def check_control_9(loan: dict) -> bool:
    """
    Control 9 — Dual Underwriter for High Value Loans:
    Loans > $750,000 require at least 2 underwriters.
    Loans at or below $750,000 automatically pass.
    """
    amount = loan.get("loan_amount")
    if amount is None:
        return True  # No amount on record → skip high-value check
    if float(amount) > 750_000:
        count = loan.get("underwriter_count")
        if count is None:
            return False
        return int(count) >= 2
    return True  # Below threshold → auto-pass


def check_control_10(loan: dict) -> bool:
    """
    Control 10 — Exception Escalation:
    If a breach was identified (breach_identified_date is set), escalation must
    occur within 1 calendar day. If no breach was identified, auto-pass.

    Note: This control checks whether the institution responded to a KNOWN breach,
    not whether the other 9 controls have just now detected one.
    """
    breach_date = _date_or_none(loan.get("breach_identified_date"))
    if breach_date is None:
        return True  # No breach identified on record → auto-pass
    escalation_date = _date_or_none(loan.get("escalation_date"))
    if escalation_date is None:
        return False  # Breach identified but no escalation → fail
    return (escalation_date - breach_date).days <= 1


# -------------------------------------------------------
# Master control runner
# -------------------------------------------------------

CONTROL_FUNCTIONS = {
    "control_1":  check_control_1,
    "control_2":  check_control_2,
    "control_3":  check_control_3,
    "control_4":  check_control_4,
    "control_5":  check_control_5,
    "control_6":  check_control_6,
    "control_7":  check_control_7,
    "control_8":  check_control_8,
    "control_9":  check_control_9,
    "control_10": check_control_10,
}


def run_all_controls(loan: dict) -> dict:
    """
    Runs all 10 compliance controls against a loan record.

    Args:
        loan: dict with all fields from public.retail_lending

    Returns:
        {
            "control_results":  {"control_1": True, "control_2": False, ...},
            "compliance_score": 8,
            "overall_status":   "BREACH",
            "severity":         "High",
            "failed_controls":  ["control_2", "control_7"]
        }
    """
    results = {}
    for name, fn in CONTROL_FUNCTIONS.items():
        try:
            results[name] = bool(fn(loan))
        except Exception as e:
            # Any unexpected exception → conservative failure
            results[name] = False

    failed = [k for k, v in results.items() if not v]
    score = len(results) - len(failed)
    overall_status = "COMPLIANT" if not failed else "BREACH"

    # Overall severity = highest severity among all failed controls
    if not failed:
        severity = "Clean"
    else:
        highest_rank = max(SEVERITY_RANK[CONTROL_SEVERITY[c]] for c in failed)
        severity = next(k for k, v in SEVERITY_RANK.items() if v == highest_rank)

    return {
        "control_results":  results,
        "compliance_score": score,
        "overall_status":   overall_status,
        "severity":         severity,
        "failed_controls":  failed,
    }


# -------------------------------------------------------
# GPT-4o-mini breach explanation
# Called ONLY when there are failed controls
# -------------------------------------------------------

def generate_breach_explanation(loan: dict, failed_controls: list) -> str:
    """
    Calls GPT-4o-mini to generate plain-English breach explanations.
    One paragraph per failed control, with bolded headers.

    Args:
        loan:            full loan dict (for context)
        failed_controls: list of control keys that failed (e.g. ["control_2", "control_7"])

    Returns:
        Formatted string suitable for storing in compliance_results.breach_explanations.
        Returns empty string if failed_controls is empty.
    """
    if not failed_controls:
        return ""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build structured breach list for the prompt
    breach_lines = []
    for ctrl in failed_controls:
        name, requirement, regulation = CONTROL_DESCRIPTIONS[ctrl]
        breach_lines.append(
            f"- {ctrl.upper()} — {name}\n"
            f"  Requirement: {requirement}\n"
            f"  Regulatory basis: {regulation}"
        )

    # Loan context for personalised explanation
    amount = loan.get("loan_amount")
    amount_str = f"${float(amount):,.2f}" if amount is not None else "N/A"
    loan_context = (
        f"Borrower: {loan.get('borrower_name', 'Unknown')}\n"
        f"Loan Amount: {amount_str}\n"
        f"Loan Type: {loan.get('loan_type', 'Unknown')}\n"
        f"DTI Ratio: {loan.get('borrower_dti', 'N/A')}%\n"
        f"Credit Score: {loan.get('borrower_credit_score', 'N/A')}\n"
        f"Stress Test Passed: {loan.get('stress_test_passed', 'N/A')}\n"
        f"Income Documents: {loan.get('income_docs_count', 'N/A')}\n"
        f"AML Completed: {loan.get('aml_completed', 'N/A')}\n"
        f"Fair Lending Reviewed: {loan.get('fair_lending_reviewed', 'N/A')}"
    )

    prompt = f"""The following mortgage loan has failed {len(failed_controls)} compliance control(s):

{chr(10).join(breach_lines)}

Loan context:
{loan_context}

For each failed control above, write a concise professional paragraph that explains:
1. What specifically went wrong for this loan (reference the actual loan data where relevant)
2. What the regulatory requirement is and why it exists
3. The risk this breach poses to the institution and/or the borrower

Format: Use a bold header for each control (e.g. **CONTROL_1 — DTI Ratio Check**), followed by the explanation paragraph. Keep each paragraph to 3-5 sentences.
Do not add a preamble, conclusion, or disclaimers."""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior mortgage compliance officer at a federally regulated "
                    "Canadian financial institution. You write precise, professional breach "
                    "notifications for the risk management team. You are familiar with OSFI "
                    "Guideline B-20, FINTRAC requirements, and standard mortgage lending controls."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
