"""
regenerate_breach_explanation.py

Regenerates and persists a fresh AI breach explanation for a specific loan.

Why this exists:
- If the OpenAI client fails (dependency mismatch, missing key, etc.), the pipeline
  stores a fallback string like "AI explanation unavailable ...".
- After fixing the environment, you can re-run generation for a given loan_id and
  store a new compliance_results row so the dashboard shows the latest explanation.

Usage:
  python regenerate_breach_explanation.py LN-10001
"""

import sys

from db import get_loan_by_id, write_compliance_result
from compliance_engine import run_all_controls, generate_breach_explanation


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python regenerate_breach_explanation.py LN-XXXXX")
        return 2

    loan_id = argv[1].strip()
    loan = get_loan_by_id(loan_id)
    if not loan:
        print(f"Loan not found: {loan_id}")
        return 1

    result = run_all_controls(loan)
    failed = result.get("failed_controls") or []
    if not failed:
        print(f"No failed controls for {loan_id}; nothing to regenerate.")
        return 0

    text = generate_breach_explanation(loan, failed)

    write_compliance_result(
        loan_id=loan_id,
        control_results=result["control_results"],
        compliance_score=result["compliance_score"],
        overall_status=result["overall_status"],
        breach_explanations=text,
        severity=result["severity"],
    )

    print(f"Regenerated breach explanation for {loan_id} (failed: {', '.join(failed)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

