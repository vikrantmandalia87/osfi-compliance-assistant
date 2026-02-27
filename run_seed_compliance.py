# run_seed_compliance.py
# One-time script to run compliance checks on the 5 seed loans.
#
# The seed loans were inserted into public.retail_lending BEFORE Debezium was
# configured, so they will NOT trigger CDC events automatically. This script
# processes them manually by running the same compliance engine the Kafka
# consumer uses.
#
# Run ONCE after:
#   1. init_db.sql has been applied (seed loans exist in DB)
#   2. kafka_consumer.py is NOT running (avoid duplicate writes)
#   OR ensure idempotency: it skips loans that already have a compliance result.
#
# Usage:
#   python run_seed_compliance.py
#   python run_seed_compliance.py --force    # Re-process even if result exists

import sys
import logging

from db import get_all_loans, get_loan_by_id, get_compliance_result_for_loan, write_compliance_result
from compliance_engine import run_all_controls, generate_breach_explanation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FORCE = "--force" in sys.argv


def process_seed_loans():
    logger.info("=" * 60)
    logger.info("OSFI Compliance AI — Seed Loan Processor")
    logger.info("=" * 60)

    loans = get_all_loans()

    if not loans:
        logger.warning("No loans found in public.retail_lending. Run init_db.sql first.")
        sys.exit(1)

    logger.info(f"Found {len(loans)} loan(s) to process.")

    processed = 0
    skipped = 0
    errors = 0

    for loan_summary in loans:
        loan_id = str(loan_summary["loan_id"])
        borrower = loan_summary.get("borrower_name", "Unknown")

        # Check if already has a compliance result
        if not FORCE:
            existing = get_compliance_result_for_loan(loan_id)
            if existing:
                logger.info(
                    f"SKIP  {borrower} ({loan_id[:8]}...) "
                    f"— already has result: {existing['overall_status']} / {existing['severity']}"
                )
                skipped += 1
                continue

        # Fetch full loan record
        loan = get_loan_by_id(loan_id)
        if loan is None:
            logger.error(f"ERROR {borrower} ({loan_id[:8]}...) — could not fetch from DB")
            errors += 1
            continue

        try:
            # Run 10 controls
            result = run_all_controls(loan)

            logger.info(
                f"  {borrower} ({loan_id[:8]}...): "
                f"Score={result['compliance_score']}/10 | "
                f"Status={result['overall_status']} | "
                f"Severity={result['severity']}"
            )

            if result["failed_controls"]:
                logger.info(f"  Failed: {', '.join(result['failed_controls'])}")

            # Generate GPT explanation for breaches
            breach_explanation = ""
            if result["failed_controls"]:
                logger.info(f"  Generating AI breach explanation...")
                try:
                    breach_explanation = generate_breach_explanation(
                        loan, result["failed_controls"]
                    )
                except Exception as e:
                    logger.error(f"  GPT explanation failed: {e}")
                    breach_explanation = (
                        f"AI explanation unavailable. "
                        f"Failed controls: {', '.join(result['failed_controls'])}"
                    )

            # Write to DB
            write_compliance_result(
                loan_id=loan_id,
                control_results=result["control_results"],
                compliance_score=result["compliance_score"],
                overall_status=result["overall_status"],
                breach_explanations=breach_explanation,
                severity=result["severity"],
            )

            logger.info(f"  Saved to compliance_results.")
            processed += 1

        except Exception as e:
            logger.error(f"ERROR {borrower} ({loan_id[:8]}...): {e}", exc_info=True)
            errors += 1

    logger.info("=" * 60)
    logger.info(f"Done. Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
    logger.info("=" * 60)
    logger.info("You can now start Streamlit: streamlit run app.py")


if __name__ == "__main__":
    process_seed_loans()
