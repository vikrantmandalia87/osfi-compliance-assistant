# kafka_consumer.py
# Kafka CDC consumer for OSFI Compliance AI
#
# Run as a standalone background process (separate terminal from Streamlit):
#   python kafka_consumer.py
#
# Flow:
#   Debezium CDC event (INSERT on public.retail_lending)
#   → Kafka topic osfi.public.retail_lending
#   → This consumer
#   → compliance_engine.run_all_controls()
#   → (if breaches) compliance_engine.generate_breach_explanation()
#   → db.write_compliance_result() → public.compliance_results
#
# Design decisions:
# - Re-fetches loan from DB after receiving CDC event to get proper Python types
#   (Debezium encodes DATE as epoch-day int, NUMERIC as string, etc.)
# - 0.5s retry handles rare race condition where CDC arrives before DB row visible
# - Errors on individual messages are caught and logged; consumer never crashes
# - Graceful shutdown via SIGINT/SIGTERM

import json
import logging
import os
import signal
import sys
import time

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from dotenv import load_dotenv

from compliance_engine import run_all_controls, generate_breach_explanation
from db import get_loan_by_id, write_compliance_result

load_dotenv()

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Config from .env
# -------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "osfi.public.retail_lending")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "compliance-consumer-group")

# -------------------------------------------------------
# Graceful shutdown
# -------------------------------------------------------
_running = True


def _handle_signal(sig, frame):
    global _running
    logger.info(f"Shutdown signal ({sig}) received. Stopping consumer...")
    _running = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# -------------------------------------------------------
# Debezium message parsing
# -------------------------------------------------------

def extract_loan_id_from_debezium(message_value: dict):
    """
    Debezium CDC JSON structure:
    {
        "schema": {...},
        "payload": {
            "op": "c",         # c=create, u=update, d=delete, r=read(snapshot)
            "before": null,
            "after": {
                "loan_id": "...",
                "borrower_name": "...",
                ...
            },
            "source": {...}
        }
    }

    We only process INSERT events (op="c").
    Returns the loan_id string from the "after" payload, or None.
    """
    try:
        payload = message_value.get("payload", {})
        op = payload.get("op")

        if op not in ("c", "r"):
            # "u" = update, "d" = delete — skip these
            logger.debug(f"Skipping non-insert CDC operation: op={op}")
            return None

        after = payload.get("after")
        if not after:
            logger.warning("CDC event has no 'after' payload — skipping")
            return None

        loan_id = after.get("loan_id")
        if not loan_id:
            logger.warning("CDC 'after' payload missing loan_id — skipping")
            return None

        return str(loan_id)

    except Exception as e:
        logger.error(f"Failed to parse Debezium message: {e}")
        return None


# -------------------------------------------------------
# Core processing pipeline
# -------------------------------------------------------

def process_loan(loan_id: str):
    """
    Full compliance processing pipeline for a single loan.

    1. Fetch loan from PostgreSQL (properly typed — avoids Debezium type issues)
    2. Run all 10 compliance controls
    3. If any controls fail → call GPT-4o-mini for breach explanation
    4. Write result to public.compliance_results
    """
    logger.info(f"Processing loan: {loan_id}")

    # Re-fetch from DB to get properly typed Python objects
    loan = get_loan_by_id(loan_id)

    if loan is None:
        # Race condition: CDC message arrived before DB row was visible
        # to a second connection. Wait and retry once.
        logger.warning(f"Loan {loan_id} not found in DB, retrying in 0.5s...")
        time.sleep(0.5)
        loan = get_loan_by_id(loan_id)

    if loan is None:
        logger.error(f"Loan {loan_id} still not found after retry — skipping")
        return

    # Run all 10 controls (deterministic, no LLM)
    result = run_all_controls(loan)

    logger.info(
        f"Loan {loan_id}: Score={result['compliance_score']}/10 | "
        f"Status={result['overall_status']} | Severity={result['severity']}"
    )

    if result["failed_controls"]:
        logger.info(
            f"Loan {loan_id}: Failed controls — {', '.join(result['failed_controls'])}"
        )

    # Generate GPT-4o-mini explanation only for breached loans
    breach_explanation = ""
    if result["failed_controls"]:
        logger.info(f"Loan {loan_id}: Generating AI breach explanation...")
        try:
            breach_explanation = generate_breach_explanation(
                loan, result["failed_controls"]
            )
            logger.info(f"Loan {loan_id}: Breach explanation generated successfully")
        except Exception as e:
            logger.error(f"Loan {loan_id}: GPT explanation failed — {e}")
            breach_explanation = (
                f"AI explanation unavailable (error: {e}). "
                f"Failed controls: {', '.join(result['failed_controls'])}"
            )

    # Persist result to PostgreSQL
    write_compliance_result(
        loan_id=str(loan_id),
        control_results=result["control_results"],
        compliance_score=result["compliance_score"],
        overall_status=result["overall_status"],
        breach_explanations=breach_explanation,
        severity=result["severity"],
    )

    logger.info(f"Loan {loan_id}: Compliance result saved to database.")


# -------------------------------------------------------
# Kafka consumer main loop
# -------------------------------------------------------

def run_consumer():
    """
    Main consumer loop. Blocks until SIGINT/SIGTERM received.
    """
    logger.info("=" * 60)
    logger.info("OSFI Compliance AI — Kafka Consumer Starting")
    logger.info(f"  Topic:             {KAFKA_TOPIC}")
    logger.info(f"  Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"  Consumer group:    {KAFKA_GROUP_ID}")
    logger.info("=" * 60)

    # Retry connecting to Kafka (Docker may not be fully ready)
    consumer = None
    for attempt in range(1, 6):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                group_id=KAFKA_GROUP_ID,
                auto_offset_reset="earliest",   # Re-process from start if group offset lost
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                max_partition_fetch_bytes=10 * 1024 * 1024,  # 10MB (Debezium JSON can be large)
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
            )
            logger.info("Connected to Kafka. Waiting for messages...")
            break
        except NoBrokersAvailable:
            logger.warning(
                f"Attempt {attempt}/5: Kafka not available at {KAFKA_BOOTSTRAP_SERVERS}. "
                f"Retrying in 5s..."
            )
            time.sleep(5)
    else:
        logger.error("Could not connect to Kafka after 5 attempts. Is Docker running?")
        sys.exit(1)

    try:
        while _running:
            # poll() with timeout lets us check the _running flag for graceful shutdown
            message_batch = consumer.poll(timeout_ms=1000)

            for topic_partition, messages in message_batch.items():
                for message in messages:
                    if not _running:
                        break
                    try:
                        loan_id = extract_loan_id_from_debezium(message.value)
                        if loan_id:
                            process_loan(loan_id)
                    except Exception as e:
                        logger.error(
                            f"Unhandled error on message offset={message.offset}: {e}",
                            exc_info=True,
                        )
                        # Continue — never crash the consumer on a single bad message
    finally:
        consumer.close()
        logger.info("Kafka consumer closed. Goodbye.")


if __name__ == "__main__":
    run_consumer()
