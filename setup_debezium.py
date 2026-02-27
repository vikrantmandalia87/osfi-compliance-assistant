# setup_debezium.py
# One-time script to register the Debezium PostgreSQL connector.
#
# Run ONCE after `docker compose up -d` (wait ~30s for Kafka Connect to start):
#   python setup_debezium.py
#
# Debezium will:
#   1. Connect to PostgreSQL via logical replication (WAL)
#   2. Watch public.retail_lending for INSERT events
#   3. Publish CDC events to Kafka topic: osfi.public.retail_lending
#
# Prerequisites:
#   - docker compose up -d is running and healthy
#   - PostgreSQL wal_level = logical  (SHOW wal_level;)
#   - replication_user role exists with REPLICATION privilege
#     CREATE ROLE replication_user WITH REPLICATION LOGIN PASSWORD 'replication_pass';
#     GRANT SELECT ON public.retail_lending TO replication_user;

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

DEBEZIUM_URL = os.getenv("DEBEZIUM_CONNECTOR_URL", "http://localhost:8083/connectors")
CONNECTOR_NAME = "osfi-retail-lending-connector"

# -------------------------------------------------------
# Connector configuration
# -------------------------------------------------------
# NOTE: Debezium connects to PostgreSQL on the HOST machine.
# Inside Docker, use "host.docker.internal" (Docker Desktop on Mac/Windows)
# or the actual host IP on Linux with native Docker.
CONNECTOR_CONFIG = {
    "name": CONNECTOR_NAME,
    "config": {
        # Connector class
        "connector.class": "io.debezium.connector.postgresql.PostgresConnector",

        # PostgreSQL connection (Debezium uses host.docker.internal to reach host PG)
        "database.hostname": "host.docker.internal",
        "database.port": os.getenv("PG_PORT", "5432"),
        "database.user": os.getenv("PG_USER", "vikrantmandalia"),   # Main DB user (also has REPLICATION via role)
        "database.password": os.getenv("PG_PASSWORD", ""),
        "database.dbname": os.getenv("PG_DB", "postgres"),
        "database.server.name": "osfi",             # Used as Kafka topic prefix

        # Replication settings
        "plugin.name": "pgoutput",                  # Built into PostgreSQL 10+ (no extra install)
        "slot.name": "osfi_debezium_slot",
        "publication.name": "osfi_debezium_pub",

        # Only watch this one table
        "table.include.list": "public.retail_lending",

        # Kafka topic naming: <prefix>.<schema>.<table> = osfi.public.retail_lending
        "topic.prefix": "osfi",

        # Message format: raw JSON (no schema envelope) for simplicity
        "key.converter": "org.apache.kafka.connect.json.JsonConverter",
        "value.converter": "org.apache.kafka.connect.json.JsonConverter",
        "key.converter.schemas.enable": "false",
        "value.converter.schemas.enable": "true",   # Keep schema in value for payload parsing

        # Snapshot: take a snapshot of existing rows on first start, then stream changes
        "snapshot.mode": "initial",

        # Heartbeat to keep replication slot alive when traffic is low
        "heartbeat.interval.ms": "10000",
    },
}


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def wait_for_connect(max_wait_seconds: int = 60):
    """Polls Kafka Connect until it responds or times out."""
    import time
    print(f"Waiting for Kafka Connect at {DEBEZIUM_URL}...")
    for i in range(max_wait_seconds // 5):
        try:
            resp = requests.get(DEBEZIUM_URL.replace("/connectors", "/"), timeout=3)
            if resp.status_code == 200:
                info = resp.json()
                print(f"Kafka Connect is ready. Version: {info.get('version', 'unknown')}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"  Not ready yet... ({(i + 1) * 5}s elapsed)")
        time.sleep(5)
    return False


def connector_exists(name: str) -> bool:
    try:
        resp = requests.get(f"{DEBEZIUM_URL}/{name}", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def delete_connector(name: str):
    resp = requests.delete(f"{DEBEZIUM_URL}/{name}", timeout=5)
    resp.raise_for_status()
    print(f"Deleted existing connector '{name}'")


def register_connector():
    """Registers (or re-registers) the Debezium connector."""

    # Check Kafka Connect is up
    if not wait_for_connect():
        print("\nERROR: Kafka Connect did not become ready within 60 seconds.")
        print("Make sure Docker Compose is running: docker compose up -d")
        print("Then check logs: docker compose logs kafka-connect")
        sys.exit(1)

    # Handle existing connector
    if connector_exists(CONNECTOR_NAME):
        print(f"\nConnector '{CONNECTOR_NAME}' already exists.")
        choice = input("Re-register it? (Deletes the old connector first) [y/N]: ").strip().lower()
        if choice == "y":
            delete_connector(CONNECTOR_NAME)
        else:
            print("Skipping registration. Existing connector unchanged.")
            _print_status()
            return

    # Register
    print(f"\nRegistering connector '{CONNECTOR_NAME}'...")
    resp = requests.post(
        DEBEZIUM_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(CONNECTOR_CONFIG),
        timeout=15,
    )

    if resp.status_code in (200, 201):
        print("Connector registered successfully!")
        print(json.dumps(resp.json(), indent=2))
        _print_status()
    else:
        print(f"\nRegistration failed: HTTP {resp.status_code}")
        print(resp.text)
        print("\nCommon causes:")
        print("  - PostgreSQL wal_level is not 'logical' (requires PG restart)")
        print("  - replication_user does not exist or lacks privileges")
        print("  - host.docker.internal cannot reach PostgreSQL (Linux native Docker issue)")
        sys.exit(1)


def _print_status():
    """Prints the current connector status."""
    try:
        import time
        time.sleep(2)  # Give connector a moment to initialise
        resp = requests.get(
            f"{DEBEZIUM_URL}/{CONNECTOR_NAME}/status", timeout=5
        )
        if resp.status_code == 200:
            status = resp.json()
            connector_state = status.get("connector", {}).get("state", "UNKNOWN")
            print(f"\nConnector state: {connector_state}")
            for task in status.get("tasks", []):
                print(f"  Task {task['id']}: {task['state']}")
        print(f"\nKafka UI (browse topic): http://localhost:8080")
        print(f"Connect REST:           {DEBEZIUM_URL}/{CONNECTOR_NAME}/status")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        register_connector()
    except requests.exceptions.ConnectionError:
        print(
            f"\nERROR: Cannot reach Kafka Connect at {DEBEZIUM_URL}.\n"
            "Run 'docker compose up -d' first and wait ~30 seconds."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
