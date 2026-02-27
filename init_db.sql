-- ============================================================
-- OSFI Compliance AI — Database Initialisation Script
-- Run: psql -U vikrantmandalia -d osfi_compliance -f init_db.sql
--
-- NOTE: public.retail_lending already exists with 50 real loans.
-- This script ONLY creates the compliance_results table and
-- grants the replication user access.
--
-- PREREQUISITE (already done):
--   ALTER SYSTEM SET wal_level = logical;
--   Restart PostgreSQL for WAL level change to take effect.
--
--   CREATE ROLE replication_user WITH REPLICATION LOGIN PASSWORD 'replication_pass';
-- ============================================================

-- ============================================================
-- Table: Compliance results (written by kafka_consumer.py)
-- loan_id matches retail_lending.loan_id which is VARCHAR(20)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.compliance_results (
    result_id           VARCHAR(36)  PRIMARY KEY,       -- UUID as string
    loan_id             VARCHAR(20)  REFERENCES public.retail_lending(loan_id),
    checked_at          TIMESTAMP    DEFAULT NOW(),
    control_results     JSONB,                           -- {"control_1": true, ...}
    compliance_score    INTEGER,                         -- 0-10
    overall_status      VARCHAR(20),                     -- 'COMPLIANT' | 'BREACH'
    breach_explanations TEXT,                            -- GPT-4o-mini explanation
    severity            VARCHAR(10)                      -- 'High' | 'Medium' | 'Low' | 'Clean'
);

CREATE INDEX IF NOT EXISTS idx_compliance_results_loan_id
    ON public.compliance_results(loan_id);

CREATE INDEX IF NOT EXISTS idx_compliance_results_checked_at
    ON public.compliance_results(checked_at DESC);

-- Grant replication user SELECT access (needed by Debezium)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'replication_user') THEN
        EXECUTE 'GRANT SELECT ON public.retail_lending TO replication_user';
        EXECUTE 'GRANT SELECT ON public.compliance_results TO replication_user';
    END IF;
END$$;

-- Confirm
SELECT 'compliance_results table ready' AS status;
SELECT COUNT(*) AS existing_loans FROM public.retail_lending;
