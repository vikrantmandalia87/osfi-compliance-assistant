-- ─────────────────────────────────────────────────────────
-- OSFI Compliance DB — Schema
-- Runs automatically on first `docker compose up`
-- ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.retail_lending (
    loan_id                  VARCHAR PRIMARY KEY,
    loan_type                VARCHAR,
    loan_amount              NUMERIC,
    funding_date             DATE,
    branch_id                VARCHAR,
    borrower_id              VARCHAR,
    borrower_name            VARCHAR,
    annual_income            NUMERIC,
    monthly_income           NUMERIC,
    monthly_debt             NUMERIC,
    borrower_dti             NUMERIC,
    borrower_credit_score    INTEGER,
    property_state           VARCHAR,
    property_value           NUMERIC,
    contract_rate            NUMERIC,
    stress_test_rate         NUMERIC,
    stress_test_passed       BOOLEAN,
    stress_test_date         DATE,
    income_doc_1             VARCHAR,
    income_doc_2             VARCHAR,
    income_docs_count        INTEGER,
    employment_verified_date DATE,
    closing_date             DATE,
    days_emp_to_closing      INTEGER,
    fair_lending_reviewed    BOOLEAN,
    fair_lending_review_date DATE,
    reviewer_id              VARCHAR,
    aml_completed            BOOLEAN,
    aml_completion_date      DATE,
    aml_officer_id           VARCHAR,
    disclosure_sent_date     DATE,
    days_before_closing      INTEGER,
    underwriter_1_id         VARCHAR,
    underwriter_2_id         VARCHAR,
    underwriter_count        INTEGER,
    breach_identified        BOOLEAN,
    breach_identified_date   DATE,
    escalation_date          DATE,
    hours_to_escalate        NUMERIC,
    escalation_documented    BOOLEAN,
    breach_list              TEXT
);

CREATE TABLE IF NOT EXISTS public.compliance_results (
    result_id           VARCHAR PRIMARY KEY,
    loan_id             VARCHAR REFERENCES public.retail_lending(loan_id),
    checked_at          TIMESTAMP,
    control_results     JSONB,
    compliance_score    INTEGER,
    overall_status      VARCHAR,
    breach_explanations TEXT,
    severity            VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_retail_loan_type   ON public.retail_lending(loan_type);
CREATE INDEX IF NOT EXISTS idx_retail_branch      ON public.retail_lending(branch_id);
CREATE INDEX IF NOT EXISTS idx_retail_state       ON public.retail_lending(property_state);
CREATE INDEX IF NOT EXISTS idx_compliance_loan_id ON public.compliance_results(loan_id);
