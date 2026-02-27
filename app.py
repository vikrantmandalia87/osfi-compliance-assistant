# app.py
# Control Monitor
# Tab 1: Loan Compliance Controls Dashboard
# Tab 2: OSFI RAG Assistant (original logic, unchanged)

# ============================================================
# Intent / Role Maps (unchanged from original)
# ============================================================
INTENT_TOPIC_MAP = {
    "underwriter": [
        "borrower assessment",
        "income verification",
        "debt serviceability",
        "credit risk",
        "underwriting standards"
    ],
    "default insured": [
        "mortgage insurance",
        "insured mortgages",
        "loan-to-value",
        "default insurance eligibility",
        "insurance underwriting"
    ],
    "minimum qualifying rate": [
        "minimum qualifying rate",
        "MQR",
        "stress test",
        "qualifying rate",
        "benchmark rate",
        "contract rate",
        "interest rate"
    ],
    "rate": [
        "interest rate",
        "qualifying rate",
        "stress testing"
    ]
}

ROLE_INTENT_MAP = {
    "underwriter": (
        "Provide a clear, structured explanation of OSFI Guideline B-20 "
        "requirements relevant to mortgage underwriters, including borrower "
        "assessment, income verification, stress testing, credit risk, and documentation."
    ),
    "under writers": (
        "Provide a clear, structured explanation of OSFI Guideline B-20 "
        "requirements relevant to mortgage underwriters, including borrower "
        "assessment, income verification, stress testing, credit risk, and documentation."
    )
}

# ============================================================
# Imports
# ============================================================
import os
import json
import time
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime, date, timedelta
import pandas as pd

# ── MCP Client (replaces direct db.py imports) ──────────────
# Streamlit now talks to PostgreSQL via MCP → FastAPI backend.
# Falls back gracefully with an error message if services are down.
try:
    from mcp_client import mcp as _mcp
    _mcp_ok, _mcp_err = _mcp.is_available()
    DB_AVAILABLE = _mcp_ok
    _DB_ERR_MSG  = _mcp_err

    # Expose the same interface app.py previously got from db.py
    class db:  # noqa: N801  (namespace shim — not a real class)
        get_next_loan_id              = staticmethod(_mcp.get_next_loan_id)
        insert_loan                   = staticmethod(_mcp.insert_loan)
        get_all_loans                 = staticmethod(_mcp.get_all_loans)
        get_loan_by_id                = staticmethod(_mcp.get_loan_by_id)
        search_loans                  = staticmethod(_mcp.search_loans)
        filter_loans                  = staticmethod(_mcp.filter_loans)
        get_all_loans_with_compliance = staticmethod(_mcp.get_all_loans_with_compliance)
        get_compliance_result_for_loan= staticmethod(_mcp.get_compliance_result_for_loan)
        get_portfolio_summary         = staticmethod(_mcp.get_portfolio_summary)

except Exception as _mcp_import_err:
    DB_AVAILABLE = False
    _DB_ERR_MSG  = (
        f"MCP client unavailable: {_mcp_import_err}\n\n"
        "Start the services:\n"
        "  1. uvicorn loan_service:app --port 8000 --reload\n"
        "  2. python mcp_server.py --http --port 3000"
    )

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Date/datetime helper
# JSON serialisation turns datetime objects into ISO strings.
# This coerces them back to datetime so .strftime() works.
# ─────────────────────────────────────────────────────────────
def _parse_dt(value):
    """Coerce an ISO-format string (or datetime) to a datetime object. Returns None if falsy."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Handles both "2026-02-25T14:30:00" and "2026-02-25T14:30:00.123456"
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Control Monitor",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# Apple-Inspired Global CSS Theme
# ============================================================
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                     'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    /* ── Page background ── */
    .stApp { background: #F5F5F7; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #FFFFFF;
        border-radius: 14px;
        padding: 6px;
        border: 1px solid #E5E5EA;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.92rem;
        color: #6E6E73;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #0071E3 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,113,227,0.30) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.4rem;
    }

    /* ── Hero / Page title banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0071E3 0%, #005BB5 55%, #003D80 100%);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 113, 227, 0.22);
    }
    .hero-banner h1 {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.4px;
        margin: 0 0 0.35rem 0;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.76);
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }
    .hero-icon { font-size: 2.4rem; margin-bottom: 0.5rem; display: block; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px solid #E5E5EA;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #8A8A8E !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #1D1D1F !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 14px; overflow: hidden; border: 1px solid #E5E5EA; }
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #E5E5EA; }
    div[data-testid="stTable"] { border-radius: 14px; overflow: hidden; border: 1px solid #E5E5EA; }

    /* ── Section cards ── */
    .section-card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* ── Answer card (RAG tab) ── */
    .answer-card-header {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.4rem 1.8rem 1rem 1.8rem;
        margin-bottom: 0;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
        border-left: 5px solid #0071E3;
    }

    /* ── Badge pills ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 13px;
        border-radius: 20px;
        font-size: 0.80rem;
        font-weight: 600;
    }
    .badge-high   { background: #D1F5E0; color: #1A7A3F; }
    .badge-medium { background: #FFF1D6; color: #A05C00; }
    .badge-low    { background: #FFE0DE; color: #A61C00; }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 1.5px solid #D2D2D7 !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.96rem !important;
        background: #FFFFFF !important;
        color: #1D1D1F !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0071E3 !important;
        box-shadow: 0 0 0 3px rgba(0,113,227,0.14) !important;
    }
    .stTextInput > label {
        font-weight: 600 !important;
        color: #1D1D1F !important;
        font-size: 0.88rem !important;
    }

    /* ── Buttons ── */
    .stFormSubmitButton > button,
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0071E3, #005BB5) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.55rem 1.8rem !important;
        font-size: 0.93rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(0,113,227,0.30) !important;
        transition: all 0.18s ease !important;
    }
    .stFormSubmitButton > button:hover,
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #005BB5, #003D80) !important;
        box-shadow: 0 6px 20px rgba(0,113,227,0.40) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        border-radius: 12px !important;
        border: 1.5px solid #D2D2D7 !important;
        color: #1D1D1F !important;
        background: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #F5F5F7 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.87rem !important;
        color: #1D1D1F !important;
        border: 1px solid #E5E5EA !important;
    }
    .streamlit-expanderContent {
        background: #FAFAFA !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid #E5E5EA !important;
        border-top: none !important;
        font-size: 0.87rem !important;
        color: #6E6E73 !important;
        line-height: 1.7 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid #D2D2D7 !important;
        background: #FFFFFF !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E5E5EA !important;
    }
    .sidebar-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8A8A8E;
        margin: 1.2rem 0 0.5rem 0;
    }
    .chip {
        display: inline-block;
        background: #EAF3FF;
        color: #0071E3;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.80rem;
        font-weight: 500;
        margin: 3px 2px;
        border: 1px solid #C5DFFF;
    }

    /* ── Query meta ── */
    .query-meta {
        font-size: 0.77rem;
        color: #8A8A8E;
        margin-top: 0.8rem;
        padding-top: 0.7rem;
        border-top: 1px solid #E5E5EA;
    }

    /* ── Alerts ── */
    .stAlert { border-radius: 12px !important; }

    /* ── Sources header ── */
    .sources-header {
        font-size: 0.98rem;
        font-weight: 600;
        color: #1D1D1F;
        margin: 1.4rem 0 0.7rem 0;
    }

    hr.apple-hr {
        border: none;
        border-top: 1px solid #E5E5EA;
        margin: 1.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Add New Loan Dialog
# ============================================================
@st.dialog("➕ Add New Loan", width="large")
def add_loan_dialog():
    st.markdown(
        "Fill in the loan details below. Once submitted, the **Kafka CDC pipeline** will "
        "automatically run all 10 compliance controls and save the result. "
        "Refresh the dashboard in ~10 seconds to see the result."
    )
    st.divider()

    if not DB_AVAILABLE:
        st.error(f"Database unavailable: {_DB_ERR_MSG}")
        return

    try:
        next_id = db.get_next_loan_id()
    except Exception as e:
        st.error(f"Could not generate loan ID: {e}")
        return

    st.info(f"**Auto-assigned Loan ID:** `{next_id}`")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Borrower Information")
        borrower_name     = st.text_input("Borrower Name *", placeholder="e.g. Jane Smith")
        branch_id         = st.text_input("Branch ID *", value="BR-001", placeholder="e.g. BR-001")
        loan_amount       = st.number_input("Loan Amount ($) *", min_value=10000.0, max_value=5000000.0,
                                             value=350000.0, step=5000.0)
        loan_type         = st.selectbox("Loan Type *", ["conventional", "insured"])
        borrower_dti      = st.number_input("Borrower DTI (%) *", min_value=0.0, max_value=100.0,
                                             value=38.0, step=0.5,
                                             help="Debt-to-Income ratio. OSFI control passes if ≤ 44%.")
        credit_score      = st.number_input("Credit Score *", min_value=300, max_value=900,
                                             value=720, step=1,
                                             help="Min 620 required for conventional loans (Control 2).")
        income_docs_count = st.number_input("Income Documents Count *", min_value=0, max_value=10,
                                             value=2, step=1,
                                             help="Must be ≥ 2 for Control 4 to pass.")
        underwriter_count = st.number_input("Underwriter Count *", min_value=1, max_value=5,
                                             value=1, step=1,
                                             help="Loans > $750K require ≥ 2 underwriters (Control 9).")

    with col_b:
        st.markdown("##### Dates & Compliance Flags")
        today             = date.today()
        funding_date      = st.date_input("Funding Date *", value=today)
        closing_date      = st.date_input("Closing Date *", value=today)
        employment_verified_date = st.date_input(
            "Employment Verified Date *", value=today - timedelta(days=5),
            help="Should be ≤ 10 days before closing date (Control 5)."
        )
        disclosure_sent_date = st.date_input(
            "Closing Disclosure Sent Date *", value=today - timedelta(days=4),
            help="Must be ≥ 3 business days before closing date (Control 8)."
        )

        st.markdown("##### Compliance Checkboxes")
        stress_test_passed    = st.checkbox("✅ Stress Test Passed (Control 3)", value=True)
        fair_lending_reviewed = st.checkbox("✅ Fair Lending Reviewed (Control 6)", value=True)
        aml_completed         = st.checkbox("✅ AML Check Completed (Control 7)", value=True)

        st.markdown("##### Breach Tracking (optional)")
        has_breach = st.checkbox("Breach Identified?", value=False)
        breach_identified_date = None
        escalation_date        = None
        if has_breach:
            breach_identified_date = st.date_input("Breach Identified Date",
                                                     value=today - timedelta(days=1))
            escalation_date = st.date_input("Escalation Date",
                                             value=today,
                                             help="Should be within 1 day of breach (Control 10).")

    st.divider()

    submitted = st.button("🚀 Submit Loan — Trigger CDC Pipeline", type="primary",
                           use_container_width=True)

    if submitted:
        if not borrower_name.strip():
            st.error("Borrower Name is required.")
            return

        loan_data = {
            "loan_id":                  next_id,
            "borrower_name":            borrower_name.strip(),
            "branch_id":                branch_id.strip(),
            "loan_amount":              float(loan_amount),
            "loan_type":                loan_type,
            "borrower_dti":             float(borrower_dti),
            "borrower_credit_score":    int(credit_score),
            "stress_test_passed":       stress_test_passed,
            "income_docs_count":        int(income_docs_count),
            "employment_verified_date": employment_verified_date,
            "closing_date":             closing_date,
            "fair_lending_reviewed":    fair_lending_reviewed,
            "aml_completed":            aml_completed,
            "disclosure_sent_date":     disclosure_sent_date,
            "underwriter_count":        int(underwriter_count),
            "breach_identified_date":   breach_identified_date,
            "escalation_date":          escalation_date,
            "funding_date":             funding_date,
        }

        try:
            inserted_id = db.insert_loan(loan_data)
            st.success(
                f"✅ **Loan `{inserted_id}` inserted successfully!**\n\n"
                f"The Kafka CDC pipeline will now automatically:\n"
                f"1. Detect this new row via Debezium\n"
                f"2. Run all 10 OSFI compliance controls\n"
                f"3. Generate an AI breach explanation (if any failures)\n"
                f"4. Save the result to the database\n\n"
                f"⏱ Refresh the dashboard in **~10 seconds** to see `{inserted_id}` appear."
            )
        except Exception as e:
            st.error(f"Failed to insert loan: {e}")


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size:2rem;">🏦</span><br>
        <span style="font-size:1.05rem; font-weight:700; color:#1D1D1F;">Control Monitor</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)

    if st.button("➕ Add New Loan", type="primary", use_container_width=True,
                  help="Open form to insert a new loan and trigger the CDC compliance pipeline"):
        add_loan_dialog()

    st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.84rem; color:#3A3A3C; line-height:1.9;">
        <b>📊 Tab 1</b> — Automated loan compliance dashboard with 10 OSFI controls.<br>
        <b>🤖 Tab 2</b> — RAG assistant for OSFI B-20 regulatory guidance.<br>
        <b>🧾 Tab 3</b> — Loan query via MCP (loan + compliance details).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:1.4rem;">Quick RAG Questions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="line-height:2.1">
        <span class="chip">Qualifying rate?</span>
        <span class="chip">Debt service ratios?</span>
        <span class="chip">Income verification?</span>
        <span class="chip">LTV requirements?</span>
        <span class="chip">Stress test rules?</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.76rem; color:#8A8A8E; line-height:1.7;">
        <b style="color:#1D1D1F;">Source:</b> OSFI Guideline B-20 (2017)<br>
        <b style="color:#1D1D1F;">Model:</b> GPT-4o-mini + Pinecone RAG<br>
        <b style="color:#1D1D1F;">Pipeline:</b> Debezium → Kafka → PostgreSQL
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RAG helpers (Tab 2)
# ============================================================
@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_answer(query):
    original_query = query
    query_lower = query.lower()
    is_summary = False
    matched_topics = []
    llm_task = query

    mortgage_topics = {
        "minimum qualifying rate": ["qualifying rate", "stress test", "buffer", "floor rate"],
        "debt service": ["gds", "tds", "gross debt service", "total debt service", "debt ratio"],
        "ltv": ["loan to value", "loan-to-value", "ltv ratio", "high ratio", "low ratio"],
        "income verification": ["income", "employment", "self-employed", "notice of assessment"],
        "mortgage insurance": ["cmhc", "mortgage default insurance", "insured mortgage", "uninsured"],
        "heloc": ["home equity line", "heloc", "revolving credit"],
        "non-conforming": ["non-conforming", "high risk", "low credit score"],
        "appraisal": ["property value", "appraisal", "valuation", "collateral"],
        "documentation": ["loan documentation", "documentation", "records"],
        "aml": ["anti-money laundering", "fintrac", "pcmltfa", "suspicious"]
    }

    for topic, keywords in mortgage_topics.items():
        if any(kw in query_lower for kw in keywords):
            matched_topics.extend([topic] + keywords[:2])
            is_summary = True

    for intent, topics in INTENT_TOPIC_MAP.items():
        if intent in query_lower:
            matched_topics.extend(topics)

    off_topic_terms = ["corporate governance", "capital adequacy", "liquidity", "bcbs", "basel", "insurance companies act"]
    if any(term in query_lower for term in off_topic_terms) and "mortgage" not in query_lower:
        return (
            "This assistant is specialized in OSFI Guideline B-20 (Residential Mortgage Underwriting). "
            "Please ask about mortgage underwriting, borrower assessment, or related topics.",
            [], "Low"
        )

    if any(p in query_lower for p in ["brief", "summary", "overview", "explain", "high level", "example"]):
        is_summary = True

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    embedding_query = "OSFI B-20 guidance on " + ", ".join(matched_topics) if matched_topics else query
    query_embedding = embeddings.embed_query(embedding_query)

    index = get_pinecone_index()
    top_k = 15 if is_summary else 5
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    if not results.matches:
        return "Not found in OSFI guidance.", [], "Low"

    matches = sorted(results.matches, key=lambda m: m.metadata.get("page", 0))
    context = "\n\n".join(
        f"[Page {m.metadata.get('page')}]\n{m.metadata['text'][:2000]}"
        for m in matches
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    Context (OSFI B-20):
    {context[:8000]}

    User question:
    {original_query}

    Task for you:
    {llm_task}

    IMPORTANT:
    - You must answer the user's question in plain English.
    - Do not return references only.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert regulatory compliance assistant specialized in OSFI Guideline B-20: Residential Mortgage Underwriting Practices and Procedures (2017).

                        This guideline covers:
                        - Residential Mortgage Underwriting Policy (RMUP)
                        - Borrower background, credit history, and income verification
                        - Debt service coverage (GDS/TDS ratios)
                        - Loan-to-Value (LTV) ratios and appraisals
                        - Mortgage insurance requirements
                        - Home Equity Lines of Credit (HELOCs)
                        - Non-conforming and high-ratio mortgages
                        - Stress testing and model validation
                        - Anti-money laundering compliance
                        - Supervisory expectations for FRFIs

                        You MUST cite specific page numbers from the guideline.

                        ━━━━━━━━━━━━━━━━━━━━━━
                        CORE OPERATING RULES
                        ━━━━━━━━━━━━━━━━━━━━━━
                        1. Answer in clear, professional, plain English.
                        2. Produce a detailed, explanatory response — never references only.
                        3. Rely ONLY on the provided OSFI B-20 context.
                        4. Do NOT invent requirements, thresholds, or interpretations.
                        5. If information is missing or unclear, state this explicitly.
                        6. Cite OSFI B-20 page numbers inline, e.g. (OSFI B-20, Page 12).

                        ━━━━━━━━━━━━━━━━━━━━━━
                        MANDATORY RESPONSE STRUCTURE
                        ━━━━━━━━━━━━━━━━━━━━━━
                        OVERVIEW
                        - 2–4 sentences summarizing the relevant OSFI B-20 guidance

                        KEY OSFI B-20 REQUIREMENTS
                        - Bullet points (minimum 4 bullets) each with inline page reference

                        PRACTICAL APPLICATION
                        - How lenders/underwriters apply these requirements

                        EXAMPLES (WHEN APPLICABLE)
                        - 1–3 concrete examples if question asks for explanation

                        LIMITATIONS / NOTES
                        - Any limitations in the retrieved guidance"""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        answer = response.choices[0].message.content
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        answer = ""

    if not answer or len(answer.strip()) < 120:
        answer = (
            "Based on OSFI Guideline B-20, mortgage underwriters are expected to "
            "apply prudent borrower assessment, verify income and employment, "
            "conduct stress testing, and ensure appropriate documentation and "
            "credit risk controls are in place. See referenced sections for details."
        )

    avg_score = sum(m.score for m in matches) / len(matches)
    confidence = "High" if avg_score > 0.3 else "Medium" if avg_score > 0.1 else "Low"
    return answer, matches, confidence


# ============================================================
# Tab layout
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Compliance Controls Dashboard", "🤖 RAG Assistant", "🧾 Loan Query via MCP"])


# ============================================================
# TAB 1 — Compliance Controls Dashboard
# ============================================================
with tab1:
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-icon">📊</span>
        <h1>Loan Compliance Controls Dashboard</h1>
        <p>Automated compliance monitoring across 10 OSFI controls — powered by Kafka CDC pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    if not DB_AVAILABLE:
        st.error(
            f"**Database connection unavailable:** {_DB_ERR_MSG}\n\n"
            "Make sure PostgreSQL is running and your `.env` file has:\n"
            "`PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`"
        )
        st.stop()

    # ── Portfolio summary metrics ──
    st.markdown("#### Portfolio Overview")
    try:
        summary = db.get_portfolio_summary()
        all_loans_with_compliance = db.get_all_loans_with_compliance()
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()

    if summary and summary.get("total_loans") and int(summary["total_loans"]) > 0:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Checked",  int(summary["total_loans"]))
        col2.metric("✅ Compliant",   int(summary["compliant_count"] or 0))
        col3.metric("❌ Breaches",    int(summary["breach_count"] or 0))
        col4.metric("🔴 High",        int(summary["high_severity"] or 0))
        col5.metric("🟡 Medium",      int(summary["medium_severity"] or 0))
        col6.metric("🔵 Low",         int(summary["low_severity"] or 0))
    else:
        st.info(
            "**No compliance results yet.**\n\n"
            "Next steps:\n"
            "1. Apply `init_db.sql` to create tables and seed loans\n"
            "2. Run `python run_seed_compliance.py` to process seed loans\n"
            "3. For live CDC: `docker compose up -d` → `python kafka_consumer.py`"
        )

    # ── All loans table ──
    if all_loans_with_compliance:
        st.markdown("#### All Loans — Compliance Summary")

        def _severity_badge(s):
            icons = {"High": "🔴", "Medium": "🟡", "Low": "🔵", "Clean": "🟢"}
            return f"{icons.get(s, '⚪')} {s}" if s else "⏳ Pending"

        def _status_badge(s):
            if s == "COMPLIANT": return "✅ COMPLIANT"
            elif s == "BREACH":  return "❌ BREACH"
            return "⏳ Pending"

        table_rows = []
        for row in all_loans_with_compliance:
            amount = row.get("loan_amount")
            table_rows.append({
                "Loan ID":  row.get("loan_id", "—"),
                "Borrower": row.get("borrower_name", "—"),
                "Branch":   row.get("branch_id", "—"),
                "Amount":   f"${float(amount):,.0f}" if amount else "—",
                "Type":     (row.get("loan_type") or "—").title(),
                "Score":    f"{row['compliance_score']}/10" if row.get("compliance_score") is not None else "—",
                "Status":   _status_badge(row.get("overall_status")),
                "Severity": _severity_badge(row.get("severity")),
            })

        df_all = pd.DataFrame(table_rows)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

    st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)

    # ── Per-loan detail ──
    st.markdown("#### Loan Detail — Control-by-Control Results")

    try:
        loans = db.get_all_loans()
    except Exception as e:
        st.error(f"Could not load loans: {e}")
        st.stop()

    if not loans:
        st.warning("No loans found in `public.retail_lending`. Run `init_db.sql` first.")
        st.stop()

    loan_options = {}
    for row in loans:
        label = (
            f"{row['loan_id']} — "
            f"{row['borrower_name']} — "
            f"${float(row['loan_amount']):,.0f}"
        )
        loan_options[label] = str(row["loan_id"])

    selected_label   = st.selectbox("Select a loan to inspect:", list(loan_options.keys()))
    selected_loan_id = loan_options[selected_label]

    try:
        result = db.get_compliance_result_for_loan(selected_loan_id)
    except Exception as e:
        st.error(f"Could not load compliance result: {e}")
        st.stop()

    if result is None:
        st.warning(
            "No compliance result found for this loan.\n\n"
            "Run `python run_seed_compliance.py` to process seed loans, "
            "or ensure `kafka_consumer.py` is running for live loans."
        )
    else:
        score    = int(result["compliance_score"])
        status   = result["overall_status"]
        severity = result["severity"]
        checked_at = _parse_dt(result.get("checked_at"))

        c1, c2, c3, c4 = st.columns(4)

        score_colour = "#34C759" if score >= 8 else "#FF9F0A" if score >= 5 else "#FF3B30"
        c1.markdown(
            f"**Compliance Score**\n\n"
            f"<span style='font-size:2.1em;font-weight:700;color:{score_colour}'>{score}/10</span>",
            unsafe_allow_html=True
        )

        status_colour = "#34C759" if status == "COMPLIANT" else "#FF3B30"
        c2.markdown(
            f"**Status**\n\n"
            f"<span style='font-size:1.5em;font-weight:700;color:{status_colour}'>"
            f"{'✅ ' if status == 'COMPLIANT' else '❌ '}{status}</span>",
            unsafe_allow_html=True
        )

        sev_colour = {
            "Clean": "#34C759", "Low": "#0071E3",
            "Medium": "#FF9F0A", "High": "#FF3B30"
        }.get(severity, "#8A8A8E")
        c3.markdown(
            f"**Severity**\n\n"
            f"<span style='font-size:1.5em;font-weight:700;color:{sev_colour}'>{severity}</span>",
            unsafe_allow_html=True
        )

        c4.markdown(
            f"**Last Checked**\n\n"
            f"{checked_at.strftime('%b %d, %Y %H:%M') if checked_at else 'N/A'}"
        )

        st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)

        CONTROL_LABELS = {
            "control_1":  "Control 1 — DTI Ratio (≤ 44%)",
            "control_2":  "Control 2 — Credit Score (≥ 620, conventional)",
            "control_3":  "Control 3 — Stress Test Passed",
            "control_4":  "Control 4 — Income Documents (≥ 2 sources)",
            "control_5":  "Control 5 — Employment Verified (≤ 10 days before closing)",
            "control_6":  "Control 6 — Fair Lending Review Documented",
            "control_7":  "Control 7 — AML Check Completed",
            "control_8":  "Control 8 — Closing Disclosure (≥ 3 business days before closing)",
            "control_9":  "Control 9 — Dual Underwriter (loans > $750K)",
            "control_10": "Control 10 — Exception Escalation (≤ 1 day)",
        }
        CONTROL_CATEGORY = {
            "control_1": "Credit Risk",   "control_2": "Credit Risk",
            "control_3": "Credit Risk",   "control_4": "Identity & Fraud",
            "control_5": "Identity & Fraud", "control_6": "Regulatory & Compliance",
            "control_7": "Regulatory & Compliance", "control_8": "Operational",
            "control_9": "Operational",   "control_10": "Monitoring",
        }
        CONTROL_SEVERITY_DISPLAY = {
            "control_1": "Medium", "control_2": "High",  "control_3": "High",
            "control_4": "Medium", "control_5": "Low",   "control_6": "Medium",
            "control_7": "High",   "control_8": "Low",   "control_9": "Low",
            "control_10": "Low",
        }

        ctrl_results = result["control_results"]
        if isinstance(ctrl_results, str):
            ctrl_results = json.loads(ctrl_results)

        rows = []
        for ctrl_key in sorted(ctrl_results.keys(), key=lambda k: int(k.split("_")[1])):
            passed = ctrl_results[ctrl_key]
            rows.append({
                "Category":           CONTROL_CATEGORY.get(ctrl_key, "—"),
                "Control":            CONTROL_LABELS.get(ctrl_key, ctrl_key),
                "Result":             "PASS" if passed else "FAIL",
                "Severity if Failed": CONTROL_SEVERITY_DISPLAY.get(ctrl_key, "—"),
            })

        df = pd.DataFrame(rows)

        def highlight_result(val):
            if val == "PASS": return "background-color:#d4edda;color:#155724;font-weight:bold"
            elif val == "FAIL": return "background-color:#f8d7da;color:#721c24;font-weight:bold"
            return ""

        styled = df.style.map(highlight_result, subset=["Result"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)
        breach_text = result.get("breach_explanations", "")
        if breach_text and breach_text.strip():
            st.subheader("🤖 AI Breach Analysis")
            st.caption("Generated by GPT-4o-mini based on detected control failures.")
            st.markdown(breach_text)
        else:
            st.success("🟢 No breaches detected. All 10 controls passed.")

    st.markdown('<hr class="apple-hr">', unsafe_allow_html=True)
    if st.button("🔄 Refresh Dashboard", type="secondary"):
        st.rerun()


# ============================================================
# TAB 2 — OSFI RAG Assistant (Apple UI + bug fix)
# ============================================================
with tab2:
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-icon">🤖</span>
        <h1>Control Monitor — RAG Assistant</h1>
        <p>AI-powered regulatory guidance grounded in OSFI Guideline B-20 — Residential Mortgage Underwriting</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ──
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = ""
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = ""
    if 'last_sources' not in st.session_state:
        st.session_state.last_sources = []

    # ── Query form ──
    with st.form(key="query_form"):
        query = st.text_input(
            "💬  Ask a compliance question",
            placeholder="e.g., What is the minimum qualifying rate for uninsured mortgages?",
        )
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            submit_button = st.form_submit_button("Analyze →", type="primary")

    # ── Process (only on submit) ──
    if submit_button and query:
        with st.spinner("Analyzing OSFI B-20 guidance…"):
            try:
                answer, sources, confidence = get_answer(query)
                # ✅ Store once in session state — fixes the double-display bug
                st.session_state.last_query      = query
                st.session_state.last_answer     = answer
                st.session_state.last_confidence = confidence
                st.session_state.last_sources    = sources
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Run ingest.py first and check your API keys in .env")

    # ── Display results ONCE from session state ──
    if st.session_state.last_answer:
        conf = st.session_state.last_confidence
        badge_class = "badge-high" if conf == "High" else "badge-medium" if conf == "Medium" else "badge-low"
        badge_icon  = "✅" if conf == "High" else "⚠️" if conf == "Medium" else "❌"

        st.markdown(f"""
        <div class="answer-card-header">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.05rem; font-weight:600; color:#1D1D1F;">Answer</span>
                <span class="badge {badge_class}">{badge_icon} {conf} Confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write(st.session_state.last_answer)

        query_id = hash(st.session_state.last_query + str(datetime.now())) % 10000
        st.markdown(
            f'<div class="query-meta">Query ID: #{query_id} &nbsp;·&nbsp; '
            f'{datetime.now().strftime("%b %d, %Y %H:%M")}</div>',
            unsafe_allow_html=True
        )

        if st.session_state.last_sources:
            st.markdown('<div class="sources-header">📄 Supporting OSFI References</div>', unsafe_allow_html=True)
            for i, match in enumerate(st.session_state.last_sources, 1):
                page  = match.metadata.get('page', 'N/A')
                score = round(match.score, 3)
                with st.expander(f"Source {i} — Page {page}  ·  relevance {score}"):
                    st.write(match.metadata["text"][:500] + "…")


# ============================================================
# TAB 3 — Loan Query via MCP
# ============================================================
def _format_money(value):
    if value is None:
        return "—"
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return str(value)


def _extract_loan_id(text):
    if not text:
        return None
    m = re.search(r"\bLN-\d{5}\b", str(text).upper())
    if m:
        return m.group(0)
    m2 = re.search(r"\b\d{5}\b", str(text))
    if m2:
        return f"LN-{m2.group(0)}"
    return None


def _extract_search_text(prompt):
    """
    Pulls out a likely search term from a natural-language query like:
      "give me loan where name = Sunita"
    Falls back to the original prompt.
    """
    if not prompt:
        return ""
    s = str(prompt).strip()

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", s)
    if quoted:
        return quoted[-1].strip()

    m = re.search(r"(?:name|borrower_name|borrower)\s*=\s*['\"]?([^'\"]+)['\"]?", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"\b\d+(?:\.\d+)?\b", s)
    if nums:
        return nums[-1].strip()

    br = re.search(r"\bBR[-\s]?\d+\b", s, flags=re.IGNORECASE)
    if br:
        return br.group(0).upper().replace(" ", "-")

    m2 = re.search(r"\bwhere\b\s+(.+)$", s, flags=re.IGNORECASE)
    if m2 and "=" in m2.group(1):
        rhs = m2.group(1).split("=", 1)[1].strip().strip("'\"")
        if rhs:
            return rhs
    return s


def _parse_structured_filters(prompt):
    """
    Parse simple structured filters from natural language.
    Supports patterns like:
      - "DTI > 50%"            → dti_gt=50
      - "loan amount >= 500000"→ loan_amount_gte=500000
      - "credit score < 620"   → credit_score_lt=620
      - "name contains Zili"   → borrower_name="Zili"
    Also returns an optional leftover free-text term as `q`.
    """
    if not prompt:
        return {}

    s = str(prompt)
    s_l = s.lower()
    filters = {}

    def _op_to_suffix(op):
        return {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "=": "gte"}.get(op)

    def _find_num(field_words, key_prefix):
        field_alt = "|".join(re.escape(w).replace("\\ ", "\\s+") for w in field_words)
        m = re.search(
            rf"(?:{field_alt})\s*(>=|<=|>|<|=)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%?",
            s_l,
        )
        if not m:
            return
        op, num = m.group(1), m.group(2).replace(",", "")
        suffix = _op_to_suffix(op)
        if not suffix:
            return
        try:
            val = float(num)
        except Exception:
            return
        filters[f"{key_prefix}_{suffix}"] = val

    _find_num(["dti", "borrower_dti"], "dti")
    _find_num(["loan amount", "loan_amount", "amount"], "loan_amount")
    _find_num(["credit score", "credit_score", "score"], "credit_score")

    # borrower name
    name_m = re.search(r"(?:name|borrower_name|borrower)\s*(?:contains|=)\s*['\"]?([a-z0-9][a-z0-9\s_-]+)['\"]?", s_l)
    if name_m:
        filters["borrower_name"] = name_m.group(1).strip()

    # leftover free-text (e.g., "Zili" in "DTI > 50% Zili")
    stop = {
        "show", "me", "records", "record", "where", "and", "or", "is", "in",
        "loan", "loans", "amount", "dti", "ratio", "credit", "score", "name",
        "borrower", "greater", "less", "than", "equal", "to",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", s)
    leftovers = [w for w in words if w.lower() not in stop]
    if leftovers and "borrower_name" not in filters:
        filters["q"] = " ".join(leftovers[-2:])  # keep it short

    return filters


def _loan_response_for_prompt(prompt):
    prompt_s = (prompt or "").strip()
    prompt_l = prompt_s.lower()

    structured = _parse_structured_filters(prompt_s)
    if structured and any(k in structured for k in ["dti_gt", "dti_gte", "dti_lt", "dti_lte", "loan_amount_gt", "loan_amount_gte", "loan_amount_lt", "loan_amount_lte", "credit_score_gt", "credit_score_gte", "credit_score_lt", "credit_score_lte", "borrower_name"]):
        if hasattr(db, "filter_loans"):
            matches = db.filter_loans(structured)
            return {"kind": "search", "q": prompt_s, "matches": matches or []}

    if any(k in prompt_l for k in ["help", "what can you do", "commands"]):
        return {
            "kind": "help",
            "text": (
                "Query loans via MCP.\n\n"
                "Examples:\n"
                "- `LN-10001` (or `10001`) — show the full loan + latest compliance\n"
                "- `name = Sunita` / `Sunita` — search across all loan columns\n"
                "- `loan amount = 375000` / `375000` — search by any value\n"
                "- `dti 38.5` / `credit score 720` / `BR-001`\n"
                "- `list loans`\n"
                "- `portfolio summary`\n"
            ),
        }

    if "portfolio" in prompt_l and "summary" in prompt_l:
        summary = db.get_portfolio_summary()
        if not summary:
            return {"kind": "text", "text": "No portfolio summary available yet (no compliance results)."}
        return {"kind": "portfolio_summary", "summary": summary}

    if "list" in prompt_l and "loan" in prompt_l:
        loans = db.get_all_loans() or []
        if not loans:
            return {"kind": "text", "text": "No loans found in `public.retail_lending`."}
        return {"kind": "list_loans", "loans": loans}

    loan_id = _extract_loan_id(prompt_s)
    if not loan_id:
        q = _extract_search_text(prompt_s)
        if not q:
            return {"kind": "text", "text": "Ask for a specific loan like `LN-10001` (or `10001`), or type `help`."}

        matches = db.search_loans(q, limit=25) if hasattr(db, "search_loans") else []
        if not matches:
            return {"kind": "search", "q": q, "matches": []}
        return {"kind": "search", "q": q, "matches": matches}

    loan = db.get_loan_by_id(loan_id)
    if not loan:
        return {"kind": "text", "text": f"Loan `{loan_id}` not found."}

    result = db.get_compliance_result_for_loan(loan_id)

    return {"kind": "loan_detail", "loan_id": loan_id, "loan": loan, "compliance": result}


with tab3:
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-icon">🧾</span>
        <h1>Loan Query via MCP</h1>
        <p>Query loan and compliance details through MCP → FastAPI → PostgreSQL</p>
    </div>
    """, unsafe_allow_html=True)

    if not DB_AVAILABLE:
        st.error(
            f"**MCP/backend unavailable:** {_DB_ERR_MSG}\n\n"
            "Start:\n"
            "1) `uvicorn loan_service:app --port 8000 --reload`\n"
            "2) `python mcp_server.py --http --port 3000`"
        )
        st.stop()

    st.caption("Examples: `LN-10001`, `10001`, `name = Sunita`, `375000`, `dti 38.5`, `list loans`, `portfolio summary`.")

    with st.form("loan_query_form", clear_on_submit=True):
        user_prompt = st.text_input("Query", placeholder="e.g., give me loan where name = Sunita")
        submit = st.form_submit_button("Search", type="primary")

    if submit and user_prompt:
        st.session_state.last_loan_query = user_prompt
        with st.spinner("Querying via MCP…"):
            try:
                st.session_state.last_loan_result = _loan_response_for_prompt(user_prompt)
            except Exception as e:
                st.session_state.last_loan_result = {"kind": "text", "text": f"Query failed: {e}"}

    last_query = st.session_state.get("last_loan_query")
    last_result = st.session_state.get("last_loan_result")

    if last_query:
        st.markdown(f"**Most recent query:** `{last_query}`")

    if not last_result:
        st.info("Enter a query above and click Search.")
        st.stop()

    kind = last_result.get("kind")

    if kind in ("text", "help"):
        st.markdown(last_result.get("text", ""))
        st.stop()

    if kind == "portfolio_summary":
        summary = last_result.get("summary") or {}
        df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in summary.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.stop()

    if kind == "list_loans":
        loans = last_result.get("loans") or []
        df = pd.DataFrame(loans)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.stop()

    if kind == "search":
        q = last_result.get("q", "")
        matches = last_result.get("matches") or []
        if not matches:
            st.warning(f"No matches found for `{q}`.")
            st.stop()

        st.markdown(f"**Matches for:** `{q}`  (showing {len(matches)})")
        df = pd.DataFrame(matches)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Show full records in a wide table (up to a reasonable limit).
        full_limit = 15
        ids = [m.get("loan_id") for m in matches if m.get("loan_id")][:full_limit]
        full_rows = []
        for loan_id_i in ids:
            try:
                full_rows.append(db.get_loan_by_id(loan_id_i))
            except Exception:
                pass
        if full_rows:
            st.markdown(f"**Full records (first {len(full_rows)}):**")
            st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)
        st.stop()

    if kind == "loan_detail":
        loan = last_result.get("loan") or {}
        compliance = last_result.get("compliance")

        st.markdown("#### Loan Record (All Columns)")
        kv = [{"Field": k, "Value": v} for k, v in loan.items()]
        st.dataframe(pd.DataFrame(kv), use_container_width=True, hide_index=True)

        st.markdown("#### Latest Compliance Result")
        if not compliance:
            st.info("No compliance result found yet for this loan.")
        else:
            comp_kv = [{"Field": k, "Value": v} for k, v in compliance.items()]
            st.dataframe(pd.DataFrame(comp_kv), use_container_width=True, hide_index=True)
            breach = (compliance.get("breach_explanations") or "").strip()
            if breach:
                st.subheader("🤖 AI Breach Analysis")
                st.markdown(breach)
        st.stop()

    st.markdown(str(last_result))
