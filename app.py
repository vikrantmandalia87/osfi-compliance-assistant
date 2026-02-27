# app.py
# Tab 1: Loan Compliance Controls Dashboard
# Tab 2: OSFI RAG Assistant
# Tab 3: Loan Database MCP Query

# ============================================================
# Intent / Role Maps
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
import sys
import asyncio
import json
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime, date, timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Database (direct psycopg2 for compliance dashboard) ──
try:
    import db
    DB_AVAILABLE = True
    _DB_ERR_MSG = ""
except Exception as _db_err:
    DB_AVAILABLE = False
    _DB_ERR_MSG = str(_db_err)

# Search up the directory tree so the .env in the repo root is found
# regardless of which worktree the app is run from.
load_dotenv()                                               # current dir / worktree
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))  # repo root

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                     'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }

    .stApp { background: #F5F5F7; }

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
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0071E3 0%, #005BB5 55%, #003D80 100%);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 113, 227, 0.22);
    }
    .hero-banner h1 { color: #FFFFFF; font-size: 2rem; font-weight: 700;
                      letter-spacing: -0.4px; margin: 0 0 0.35rem 0; }
    .hero-banner p  { color: rgba(255,255,255,0.76); font-size: 1rem;
                      font-weight: 400; margin: 0; }
    .hero-icon { font-size: 2.4rem; margin-bottom: 0.5rem; display: block; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #FFFFFF; border-radius: 16px; padding: 1rem 1.2rem;
        border: 1px solid #E5E5EA; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important; font-weight: 600 !important;
        color: #8A8A8E !important; text-transform: uppercase; letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.9rem !important; font-weight: 700 !important; color: #1D1D1F !important;
    }

    .stDataFrame { border-radius: 14px; overflow: hidden; border: 1px solid #E5E5EA; }

    /* ── Answer card ── */
    .answer-card-header {
        background: #FFFFFF; border-radius: 18px; padding: 1.4rem 1.8rem 1rem 1.8rem;
        margin-bottom: 0; box-shadow: 0 2px 16px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05); border-left: 5px solid #0071E3;
    }

    /* ── Badges ── */
    .badge { display: inline-flex; align-items: center; gap: 5px;
             padding: 4px 13px; border-radius: 20px; font-size: 0.80rem; font-weight: 600; }
    .badge-high   { background: #D1F5E0; color: #1A7A3F; }
    .badge-medium { background: #FFF1D6; color: #A05C00; }
    .badge-low    { background: #FFE0DE; color: #A61C00; }

    /* ── Inputs ── */
    .stTextInput > div > div > input {
        border-radius: 12px !important; border: 1.5px solid #D2D2D7 !important;
        padding: 0.7rem 1rem !important; font-size: 0.96rem !important;
        background: #FFFFFF !important; color: #1D1D1F !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #0071E3 !important;
        box-shadow: 0 0 0 3px rgba(0,113,227,0.14) !important;
    }
    .stTextInput > label { font-weight: 600 !important; color: #1D1D1F !important;
                           font-size: 0.88rem !important; }

    /* ── Buttons ── */
    .stFormSubmitButton > button, .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0071E3, #005BB5) !important;
        color: #FFFFFF !important; border: none !important; border-radius: 12px !important;
        padding: 0.55rem 1.8rem !important; font-size: 0.93rem !important;
        font-weight: 600 !important; box-shadow: 0 4px 14px rgba(0,113,227,0.30) !important;
    }
    .stButton > button[kind="secondary"] {
        border-radius: 12px !important; border: 1.5px solid #D2D2D7 !important;
        color: #1D1D1F !important; background: #FFFFFF !important; font-weight: 500 !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #F5F5F7 !important; border-radius: 10px !important;
        font-weight: 600 !important; font-size: 0.87rem !important;
        color: #1D1D1F !important; border: 1px solid #E5E5EA !important;
    }
    .streamlit-expanderContent {
        background: #FAFAFA !important; border-radius: 0 0 10px 10px !important;
        border: 1px solid #E5E5EA !important; border-top: none !important;
        font-size: 0.87rem !important; color: #6E6E73 !important; line-height: 1.7 !important;
    }

    .stSelectbox > div > div {
        border-radius: 12px !important; border: 1.5px solid #D2D2D7 !important;
        background: #FFFFFF !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important; border-right: 1px solid #E5E5EA !important;
    }
    .sidebar-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: #8A8A8E; margin: 1.2rem 0 0.5rem 0;
    }
    .chip {
        display: inline-block; background: #EAF3FF; color: #0071E3;
        border-radius: 20px; padding: 4px 12px; font-size: 0.80rem;
        font-weight: 500; margin: 3px 2px; border: 1px solid #C5DFFF;
    }

    .query-meta {
        font-size: 0.77rem; color: #8A8A8E; margin-top: 0.8rem;
        padding-top: 0.7rem; border-top: 1px solid #E5E5EA;
    }

    .stAlert { border-radius: 12px !important; }

    .sources-header {
        font-size: 0.98rem; font-weight: 600; color: #1D1D1F; margin: 1.4rem 0 0.7rem 0;
    }

    hr.apple-hr { border: none; border-top: 1px solid #E5E5EA; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Date helper
# ============================================================
def _parse_dt(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


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
        loan_amount       = st.number_input("Loan Amount ($) *", min_value=10000.0,
                                             max_value=5000000.0, value=350000.0, step=5000.0)
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
        today                    = date.today()
        funding_date             = st.date_input("Funding Date *", value=today)
        closing_date             = st.date_input("Closing Date *", value=today)
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
            escalation_date = st.date_input("Escalation Date", value=today,
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
        <b>📊 Tab 1</b> — Compliance dashboard with 10 OSFI controls.<br>
        <b>🤖 Tab 2</b> — RAG assistant for OSFI B-20 guidance.<br>
        <b>🔍 Tab 3</b> — Natural-language loan database queries via MCP.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:1.4rem;">Quick RAG Questions</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="line-height:2.1">
        <span class="chip">Qualifying rate?</span>
        <span class="chip">Debt service ratios?</span>
        <span class="chip">Income verification?</span>
        <span class="chip">LTV requirements?</span>
        <span class="chip">Stress test rules?</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:1.4rem;">Example Loan Queries</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="line-height:2.1">
        <span class="chip">Loans > $300K</span>
        <span class="chip">Loans in Ontario</span>
        <span class="chip">Failed stress test</span>
        <span class="chip">Portfolio stats</span>
        <span class="chip">Find loan LN-10001</span>
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

    off_topic_terms = ["corporate governance", "capital adequacy", "liquidity",
                       "bcbs", "basel", "insurance companies act"]
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
                    "content": """You are an expert regulatory compliance assistant specialized in OSFI Guideline B-20.

                        ━━━━━━━━━━━━━━━━━━━━━━
                        CORE OPERATING RULES
                        ━━━━━━━━━━━━━━━━━━━━━━
                        1. Answer in clear, professional, plain English.
                        2. Produce a detailed, explanatory response — never references only.
                        3. Rely ONLY on the provided OSFI B-20 context.
                        4. Do NOT invent requirements, thresholds, or interpretations.
                        5. Cite OSFI B-20 page numbers inline, e.g. (OSFI B-20, Page 12).

                        ━━━━━━━━━━━━━━━━━━━━━━
                        MANDATORY RESPONSE STRUCTURE
                        ━━━━━━━━━━━━━━━━━━━━━━
                        OVERVIEW
                        - 2–4 sentences summarizing the relevant OSFI B-20 guidance

                        KEY OSFI B-20 REQUIREMENTS
                        - Bullet points (minimum 4) each with inline page reference

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
# MCP Client — Loan Database (Tab 3)
# ============================================================
MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_mcp_server.py")

_LOAN_DB_TRIGGERS = [
    "show me loans", "show loans", "find loans", "list loans",
    "which loans", "how many loans", "get loans", "fetch loans",
    "loans where", "loans with", "loans in", "loans that",
    "loans above", "loans below", "loans greater", "loans more than",
    "loans less than", "loan amount", "loan value",
    "loan portfolio", "portfolio stats", "portfolio statistics",
    "borrower named", "search borrower", "find borrower",
    "find loan ", "get loan ", "loan id", "fetch loan",
    "stress test passed", "stress test failed", "breach identified",
    "breach found", "all loans", "loans with breach",
    "from branch", "branch id", "loans from branch", "loans from br",
]


def is_loan_query(query: str) -> bool:
    q = query.lower()
    return any(trigger in q for trigger in _LOAN_DB_TRIGGERS)


def extract_loan_params(query: str) -> dict:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """You are a loan database query parameter extractor.
Given a natural-language question about loans, return a JSON object with:
  "tool"   : one of ["query_loans", "get_loan_by_id", "get_loan_stats", "search_borrower"]
  "params" : object with the relevant filter parameters

TOOL SELECTION RULES:
- Use "get_loan_by_id" when the user asks for a specific loan ID (e.g. "LN-10001", "loan LN-10005").
- Use "search_borrower" when the user asks for loans by a person's name.
- Use "get_loan_stats" ONLY when the user asks for overall portfolio-wide statistics with NO specific filter (e.g. "show me portfolio stats", "overall summary").
- Use "query_loans" for ALL other cases, including:
    * "how many loans from branch X"  → query_loans with branch_id
    * "loans in Ontario"              → query_loans with property_state
    * "loans above $500K"             → query_loans with min_amount
    * "failed stress test"            → query_loans with stress_test_passed=false
    * "all loans" / "list loans"      → query_loans with no filters

Available parameters for query_loans:
  min_amount        (float)  – minimum loan amount in dollars
  max_amount        (float)  – maximum loan amount in dollars
  loan_type         (string) – e.g. "Jumbo", "Conventional"
  property_state    (string) – province e.g. "Ontario", "Alberta"
  branch_id         (string) – branch ID e.g. "BR-001". If user says "branch 1" or "branch 001" use "BR-001".
  stress_test_passed (bool)  – true = passed, false = failed
  breach_identified  (bool)  – true = has breach, false = no breach
  min_credit_score  (int)
  max_credit_score  (int)
  limit             (int)    – max rows (default 50)

Available parameters for get_loan_by_id:
  loan_id (string) – e.g. "LN-10001"

Available parameters for get_loan_stats:
  (no parameters — portfolio-wide only)

Available parameters for search_borrower:
  name  (string) – borrower name or partial name
  limit (int)

Return ONLY valid JSON. Examples:
  {"tool": "query_loans", "params": {"min_amount": 300000}}
  {"tool": "query_loans", "params": {"branch_id": "BR-001"}}
  {"tool": "get_loan_by_id", "params": {"loan_id": "LN-10001"}}"""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


async def _call_mcp_tool_async(tool_name: str, arguments: dict):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result


def call_loan_tool(tool_name: str, arguments: dict):
    return asyncio.run(_call_mcp_tool_async(tool_name, arguments))


def render_loan_results(tool_name: str, raw_result) -> None:
    text = raw_result.content[0].text if raw_result.content else "{}"
    data = json.loads(text)

    if tool_name == "get_loan_stats":
        st.subheader("Loan Portfolio Statistics")
        s = data.get("summary", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Loans", int(s.get("total_loans", 0)))
        col2.metric("Total Portfolio", f"${float(s.get('total_portfolio_value', 0)):,.0f}")
        col3.metric("Avg Loan Amount", f"${float(s.get('avg_loan_amount', 0)):,.0f}")
        col4.metric("Breaches", int(s.get("breach_count", 0)))

        col5, col6, col7 = st.columns(3)
        col5.metric("Avg Credit Score", int(float(s.get("avg_credit_score", 0))))
        col6.metric("Avg DTI", f"{float(s.get('avg_dti', 0)):.1f}%")
        col7.metric("Stress Test Passed", int(s.get("stress_test_passed_count", 0)))

        if data.get("by_loan_type"):
            st.markdown("**By Loan Type**")
            st.dataframe(pd.DataFrame(data["by_loan_type"]), use_container_width=True, hide_index=True)
        if data.get("by_province"):
            st.markdown("**By Province**")
            st.dataframe(pd.DataFrame(data["by_province"]), use_container_width=True, hide_index=True)

    elif tool_name == "get_loan_by_id":
        if "error" in data:
            st.warning(data["error"])
        else:
            st.subheader(f"Loan Details — {data.get('loan_id')}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Loan Amount", f"${float(data.get('loan_amount', 0)):,.2f}")
            col2.metric("Property Value", f"${float(data.get('property_value', 0)):,.2f}")
            col3.metric("Credit Score", data.get("borrower_credit_score", "N/A"))
            st.write(f"**Borrower:** {data.get('borrower_name')}  |  "
                     f"**Province:** {data.get('property_state')}  |  "
                     f"**Stress Test:** {'✅ Passed' if data.get('stress_test_passed') else '❌ Failed'}  |  "
                     f"**Breach:** {'⚠️ Yes' if data.get('breach_identified') else 'None'}")
            with st.expander("Full loan record"):
                st.json(data)

    else:
        count = data.get("count", 0)
        loans = data.get("loans", [])
        message = data.get("message", "")

        if count == 0:
            st.info(message or "No loans found matching your criteria.")
            return

        st.subheader(f"Loan Query Results — {count} loan(s) found")
        df = pd.DataFrame(loans)

        for col in ["loan_amount", "property_value"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if x is not None else "")

        rename_map = {
            "loan_id": "Loan ID", "loan_type": "Type", "loan_amount": "Loan Amount",
            "borrower_name": "Borrower", "branch_id": "Branch",
            "borrower_credit_score": "Credit Score", "borrower_dti": "DTI%",
            "property_value": "Property Value", "property_state": "Province",
            "contract_rate": "Rate", "stress_test_passed": "Stress Test",
            "breach_identified": "Breach", "funding_date": "Funded",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Put most useful columns first
        preferred_order = ["Loan ID", "Branch", "Borrower", "Type", "Loan Amount",
                           "Province", "Credit Score", "DTI%", "Stress Test", "Breach", "Funded"]
        cols = [c for c in preferred_order if c in df.columns] + \
               [c for c in df.columns if c not in preferred_order]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)


def handle_loan_query(query: str) -> None:
    with st.spinner("Querying loan database via MCP..."):
        try:
            parsed = extract_loan_params(query)
            tool_name = parsed.get("tool", "query_loans")
            params = parsed.get("params", {})

            st.caption(f"MCP tool: `{tool_name}` | params: `{json.dumps(params)}`")

            result = call_loan_tool(tool_name, params)
            render_loan_results(tool_name, result)

        except Exception as e:
            st.error(f"Loan database query failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# Tab layout
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Compliance Controls Dashboard",
    "🤖 RAG Assistant",
    "🔍 Loan Query (MCP)",
])


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
            "Make sure PostgreSQL is running on localhost:5432."
        )
        st.stop()

    # ── Portfolio summary ──
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
            "Run `python run_seed_compliance.py` to process existing loans, "
            "or start the Kafka CDC pipeline."
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
        st.warning("No loans found in `public.retail_lending`.")
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
            "Run `python run_seed_compliance.py` to process seed loans."
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
# TAB 2 — OSFI RAG Assistant
# ============================================================
with tab2:
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-icon">🤖</span>
        <h1>Control Monitor — RAG Assistant</h1>
        <p>AI-powered regulatory guidance grounded in OSFI Guideline B-20</p>
    </div>
    """, unsafe_allow_html=True)

    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_answer' not in st.session_state:
        st.session_state.last_answer = ""
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = ""
    if 'last_sources' not in st.session_state:
        st.session_state.last_sources = []

    with st.form(key="query_form"):
        query = st.text_input(
            "💬  Ask a compliance question",
            placeholder="e.g., What is the minimum qualifying rate for uninsured mortgages?",
        )
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            submit_button = st.form_submit_button("Analyze →", type="primary")

    if submit_button and query:
        with st.spinner("Analyzing OSFI B-20 guidance…"):
            try:
                answer, sources, confidence = get_answer(query)
                st.session_state.last_query      = query
                st.session_state.last_answer     = answer
                st.session_state.last_confidence = confidence
                st.session_state.last_sources    = sources
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Run ingest.py first and check your API keys in .env")

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
            st.markdown('<div class="sources-header">📄 Supporting OSFI References</div>',
                        unsafe_allow_html=True)
            for i, match in enumerate(st.session_state.last_sources, 1):
                page  = match.metadata.get('page', 'N/A')
                score = round(match.score, 3)
                with st.expander(f"Source {i} — Page {page}  ·  relevance {score}"):
                    st.write(match.metadata["text"][:500] + "…")


# ============================================================
# TAB 3 — Loan Query via MCP
# ============================================================
with tab3:
    st.markdown("""
    <div class="hero-banner">
        <span class="hero-icon">🔍</span>
        <h1>Loan Database Query</h1>
        <p>Ask questions in plain English — the MCP server translates them into live database queries</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key="loan_query_form"):
        loan_query = st.text_input(
            "💬  Ask about loans",
            placeholder="e.g., Show me loans where loan amount is more than 300K",
        )
        col_btn3, _ = st.columns([1, 5])
        with col_btn3:
            loan_submit = st.form_submit_button("Query →", type="primary")

    if loan_submit and loan_query:
        handle_loan_query(loan_query)
