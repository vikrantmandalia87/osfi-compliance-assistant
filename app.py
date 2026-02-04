# app.py
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

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="OSFI Compliance AI Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 OSFI Compliance AI Assistant")
st.subheader("AI-powered regulatory guidance for mortgage underwriting")

with st.sidebar:
    st.header("About")
    st.write("This assistant uses RAG to answer questions based on OSFI Guideline B-20.")
    st.header("Example Questions")
    st.code("What is the minimum qualifying rate?")
    st.code("What are the debt service ratio limits?")

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def get_answer(query):
    # ---------------------------
    # 1️⃣ ALWAYS initialize first
    # ---------------------------
    original_query = query
    query_lower = query.lower()

    is_summary = False
    matched_topics = []
    llm_task = query

    # ---------------------------
    # 2️⃣ Detect intent → task
    # ---------------------------
    # Detect rate-related queries explicitly
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

    # ---------------------------
    # 3️⃣ Topic expansion (RAG help)
    # ---------------------------
    for intent, topics in INTENT_TOPIC_MAP.items():
        if intent in query_lower:
            matched_topics.extend(topics)

    # ---------------------------
    # 4️⃣ Guardrail: OSFI scope
    # ---------------------------
    off_topic_terms = ["corporate governance", "capital adequacy", "liquidity", "bcbs", "basel", "insurance companies act"]
    if any(term in query_lower for term in off_topic_terms) and "mortgage" not in query_lower:
        return (
            "This assistant is specialized in OSFI Guideline B-20 (Residential Mortgage Underwriting). "
            "Please ask about mortgage underwriting, borrower assessment, or related topics.",
            [],
            "Low"
        )

    # ---------------------------
    # 5️⃣ Summary detection
    # ---------------------------
    if any(p in query_lower for p in ["brief", "summary", "overview", "explain", "high level", "example"]):
        is_summary = True

    # ---------------------------
    # 6️⃣ Embedding query
    # ---------------------------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    if matched_topics:
        embedding_query = "OSFI B-20 guidance on " + ", ".join(matched_topics)
    else:
        embedding_query = query

    query_embedding = embeddings.embed_query(embedding_query)

    # ---------------------------
    # 7️⃣ Pinecone retrieval
    # ---------------------------
    index = get_pinecone_index()
    top_k = 15 if is_summary else 5

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    if not results.matches:
        return "Not found in OSFI guidance.", [], "Low"

    matches = sorted(results.matches, key=lambda m: m.metadata.get("page", 0))

    context = "\n\n".join(
        f"[Page {m.metadata.get('page')}]\n{m.metadata['text'][:2000]}"
        for m in matches
    )

    # ---------------------------
    # 8️⃣ LLM call (answer synthesis)
    # ---------------------------
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
                        - Supervisory expectations for FRFIs (Federally Regulated Financial Institutions)

                        You MUST cite specific page numbers from the guideline.

                        ━━━━━━━━━━━━━━━━━━━━━━
                        CORE OPERATING RULES
                        ━━━━━━━━━━━━━━━━━━━━━━
                        1. You MUST answer in clear, professional, plain English.
                        2. You MUST produce a detailed, explanatory response — never references only.
                        3. You MUST rely ONLY on the provided OSFI B-20 context.
                        4. You MUST NOT invent requirements, thresholds, or interpretations.
                        5. If information is missing or unclear in the context, state this explicitly.
                        6. Cite OSFI B-20 page numbers inline, e.g. (OSFI B-20, Page 12).

                        ━━━━━━━━━━━━━━━━━━━━━━
                        MANDATORY RESPONSE STRUCTURE
                        ━━━━━━━━━━━━━━━━━━━━━━
                        You MUST use the following structure in EVERY response:

                        OVERVIEW
                        - 2–4 sentences summarizing the relevant OSFI B-20 guidance

                        KEY OSFI B-20 REQUIREMENTS
                        - Bullet points (minimum 4 bullets)
                        - Each bullet must include an inline page reference

                        PRACTICAL APPLICATION
                        - Explain how lenders or underwriters apply these requirements

                        EXAMPLES (WHEN APPLICABLE)
                        - Provide 1–3 concrete examples if the question asks for explanation

                        LIMITATIONS / NOTES
                        - Clearly state any limitations in the retrieved guidance

                        Remember:
                        Your role is to EXPLAIN OSFI B-20, not to interpret beyond it."""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
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

    # Fallback for short/empty answers
    
    if not answer or len(answer.strip()) < 120:
        answer = (
            "Based on OSFI Guideline B-20, mortgage underwriters are expected to "
            "apply prudent borrower assessment, verify income and employment, "
            "conduct stress testing, and ensure appropriate documentation and "
            "credit risk controls are in place. See referenced sections for details."
        )

    # ---------------------------
    # 9️⃣ Confidence scoring
    # ---------------------------
    avg_score = sum(m.score for m in matches) / len(matches)

    confidence = (
        "High" if avg_score > 0.3
        else "Medium" if avg_score > 0.1
        else "Low"
    )

    return answer, matches, confidence


# Initialize session state for query
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# Create form for query input
with st.form(key="query_form"):
    query = st.text_input("Enter your compliance question:", 
                          placeholder="e.g., What is the minimum qualifying rate?")
    submit_button = st.form_submit_button("Submit", type="primary")

# Process when form is submitted (Enter key or button click)
if submit_button and query:
    with st.spinner("Analyzing..."):
        try:
            answer, sources, confidence = get_answer(query)
            
            st.subheader("Answer")
            st.write(answer)
            
            color = "green" if confidence == "High" else "orange" if confidence == "Medium" else "red"
            st.markdown(f"<span style='color:{color};font-weight:bold'>Confidence: {confidence}</span>", 
                       unsafe_allow_html=True)
            
            if answer.strip():
                st.subheader("Supporting OSFI References")
                for i, match in enumerate(sources, 1):
                    with st.expander(f"Source {i} (Page {match.metadata.get('page', 'N/A')})"):
                        st.write(match.metadata["text"][:500] + "...")
            
            st.caption(f"Query ID: {hash(query + str(datetime.now())) % 10000}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Run ingest.py first and check API keys.")

# Display results if they exist
if 'last_answer' in st.session_state and st.session_state.last_answer:
    st.subheader("Answer")
    st.write(st.session_state.last_answer)
    
    color = "green" if st.session_state.last_confidence == "High" else "orange" if st.session_state.last_confidence == "Medium" else "red"
    st.markdown(f"<span style='color:{color};font-weight:bold'>Confidence: {st.session_state.last_confidence}</span>", 
               unsafe_allow_html=True)
    
    if st.session_state.last_answer.strip():
        st.subheader("Supporting OSFI References")
        for i, match in enumerate(st.session_state.last_sources, 1):
            with st.expander(f"Source {i} (Page {match.metadata.get('page', 'N/A')})"):
                st.write(match.metadata["text"][:500] + "...")
    
    st.caption(f"Query ID: {hash(st.session_state.last_query + str(datetime.now())) % 10000}")