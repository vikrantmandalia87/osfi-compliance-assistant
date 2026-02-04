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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

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
    rate_terms = ["minimum qualifying rate", "mqr", "stress test rate", "qualifying rate", "interest rate", "rate"]
    if any(term in query_lower for term in rate_terms):
        matched_topics.extend(["minimum qualifying rate", "stress testing", "interest rate"])
        is_summary = True
    
    for key, instruction in ROLE_INTENT_MAP.items():
        if key in query_lower:
            llm_task = instruction
            is_summary = True
            break

    # ---------------------------
    # 3️⃣ Topic expansion (RAG help)
    # ---------------------------
    for intent, topics in INTENT_TOPIC_MAP.items():
        if intent in query_lower:
            matched_topics.extend(topics)

    # ---------------------------
    # 4️⃣ Guardrail: OSFI scope
    # ---------------------------
    if "osfi" in query_lower and "b-20" not in query_lower:
        return (
            "This assistant is limited to OSFI Guideline B-20 (Residential Mortgage Underwriting). "
            "Please ask a B-20–specific question.",
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
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": """You are an expert regulatory compliance assistant specialized exclusively in OSFI Guideline B-20 (Residential Mortgage Underwriting Practices and Procedures).

Your purpose is to help users understand, interpret, and apply OSFI B-20 requirements in a clear, practical, and regulator-safe manner.

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
UNDERSTANDING USER INTENT
━━━━━━━━━━━━━━━━━━━━━━
The user may ask:
- Broad questions (e.g., "guidelines for underwriters")
- Topic-specific questions (e.g., income verification, stress testing)
- Product-specific questions (e.g., default insured mortgages)
- Scenario-based questions (e.g., how a rule applies in practice)
- Requests for examples or explanations

You must infer the intent and respond appropriately while staying within OSFI B-20.

━━━━━━━━━━━━━━━━━━━━━━
MANDATORY RESPONSE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━
You MUST use the following structure in EVERY response:

OVERVIEW
- 2–4 sentences summarizing the relevant OSFI B-20 guidance
- Describe the regulatory objective and risk being addressed

KEY OSFI B-20 REQUIREMENTS
- Bullet points (minimum 4 bullets)
- Each bullet must clearly explain a requirement or expectation
- Each bullet must include an inline page reference

PRACTICAL APPLICATION
- Explain how lenders or underwriters apply these requirements in real workflows
- Describe what decisions, checks, or documentation are expected

EXAMPLES (WHEN APPLICABLE)
- Provide 1–3 concrete examples if the question asks for explanation or illustration
- Use simple numbers or scenarios ONLY if supported by the context
- If numerical examples are not provided in the context, explicitly say:
  "OSFI B-20 does not prescribe specific numerical examples in the retrieved sections."

LIMITATIONS / NOTES
- Clearly state any limitations in the retrieved guidance
- Identify where OSFI principles are high-level rather than prescriptive

━━━━━━━━━━━━━━━━━━━━━━
TONE AND STYLE
━━━━━━━━━━━━━━━━━━━━━━
- Professional and neutral (regulator-facing)
- Clear enough for business users and underwriters
- No legal advice language
- No speculation or opinion

━━━━━━━━━━━━━━━━━━━━━━
FAIL-SAFE BEHAVIOR
━━━━━━━━━━━━━━━━━━━━━━
If the context does not sufficiently answer the question:
- Say so explicitly
- Explain what aspect is not covered in the retrieved OSFI B-20 sections
- Do NOT attempt to fill gaps from general knowledge

Remember:
Your role is to EXPLAIN OSFI B-20, not to interpret beyond it."""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_output_tokens=2000,
            reasoning={"effort": "low"}
        )
        
        answer = response.output_text

    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
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


query = st.text_input("Enter your compliance question:", 
                      placeholder="e.g., What is the minimum qualifying rate?")

if query:
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

st.caption("Built for demonstration. Not official OSFI guidance.")