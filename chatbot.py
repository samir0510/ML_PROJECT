"""CropCare AI RAG chatbot engine grounded in Tomato & Potato Master Guide."""

import os
from typing import Optional, Dict, Any

MASTER_GUIDE_PDF = os.path.join(os.path.dirname(__file__), "tomato_potato_farmer_chatbot_master_guide.pdf")

_CACHED_MASTER_GUIDE = None
_CACHED_GEMINI_CLIENT = None


def get_gemini_api_key() -> str:
    """Retrieves Gemini API key from environment variable or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return ""


def load_master_guide_text() -> str:
    """Extract and cache the full Master Guide text from the PDF."""
    global _CACHED_MASTER_GUIDE
    if _CACHED_MASTER_GUIDE is not None:
        return _CACHED_MASTER_GUIDE

    if not os.path.exists(MASTER_GUIDE_PDF):
        return ""
    try:
        import pdfplumber
        with pdfplumber.open(MASTER_GUIDE_PDF) as pdf:
            pages = []
            for idx, p in enumerate(pdf.pages):
                txt = p.extract_text() or ""
                pages.append(f"--- Section / Page {idx+1} ---\n{txt}")
            _CACHED_MASTER_GUIDE = "\n\n".join(pages)
            return _CACHED_MASTER_GUIDE
    except Exception as e:
        print(f"Error reading Master Guide PDF: {e}")
        return ""


def get_gemini_client():
    """Initialize and cache the Google GenAI client."""
    global _CACHED_GEMINI_CLIENT
    if _CACHED_GEMINI_CLIENT is not None:
        return _CACHED_GEMINI_CLIENT
    try:
        from google import genai
        api_key = get_gemini_api_key()
        if not api_key:
            return None
        _CACHED_GEMINI_CLIENT = genai.Client(api_key=api_key)
        return _CACHED_GEMINI_CLIENT
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

FARMER_KNOWLEDGE_BASE = {
    "Late Blight Urgency": """**🚨 Urgent Action for Late Blight (*Phytophthora infestans*):**
• Destroy infected vines immediately; strictly do NOT compost them.
• Apply protective fungicide (Chlorothalonil/Mancozeb) before rain or systemic (Metalaxyl+Mancozeb).
• Stop overhead irrigation and warn neighboring farms.""",

    "Early Blight Target Rings": """**🎯 Early Blight Management (*Alternaria solani*):**
• Target rings start on oldest lower leaves under warm, humid conditions (24–29°C).
• Prune lower foliage up to 30 cm and mulch to prevent soil-splash.
• Apply Copper octanoate or Azoxystrobin/Difenoconazole at first sign.""",

    "Mosaic Virus Spread": """**🦠 Tomato Mosaic Virus (ToMV) Control:**
• Highly contagious mechanical spread via tools, hands, and seeds. No chemical cure exists.
• Rogue out and bag infected plants immediately.
• Disinfect tools with 1:4 bleach or skim milk; wash hands with soap."""
}

def get_farmer_advisory(user_prompt: str, current_diagnosis: Optional[Dict[str, Any]] = None) -> str:
    """Generates RAG advisory grounded in Master Guide PDF and active diagnosis."""
    client = get_gemini_client()
    master_guide = load_master_guide_text()

    if client and master_guide:
        diag_context = ""
        if current_diagnosis and current_diagnosis.get("class_name"):
            diag_context = f"""
[CURRENT LEAF DIAGNOSIS IN SESSION]
- Crop: {current_diagnosis.get('crop')}
- Disease / Status: {current_diagnosis.get('disease_name')}
- Scientific Name: {current_diagnosis.get('scientific_name')}
- Model Confidence: {current_diagnosis.get('confidence')}%
- Severity: {current_diagnosis.get('severity')}
"""

        rag_prompt = f"""You are CropCare AI Advisor for tomato and potato farmers.
Answer grounded strictly in this Master Guide:
{master_guide}

{diag_context}
Question: {user_prompt}
Provide practical, easy-to-understand advice prioritizing field sanitation, weather warnings, and registered label sprays."""
        models_to_try = ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-3.8-flash"]
        for m_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=rag_prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as err:
                print(f"[Gemini RAG] Attempt with {m_name} failed: {err}")
                continue

    # Fallback to local grounded knowledge base if offline / rate limited
    lower = user_prompt.lower()
    if "late blight" in lower:
        return FARMER_KNOWLEDGE_BASE["Late Blight Urgency"]
    elif "early blight" in lower:
        return FARMER_KNOWLEDGE_BASE["Early Blight Target Rings"]
    elif "mosaic" in lower or "virus" in lower:
        return FARMER_KNOWLEDGE_BASE["Mosaic Virus Spread"]
    elif "healthy" in lower:
        return "🌱 **Healthy Crop Management:** Keep regular monitoring, ensure balanced drip irrigation, avoid overhead wetting, and rotate solanaceous crops (tomatoes, potatoes) annually to prevent soil-borne pathogen buildup."
    else:
        return f"🌾 **CropCare Advisor (Offline Mode):** Thank you for your question: *'{user_prompt}'*.\n\n" \
               "For tomato and potato crops, always ensure:\n" \
               "1. **Inspection:** Look for discoloration, necrotic spots, or white spores under leaf surfaces.\n" \
               "2. **Watering:** Apply water at soil level via drip irrigation; avoid wet leaves overnight.\n" \
               "3. **Air Circulation:** Stake plants and prune lower suckers to maintain airflow.\n" \
               "4. **Diagnosis:** Upload a close-up photo of the affected leaf on the left for instant automated disease identification."

def generate_report_content_with_chatbot(diag: dict) -> str:
    """Generates dynamic agronomic report grounded in Master Guide PDF."""
    client = get_gemini_client()
    master_guide = load_master_guide_text()

    crop = diag.get("crop", "Solanaceae")
    disease = diag.get("disease_name", "Unknown")
    scientific = diag.get("scientific_name", "")
    confidence = diag.get("confidence", 0.0)
    severity = diag.get("severity", "Unknown")

    prompt = f"""You are CropCare AI Advisor. Generate a diagnostic report grounded strictly in the Master Guide:
{master_guide}

Diagnosis: {crop} — {disease} ({scientific}) | Confidence: {confidence}% | Severity: {severity}

Format using clean bullet points under these sections:
### Executive Pathology & Visual Symptoms
### Weather & Microclimate Vulnerability
### Master Guide Follow-Up Scouting Schedule
### Field Sanitation & Biosecurity Protocol
### Irrigation & Canopy Moisture Management
### Chemical & Fungicide Interventions
### Long-Term Management & Resistant Varieties
### Farmer Safety, PPE & Local KVK Escalation"""

    models_to_try = ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-3.8-flash"]
    for m in models_to_try:
        try:
            if client:
                res = client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    return res.text.strip()
        except Exception as e:
            print(f"[Dynamic RAG Report] {m} failed: {e}")
            continue

    # Fallback template if offline or rate-limited
    return f"""### Executive Pathology & Visual Symptoms
Inspected leaf indicates **{disease}** (*{scientific}*) with **{confidence}%** confidence ({severity} severity).

### Weather & Microclimate Vulnerability
Monitor humidity and leaf wetness. Keep foliage dry to limit spore germination.

### Master Guide Follow-Up Scouting Schedule
Inspect crop every 5–7 days, with an extra photo check 2–3 days after rain or cloudy spells.

### Field Sanitation & Biosecurity Protocol
Remove and safely dispose of infected foliage off-site. Disinfect pruning tools between plants.

### Chemical & Fungicide Interventions
Consult local registered fungicides and alternate FRAC groups. Note: Viral diseases have no cure spray.

### Safety, PPE & Local KVK Escalation
Wear proper PPE, observe label pre-harvest intervals (PHI), and consult your local KVK extension officer."""


def init_chatbot_state():
    """Initializes chatbot message history in Streamlit session state."""
    import streamlit as st
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am CropCare AI. Upload a crop photo on the left for instant leaf and disease detection, or ask me any farming question directly!"
            }
        ]


def render_chatbot_ui(current_diagnosis: Optional[Dict[str, Any]] = None):
    """Renders interactive Chatbot UI in Streamlit."""
    import streamlit as st

    init_chatbot_state()

    col_chat_title, col_chat_clear = st.columns([3, 1])
    with col_chat_title:
        st.markdown('<div class="section-title">💬 Step 3: Farmer Advisory & Chatbot</div>', unsafe_allow_html=True)
    with col_chat_clear:
        if st.button("🧹 Clear Chat", key="btn_clear_chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am CropCare AI. Upload a crop photo on the left for instant leaf and disease detection, or ask me any farming question directly!"
                }
            ]
            st.rerun()

    # Chat Messages Scroll Area
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Quick Questions Section
    st.markdown('<div class="quick-title">Quick Questions:</div>', unsafe_allow_html=True)
    qq_col1, qq_col2, qq_col3 = st.columns(3)
    
    prompt_to_send = None
    with qq_col1:
        if st.button("🥔 Late Blight Urgency", use_container_width=True):
            prompt_to_send = "What is the urgency and immediate treatment for Late Blight?"
    with qq_col2:
        if st.button("🍅 Early Blight Target Rings", use_container_width=True):
            prompt_to_send = "How do I recognize and manage Early Blight target rings?"
    with qq_col3:
        if st.button("🦠 Mosaic Virus Spread", use_container_width=True):
            prompt_to_send = "How does Mosaic Virus spread and how can I protect nearby crops?"

    # Chat Input Box
    user_query = st.chat_input("Type your question here (e.g., 'What spray should I use?' or 'How do I protect nearby plants?')...")
    if user_query:
        prompt_to_send = user_query

    # Handle Message Send
    if prompt_to_send:
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.spinner("Consulting Farmer Master Guide..."):
            advisory_answer = get_farmer_advisory(prompt_to_send, current_diagnosis=current_diagnosis)
        st.session_state.messages.append({"role": "assistant", "content": advisory_answer})
        st.rerun()

