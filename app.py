# ============================================================
# CROPCARE AI — LIGHT COLOR THEME DASHBOARD
# ============================================================
# Features:
#   1. Leaf Verification (Model/Leaf_Detector_v3.keras)
#      - Confirms if uploaded image is a valid plant leaf.
#      - If not: prompts user "Please capture or upload a leaf image".
#   2. Disease Identification (Model/Tomato_Potato_Disease_Model.keras)
#      - Classifies among 7 classes for Tomato & Potato.
#   3. RAG Farmer Chatbot (Gemini 3.5 Flash + FAISS Vector Store)
#      - Answers farmer questions with master guide PDF page references.
# ============================================================

import os
import sys
from PIL import Image

# Ensure safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Import our modular predictor, chatbot, and PDF report generator
from predictor import CropVisionPipeline
from chatbot.step3_chatbot import FarmerChatbot
from chatbot.step2_vector_store import is_vector_store_ready
from report_generator import generate_disease_advisory_pdf

# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="CropCare AI — Smart Leaf Diagnostic & Advisor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom Light Theme Styling ──────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit Sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Global Light Theme Colors */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8FAF8 !important;
        color: #1F2937 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* General Headings & Labels */
    h1, h2, h3, h4, h5, h6, label, .stText {
        color: #1F2937 !important;
    }
    
    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F8F3 100%);
        border: 1px solid #D1E7D7;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.05);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B5E20 !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #4B5563 !important;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    
    /* Model Status Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-green {
        background-color: #E8F5E9 !important;
        color: #2E7D32 !important;
        border: 1px solid #C8E6C9 !important;
    }
    .badge-blue {
        background-color: #E3F2FD !important;
        color: #1565C0 !important;
        border: 1px solid #BBDEFB !important;
    }
    .badge-purple {
        background-color: #F3E5F5 !important;
        color: #7B1FA2 !important;
        border: 1px solid #E1BEE7 !important;
    }

    /* Content Cards */
    .clean-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .clean-card * {
        color: #1F2937 !important;
    }
    
    /* Alert Boxes */
    .alert-warning-card {
        background-color: #FFFBEB !important;
        border-left: 5px solid #F59E0B !important;
        border: 1px solid #FDE68A !important;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 16px 0;
    }
    .alert-success-card {
        background-color: #F0FDF4 !important;
        border-left: 5px solid #22C55E !important;
        border: 1px solid #BBF7D0 !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    
    /* Result Highlights */
    .diagnosis-result-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827 !important;
        margin-bottom: 4px;
    }
    .severity-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin-bottom: 12px;
    }

    /* ============================================================
       SLEEK BLACK / DARK COMPONENTS WITH CRISP WHITE TEXT
       ============================================================ */

    /* ── 1. CHAT MESSAGES (Sleek Dark Card + Crisp White Text) ── */
    [data-testid="stChatMessage"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background-color: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] *,
    [data-testid="stChatMessage"] [data-testid="stExpander"] p,
    [data-testid="stChatMessage"] [data-testid="stExpander"] span,
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
    }
    [data-testid="stChatMessage"] code {
        background-color: #1F2937 !important;
        color: #86EFAC !important;
        -webkit-text-fill-color: #86EFAC !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* ── 2. BUTTONS (Sleek Dark Background + Crisp White Text) ── */
    .stButton > button {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button,
    .stButton > button *,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div,
    .stButton > button [data-testid="stMarkdownContainer"],
    .stButton > button [data-testid="stMarkdownContainer"] p,
    .stButton > button [data-testid="stMarkdownContainer"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #1F2937 !important;
        border-color: #4ADE80 !important;
    }
    .stButton > button:hover,
    .stButton > button:hover *,
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div,
    .stButton > button:hover [data-testid="stMarkdownContainer"] p {
        color: #4ADE80 !important;
        -webkit-text-fill-color: #4ADE80 !important;
    }
    .stButton > button:focus,
    .stButton > button:active {
        border-color: #4ADE80 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.3) !important;
    }

    /* ── 3. FILE UPLOADER (Sleek Dark Background + Crisp White Text) ── */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #111827 !important;
        border: 2px dashed #374151 !important;
        border-radius: 12px !important;
        padding: 18px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"],
    [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: #9CA3AF !important;
        -webkit-text-fill-color: #9CA3AF !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #1F2937 !important;
        border: 1px solid #4B5563 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stFileUploaderDropzone"] button div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #2D3748 !important;
        border-color: #4ADE80 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover * {
        color: #4ADE80 !important;
        -webkit-text-fill-color: #4ADE80 !important;
    }

    /* ── 4. CAMERA INPUT (Sleek Dark Background + Crisp White Text) ── */
    [data-testid="stCameraInput"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    [data-testid="stCameraInput"],
    [data-testid="stCameraInput"] *,
    [data-testid="stCameraInput"] p,
    [data-testid="stCameraInput"] span,
    [data-testid="stCameraInput"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    [data-testid="stCameraInput"] button {
        background-color: #1F2937 !important;
        border: 1px solid #4B5563 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* ── 5. CHAT INPUT (Sleek Dark Background + Crisp White Text) ── */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9CA3AF !important;
        -webkit-text-fill-color: #9CA3AF !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* ── 6. TABS & EXPANDERS (General) ───────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid #E5E7EB !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4B5563 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2E7D32 !important;
        font-weight: 700 !important;
        border-bottom-color: #2E7D32 !important;
    }

    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] * {
        color: #1F2937 !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #2E7D32 !important;
    }

    /* ── 7. CLOSE SIGN / RESET BUTTON (Image Preview Corner) ─── */
    .close-sign-btn .stButton > button {
        background-color: #111827 !important;
        color: #F87171 !important;
        -webkit-text-fill-color: #F87171 !important;
        border: 1px solid #7F1D1D !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 5px 12px !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    .close-sign-btn .stButton > button * {
        color: #F87171 !important;
        -webkit-text-fill-color: #F87171 !important;
    }
    .close-sign-btn .stButton > button:hover {
        background-color: #7F1D1D !important;
        border-color: #EF4444 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    .close-sign-btn .stButton > button:hover * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

</style>
""", unsafe_allow_html=True)


# ── Cached Model & Chatbot Loaders ───────────────────────────
@st.cache_resource(show_spinner=False)
def get_vision_pipeline() -> CropVisionPipeline:
    """Load vision models into memory once."""
    pipeline = CropVisionPipeline()
    pipeline.load_models()
    return pipeline

@st.cache_resource(show_spinner=False)
def get_chatbot_instance(api_key: str):
    """Load RAG chatbot engine with local FAISS index."""
    bot = FarmerChatbot(api_key=api_key, model_name="gemini-3.5-flash")
    bot.initialize_knowledge_base("tomato_potato_farmer_chatbot_master_guide.pdf")
    return bot


# ── Configuration (Loaded directly from environment) ────────
api_key = os.getenv("GEMINI_API_KEY", "")
leaf_threshold = 0.5


# ── Top Banner Header ───────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="header-title">🌿 CropCare AI — Smart Crop Diagnostic & Advisor</div>
    <div class="header-subtitle">
        Intelligent dual-stage vision pipeline: Verifies genuine crop leaves, identifies diseases with deep learning, and provides grounded farmer advice from the official Master Guide.
    </div>
    <div class="badge-container">
        <span class="badge badge-green">🌱 Stage 1: Leaf Detector v3</span>
        <span class="badge badge-blue">🔬 Stage 2: Disease Identifier (7 Classes)</span>
        <span class="badge badge-purple">🤖 Stage 3: Gemini 3.5 RAG Guide Chatbot</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Initialize Session Chat State ───────────────────────────
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am CropCare AI. Upload a crop photo on the left for instant leaf and disease detection, or ask me any farming question directly!",
            "sources": []
        }
    ]

if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None


# ── Two-Column Main Layout ──────────────────────────────────
col_vision, col_chat = st.columns([1, 1], gap="large")

# ============================================================
# LEFT COLUMN: 📷 Leaf Detection & Disease Diagnostic
# ============================================================
with col_vision:
    st.subheader("📷 Step 1 & 2: Crop Photo Inspection")

    # Image input method tabs
    tab_upload, tab_camera = st.tabs(["📁 Upload Image File", "📸 Use Camera"])
    
    input_image = None
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload leaf photo (JPG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key=f"leaf_uploader_{st.session_state.uploader_key}"
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

    with tab_camera:
        camera_file = st.camera_input(
            "Capture photo using webcam:",
            label_visibility="collapsed",
            key=f"leaf_camera_{st.session_state.uploader_key}"
        )
        if camera_file is not None:
            input_image = Image.open(camera_file)

    if input_image is not None:
        # Header bar with corner Close / Reset sign button
        col_img_lbl, col_img_btn = st.columns([3, 1])
        with col_img_lbl:
            st.markdown(
                "<div style='font-size: 0.95rem; font-weight: 700; color: #1F2937; padding-top: 8px;'>"
                "📸 Crop Photo Selected"
                "</div>",
                unsafe_allow_html=True
            )
        with col_img_btn:
            st.markdown('<div class="close-sign-btn">', unsafe_allow_html=True)
            if st.button("✕ Remove", key="btn_remove_picture", use_container_width=True, help="Remove picture and reset diagnosis"):
                st.session_state.uploader_key += 1
                st.session_state.last_diagnosis = None
                st.session_state.pdf_report = None
                st.session_state.pdf_report_key = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Display image preview
        st.image(input_image, use_container_width=True)

        # Run Dual-Model Pipeline
        with st.spinner("Analyzing image through dual vision models..."):
            try:
                pipeline = get_vision_pipeline()

                # ── STAGE 1: Leaf Detection ──────────────────────
                is_leaf, leaf_prob, raw_score = pipeline.check_is_leaf(input_image, threshold=leaf_threshold)
                leaf_percentage = round(leaf_prob * 100, 1)

                if not is_leaf:
                    # ⚠️ Image is NOT a leaf
                    st.markdown(f"""
                    <div class="alert-warning-card">
                        <h4 style="color: #B45309; margin: 0 0 8px 0;">⚠️ Please capture or upload a leaf image</h4>
                        <p style="margin: 0; color: #78350F; font-size: 0.95rem;">
                            Our leaf detector model determined that this photo does <b>not</b> clearly show a plant leaf (Leaf Score: <b>{leaf_percentage}%</b>).
                        </p>
                        <hr style="border: 0; border-top: 1px solid #FDE68A; margin: 12px 0;">
                        <p style="margin: 0; font-size: 0.88rem; color: #92400E;">
                            <b>Helpful Tips for Farmers:</b><br>
                            • Hold the camera close to the affected plant leaf.<br>
                            • Ensure clear daylight lighting and avoid heavy blur or background clutter.<br>
                            • Keep the leaf centered in the frame.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.last_diagnosis = None

                else:
                    # ✅ Image IS a leaf -> Proceed to Stage 2
                    st.markdown(f"""
                    <div class="alert-success-card">
                        <b style="color: #15803D;">✅ Leaf Verified!</b> 
                        <span style="color: #166534; font-size: 0.9rem;">(Confidence: <b>{leaf_percentage}%</b>)</span><br>
                        <span style="color: #14532D; font-size: 0.85rem;">Proceeding to Disease Identification model...</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── STAGE 2: Disease Identification ──────────
                    diag = pipeline.predict_disease(input_image)
                    st.session_state.last_diagnosis = diag

                    # Display Diagnosis Card
                    st.markdown(f"""
                    <div class="clean-card" style="border-top: 4px solid {diag['severity_color']};">
                        <div style="font-size: 0.85rem; color: #6B7280; font-weight: 600; text-transform: uppercase;">
                            {diag['crop']} Diagnosis
                        </div>
                        <div class="diagnosis-result-title">{diag['disease_name']}</div>
                        <div style="color: #4B5563; font-style: italic; font-size: 0.88rem; margin-bottom: 8px;">
                            {diag['scientific_name']}
                        </div>
                        <span class="severity-pill" style="background-color: {diag['severity_color']};">
                            Severity: {diag['severity']}
                        </span>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #1F2937; margin: 6px 0 10px 0;">
                            Prediction Confidence: <span style="color: #2E7D32;">{diag['confidence']}%</span>
                        </div>
                        <p style="color: #374151; font-size: 0.92rem; line-height: 1.4; margin: 0;">
                            {diag['summary']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Expandable: Probability Breakdown
                    with st.expander("📊 View Detailed Probabilities (All 7 Classes)"):
                        for item in diag["breakdown"]:
                            st.write(f"**{item['class_name']}**: {item['percentage']}%")
                            st.progress(item["probability"])

                    # ── STAGE 3: AUTOMATED PDF REPORT BY CHATBOT ──
                    diag_key = f"{diag['crop']}_{diag['disease_name']}_{diag['confidence']}"
                    
                    if st.session_state.get("pdf_report_key") != diag_key:
                        with st.spinner("🤖 Chatbot is analyzing Master Guide and generating complete PDF advisory report..."):
                            try:
                                if api_key and is_vector_store_ready():
                                    bot = get_chatbot_instance(api_key)
                                    adv_query = (
                                        f"A farmer leaf is diagnosed with {diag['crop']} {diag['disease_name']} "
                                        f"({diag['scientific_name']}) at {diag['confidence']}% confidence. "
                                        f"From the official Farmer Master Guide, generate a comprehensive agronomic action plan with sections:\n"
                                        f"### 1. Visual Symptoms & Rapid Field Identification\n"
                                        f"### 2. Urgent Actions for Today\n"
                                        f"### 3. Cultural & Sanitation Practices (Watering, Pruning, Spacing)\n"
                                        f"### 4. Recommended Treatments from Master Guide (Organic & Chemical Sprays)\n"
                                        f"### 5. Spread Prevention & Neighboring Plant Protection\n"
                                        f"Provide direct, practical, and clear steps."
                                    )
                                    report_res = bot.ask(adv_query, top_k=4)
                                    adv_text = report_res["answer"]
                                    adv_sources = report_res["sources"]
                                else:
                                    adv_text = f"Diagnosis for {diag['crop']} {diag['disease_name']}.\n\n{diag['summary']}"
                                    adv_sources = []

                                pdf_bytes = generate_disease_advisory_pdf(
                                    diagnosis=diag,
                                    advisory_text=adv_text,
                                    sources=adv_sources,
                                    leaf_image=input_image
                                )

                                st.session_state.pdf_report = {
                                    "key": diag_key,
                                    "bytes": pdf_bytes,
                                    "advisory": adv_text,
                                    "sources": adv_sources,
                                    "filename": f"CropCare_{diag['crop']}_{diag['disease_name'].replace(' ', '_')}_Advisory.pdf"
                                }
                                st.session_state.pdf_report_key = diag_key
                            except Exception as pdf_err:
                                st.warning(f"Could not generate AI PDF advisory automatically: {pdf_err}")
                                st.session_state.pdf_report = None

                    # Display PDF Report Card & Download Button
                    if st.session_state.get("pdf_report"):
                        rep = st.session_state.pdf_report
                        st.markdown("""
                        <div style="background-color: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 16px 18px; margin-top: 14px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                                <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">
                                    📄 Farmer Advisory PDF Report Ready
                                </div>
                                <span style="background-color: #064E3B; color: #6EE7B7; border: 1px solid #059669; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600;">
                                    ✓ Generated by Chatbot
                                </span>
                            </div>
                            <p style="color: #9CA3AF; font-size: 0.85rem; margin: 0 0 12px 0;">
                                Complete management guide generated from the Farmer Master Guide including symptoms, immediate first steps, organic & chemical treatments, and page citations.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.download_button(
                            label="📥 Download Full Farmer Advisory PDF Report",
                            data=rep["bytes"],
                            file_name=rep["filename"],
                            mime="application/pdf",
                            use_container_width=True
                        )

                        with st.expander("📖 Read Generated Advisory on Screen"):
                            st.markdown(rep["advisory"])
                            if rep.get("sources"):
                                st.markdown("---")
                                st.markdown("**Master Guide Page References Used:**")
                                for s in rep["sources"]:
                                    st.caption(f"&bull; **Page {s['page']}**: {s['content'][:140]}...")

            except Exception as e:
                st.error(f"Error during vision inspection: {e}")

    else:
        # Placeholder card when no image is uploaded
        st.markdown("""
        <div class="clean-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🌱</div>
            <h4 style="color: #374151; margin-bottom: 6px;">No Image Uploaded Yet</h4>
            <p style="color: #6B7280; font-size: 0.9rem; max-width: 380px; margin: 0 auto;">
                Upload a tomato or potato leaf photo or snap a picture with your camera above to start automated health diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RIGHT COLUMN: 🤖 RAG Chatbot & Agricultural Care Advice
# ============================================================
with col_chat:
    chat_hcol1, chat_hcol2 = st.columns([3, 1])
    with chat_hcol1:
        st.subheader("💬 Step 3: Farmer Advisory & Chatbot")
    with chat_hcol2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I am CropCare AI. Upload a crop photo on the left for instant leaf and disease detection, or ask me any question directly below!",
                    "sources": []
                }
            ]
            st.rerun()

    # Active Photo Diagnosis Banner (Chatbot remembers this photo until removed)
    diag_context = st.session_state.last_diagnosis
    if diag_context:
        if not diag_context.get("is_healthy", False):
            st.markdown(f"""
            <div style="background-color: #111827; border: 1px solid #374151; border-left: 4px solid {diag_context['severity_color']}; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                <b style="color: #FFFFFF;">🎯 Active Photo Context:</b> 
                <span style="color: #FBBF24; font-weight: 600;">{diag_context['crop']} &mdash; {diag_context['disease_name']}</span> 
                <span style="color: #9CA3AF; font-size: 0.85rem;">({diag_context['confidence']}% confidence)</span><br>
                <span style="color: #D1D5DB; font-size: 0.82rem;">The chatbot is specifically focused on this diagnosed leaf until you click Remove.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #111827; border: 1px solid #374151; border-left: 4px solid #10B981; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
                <b style="color: #FFFFFF;">🎯 Active Photo Context:</b> 
                <span style="color: #34D399; font-weight: 600;">{diag_context['crop']} &mdash; Healthy Foliage</span><br>
                <span style="color: #D1D5DB; font-size: 0.82rem;">The chatbot is currently focused on keeping your {diag_context['crop']} healthy and disease-free.</span>
            </div>
            """, unsafe_allow_html=True)

    # Display Chat Messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander("📖 Sources from Master Guide (Page References)"):
                        for idx, src in enumerate(msg["sources"], start=1):
                            st.markdown(f"**Source {idx} — Page {src['page']}** ({src['source']})")
                            st.caption(f"> {src['content']}")

    # Quick prompt buttons
    st.write("**Quick Questions:**")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    quick_selected = None

    if diag_context and not diag_context.get("is_healthy", False):
        # Disease-specific quick prompts
        d_name = diag_context["disease_name"]
        if quick_col1.button(f"💊 Treatment for {d_name}", use_container_width=True):
            quick_selected = f"What are the recommended cultural and safe treatment steps for {d_name}?"
        if quick_col2.button("🌿 Organic Alternatives", use_container_width=True):
            quick_selected = f"Are there organic or non-chemical options to manage {d_name}?"
        if quick_col3.button("⚠️ Will It Spread?", use_container_width=True):
            quick_selected = f"How fast does {d_name} spread, and what weather makes it worse?"
    else:
        # General quick prompts
        if quick_col1.button("🥔 Late Blight Urgency", use_container_width=True):
            quick_selected = "What are the urgent first steps for late blight on potatoes?"
        if quick_col2.button("🍅 Early Blight Target Rings", use_container_width=True):
            quick_selected = "How do I recognize early blight target spots on tomato leaves?"
        if quick_col3.button("🦠 Mosaic Virus Spread", use_container_width=True):
            quick_selected = "How does Tomato Mosaic Virus spread and can it be cured?"

    # Chat Input Box with dynamic context awareness
    if diag_context:
        d_label = diag_context['disease_name'] if not diag_context.get("is_healthy", False) else "Healthy"
        input_prompt_txt = f"Ask anything about your {diag_context['crop']} ({d_label}) e.g. 'What spray to use?', 'Will it spread?'..."
    else:
        input_prompt_txt = "Type your question here (e.g., 'What spray should I use?' or 'How do I protect nearby plants?')..."

    chat_input = st.chat_input(input_prompt_txt)
    question_to_ask = quick_selected or chat_input

    if question_to_ask:
        if not api_key:
            st.error("Please configure your GEMINI_API_KEY in the .env file to enable the Chatbot.")
        elif not is_vector_store_ready():
            st.error("Vector store is not ready. Please ensure FAISS index is built in data/vector_store.")
        else:
            # Append User Message
            st.session_state.messages.append({"role": "user", "content": question_to_ask, "sources": []})
            with st.chat_message("user"):
                st.write(question_to_ask)

            # Generate Assistant Response with Active Photo Context
            with st.chat_message("assistant"):
                with st.spinner("Searching Master Guide and generating advice..."):
                    try:
                        bot = get_chatbot_instance(api_key)
                        result = bot.ask(
                            question_to_ask,
                            top_k=3,
                            diagnosis_context=st.session_state.last_diagnosis
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        st.write(answer)

                        if sources:
                            with st.expander("📖 Sources from Master Guide (Page References)"):
                                for idx, src in enumerate(sources, start=1):
                                    st.markdown(f"**Source {idx} — Page {src['page']}** ({src['source']})")
                                    st.caption(f"> {src['content']}")

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    except Exception as err:
                        st.error(f"Error generating advice: {err}")
