"""CropCare AI — Smart Crop Diagnostic & Advisor."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

# Ensure safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import streamlit as st
from predictor import CropVisionPipeline
from report_generator import generate_pdf_report
from chatbot import init_chatbot_state, render_chatbot_ui

# Page configuration
st.set_page_config(
    page_title="CropCare AI — Smart Crop Diagnostic & Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme styling
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
        font-size: 2.1rem;
        font-weight: 800;
        color: #1B5E20 !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 0.96rem;
        color: #4B5563 !important;
        margin-top: 6px;
        margin-bottom: 12px;
        line-height: 1.45;
    }
    
    /* Model Status Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
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

    /* Section Subheaders */
    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
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
        font-size: 1.55rem;
        font-weight: 700;
        color: #111827 !important;
        margin-bottom: 4px;
    }

    /* Empty state upload box */
    .empty-state-card {
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        background: #FFFFFF;
        padding: 45px 24px;
        text-align: center;
        margin-top: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    }
    .empty-state-icon {
        font-size: 2.8rem;
        margin-bottom: 12px;
    }
    .empty-state-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 8px;
    }
    .empty-state-text {
        font-size: 0.92rem;
        color: #6B7280;
        max-width: 380px;
        margin: 0 auto;
        line-height: 1.45;
    }

    /* ============================================================
       SLEEK BLACK / DARK COMPONENTS WITH CRISP WHITE TEXT
       ============================================================ */

    /* 1. CHAT MESSAGES */
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

    /* 2. BUTTONS (Sleek Dark + White Text + Green Hover) */
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
    .stButton > button:hover *,
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div,
    .stButton > button:hover [data-testid="stMarkdownContainer"] p {
        color: #4ADE80 !important;
        -webkit-text-fill-color: #4ADE80 !important;
    }

    /* 3. FILE UPLOADER (Sleek Dark Background + Crisp White Text) */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
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
    [data-testid="stFileUploaderDropzone"] button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #2D3748 !important;
        border-color: #4ADE80 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover * {
        color: #4ADE80 !important;
        -webkit-text-fill-color: #4ADE80 !important;
    }

    /* 4. CAMERA INPUT */
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

    /* 5. CHAT INPUT */
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

    /* 6. TABS */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 2px solid #E5E7EB !important;
        gap: 14px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4B5563 !important;
        font-weight: 600 !important;
        padding-bottom: 8px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2E7D32 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #2E7D32 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 10px !important;
        margin-top: 10px !important;
    }
    [data-testid="stExpander"] * {
        color: #1F2937 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #2E7D32 !important;
    }

    /* Reset button in image preview */
    .reset-btn .stButton > button {
        background-color: #111827 !important;
        color: #F87171 !important;
        -webkit-text-fill-color: #F87171 !important;
        border: 1px solid #7F1D1D !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 4px 10px !important;
        font-size: 0.85rem !important;
    }
    .reset-btn .stButton > button:hover {
        background-color: #7F1D1D !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* Quick Question title */
    .quick-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #374151;
        margin-top: 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Load vision pipeline
@st.cache_resource(show_spinner=False)
def get_vision_pipeline() -> CropVisionPipeline:
    """Load vision models into memory once."""
    pipeline = CropVisionPipeline()
    pipeline.load_models()
    return pipeline


# Header banner
st.markdown("""
<div class="header-card">
    <div class="header-title">🌱 CropCare AI — Smart Crop Diagnostic & Advisor</div>
    <div class="header-subtitle">
        Intelligent dual-stage vision pipeline: Verifies genuine crop leaves, identifies diseases with deep learning, and provides grounded farmer advice from the official Master Guide.
    </div>
    <div class="badge-container">
        <span class="badge badge-green">🌱 Stage 1: Leaf Detector v3</span>
        <span class="badge badge-blue">🔬 Stage 2: Auto Crop & Disease Identifier</span>
        <span class="badge badge-purple">💬 Stage 3: Gemini 3.5 RAG Guide Chatbot</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Session state setup
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "sample_image_path" not in st.session_state:
    st.session_state.sample_image_path = None

init_chatbot_state()

if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None

# Two-column layout
col_vision, col_chat = st.columns([1, 1], gap="large")

# Left column: Crop photo inspection
with col_vision:
    st.markdown('<div class="section-title">📷 Step 1 & 2: Crop Photo Inspection</div>', unsafe_allow_html=True)

    # 3 Input Methods Tabs
    tab_upload, tab_camera, tab_samples = st.tabs([
        "📁 Upload Image File",
        "📷 Use Camera",
        "🖼️ Sample Leaf Photos"
    ])

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
            st.session_state.sample_image_path = None

    with tab_camera:
        camera_file = st.camera_input(
            "Capture photo using webcam:",
            label_visibility="collapsed",
            key=f"leaf_camera_{st.session_state.uploader_key}"
        )
        if camera_file is not None:
            input_image = Image.open(camera_file)
            st.session_state.sample_image_path = None

    with tab_samples:
        st.markdown("<p style='font-size:0.88rem; color:#4B5563; margin-bottom:10px;'>Select any verified leaf image to test immediate diagnosis:</p>", unsafe_allow_html=True)
        samples_dir = os.path.join(os.path.dirname(__file__), "sample_leaf_images")
        sample_items = [
            ("potato_early_blight_sample.jpg", "🥔 Potato Early Blight (New Photo)"),
            ("potato_late_blight_sample.jpg", "🥔 Potato Late Blight (Photo 1)"),
            ("Tomato_Late_Blight_User_Sample.jpg", "🍅 Tomato Late Blight (User Photo)"),
            ("Tomato_Early_Blight.png", "🍅 Tomato Early Blight"),
            ("Tomato_Mosaic_Virus.png", "🍅 Tomato Mosaic Virus"),
            ("Tomato_Healthy.png", "🍅 Tomato Healthy"),
            ("Potato_Late_Blight.png", "🥔 Potato Late Blight"),
            ("Potato_Early_Blight.png", "🥔 Potato Early Blight"),
            ("test_sample_potato_healthy.png", "🥔 Potato Healthy"),
        ]
        grid_cols = st.columns(2)
        for s_idx, (s_file, s_label) in enumerate(sample_items):
            s_path = os.path.join(samples_dir, s_file)
            if os.path.exists(s_path):
                with grid_cols[s_idx % 2]:
                    if st.button(f"{s_label}", key=f"btn_sample_{s_file}", use_container_width=True):
                        st.session_state.sample_image_path = s_path
                        st.session_state.last_diagnosis = None
                        st.rerun()

    if input_image is None and st.session_state.sample_image_path and os.path.exists(st.session_state.sample_image_path):
        input_image = Image.open(st.session_state.sample_image_path)

    # Image display and diagnosis
    if input_image is not None:
        # Header bar with Reset / Remove button
        col_img_lbl, col_img_btn = st.columns([3, 1])
        with col_img_lbl:
            st.markdown("<p style='font-weight:700; color:#1F2937; margin:6px 0 0 0;'>Inspected Leaf Photo:</p>", unsafe_allow_html=True)
        with col_img_btn:
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            if st.button("✕ Remove", key="btn_remove_photo", use_container_width=True):
                st.session_state.sample_image_path = None
                st.session_state.uploader_key += 1
                st.session_state.last_diagnosis = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.image(input_image, use_container_width=True)

        # Dual-stage vision pipeline
        with st.spinner("Analyzing image through dual-stage vision pipeline..."):
            pipeline = get_vision_pipeline()

            # STAGE 1: Leaf Verification
            is_leaf, leaf_conf, raw_pred = pipeline.check_is_leaf(input_image, threshold=0.5)
            leaf_percentage = round(leaf_conf * 100, 1)

            if not is_leaf:
                st.markdown(f"""
                <div class="alert-warning-card">
                    <b style="color: #92400E; font-size: 1.05rem;">⚠️ Non-Leaf Image Detected</b><br>
                    <p style="margin: 6px 0; color: #78350F; font-size: 0.95rem;">
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
                st.markdown(f"""
                <div class="alert-success-card">
                    <b style="color: #15803D;">✅ Leaf Verified!</b> 
                    <span style="color: #166534; font-size: 0.9rem;">(Confidence: <b>{leaf_percentage}%</b>)</span><br>
                    <span style="color: #14532D; font-size: 0.85rem;">Automatically identifying crop species & disease...</span>
                </div>
                """, unsafe_allow_html=True)

                # STAGE 2: Automated Disease Identification
                diag = pipeline.predict_disease(input_image, crop_filter=None)
                st.session_state.last_diagnosis = diag

                # Diagnosis Card
                st.markdown(f"""
                <div class="clean-card" style="border-top: 4px solid {diag['severity_color']};">
                    <div class="diagnosis-result-title">{diag['crop']} — {diag['disease_name']}</div>
                    <div style="color: #4B5563; font-style: italic; font-size: 0.88rem; margin-bottom: 8px;">
                        {diag['scientific_name']}
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #1F2937; margin: 6px 0 10px 0;">
                        Disease Confidence: <span style="color: #2E7D32;">{diag['confidence']}%</span>
                    </div>
                    <p style="color: #374151; font-size: 0.92rem; line-height: 1.45; margin: 0;">
                        {diag['summary']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Expandable Probability Breakdown
                with st.expander("📊 View Detailed Probabilities (All 7 Classes)"):
                    for item in diag["breakdown"]:
                        st.write(f"**{item['class_name']}**: {item['percentage']}%")
                        st.progress(item["probability"])

                # Generate dynamic PDF report
                if "cached_pdf_report" not in st.session_state or st.session_state.get("cached_pdf_diag_class") != diag["class_name"]:
                    with st.spinner("🤖 Chatbot is analyzing Master Guide & compiling PDF report..."):
                        st.session_state.cached_pdf_report = generate_pdf_report(
                            diag, input_image, leaf_confidence=leaf_percentage
                        )
                        st.session_state.cached_pdf_diag_class = diag["class_name"]

                report_filename = f"CropCare_Diagnostic_Report_{diag['crop']}_{diag['disease_name'].replace(' ', '_')}.pdf"

                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="📄 Download AI Chatbot Diagnostic Report (PDF)",
                    data=st.session_state.cached_pdf_report,
                    file_name=report_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_download_report"
                )

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state-card">
            <div class="empty-state-icon">🌱</div>
            <div class="empty-state-title">No Image Uploaded Yet</div>
            <div class="empty-state-text">
                Upload a tomato or potato leaf photo or snap a picture with your camera above to start automated health diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# Right column: Farmer advisory & chatbot
with col_chat:
    render_chatbot_ui(current_diagnosis=st.session_state.last_diagnosis)
