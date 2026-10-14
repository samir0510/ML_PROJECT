# 🌱 CropCare AI — Smart Crop Diagnostic & Advisor

CropCare AI is an intelligent agricultural intelligence system for tomato and potato crops featuring a dual-stage deep learning vision pipeline, a grounded RAG chatbot powered by Gemini, and dynamic diagnostic PDF reporting.

---

## 🚀 Key Features

1. **Stage 1 — Leaf Detector (`Leaf_Detector_v3.keras`)**:
   - Verifies whether an uploaded image contains a genuine plant leaf.
   - Guardrails against non-leaf objects, synthetic surfaces, skin tones, and plain backgrounds.

2. **Stage 2 — Automated Crop & Disease Classifier (`Tomato_Potato_Disease_Model.keras`)**:
   - Classifies across 7 supported classes:
     - 🥔 Potato Early Blight (*Alternaria solani*)
     - 🥔 Potato Late Blight (*Phytophthora infestans*)
     - 🥔 Potato Healthy Foliage
     - 🍅 Tomato Early Blight (*Alternaria solani*)
     - 🍅 Tomato Late Blight (*Phytophthora infestans*)
     - 🍅 Tomato Mosaic Virus (*ToMV*)
     - 🍅 Tomato Healthy Foliage

3. **Stage 3 — Grounded RAG Chatbot (`chatbot.py`)**:
   - Grounded in the official **Tomato & Potato Farmer Master Guide PDF**.
   - Contextualizes advice with the live leaf diagnosis.
   - Provides clear, actionable steps on field sanitation, weather warnings, and registered treatments.

4. **Dynamic Diagnostic Report Generator (`report_generator.py`)**:
   - Dynamically writes publication-grade PDF diagnostic reports.
   - Includes **Weather & Microclimate Risk Matrix** (temperatures, relative humidity, leaf wetness thresholds, windborne spore dispersal).
   - Details **Follow-up Scouting Photo Schedules** (Section 10 of Master Guide).
   - Incorporates strict biosecurity rules (e.g., No-Compost protocol for Late Blight).

---

## 📂 Project Structure

```
├── app.py                                         # Streamlit UI application
├── chatbot.py                                     # Master Guide RAG Chatbot & PDF synthesis
├── predictor.py                                   # Dual-stage deep learning vision pipeline
├── report_generator.py                            # ReportLab PDF report generator
├── requirements.txt                               # Python dependencies
├── tomato_potato_farmer_chatbot_master_guide.pdf  # Official agronomic master guide
├── Model/                                         # Trained deep learning models
│   ├── Crop_Feature_Extractor.keras
│   ├── Leaf_Detector_v3.keras
│   ├── Leaf_Detector_v3.tflite
│   └── Tomato_Potato_Disease_Model.keras
├── sample_leaf_images/                            # Sample evaluation leaf photos
└── .streamlit/
    └── config.toml                                # Theme configuration
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hetu695055/ML-Project-CropCare-AI-.git
   cd ML-Project-CropCare-AI-
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
