# 🌿 CropCare AI — Smart Crop Diagnostic & Agronomic Advisor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Keras%20%2F%20TensorFlow-2.16%2B-orange.svg)](https://keras.io/)
[![LLM & RAG](https://img.shields.io/badge/LangChain%20%2B%20Gemini-Flash-4285F4.svg)](https://ai.google.dev/)
[![Vector Store](https://img.shields.io/badge/FAISS-Vector%20Search-black.svg)](https://github.com/facebookresearch/faiss)

An intelligent, grounded agricultural decision-support system designed specifically for **Tomato and Potato** crops. CropCare AI combines a **dual-stage deep learning vision pipeline**, an **automated PDF agronomic advisory report generator**, and a **RAG-powered conversational advisor** grounded strictly in official agronomic literature.

---

## 🌟 Key Features

### 1. 🔍 Dual-Stage Deep Learning Vision Pipeline
* **Stage 1 — Leaf Verification (`Model/Leaf_Detector_v3.keras`)**:
  * Evaluates uploaded photos before classification to verify the presence of a genuine plant leaf.
  * Rejects irrelevant images (e.g., people, indoor objects, vehicles) and prompts the user to capture a leaf.
* **Stage 2 — Disease Classification (`Model/Tomato_Potato_Disease_Model.keras`)**:
  * Accurate multi-class identification across **7 distinct health categories**:
    * **Potato Early Blight** (*Alternaria solani*)
    * **Potato Late Blight** (*Phytophthora infestans*)
    * **Potato Healthy** (*Solanum tuberosum*)
    * **Tomato Early Blight** (*Alternaria solani*)
    * **Tomato Late Blight** (*Phytophthora infestans*)
    * **Tomato Mosaic Virus** (*Tobamovirus*)
    * **Tomato Healthy** (*Solanum lycopersicum*)

### 2. 🤖 RAG-Grounded Farmer Chatbot
* Powered by **Google Gemini** embeddings (`models/gemini-embedding-001`) and LLM via **LangChain**.
* Backed by a high-performance **FAISS vector store** indexed from the included **Tomato & Potato Farmer Master Guide**.
* Answers agronomic queries with citations and page references from the source literature.

### 3. 📄 Automated PDF Advisory Reports
* Built with **ReportLab** to generate publication-grade agronomic advisory sheets.
* Includes diagnosis summary, severity metrics, thumbnail of the analyzed leaf, recommended treatments (chemical & organic), and preventive cultural practices.

### 4. 💻 Interactive User Interfaces
* **Modern Web Dashboard (`app.py`)**: Built with Streamlit featuring a light color theme, real-time image diagnostic pipeline, interactive chatbot, and instant PDF download.
* **Terminal CLI (`chatbot.py`)**: Direct conversational interface for command-line users.

---

## 📂 Repository Structure

```
ML_PROJECT/
├── Model/                                          # Trained deep learning models
│   ├── Leaf_Detector_v3.keras                     # Model 1: Binary leaf verifier (~49 MB)
│   └── Tomato_Potato_Disease_Model.keras          # Model 2: 7-class disease classifier (~31 MB)
│
├── chatbot/                                       # RAG engine & vector search modules
│   ├── __init__.py                                # Package marker
│   ├── step1_pdf_loader.py                        # Document loader & chunker (pdfplumber)
│   ├── step2_vector_store.py                      # FAISS embedding generator & loader
│   └── step3_chatbot.py                           # LangChain RAG pipeline & Gemini chat
│
├── data/
│   └── vector_store/                              # Local FAISS index files
│       ├── index.faiss                            # Vector similarity index (~1 MB)
│       └── index.pkl                              # Document chunk metadata (~60 KB)
│
├── .streamlit/
│   └── config.toml                                # Theme and server configuration
│
├── app.py                                         # Streamlit web application
├── predictor.py                                   # Vision inference pipeline wrapper
├── report_generator.py                            # ReportLab PDF advisory generator
├── chatbot.py                                     # Command-line chatbot runner
├── tomato_potato_farmer_chatbot_master_guide.pdf  # Source agronomic knowledge guide
├── requirements.txt                               # Python dependencies
├── .env.example                                   # Environment variable template
├── .gitignore                                     # Excluded files & secrets
└── README.md                                      # Documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
* **Python 3.9 – 3.11** installed on your system.
* A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1. Clone the Repository
```bash
git clone https://github.com/samir0510/ML_PROJECT.git
cd ML_PROJECT
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
```
Open `.env` and set:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 🚀 How to Run

### Option A: Launch the Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.
* Upload or snap a leaf photo.
* Inspect verification status, diagnosis, and severity breakdown.
* Download a comprehensive PDF advisory report.
* Chat with the AI advisor grounded in agronomic data.

### Option B: Run the Command-Line Chatbot
```bash
python chatbot.py
```
Interact directly with the master-guide grounded RAG assistant in your terminal. Type `exit` or `quit` to end the session.

### Option C: Rebuild the FAISS Vector Index (Optional)
If you update or replace `tomato_potato_farmer_chatbot_master_guide.pdf`:
```bash
python -m chatbot.step2_vector_store
```

---

## 📦 Large File & GitHub Considerations

* **`Model/Leaf_Detector_v3.keras` (~51.2 MB)**:
  * Exceeds GitHub's 50 MB soft warning threshold, but is comfortably below GitHub's 100 MB hard rejection limit. Standard `git push` will succeed with an advisory warning.
  * For strict enterprise repository practices, [Git LFS](https://git-lfs.github.com/) can optionally be enabled (`git lfs track "*.keras"`).
* **`Model/Tomato_Potato_Disease_Model.keras` (~32.9 MB)**:
  * Below the 50 MB warning threshold and safe for standard Git tracking.
* **Vector Store (`data/vector_store/`) (~1.06 MB)**:
  * Included to allow instant plug-and-play chatbot functionality without spending user API quota during initial setup.

---

## 🔒 Security Notice
* **Never commit `.env` or API credentials** to version control. The repository's `.gitignore` explicitly excludes `.env`, secret files, virtual environments, and local caches.
* Use `.env.example` as a template for public reference.

---

## 📄 License
This project is licensed under the MIT License — see the repository for details.
