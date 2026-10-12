# 🌿 CropCare AI — Smart Crop Diagnostic & Farmer Advisor

An intelligent, grounded agricultural decision support system for **Tomato and Potato** crops.

Combines a dual-stage deep learning vision pipeline, an automated PDF agronomic report generator, and a RAG agricultural chatbot grounded in the official **Farmer Master Guide**.

---

## 📂 Project Structure

```
ML Project/
│
├── Model/                                          ← Deep learning vision models
│   ├── Leaf_Detector_v3.keras                     ← Model 1: Leaf verification (filters non-leaf photos)
│   └── Tomato_Potato_Disease_Model.keras          ← Model 2: 7-class disease identification
│
├── chatbot/                                       ← RAG Chatbot engine
│   ├── __init__.py                                ← Package marker
│   ├── step1_pdf_loader.py                        ← Step 1: Chunk Master Guide PDF
│   ├── step2_vector_store.py                      ← Step 2: FAISS vector database
│   └── step3_chatbot.py                           ← Step 3: RAG answering logic
│
├── data/
│   └── vector_store/                              ← Saved FAISS index (index.faiss, index.pkl)
│
├── app.py                                         ← Streamlit web dashboard
├── predictor.py                                   ← Dual-model vision pipeline wrapper
├── report_generator.py                            ← Automated PDF advisory report generator
├── chatbot.py                                     ← Terminal CLI chatbot runner
├── tomato_potato_farmer_chatbot_master_guide.pdf  ← Source knowledge guide
├── requirements.txt                               ← Python dependencies
├── .env                                           ← Environment variables (Gemini API key)
└── README.md                                      ← Documentation
```

---

## 🚀 How to Run

### 1. Launch the Web Dashboard (Recommended)
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
- Upload or capture a plant leaf image.
- Review leaf verification score & disease diagnosis.
- Download the complete automated PDF advisory report.
- Chat with the advisor with continuous photo context awareness.

### 2. Run the Terminal Chatbot
```powershell
python chatbot.py
```
Type your farming questions directly in PowerShell. Type `exit` or `quit` to stop.
