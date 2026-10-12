# ============================================================
# STEP 3: RAG Chatbot (Gemini + Vector Store)
# ============================================================
# This file brings everything together:
#
#   User asks question
#          ↓
#   Find matching chunks in FAISS vector store (from Step 2)
#          ↓
#   Send both (Question + Chunks) to Google Gemini LLM
#          ↓
#   Gemini generates a clear, farmer-friendly answer
#          ↓
#   Display answer along with PDF source page references!
# ============================================================

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from .step1_pdf_loader import load_pdf_as_chunks
    from .step2_vector_store import build_vector_store, load_vector_store, is_vector_store_ready
except ImportError:
    from step1_pdf_loader import load_pdf_as_chunks
    from step2_vector_store import build_vector_store, load_vector_store, is_vector_store_ready


# ── Farmer-Friendly System Prompt ───────────────────────────
SYSTEM_PROMPT = """You are CropCare AI, a helpful, farmer-friendly agricultural assistant specializing in Tomato and Potato crops.

You have access to excerpts from the official Tomato and Potato Farmer Master Guide.

RULES FOR YOUR ANSWERS:
1. Use simple everyday language, short sentences, and bullet points where helpful.
2. Explain technical terms simply (for example, explain 'fungicide' as 'a spray used to prevent fungal rot').
3. Be honest: A photo or symptom description is a guide, not a 100% lab diagnosis.
4. Give actionable next steps:
   - What to check on the plant (leaves, stems, fruits/tubers).
   - Safe cultural practices (removing infected leaves, avoiding overhead watering, cleaning tools).
   - If chemical treatments are discussed, remind the farmer to verify with product labels and consult their local Krishi Vigyan Kendra (KVK) or agriculture officer.
5. If the user asks something outside tomato and potato crops or outside the guide context, politely explain what you know and suggest consulting a local farm specialist.

CONTEXT FROM FARMER GUIDE:
{context}

ACTIVE PHOTO & DIAGNOSIS CONTEXT:
{active_diagnosis}

FARMER'S QUESTION:
{question}

HELPFUL FARMER-FRIENDLY ANSWER:"""


class FarmerChatbot:
    """
    RAG-based Chatbot powered by LangChain and Google Gemini.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.vector_store = None

        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "[ERROR] GEMINI_API_KEY is not set!\n"
                "Please configure GEMINI_API_KEY in your .env file or pass it to FarmerChatbot(api_key=...)."
            )

        # Initialize the Gemini LLM with gemini-3.5-flash
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.3,
            max_retries=2,  # Prevent endless retries on rate limits
        )

        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    def initialize_knowledge_base(self, pdf_path: str, force_rebuild: bool = False):
        """
        Load vector store from disk or build it fresh from PDF.
        """
        if is_vector_store_ready() and not force_rebuild:
            print("[INFO] Loading existing vector store from disk...")
            self.vector_store = load_vector_store(api_key=self.api_key)
        else:
            print(f"[INFO] Processing PDF '{pdf_path}' and building vector store...")
            chunks = load_pdf_as_chunks(pdf_path)
            self.vector_store = build_vector_store(chunks, api_key=self.api_key)

    def ask(
        self,
        question: str,
        top_k: int = 4,
        diagnosis_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and receive an answer with source page references.

        Args:
            question          : Farmer's question
            top_k             : Number of relevant text chunks to retrieve (default 4)
            diagnosis_context : Active prediction from vision models (crop, disease, confidence, etc.)

        Returns:
            Dict containing:
              - 'answer': Gemini's generated answer string
              - 'sources': List of retrieved source excerpts and page numbers
        """
        if self.vector_store is None:
            raise RuntimeError("Knowledge base is not initialized! Call initialize_knowledge_base() first.")

        # Build retrieval query and diagnosis context string
        if diagnosis_context and not diagnosis_context.get("is_healthy", False):
            crop = diagnosis_context.get("crop", "")
            disease = diagnosis_context.get("disease_name", "")
            scientific = diagnosis_context.get("scientific_name", "")
            confidence = diagnosis_context.get("confidence", "")
            severity = diagnosis_context.get("severity", "")
            summary = diagnosis_context.get("summary", "")

            search_query = f"{crop} {disease} {question}"
            diag_str = (
                f"CURRENTLY INSPECTED LEAF PHOTO ON SCREEN:\n"
                f"- Crop: {crop}\n"
                f"- Diagnosed Disease: {disease} ({scientific})\n"
                f"- Model Confidence: {confidence}%\n"
                f"- Severity: {severity}\n"
                f"- Symptoms Summary: {summary}\n\n"
                f"CRITICAL INSTRUCTION: The farmer has this exact photo active on screen. "
                f"Answer their questions directly regarding this {crop} affected by {disease}. "
                f"Assume pronouns like 'it', 'my plant', 'this disease', or 'what spray' refer to this diagnosed {crop} {disease}."
            )
        elif diagnosis_context and diagnosis_context.get("is_healthy", False):
            crop = diagnosis_context.get("crop", "")
            search_query = f"{crop} healthy maintenance {question}"
            diag_str = (
                f"CURRENTLY INSPECTED LEAF PHOTO ON SCREEN:\n"
                f"- Crop: {crop}\n"
                f"- Condition: Healthy Foliage ({diagnosis_context.get('confidence', '')}% confidence)\n"
                f"The farmer's leaf is healthy. Advise them on keeping their {crop} crop healthy and disease-free."
            )
        else:
            search_query = question
            diag_str = "No active photo uploaded. The farmer is asking general crop questions."

        # Step 1: Retrieve top-k most relevant chunks from FAISS
        relevant_docs = self.vector_store.similarity_search(search_query, k=top_k)

        # Step 2: Combine chunks into a single context string
        context_parts = []
        source_details = []

        for doc in relevant_docs:
            page_num = doc.metadata.get("page", "Unknown")
            source_file = doc.metadata.get("source", "PDF Guide")
            text = doc.page_content.strip()

            context_parts.append(f"[Page {page_num}]\n{text}")
            source_details.append({
                "page": page_num,
                "source": source_file,
                "content": text,
            })

        combined_context = "\n\n---\n\n".join(context_parts)

        # Step 3: Format prompt and call Gemini
        formatted_prompt = self.prompt_template.format_messages(
            context=combined_context,
            active_diagnosis=diag_str,
            question=question
        )

        response = self.llm.invoke(formatted_prompt)

        # Parse text content safely
        raw_content = response.content
        if isinstance(raw_content, list):
            answer_text = "".join([c["text"] if isinstance(c, dict) and "text" in c else str(c) for c in raw_content])
        else:
            answer_text = str(raw_content)

        return {
            "answer": answer_text,
            "sources": source_details
        }


# ── Interactive Terminal Chat Loop ───────────────────────────────
def run_cli_chat():
    """
    Simple command-line interface to chat directly in your terminal.
    """
    print("=" * 60)
    print("  TOMATO & POTATO FARMER CHATBOT (RAG + GEMINI)")
    print("=" * 60)

    # Locate default PDF in project folder
    pdf_files = list(PROJECT_ROOT.glob("*.pdf"))
    if not pdf_files:
        print("[ERROR] No PDF found in project folder.")
        return

    pdf_path = str(pdf_files[0])

    try:
        bot = FarmerChatbot()
        bot.initialize_knowledge_base(pdf_path)

        print("\n[OK] Chatbot is ready! Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                user_query = input("Farmer Question > ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_query:
                continue

            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            print("\nSearching guide and generating answer...\n")
            result = bot.ask(user_query)

            print("=" * 60)
            print("CROPCARE ADVICE:")
            print("=" * 60)
            print(result["answer"])
            print("\n" + "-" * 60)
            print("GUIDE REFERENCES USED:")
            for s in result["sources"]:
                print(f" * Page {s['page']}: {s['content'][:120]}...")
            print("-" * 60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    run_cli_chat()
